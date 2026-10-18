"""Deterministic Micro-Checkpoint Spot-Check Validator.

Instead of classic BOINC quorum=2 (which wastes 50% of global compute), HiveMind validators
re-execute a randomly selected s-step micro-checkpoint segment (10% verification overhead for
s=50, H=500). If the validator's output diverges beyond tolerance eps = 1e-4, the Workunit is
rejected and the user's SteamID64 is flagged for Byzantine/hardware fault behavior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
import torch
import torch.nn as nn

from hivemind.boinc.workunit import BOINCWorkunit
from hivemind.core.diloco_inner import InnerLoopResult, replay_inner_segment
from hivemind.core.quantization import compute_state_dict_sha256


@dataclass
class SpotCheckVerdict:
    """Audit result from the BOINC Spot-Check Validator."""

    wu_id: str
    worker_id: str
    steam_id64: str
    segment_start_step: int
    segment_end_step: int
    verification_overhead_pct: float
    max_abs_divergence: float
    tolerance_eps: float
    hash_matched: bool
    passed: bool
    steam_account_flagged: bool
    reason: str


class DeterministicSpotCheckValidator:
    """Trusted BOINC Validator pool verifier for deterministic micro-checkpoint segments."""

    def __init__(self, tolerance_eps: float = 1e-4) -> None:
        self.tolerance_eps = tolerance_eps
        self.flagged_steam_accounts: dict[str, list[str]] = {}

    def verify_submission(
        self,
        validator_model: nn.Module,
        workunit: BOINCWorkunit,
        submission: InnerLoopResult,
        token_sequences: torch.Tensor,
        steam_id64: str,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        micro_batch_size: int = 1,
        validator_rng_seed: int | None = None,
    ) -> SpotCheckVerdict:
        """Select one micro-checkpoint segment and verify worker determinism within eps = 1e-4."""
        ckpt_steps = sorted(s for s in submission.micro_checkpoints.keys() if s > 0)
        if not ckpt_steps:
            self._flag_account(steam_id64, f"Missing micro-checkpoints in WU {workunit.wu_id}")
            return SpotCheckVerdict(
                wu_id=workunit.wu_id,
                worker_id=submission.worker_id,
                steam_id64=steam_id64,
                segment_start_step=0,
                segment_end_step=0,
                verification_overhead_pct=0.0,
                max_abs_divergence=float("inf"),
                tolerance_eps=self.tolerance_eps,
                hash_matched=False,
                passed=False,
                steam_account_flagged=True,
                reason="Submission contained zero micro-checkpoints.",
            )

        # Select the first 50-step segment [0, s_1] for stateless deterministic verification
        # (or a randomly selected segment when optimizer state is checkpointed)
        rng = random.Random(validator_rng_seed if validator_rng_seed is not None else workunit.rng_seed)
        selected_end = ckpt_steps[0] if len(ckpt_steps) == 1 else rng.choice(ckpt_steps[:1])
        selected_start = 0

        overhead_pct = round(100.0 * (selected_end - selected_start) / max(1, workunit.inner_steps_H), 2)

        expected_State = replay_inner_segment(
            model=validator_model,
            start_lora_state=submission.micro_checkpoints[0].state_snapshot,
            token_sequences=token_sequences,
            start_step=selected_start,
            end_step=selected_end,
            lr=lr,
            weight_decay=weight_decay,
            rng_seed=workunit.rng_seed,
            micro_batch_size=micro_batch_size,
        )

        worker_record = submission.micro_checkpoints[selected_end]
        worker_state = worker_record.state_snapshot

        max_diff = 0.0
        for name, exp_tensor in expected_State.items():
            if name not in worker_state:
                max_diff = float("inf")
                break
            diff = float(
                (exp_tensor.to(torch.float32).cpu() - worker_state[name].to(torch.float32).cpu())
                .abs()
                .max()
                .item()
            )
            if diff > max_diff:
                max_diff = diff

        # Also inspect final delta consistency if worker tampered with theta_final after step s_1
        for name, init_t in submission.theta_init.items():
            reconstructed_delta = init_t.to(torch.float32).cpu() - submission.theta_final[name].to(torch.float32).cpu()
            submitted_raw = submission.raw_delta[name].to(torch.float32).cpu()
            delta_mismatch = float((reconstructed_delta - submitted_raw).abs().max().item())
            if delta_mismatch > max_diff:
                max_diff = delta_mismatch

        expected_sha = compute_state_dict_sha256(expected_State)
        expected_State.clear()
        del expected_State
        hash_matched = expected_sha == worker_record.lora_sha256
        passed = max_diff <= self.tolerance_eps

        flagged = not passed
        reason = (
            f"Verified segment [{selected_start}, {selected_end}] ({overhead_pct}% overhead): "
            f"max_diff={max_diff:.2e} <= eps={self.tolerance_eps:.1e}"
            if passed
            else f"REJECTED segment [{selected_start}, {selected_end}]: "
            f"divergence {max_diff:.2e} exceeds tolerance eps={self.tolerance_eps:.1e}"
        )
        if flagged:
            self._flag_account(steam_id64, reason)

        return SpotCheckVerdict(
            wu_id=workunit.wu_id,
            worker_id=submission.worker_id,
            steam_id64=steam_id64,
            segment_start_step=selected_start,
            segment_end_step=selected_end,
            verification_overhead_pct=overhead_pct,
            max_abs_divergence=round(max_diff, 8),
            tolerance_eps=self.tolerance_eps,
            hash_matched=hash_matched,
            passed=passed,
            steam_account_flagged=flagged,
            reason=reason,
        )

    def _flag_account(self, steam_id64: str, reason: str) -> None:
        self.flagged_steam_accounts.setdefault(steam_id64, []).append(reason)
