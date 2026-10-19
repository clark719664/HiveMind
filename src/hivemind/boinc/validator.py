"""Validate worker checkpoints and bind the accepted delta to deterministic replay.

Optimizer/RNG snapshots are not part of the current wire format. Consequently the
safe fallback replays the complete prefix through the final checkpoint; it does
not claim the lower cost of independently replayable random segments.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from hivemind.boinc.workunit import BOINCWorkunit
from hivemind.core.compression import compress_lora_delta
from hivemind.core.diloco_inner import InnerLoopResult, replay_inner_segment
from hivemind.core.quantization import compute_state_dict_sha256, extract_lora_state_dict


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
    """Trusted replay verifier; the coordinator supplies the assigned initial state."""

    def __init__(self, tolerance_eps: float = 1e-4) -> None:
        if not 0 <= tolerance_eps < float("inf"):
            raise ValueError("Validation tolerance must be finite and nonnegative.")
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
        trusted_initial_state: dict[str, torch.Tensor] | None = None,
    ) -> SpotCheckVerdict:
        """Replay all completed steps and verify the exact compressed delta uploaded.

        Direct/local callers may omit ``trusted_initial_state`` to check internal
        consistency only. A coordinator must supply its own assigned state.
        ``validator_rng_seed`` remains accepted for API compatibility.
        """
        del validator_rng_seed
        selected_end = 0
        overhead_pct = 0.0
        max_diff = 0.0
        hash_matched = False
        replay_started = False

        def compare_states(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> float:
            if set(expected) != set(actual) or not expected:
                raise ValueError("Parameter names differ or the submitted state is empty.")
            largest = 0.0
            for name, expected_tensor in expected.items():
                actual_tensor = actual[name]
                if not isinstance(actual_tensor, torch.Tensor) or expected_tensor.shape != actual_tensor.shape:
                    raise ValueError(f"Invalid parameter shape for {name}.")
                a = expected_tensor.detach().to(device="cpu", dtype=torch.float32)
                b = actual_tensor.detach().to(device="cpu", dtype=torch.float32)
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError(f"Non-finite parameter values for {name}.")
                if a.numel():
                    largest = max(largest, float((a - b).abs().max().item()))
            return largest

        try:
            if submission.branch_name != workunit.branch_name or submission.outer_step != workunit.outer_step:
                raise ValueError("Submission does not match the assigned workunit.")
            steps = submission.steps_completed
            if type(steps) is not int or not 0 < steps <= workunit.inner_steps_H:
                raise ValueError("Invalid completed step count.")
            if steps != workunit.inner_steps_H and not submission.preempted_by_daemon:
                raise ValueError("Incomplete work requires a preemption record.")
            if workunit.spot_check_interval_s <= 0:
                raise ValueError("Invalid checkpoint interval.")
            required_steps = set(range(0, steps + 1, workunit.spot_check_interval_s)) | {steps}
            if set(submission.micro_checkpoints) != required_steps:
                raise ValueError("Missing or unexpected micro-checkpoints.")
            model_state = extract_lora_state_dict(validator_model)
            # Establish names, shapes and finite values without trusting current model values.
            compare_states(model_state, submission.theta_init)
            if trusted_initial_state is not None and compare_states(trusted_initial_state, submission.theta_init) != 0:
                raise ValueError("Initial state does not match the coordinator checkpoint.")
            for step, record in submission.micro_checkpoints.items():
                if record.step != step:
                    raise ValueError("Checkpoint step metadata mismatch.")
                compare_states(submission.theta_init, record.state_snapshot)
                if compute_state_dict_sha256(record.state_snapshot) != record.lora_sha256:
                    raise ValueError(f"Checkpoint hash mismatch at step {step}.")
            if compare_states(submission.theta_init, submission.micro_checkpoints[0].state_snapshot) != 0:
                raise ValueError("Checkpoint zero does not match theta_init.")
            if compare_states(submission.theta_final, submission.micro_checkpoints[steps].state_snapshot) != 0:
                raise ValueError("Final checkpoint does not match theta_final.")
            delta = {
                name: init.detach().float().cpu() - submission.theta_final[name].detach().float().cpu()
                for name, init in submission.theta_init.items()
            }
            if compare_states(delta, submission.raw_delta) != 0:
                raise ValueError("Raw delta does not match the initial and final states.")
            package = submission.compressed_delta
            expected_package = compress_lora_delta(
                delta, submission.worker_id, submission.branch_name, submission.outer_step,
                inner_steps_completed=steps, topk_ratio=package.topk_ratio,
            )
            for name in ("worker_id", "branch_name", "outer_step", "inner_steps_completed", "uncompressed_bytes", "compressed_bytes"):
                if getattr(package, name) != getattr(expected_package, name):
                    raise ValueError(f"Compressed delta metadata mismatch: {name}.")
            if set(package.tensors) != set(expected_package.tensors):
                raise ValueError("Compressed delta parameter names differ.")
            for name, expected in expected_package.tensors.items():
                actual = package.tensors[name]
                if (actual.original_shape != expected.original_shape or actual.original_numel != expected.original_numel
                        or actual.scale != expected.scale or actual.indices.dtype != expected.indices.dtype
                        or actual.fp8_values.dtype != expected.fp8_values.dtype
                        or not torch.equal(actual.indices.cpu(), expected.indices)
                        or not torch.equal(actual.fp8_values.cpu().view(torch.uint8), expected.fp8_values.view(torch.uint8))):
                    raise ValueError(f"Compressed delta payload mismatch: {name}.")

            selected_end = steps
            overhead_pct = round(100.0 * steps / workunit.inner_steps_H, 2)

            def verify_checkpoint(step: int) -> None:
                nonlocal max_diff
                if step in submission.micro_checkpoints:
                    divergence = compare_states(
                        extract_lora_state_dict(validator_model), submission.micro_checkpoints[step].state_snapshot,
                    )
                    max_diff = max(max_diff, divergence)

            replay_started = True
            expected_state = replay_inner_segment(
                model=validator_model,
                start_lora_state=trusted_initial_state if trusted_initial_state is not None else submission.theta_init,
                token_sequences=token_sequences,
                start_step=0, end_step=steps, lr=lr, weight_decay=weight_decay,
                rng_seed=workunit.rng_seed, micro_batch_size=micro_batch_size,
                step_callback=verify_checkpoint,
            )
            max_diff = max(max_diff, compare_states(expected_state, submission.theta_final))
            hash_matched = compute_state_dict_sha256(expected_state) == submission.micro_checkpoints[steps].lora_sha256
            passed = max_diff <= self.tolerance_eps
            reason = (
                f"{'Verified' if passed else 'REJECTED'} full replay [0, {steps}] "
                f"({overhead_pct}% replay overhead): max_diff={max_diff:.2e}, eps={self.tolerance_eps:.1e}"
            )
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
            if replay_started and isinstance(exc, RuntimeError):
                # A server/model failure is retryable, not evidence of worker fraud.
                raise
            passed = False
            max_diff = float("inf")
            reason = f"REJECTED malformed or inconsistent submission: {exc}"

        if not passed:
            self._flag_account(steam_id64, reason)
        return SpotCheckVerdict(
            wu_id=workunit.wu_id, worker_id=submission.worker_id, steam_id64=steam_id64,
            segment_start_step=0, segment_end_step=selected_end,
            verification_overhead_pct=overhead_pct, max_abs_divergence=round(max_diff, 8),
            tolerance_eps=self.tolerance_eps, hash_matched=hash_matched, passed=passed,
            steam_account_flagged=not passed, reason=reason,
        )

    def _flag_account(self, steam_id64: str, reason: str) -> None:
        self.flagged_steam_accounts.setdefault(steam_id64, []).append(reason)
