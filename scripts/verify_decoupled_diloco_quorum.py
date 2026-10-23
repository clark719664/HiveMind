"""Controlled check `H11`: Decoupled DiLoCo quorum, grace window, token weights and RDA.

Writes ``artifacts/decoupled_diloco_quorum_check.json``. Every number is computed in this
process from synthetic tensors; nothing is measured over a network. The check has a positive
arm (orthogonal equal-norm learner deltas, where RDA must restore the common norm `R` while
the direct average shrinks to `R / sqrt(M)`) and two negative controls (a straggler outside
the grace window must be excluded; a quorum of `K = M` with one missing learner must *not*
merge). A failing arm sets ``all_passed = false`` instead of being hidden.

Run: ``PYTHONPATH=src python scripts/verify_decoupled_diloco_quorum.py``
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivemind.swarm.decoupled_diloco import (  # noqa: E402
    DECOUPLED_DILOCO_REFERENCE,
    AdaptiveGraceWindowController,
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    direct_weighted_average,
    radial_directional_average,
)
from hivemind.swarm.diloco_nextgen import CPUOffloadedFlatBucketOuterState  # noqa: E402

ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "decoupled_diloco_quorum_check.json"


def _rda_vs_direct_norms(num_learners: int, dim: int, radius: float) -> Dict[str, float]:
    basis = torch.eye(dim)[:num_learners] * radius
    direct = direct_weighted_average(list(basis))
    rda = radial_directional_average(list(basis))
    return {
        "num_learners_m": float(num_learners),
        "common_norm_r": radius,
        "direct_average_norm": float(direct.norm()),
        "direct_average_predicted_r_over_sqrt_m": radius / math.sqrt(num_learners),
        "rda_norm": float(rda.norm()),
        "rda_over_direct_ratio": float(rda.norm() / direct.norm()),
    }


def run_check(seed: int = 2026) -> Dict[str, Any]:
    torch.manual_seed(seed)
    started = time.perf_counter()
    arms: List[Dict[str, Any]] = []

    # Arm 1 (positive): RDA restores R for M in {2, 4, 8, 16} orthogonal learners.
    rda_rows = [_rda_vs_direct_norms(m, 64, 3.0) for m in (2, 4, 8, 16)]
    rda_ok = all(
        math.isclose(r["rda_norm"], r["common_norm_r"], rel_tol=1e-5)
        and math.isclose(r["direct_average_norm"], r["direct_average_predicted_r_over_sqrt_m"], rel_tol=1e-5)
        for r in rda_rows
    )
    arms.append({"arm_id": "H11-A", "kind": "positive", "name": "RDA norm invariance vs direct-average shrinkage", "passed": rda_ok, "rows": rda_rows})

    # Arm 2 (negative control): straggler outside the grace window is excluded; its tokens are deferred.
    dim = 32
    global_prev = torch.zeros(dim)
    e = torch.eye(dim)
    deltas = {f"fast{i}": e[i] * 2.0 for i in range(6)}
    deltas["slow0"] = e[6] * 500.0
    deltas["slow1"] = e[7] * 500.0
    frags = {lid: global_prev - d for lid, d in deltas.items()}
    arrivals = [LearnerArrival(f"fast{i}", 0.1 * i, steps_since_sync=50, tokens_since_sync=5000) for i in range(6)]
    arrivals += [LearnerArrival("slow0", 25.0, 50, 5000), LearnerArrival("slow1", 40.0, 50, 5000)]
    ctrl = AdaptiveGraceWindowController(overlap_steps_tau=2, safety_margin_gamma=0.5, ema_beta=0.0, sync_duration_s=1.0)
    ctrl.observe_step_duration(4.0)
    ctrl.observe_quorum_duration(0.3)
    syncer = DecoupledDiLoCoQuorumSyncer(num_learners_m=8, min_quorum_k=4, merge_mode="rda", grace_controller=ctrl)
    delta, sel, tel = syncer.merge_fragment("w", global_prev, frags, arrivals)
    straggler_leak = float(delta[6].abs() + delta[7].abs()) if delta is not None else float("nan")
    arm2_ok = (
        delta is not None
        and sel.admitted_learner_ids == tuple(f"fast{i}" for i in range(6))
        and sel.deferred_learner_ids == ("slow0", "slow1")
        and straggler_leak == 0.0
        and math.isclose(tel.merged_delta_norm, 2.0, rel_tol=1e-5)
    )
    state = CPUOffloadedFlatBucketOuterState({"w": global_prev.clone()})
    updated, _ = state.outer_nesterov_step({"w": delta if delta is not None else torch.zeros(dim)}, lr=0.7, momentum=0.9)
    arms.append(
        {
            "arm_id": "H11-B",
            "kind": "negative_control",
            "name": "Late stragglers excluded by adaptive grace window; Delta feeds flat-bucket Nesterov step",
            "passed": arm2_ok,
            "grace_controller": ctrl.telemetry(),
            "admitted_learner_ids": list(sel.admitted_learner_ids),
            "deferred_learner_ids": list(sel.deferred_learner_ids),
            "admitted_tokens": tel.admitted_tokens,
            "deferred_tokens": tel.deferred_tokens,
            "straggler_coordinate_leak": straggler_leak,
            "merged_delta_norm": tel.merged_delta_norm,
            "direct_average_norm": tel.direct_average_norm,
            "learner_weight_each": next(iter(tel.learner_weights.values())),
            "outer_step_linf": float((updated["w"] - global_prev).abs().max()),
            "extra_gpu_vram_bytes": state.memory_telemetry()["extra_gpu_vram_bytes"],
        }
    )

    # Arm 3 (negative control): K = M with one learner missing must not merge.
    syncer_full = DecoupledDiLoCoQuorumSyncer(num_learners_m=8, min_quorum_k=8, merge_mode="rda")
    delta3, sel3, tel3 = syncer_full.merge_fragment("w", global_prev, frags, arrivals[:7], grace_window_s=1e6)
    arm3_ok = delta3 is None and not sel3.reached and tel3.num_admitted == 0 and tel3.num_deferred == 7
    arms.append(
        {
            "arm_id": "H11-C",
            "kind": "negative_control",
            "name": "Quorum K = M not reached with 7/8 learners -> syncer waits (no merge)",
            "passed": arm3_ok,
            "num_arrivals": tel3.num_arrivals,
            "num_admitted": tel3.num_admitted,
            "num_deferred": tel3.num_deferred,
        }
    )

    report: Dict[str, Any] = {
        "schema_version": "hivemind-decoupled-diloco-quorum-check/v1.0",
        "hypothesis_id": "H11",
        "reference": DECOUPLED_DILOCO_REFERENCE,
        "seed": seed,
        "arms": arms,
        "passed_arms": sum(1 for a in arms if a["passed"]),
        "total_arms": len(arms),
        "all_passed": all(a["passed"] for a in arms),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "scope_note": "Single-process controlled check on synthetic tensors; no network, no trained model.",
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def main() -> int:
    report = run_check()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("hypothesis_id", "passed_arms", "total_arms", "all_passed", "receipt_sha256")}, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
