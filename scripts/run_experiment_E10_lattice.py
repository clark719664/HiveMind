"""Experiment E10-lattice: E8 vs Z8 vs 1-bit sign quantisation of outer deltas at matched rate, with and without
error feedback, on the E5-het testbed (pre-registered in ``artifacts/experiments/E10_lattice_prereg.json``).

Leverages the lattice decoders of the author's OptimalEnergyAI project (ported to
``hivemind.compress.lattice_quantizer`` with attribution) and the signSGD 1-bit compressor.

Run: PYTHONPATH=src python scripts/run_experiment_E10_lattice.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hivemind.compress import LatticeDeltaQuantizer, SignDeltaQuantizer, brute_force_decoder_check, calibrate_scale_for_rate  # noqa: E402
from run_experiment_E5_het import HetToyRun  # noqa: E402
from run_experiment_E5_toy import mean_ci  # noqa: E402

PREREG_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E10_lattice_prereg.json"
E5_HET_PREREG = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_prereg.json"
E5_HET_REPORT = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_report.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E10_lattice_report.json"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
RHO, M = 4.0, 8
RATES = (1.0, 2.0)


def _arm_quantizers(arm: str, scales: Dict[str, float], ef: bool, num_learners: int) -> List[Any]:
    if arm == "none":
        return []
    if arm == "sign":
        return [SignDeltaQuantizer(error_feedback=ef) for _ in range(num_learners)]
    kind, rate = arm.split("@")
    return [LatticeDeltaQuantizer(kind, scales[arm], error_feedback=ef) for _ in range(num_learners)]


def run_arm(seed: int, arm: str, ef: bool, scales: Dict[str, float], outer_lr: float, prereg_het: Dict[str, Any]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    tr = HetToyRun(RHO, M, seed, prereg_het)
    qs = _arm_quantizers(arm, scales, ef, M)
    rel: List[float] = []
    rates: List[float] = []

    def transform(i: int, v: torch.Tensor) -> torch.Tensor:
        if not qs:
            return v
        out = qs[i].quantize(v)
        rel.append(out.relative_mse)
        rates.append(out.rate_bits_per_dim)
        return out.dequantized

    res = tr.run("direct", "all", "ungated", outer_lr, delta_transform=transform)
    return {"seed": seed, "arm": arm, "error_feedback": ef, "final_loss": res["final_loss"], "loss_curve": res["loss_curve"],
            "mean_relative_mse": statistics.fmean(rel) if rel else 0.0, "mean_rate_bits_per_dim": statistics.fmean(rates) if rates else 32.0}


def _task(t: Tuple[int, str, bool, Dict[str, float], float, Dict[str, Any]]) -> Dict[str, Any]:
    return run_arm(*t)


def main() -> int:
    started = time.perf_counter()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    prereg_het = json.loads(E5_HET_PREREG.read_text(encoding="utf-8"))
    het = json.loads(E5_HET_REPORT.read_text(encoding="utf-8"))
    outer_lr = float(het["analysis"]["P2"]["best_outer_lr_at_rho_max"]["direct"][f"M{M}"])
    seeds = [1, 2, 3]

    # calibration on the round-0 delta of learner 0, seed 1
    cal_run = HetToyRun(RHO, M, 1, prereg_het)
    sample = cal_run._round0_deltas[0]
    scales: Dict[str, float] = {}
    calibration: Dict[str, Any] = {}
    for kind, rate in itertools.product(("Z", "E8"), RATES):
        s, r = calibrate_scale_for_rate(kind, sample, rate)
        scales[f"{kind}@{rate}"] = s
        calibration[f"{kind}@{rate}"] = {"scale": s, "rate_on_calibration_sample": r}
    s_fine, r_fine = calibrate_scale_for_rate("E8", sample, 6.0)
    scales["E8@6.0"] = s_fine
    calibration["E8@6.0 (negative control)"] = {"scale": s_fine, "rate_on_calibration_sample": r_fine}

    arms = ["none", "sign"] + [f"{k}@{r}" for k in ("Z", "E8") for r in RATES]
    tasks = [(seed, arm, ef, scales, outer_lr, prereg_het) for seed in seeds for arm in arms for ef in (False, True) if not (arm == "none" and ef)]
    tasks += [(seed, "E8@6.0", False, scales, outer_lr, prereg_het) for seed in seeds]
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(_task, tasks):
            results.append(r)
            print(f"seed {r['seed']} {r['arm']:7s} EF={r['error_feedback']!s:5s} loss {r['final_loss']:.4f} relMSE {r['mean_relative_mse']:.4f} rate {r['mean_rate_bits_per_dim']:.3f}", flush=True)

    def get(arm: str, ef: bool, seed: int) -> Dict[str, Any]:
        return next(r for r in results if r["arm"] == arm and r["error_feedback"] == ef and r["seed"] == seed)

    cells: Dict[str, Any] = {}
    for arm in arms + ["E8@6.0"]:
        for ef in (False, True):
            rows = [r for r in results if r["arm"] == arm and r["error_feedback"] == ef]
            if not rows:
                continue
            lm, lh = mean_ci([r["final_loss"] for r in rows])
            cells[f"{arm}|EF={'on' if ef else 'off'}"] = {"arm": arm, "error_feedback": ef, "final_loss_mean": lm, "final_loss_ci95": lh,
                                                           "final_loss_per_seed": [r["final_loss"] for r in rows],
                                                           "mean_relative_mse": statistics.fmean(r["mean_relative_mse"] for r in rows),
                                                           "mean_rate_bits_per_dim": statistics.fmean(r["mean_rate_bits_per_dim"] for r in rows)}

    # matched-rate control first: an R whose Z8 and E8 arms realise rates more than 0.1 bit/dim apart is invalid
    neg3 = {}
    valid_rates: List[float] = []
    for rate in RATES:
        rz = cells[f"Z@{rate}|EF=off"]["mean_rate_bits_per_dim"]
        re8 = cells[f"E8@{rate}|EF=off"]["mean_rate_bits_per_dim"]
        ok = abs(rz - re8) <= 0.1
        neg3[f"R={rate}"] = {"z8_rate": rz, "e8_rate": re8, "matched": ok}
        if ok:
            valid_rates.append(rate)
    neg3_ok = 1.0 in valid_rates  # the headline rate must be matched; other rates are reported invalid individually

    # P10a (over matched rates only)
    p10a: Dict[str, Any] = {}
    a_ok = bool(valid_rates)
    a_refuted = False
    for rate in RATES:
        for ef in (False, True):
            ratios = [get(f"E8@{rate}", ef, s)["mean_relative_mse"] / get(f"Z@{rate}", ef, s)["mean_relative_mse"] for s in seeds]
            m, h = mean_ci(ratios)
            p10a[f"R={rate}|EF={'on' if ef else 'off'}"] = {"ratio_mean": m, "ci95": h, "per_seed": ratios, "gaussian_prediction": 0.86, "rate_matched": rate in valid_rates}
            if not ef and rate in valid_rates:
                a_ok &= (m + h) < 1.0
                a_refuted |= (m - h) > 1.0
    invalid_txt = "" if len(valid_rates) == len(RATES) else f"; R in {[r for r in RATES if r not in valid_rates]} invalid (matched-rate control failed)"
    p10a_status = ("supported" if a_ok else ("refuted" if a_refuted else "inconclusive (CI includes 1 at some matched rate)")) + invalid_txt

    # P10b
    p10b: Dict[str, Any] = {}
    for rate in RATES:
        for ef in (False, True):
            diffs = [get(f"Z@{rate}", ef, s)["final_loss"] - get(f"E8@{rate}", ef, s)["final_loss"] for s in seeds]
            m, h = mean_ci(diffs)
            st = "supported" if m - h > 0 else ("refuted" if m + h < 0 else "inconclusive (CI includes 0)")
            if rate not in valid_rates:
                st = "invalid (matched-rate control failed)"
            p10b[f"R={rate}|EF={'on' if ef else 'off'}"] = {"z8_minus_e8_mean": m, "ci95": h, "per_seed": diffs, "status": st}
    p10b_status = p10b["R=1.0|EF=on"]["status"]

    # P10c
    diffs_c = [get("E8@1.0", True, s)["final_loss"] - get("none", False, s)["final_loss"] for s in seeds]
    mc, hc = mean_ci(diffs_c)
    p10c_status = "refuted" if mc - hc > 0 else "supported"

    # controls
    pos_rows = [{"seed": s, "sign_noEF": get("sign", False, s)["final_loss"], "none": get("none", False, s)["final_loss"]} for s in seeds]
    positive_ok = all(r["sign_noEF"] > r["none"] for r in pos_rows)
    neg1_rows = [{"seed": s, "e8_fine": get("E8@6.0", False, s)["final_loss"], "none": get("none", False, s)["final_loss"]} for s in seeds]
    neg1_ok = all(abs(r["e8_fine"] - r["none"]) <= 1e-3 for r in neg1_rows)
    neg2 = {k: brute_force_decoder_check(k, 1000, 2026) for k in ("Z", "E8")}
    neg2_ok = all(v["passed"] for v in neg2.values())
    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E10-lattice/v1.0",
        "experiment_id": "E10-lattice",
        "prereg_sha256": prereg_sha,
        "e5_het_receipt_sha256": het["receipt_sha256"],
        "outer_lr_from_e5_het": outer_lr,
        "calibration": calibration,
        "num_runs": len(results),
        "cells": cells,
        "runs": results,
        "analysis": {"P10a": {"cells": p10a, "status": p10a_status}, "P10b": {"cells": p10b, "status": p10b_status},
                     "P10c": {"e8_r1_ef_minus_none_mean": mc, "ci95": hc, "status": p10c_status}},
        "controls": {"positive_sign_hurts": {"rows": pos_rows, "passed": positive_ok},
                     "negative_fine_scale_e8_equals_none": {"rows": neg1_rows, "passed": neg1_ok},
                     "negative_decoder_brute_force": {**neg2, "passed": neg2_ok},
                     "negative_matched_rate": {**neg3, "valid_rates": valid_rates, "passed": neg3_ok, "rule": "an unmatched R invalidates the P10 comparisons at that R; the headline R = 1 must be matched"}},
        "controls_all_passed": bool(positive_ok and neg1_ok and neg2_ok and neg3_ok),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scope_note": prereg["scope"],
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    order = ["none|EF=off", "sign|EF=off", "sign|EF=on", "Z@1.0|EF=off", "E8@1.0|EF=off", "Z@1.0|EF=on", "E8@1.0|EF=on", "Z@2.0|EF=off", "E8@2.0|EF=off", "Z@2.0|EF=on", "E8@2.0|EF=on"]
    labels = [k.replace("|EF=off", "").replace("|EF=on", "+EF").replace("@", " ") for k in order]
    ax = axes[0]
    ax.bar(range(len(order)), [cells[k]["final_loss_mean"] for k in order], yerr=[cells[k]["final_loss_ci95"] for k in order], capsize=3,
           color=["#7f8c8d" if k.startswith("none") else ("#c0392b" if k.startswith("sign") else ("#e67e22" if k.startswith("Z") else "#2980b9")) for k in order])
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7); ax.set_ylabel("held-out loss after 12 rounds"); ax.set_title("E10: final loss by codec (3 seeds, 95% CI)"); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    for kind, col in (("Z", "#e67e22"), ("E8", "#2980b9")):
        ax.plot([cells[f"{kind}@{r}|EF=off"]["mean_rate_bits_per_dim"] for r in RATES], [cells[f"{kind}@{r}|EF=off"]["mean_relative_mse"] for r in RATES], marker="o", color=col, label=f"{kind}8 (EF off)")
    ax.set_xlabel("measured rate (bits/dim)"); ax.set_ylabel("relative quantisation MSE"); ax.set_title("distortion vs rate"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig7_e10_lattice_codecs.{ext}", dpi=160)
    plt.close(fig)
    print(json.dumps({"P10a": p10a_status, "P10b": p10b_status, "P10c": p10c_status, "controls_all_passed": report["controls_all_passed"], "runs": len(results), "elapsed_s": report["elapsed_seconds"], "receipt": report["receipt_sha256"]}, indent=2))
    return 0 if report["controls_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
