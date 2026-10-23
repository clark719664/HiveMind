"""Experiment E5-het: the E5 outer-loop factorial on a heterogeneity-controlled toy testbed
(pre-registered in ``artifacts/experiments/E5_het_prereg.json``).

E5-toy (``run_experiment_E5_toy.py``) could not discriminate between merges because every
learner saw the same teacher: the mean pairwise cosine of learner deltas was 0.909. E5-het adds
one knob, ``rho``: learner ``m`` is labelled by its own teacher whose LoRA ``B`` factors are
``B_shared + rho * B_m`` (``B_m`` drawn once per learner, same scale as ``B_shared``), and the
held-out set mixes every learner's teacher equally. ``rho = 0`` reproduces E5-toy bit for bit
(regression control); larger ``rho`` makes learner deltas less aligned so that the RDA-vs-direct
question (P2), the grace controller (P3), the PA-CGAD gate (P4) and the new norm-ratio
prediction (P9) can be tested at toy cost.

Run:        PYTHONPATH=src python scripts/run_experiment_E5_het.py
Calibrate:  PYTHONPATH=src python scripts/run_experiment_E5_het.py --calibrate   (cosines only, no losses)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_experiment_E5_toy import ToyRun, mean_ci, mean_pairwise_cosine  # noqa: E402
from toy_model import SyntheticLoRACausalLM  # noqa: E402

PREREG_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_prereg.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_report.json"
RAW_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_raw_runs.json"
CALIB_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_calibration.json"
E5_TOY_REPORT = PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_report.json"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"


def build_heterogeneous_dataset(
    rho: float,
    vocab_size: int = 64,
    seq_len: int = 16,
    n_train_per_worker: int = 64,
    n_workers: int = 4,
    n_val: int = 128,
    seed: int = 1001,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    """Per-learner teachers ``B_shared + rho * B_m``; identical to the E5-toy builder when ``rho = 0``.

    The shared teacher, the random-number stream and the input sequences are produced exactly as in
    ``toy_model._build_synthetic_sequence_dataset``; the per-learner perturbations come from a
    separate generator so that the shared stream is untouched. Validation row ``j`` is labelled by
    the teacher of learner ``j mod n_workers`` (equal mixture of every learner's domain).
    """
    teacher = SyntheticLoRACausalLM(vocab_size=vocab_size, d_model=64, rank_r=8, seed=seed)
    gen = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        teacher.l1_B.copy_(torch.randn(teacher.l1_B.shape, generator=gen) * 0.35)
        teacher.l2_B.copy_(torch.randn(teacher.l2_B.shape, generator=gen) * 0.35)
    total_train = n_train_per_worker * n_workers
    all_x = torch.randint(0, vocab_size, (total_train + n_val, seq_len), generator=gen)

    het_gen = torch.Generator().manual_seed(seed + 7919)
    labels_per_worker: List[torch.Tensor] = []
    for _ in range(n_workers):
        t = copy.deepcopy(teacher)
        with torch.no_grad():
            t.l1_B.add_(rho * torch.randn(t.l1_B.shape, generator=het_gen) * 0.35)
            t.l2_B.add_(rho * torch.randn(t.l2_B.shape, generator=het_gen) * 0.35)
            logits, _ = t(all_x)
        labels_per_worker.append(logits.argmax(dim=-1))

    worker_splits: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k in range(n_workers):
        lo, hi = k * n_train_per_worker, (k + 1) * n_train_per_worker
        worker_splits.append((all_x[lo:hi].clone(), labels_per_worker[k][lo:hi].clone()))
    val_x = all_x[total_train:].clone()
    val_y = torch.stack([labels_per_worker[j % n_workers][total_train + j] for j in range(n_val)]).clone()
    return worker_splits, (val_x, val_y)


class HetToyRun(ToyRun):
    """E5 factorial cell on the heterogeneity-controlled dataset."""

    def __init__(self, rho: float, num_learners: int, seed: int, prereg: Dict[str, Any]) -> None:
        self.rho = float(rho)
        super().__init__(num_learners, seed, prereg)

    def _build_data(self, num_learners: int, seed: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        return build_heterogeneous_dataset(self.rho, vocab_size=64, n_train_per_worker=64, n_workers=num_learners, n_val=128, seed=2026 + seed)


def round0_cosine(rho: float, num_learners: int, seed: int, prereg: Dict[str, Any]) -> Tuple[float, float]:
    """Mean pairwise cosine and RDA/direct norm ratio of the round-0 learner deltas (manipulation check)."""
    tr = HetToyRun(rho, num_learners, seed, prereg)
    vecs = [tr._round0_deltas[i] for i in range(num_learners)]
    from hivemind.swarm.decoupled_diloco import direct_weighted_average, radial_directional_average

    ratio = float(radial_directional_average(vecs).norm()) / float(direct_weighted_average(vecs).norm())
    return mean_pairwise_cosine(vecs), ratio


def _calibrate_task(task: Tuple[float, int, int, Dict[str, Any]]) -> Tuple[float, int, int, float, float]:
    torch.set_num_threads(1)
    rho, m, seed, prereg = task
    c, r = round0_cosine(rho, m, seed, prereg)
    return rho, m, seed, c, r


def calibrate(prereg: Dict[str, Any], rhos: Sequence[float]) -> int:
    f = prereg["factors"]
    tasks = [(rho, m, seed, prereg) for rho, m, seed in itertools.product(rhos, f["num_learners_m"], f["seeds"])]
    rows: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for rho, m, seed, c, r in pool.map(_calibrate_task, tasks):
            rows.append({"rho": rho, "M": m, "seed": seed, "round0_cosine": c, "round0_rda_over_direct": r})
    summary = {}
    for rho in rhos:
        for m in f["num_learners_m"]:
            sel = [x for x in rows if x["rho"] == rho and x["M"] == m]
            summary[f"rho{rho}|M{m}"] = {"cosine_mean": statistics.fmean(x["round0_cosine"] for x in sel), "ratio_mean": statistics.fmean(x["round0_rda_over_direct"] for x in sel)}
            print(f"rho={rho:<5} M={m}: round-0 cosine {summary[f'rho{rho}|M{m}']['cosine_mean']:.3f}  RDA/direct norm ratio {summary[f'rho{rho}|M{m}']['ratio_mean']:.3f}")
    CALIB_PATH.write_text(json.dumps({"note": "manipulation-check calibration run before pre-registration: round-0 delta cosines only, no loss was computed or looked at", "rows": rows, "summary": summary, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    return 0


def loss_at_budget(loss_curve: Sequence[float], wall_curve: Sequence[float], budget: float) -> float:
    best = loss_curve[0]
    for loss, w in zip(loss_curve, wall_curve):
        if w <= budget + 1e-9:
            best = loss
    return best


def _run_bundle(task: Tuple[float, int, int, Dict[str, Any]]) -> Tuple[float, int, int, Dict[Tuple[str, str, str, float], Dict[str, Any]], float, Optional[Tuple[float, float]], Optional[float]]:
    torch.set_num_threads(1)
    rho, m, seed, prereg = task
    f = prereg["factors"]
    tr = HetToyRun(rho, m, seed, prereg)
    res: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}
    for merge, adm, gate, lr in itertools.product(f["merge"], f["admission"], f["gate"], f["outer_lr"]):
        if adm == "all" and gate == "cgad" and (merge, "all", "ungated", lr) in res:
            res[(merge, adm, gate, lr)] = copy.deepcopy(res[(merge, "all", "ungated", lr)])
        else:
            res[(merge, adm, gate, lr)] = tr.run(merge, adm, gate, lr)
    burst_loss = tr.run("direct", "all", "ungated", float(prereg["reference_outer_lr"]), stale_burst=True)["final_loss"]
    ident_pair: Optional[Tuple[float, float]] = None
    if m == 4 and seed == 1 and rho == max(f["rho"]):
        ident_pair = (tr.run("rda", "all", "ungated", 0.7, identical_shards=True)["final_loss"], tr.run("direct", "all", "ungated", 0.7, identical_shards=True)["final_loss"])
    regression: Optional[float] = None
    if rho == 0.0 and m == 4 and seed == 1:
        regression = tr.run("direct", "all", "ungated", 0.7)["final_loss"]
    return rho, m, seed, res, burst_loss, ident_pair, regression


def _run_ext_cell(task: Tuple[float, int, int, str, float, Dict[str, Any]]) -> Tuple[float, int, int, str, float, Dict[str, Any]]:
    torch.set_num_threads(1)
    rho, m, seed, merge, lr, prereg = task
    tr = HetToyRun(rho, m, seed, prereg)
    return rho, m, seed, merge, lr, tr.run(merge, "all", "ungated", lr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="manipulation check only (round-0 cosines), no losses")
    ap.add_argument("--rhos", type=float, nargs="*", default=None)
    ap.add_argument("--from-raw", action="store_true", help="skip training; analyse artifacts/experiments/E5_het_raw_runs.json")
    ap.add_argument("--extend-p2-grid", action="store_true", help="amendment 2: load the raw runs, add the rho_max all/ungated cells on the extended LR grid, re-analyse")
    args = ap.parse_args()
    started = time.perf_counter()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    f = prereg["factors"]
    if args.calibrate:
        return calibrate(prereg, args.rhos or f["rho"])

    tasks = [(rho, m, seed, prereg) for rho, m, seed in itertools.product(f["rho"], f["num_learners_m"], f["seeds"])]
    bundle: Dict[Tuple[float, int, int], Dict[Tuple[str, str, str, float], Dict[str, Any]]] = {}
    burst: Dict[Tuple[float, int, int], float] = {}
    ident: Tuple[float, float] = (0.0, 0.0)
    regression_loss: Optional[float] = None
    ext_lrs: List[float] = list(prereg.get("p2_grid_extension_outer_lr", [])) if args.extend_p2_grid else []
    if args.from_raw and not args.extend_p2_grid and RAW_PATH.exists():
        ext_lrs = list(json.loads(RAW_PATH.read_text(encoding="utf-8")).get("extension_outer_lr", []))
    if args.from_raw or args.extend_p2_grid:
        raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
        for b_key, res_s in raw["bundles"].items():
            rho_s, m_s, seed_s = b_key.split("|")
            bundle[(float(rho_s), int(m_s), int(seed_s))] = {tuple(json.loads(k)): v for k, v in res_s.items()}
        burst = {tuple(json.loads(k)): v for k, v in raw["burst"].items()}
        ident = tuple(raw["ident"])
        regression_loss = raw["regression_loss"]
        train_elapsed = raw["train_elapsed_seconds"]
        if ext_lrs and args.extend_p2_grid:
            rho_max_ = max(f["rho"])
            ext_tasks = [(rho_max_, m, seed, merge, lr, prereg) for m in f["num_learners_m"] for seed in f["seeds"] for merge in f["merge"] for lr in ext_lrs]
            with ProcessPoolExecutor(max_workers=8) as pool:
                for rho_, m_, seed_, merge_, lr_, res_ in pool.map(_run_ext_cell, ext_tasks):
                    bundle[(rho_, m_, seed_)][(merge_, "all", "ungated", lr_)] = res_
            train_elapsed = round(train_elapsed + (time.perf_counter() - started), 2)
            print(f"extension cells done ({len(ext_tasks)} runs)", flush=True)
            raw["bundles"] = {f"{rho}|{m}|{seed}": {json.dumps(list(k)): v for k, v in res.items()} for (rho, m, seed), res in bundle.items()}
            raw["extension_outer_lr"] = ext_lrs
            raw["train_elapsed_seconds"] = train_elapsed
            RAW_PATH.write_text(json.dumps(raw), encoding="utf-8")
    else:
        with ProcessPoolExecutor(max_workers=8) as pool:
            for rho, m, seed, res, b, ip, reg in pool.map(_run_bundle, tasks):
                bundle[(rho, m, seed)] = res
                burst[(rho, m, seed)] = b
                if ip is not None:
                    ident = ip
                if reg is not None:
                    regression_loss = reg
                print(f"bundle rho={rho} M={m} seed={seed} done ({time.perf_counter() - started:.0f}s)", flush=True)
        train_elapsed = round(time.perf_counter() - started, 2)
        RAW_PATH.write_text(json.dumps({
            "prereg_sha256": prereg_sha,
            "bundles": {f"{rho}|{m}|{seed}": {json.dumps(list(k)): v for k, v in res.items()} for (rho, m, seed), res in bundle.items()},
            "burst": {json.dumps(list(k)): v for k, v in burst.items()},
            "ident": list(ident), "regression_loss": regression_loss, "train_elapsed_seconds": train_elapsed,
        }), encoding="utf-8")
        print(f"raw runs dumped to {RAW_PATH}", flush=True)

    rho_max = max(f["rho"])

    def grid_for(rho: float, adm: str, gate: str) -> List[float]:
        return list(f["outer_lr"]) + (ext_lrs if (ext_lrs and rho == rho_max and adm == "all" and gate == "ungated") else [])

    cells: Dict[str, Dict[str, Any]] = {}
    n_runs = 0
    for rho, m, merge, adm, gate in itertools.product(f["rho"], f["num_learners_m"], f["merge"], f["admission"], f["gate"]):
      for lr in grid_for(rho, adm, gate):
        key = f"rho{rho}|M{m}|{merge}|{adm}|{gate}|lr{lr}"
        per_seed = [bundle[(rho, m, seed)][(merge, adm, gate, lr)] for seed in f["seeds"]]
        n_runs += len(f["seeds"])
        mean, hw = mean_ci([p["final_loss"] for p in per_seed])
        cells[key] = {
            "rho": rho, "M": m, "merge": merge, "admission": adm, "gate": gate, "outer_lr": lr,
            "final_loss_mean": mean, "final_loss_ci95": hw, "final_loss_per_seed": [p["final_loss"] for p in per_seed],
            "total_wall_mean": statistics.fmean(p["total_wall"] for p in per_seed),
            "admitted_token_fraction_mean": statistics.fmean(p["admitted_token_fraction"] for p in per_seed),
            "mean_pairwise_cosine": statistics.fmean(p["mean_pairwise_cosine"] for p in per_seed),
            "mean_rda_over_direct_norm_ratio": statistics.fmean(p["mean_rda_over_direct_norm_ratio"] for p in per_seed),
            "per_seed": per_seed,
        }

    def cell(rho: float, m: int, merge: str, adm: str, gate: str, lr: float) -> Dict[str, Any]:
        return cells[f"rho{rho}|M{m}|{merge}|{adm}|{gate}|lr{lr}"]

    def best_lr(rho: float, m: int, merge: str, adm: str, gate: str) -> Tuple[float, Dict[str, Any]]:
        _, lr = min((cell(rho, m, merge, adm, gate, lr)["final_loss_mean"], lr) for lr in grid_for(rho, adm, gate))
        return lr, cell(rho, m, merge, adm, gate, lr)

    def best_lr_base(rho: float, m: int, merge: str, adm: str, gate: str) -> Tuple[float, Dict[str, Any]]:
        """Best LR on the original grid only (P3/P4 cells have no extension)."""
        _, lr = min((cell(rho, m, merge, adm, gate, lr)["final_loss_mean"], lr) for lr in f["outer_lr"])
        return lr, cell(rho, m, merge, adm, gate, lr)

    p2_grid = grid_for(rho_max, "all", "ungated")
    grid_lo, grid_hi = min(p2_grid), max(p2_grid)

    # manipulation check: cosine monotone non-increasing in rho (direct/all/ungated at the reference lr), <= 0.3 at rho_max
    ref_lr = prereg["reference_outer_lr"]
    cos_by_rho = {rho: statistics.fmean(cell(rho, m, "direct", "all", "ungated", ref_lr)["mean_pairwise_cosine"] for m in f["num_learners_m"]) for rho in f["rho"]}
    ratio_by_rho_m = {f"rho{rho}|M{m}": cell(rho, m, "direct", "all", "ungated", ref_lr)["mean_rda_over_direct_norm_ratio"] for rho in f["rho"] for m in f["num_learners_m"]}
    monotone = all(cos_by_rho[a] >= cos_by_rho[b] for a, b in zip(f["rho"], f["rho"][1:]))
    p2_applicable = abs(cos_by_rho[rho_max]) <= 0.3
    manipulation = {"cosine_by_rho": cos_by_rho, "monotone_non_increasing": monotone, "cosine_at_rho_max_le_0.3": p2_applicable, "rda_over_direct_norm_ratio": ratio_by_rho_m}

    # P2 at rho_max
    p2_best = {merge: {f"M{m}": best_lr(rho_max, m, merge, "all", "ungated")[0] for m in f["num_learners_m"]} for merge in f["merge"]}
    boundary = any(v in (grid_lo, grid_hi) for d in p2_best.values() for v in d.values())
    direct_grows = p2_best["direct"]["M8"] > p2_best["direct"]["M4"]
    rda_stable = p2_best["rda"]["M8"] == p2_best["rda"]["M4"]
    if not p2_applicable:
        p2_status = "inapplicable (cosine at rho_max > 0.3)"
    elif boundary:
        p2_status = "inconclusive (a best outer LR sits on the grid boundary)"
    else:
        p2_status = "supported" if (direct_grows and rda_stable) else "refuted"
    p2_all_rho = {f"rho{rho}": {merge: {f"M{m}": best_lr(rho, m, merge, "all", "ungated")[0] for m in f["num_learners_m"]} for merge in f["merge"]} for rho in f["rho"]}

    # P9 at rho_max: lr*_direct / lr*_rda within one grid step (factor sqrt 2) of the mean RDA/direct norm ratio
    step = math.log(math.sqrt(2.0))
    p9_rows = []
    p9_ok = True
    for m in f["num_learners_m"]:
        ld, lr_ = p2_best["direct"][f"M{m}"], p2_best["rda"][f"M{m}"]
        rbar = ratio_by_rho_m[f"rho{rho_max}|M{m}"]
        dev = abs(math.log(ld / lr_) - math.log(rbar))
        p9_rows.append({"M": m, "best_lr_direct": ld, "best_lr_rda": lr_, "mean_norm_ratio": rbar, "log_deviation": dev, "within_one_grid_step": dev <= step + 1e-9})
        p9_ok &= dev <= step + 1e-9
    p9_status = "inconclusive (grid boundary)" if boundary else ("supported" if p9_ok else "refuted")

    # P3 at rho_max: adaptive vs matched random at each cell's best lr, loss at matched wall budget
    diffs: List[float] = []
    p3_rows = []
    for m in f["num_learners_m"]:
        for gate in f["gate"]:
            for merge in f["merge"]:
                _, ca = best_lr(rho_max, m, merge, "quorum_grace", gate)
                _, cr = best_lr(rho_max, m, merge, "quorum_random", gate)
                for s_idx, seed in enumerate(f["seeds"]):
                    pa, pr = ca["per_seed"][s_idx], cr["per_seed"][s_idx]
                    budget = min(pa["total_wall"], pr["total_wall"])
                    la = loss_at_budget(pa["loss_curve"], pa["wall_curve"], budget)
                    lrnd = loss_at_budget(pr["loss_curve"], pr["wall_curve"], budget)
                    diffs.append(lrnd - la)
                    p3_rows.append({"M": m, "gate": gate, "merge": merge, "seed": seed, "budget": budget, "loss_adaptive": la, "loss_random": lrnd})
    p3_mean, p3_hw = mean_ci(diffs)
    p3_status = "supported" if p3_mean - p3_hw > 0 else ("refuted" if p3_mean + p3_hw < 0 else "inconclusive (CI includes 0)")

    # P4 at every rho: gate gain under quorum_grace; 'all' is the negative control (must be exactly 0)
    p4: Dict[str, Any] = {}
    for rho in f["rho"]:
        for adm in ("quorum_grace", "all"):
            gains: List[float] = []
            for m in f["num_learners_m"]:
                for merge in f["merge"]:
                    lr_u, cu = best_lr_base(rho, m, merge, adm, "ungated")
                    cg = cell(rho, m, merge, adm, "cgad", lr_u)
                    gains.extend(cu["per_seed"][i]["final_loss"] - cg["per_seed"][i]["final_loss"] for i in range(len(f["seeds"])))
            gm, gh = mean_ci(gains)
            p4[f"rho{rho}|{adm}"] = {"gain_mean": gm, "gain_ci95": gh, "n": len(gains)}
    neg_all_zero = all(abs(p4[f"rho{rho}|all"]["gain_mean"]) < 1e-9 for rho in f["rho"])
    g = p4[f"rho{rho_max}|quorum_grace"]
    p4_status = "supported" if g["gain_mean"] - g["gain_ci95"] > 0 else ("refuted" if g["gain_mean"] + g["gain_ci95"] < 0 else "inconclusive (CI includes 0)")

    # controls
    pos_rows = []
    for rho in f["rho"]:
        for m in f["num_learners_m"]:
            for s_idx, seed in enumerate(f["seeds"]):
                clean = cell(rho, m, "direct", "all", "ungated", ref_lr)["per_seed"][s_idx]["final_loss"]
                pos_rows.append({"rho": rho, "M": m, "seed": seed, "clean_loss": clean, "stale_burst_loss": burst[(rho, m, seed)]})
    positive_ok = all(r["stale_burst_loss"] > r["clean_loss"] for r in pos_rows)
    ident_rda, ident_dir = ident
    neg1_ok = abs(ident_rda - ident_dir) <= 1e-5 * max(1.0, abs(ident_dir))
    toy = json.loads(E5_TOY_REPORT.read_text(encoding="utf-8"))
    toy_ref = toy["cells"]["M4|direct|all|ungated|lr0.7"]["final_loss_per_seed"][0]
    neg2_ok = regression_loss is not None and abs(regression_loss - toy_ref) <= 1e-6

    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E5-het/v1.0",
        "experiment_id": "E5-het",
        "prereg_sha256": prereg_sha,
        "num_runs": n_runs,
        "cells": cells,
        "manipulation_check": manipulation,
        "analysis": {
            "P2": {"best_outer_lr_at_rho_max": p2_best, "best_outer_lr_all_rho": p2_all_rho, "mean_pairwise_cosine_at_rho_max": cos_by_rho[rho_max], "applicable": p2_applicable, "grid_boundary_hit": boundary, "outer_lr_grid_used": p2_grid, "status": p2_status},
            "P3": {"paired_diff_random_minus_adaptive_mean": p3_mean, "ci95_halfwidth_df2": p3_hw, "n_pairs": len(diffs), "status": p3_status, "rows": p3_rows},
            "P4": {"gains": p4, "negative_control_all_gain_is_zero": neg_all_zero, "status": p4_status},
            "P9": {"rows": p9_rows, "status": p9_status},
        },
        "controls": {
            "positive_stale_burst": {"rows": pos_rows, "passed": positive_ok},
            "negative_rda_identity": {"rda_loss": ident_rda, "direct_loss": ident_dir, "passed": neg1_ok},
            "negative_rho0_reproduces_e5_toy": {"e5_het_loss": regression_loss, "e5_toy_loss": toy_ref, "e5_toy_receipt": toy["receipt_sha256"], "passed": neg2_ok},
        },
        "controls_all_passed": bool(positive_ok and neg1_ok and neg2_ok),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "training_elapsed_seconds": train_elapsed,
        "amendments": prereg.get("amendments", []),
        "p2_grid_extended": bool(ext_lrs),
        "scope_note": "Toy synthetic LM on one 8-CPU host; heterogeneity is a synthetic per-learner teacher perturbation, not real domain shift; decides RQ2-RQ4 and P9 at this scale only.",
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    colors = {"direct": "#e67e22", "rda": "#2980b9"}
    ax = axes[0]
    for m, ls in zip(f["num_learners_m"], ("-", "--")):
        ax.plot(f["rho"], [cell(rho, m, "direct", "all", "ungated", ref_lr)["mean_pairwise_cosine"] for rho in f["rho"]], marker="o", ls=ls, color="#444", label=f"cosine, M={m}")
        ax.plot(f["rho"], [ratio_by_rho_m[f"rho{rho}|M{m}"] for rho in f["rho"]], marker="s", ls=ls, color="#8e44ad", label=f"RDA/direct norm, M={m}")
    ax.axhline(0.3, color="#c0392b", lw=0.8, ls=":", label="P2 applicability bound")
    ax.set_xlabel("heterogeneity $\\rho$"); ax.set_title("manipulation check"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax, m in zip(axes[1:], f["num_learners_m"]):
        for merge in f["merge"]:
            means = [cell(rho_max, m, merge, "all", "ungated", lr)["final_loss_mean"] for lr in p2_grid]
            hws = [cell(rho_max, m, merge, "all", "ungated", lr)["final_loss_ci95"] for lr in p2_grid]
            ax.errorbar(p2_grid, means, yerr=hws, marker="o", capsize=3, color=colors[merge], label=f"{merge} merge")
        ax.set_xscale("log"); ax.set_title(f"$\\rho$ = {rho_max}, M = {m}: loss vs outer LR"); ax.set_xlabel("outer learning rate (log)"); ax.set_ylabel("held-out loss after 12 rounds"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig6_e5_het_rda_vs_direct.{ext}", dpi=160)
    plt.close(fig)

    print(json.dumps({k: report["analysis"][k]["status"] for k in ("P2", "P3", "P4", "P9")} | {"manipulation": manipulation["cosine_by_rho"], "controls_all_passed": report["controls_all_passed"], "runs": n_runs, "elapsed_s": report["elapsed_seconds"], "receipt": report["receipt_sha256"]}, indent=2))
    return 0 if report["controls_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
