"""Experiment E5-toy: outer-loop factorial under simulated churn (pre-registered in
``artifacts/experiments/E5_toy_prereg.json``; decides RQ2, RQ3, RQ4 at toy scale).

Reuses the E1 synthetic LoRA causal LM and runs the factorial
merge x admission x gate x outer_lr x M x seed with a replayed arrival model, plus the
positive control (stale_burst) and two negative controls (rda_identity, grace_gamma0).
Writes ``artifacts/experiments/E5_toy_report.json`` (hash of the pre-registration included)
and ``paper/figures/fig5_e5_toy_rda_vs_direct.{pdf,png}``.

Run: ``PYTHONPATH=src python scripts/run_experiment_E5_toy.py``
"""

from __future__ import annotations

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hivemind.swarm.decoupled_diloco import (  # noqa: E402
    AdaptiveGraceWindowController,
    LearnerArrival,
    direct_weighted_average,
    radial_directional_average,
    select_quorum,
)
from hivemind.swarm.diloco_nextgen import CosineGatedAdamDecayOuterOptimizer  # noqa: E402
from toy_model import (  # noqa: E402
    SyntheticLoRACausalLM,
    _build_synthetic_sequence_dataset,
)

PREREG_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_prereg.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_report.json"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"

T_CRIT_DF2 = 4.302652729911275  # two-sided 95% t critical value, df = 2 (three seeds)


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and 95% t half-width for n = 3 (df = 2); half-width is 0 for n < 2."""
    vals = [float(v) for v in values]
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))  # explicit: statistics.stdev raised on some float inputs under Python 3.12
    return m, T_CRIT_DF2 * sd / math.sqrt(len(vals))


def sample_arrivals(num_learners: int, outer_rounds: int, seed: int) -> List[List[float]]:
    """Arrival time of each learner per round in compute units (pre-registered model)."""
    g = torch.Generator().manual_seed(10_000 + 97 * num_learners + seed)
    rounds: List[List[float]] = []
    for _ in range(outer_rounds):
        base = torch.exp(torch.randn(num_learners, generator=g) * 0.25)  # lognormal, median 1.0
        straggler = torch.rand(num_learners, generator=g) < 0.2
        factor = 3.0 + 5.0 * torch.rand(num_learners, generator=g)
        t = torch.where(straggler, base * factor, base)
        rounds.append([float(x) for x in t])
    return rounds


def adaptive_windows(arrivals: List[List[float]], k: int, sync_units: float) -> List[float]:
    """Grace window the adaptive controller would open each round, given arrivals only."""
    ctrl = AdaptiveGraceWindowController(overlap_steps_tau=2, safety_margin_gamma=0.5, ema_beta=0.5, sync_duration_s=sync_units)
    windows: List[float] = []
    for t in arrivals:
        ordered = sorted(t)
        ctrl.observe_step_duration(statistics.median(ordered))
        ctrl.observe_quorum_duration(ordered[k - 1])
        windows.append(ctrl.grace_window_seconds())
    return windows


def random_windows(mean_window: float, outer_rounds: int, seed: int) -> List[float]:
    g = torch.Generator().manual_seed(20_000 + seed)
    return [float(2.0 * mean_window * torch.rand(1, generator=g).item()) for _ in range(outer_rounds)]


def inner_train(
    w_model: SyntheticLoRACausalLM,
    start_params: Dict[str, torch.Tensor],
    shard: Tuple[torch.Tensor, torch.Tensor],
    round_idx: int,
    inner_steps: int,
    batch_size: int,
    inner_lr: float,
) -> Dict[str, torch.Tensor]:
    """Run H inner SGD steps from ``start_params`` and return the pseudo-gradient start - end."""
    with torch.no_grad():
        for name, p in w_model.named_parameters():
            p.copy_(start_params[name])
    wx, wy = shard
    for s in range(inner_steps):
        idx = ((round_idx * inner_steps + s) * batch_size) % wx.shape[0]
        bx, by = wx[idx : idx + batch_size], wy[idx : idx + batch_size]
        w_model.zero_grad()
        _, loss = w_model(bx, by)
        loss.backward()
        with torch.no_grad():
            for p in w_model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-inner_lr)
    with torch.no_grad():
        return {name: (start_params[name] - p.detach()).clone() for name, p in w_model.named_parameters()}


def flat(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in d.values()])


def unflat(vec: torch.Tensor, like: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    pos = 0
    for k, v in like.items():
        n = v.numel()
        out[k] = vec[pos : pos + n].reshape(v.shape).clone()
        pos += n
    return out


def mean_pairwise_cosine(vectors: List[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 1.0
    cos: List[float] = []
    for a, b in itertools.combinations(vectors, 2):
        na, nb = float(a.norm()), float(b.norm())
        cos.append(float(torch.dot(a, b)) / (na * nb) if na > 0 and nb > 0 else 0.0)
    return statistics.fmean(cos)


class ToyRun:
    """One factorial cell for one (M, seed, outer_lr)."""

    def __init__(self, num_learners: int, seed: int, prereg: Dict[str, Any]) -> None:
        fixed = prereg["fixed"]
        self.m = num_learners
        self.seed = seed
        self.rounds = int(fixed["outer_rounds"])
        self.h = int(fixed["inner_steps_h"])
        self.batch = int(fixed["batch_size"])
        self.inner_lr = float(fixed["inner_lr"])
        self.mu = float(fixed["outer_momentum"])
        self.sync = float(fixed["sync_duration_units"])
        self.shards, (self.val_x, self.val_y) = self._build_data(num_learners, seed)
        self.init_model = SyntheticLoRACausalLM(vocab_size=64, d_model=64, rank_r=8, seed=2026 + seed)
        self._work_model = copy.deepcopy(self.init_model)
        self.init_state = {k: v.detach().clone() for k, v in self.init_model.named_parameters()}
        self.arrivals = sample_arrivals(num_learners, self.rounds, seed)
        self.k = math.ceil(num_learners / 2)
        self.adaptive = adaptive_windows(self.arrivals, self.k, self.sync)
        self.random = random_windows(statistics.fmean(self.adaptive), self.rounds, seed)
        self.gate = CosineGatedAdamDecayOuterOptimizer(lr=1e-3, alpha=0.2, tau_cut=32.0)
        self._round0_deltas: Dict[int, torch.Tensor] = {
            m_idx: flat(inner_train(self._work_model, self.init_state, self.shards[m_idx], 0, self.h, self.batch, self.inner_lr))
            for m_idx in range(self.m)
        }

    def _build_data(self, num_learners: int, seed: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        """IID shards from one teacher (E5-toy); E5-het overrides this with per-learner teachers."""
        return _build_synthetic_sequence_dataset(vocab_size=64, n_train_per_worker=64, n_workers=num_learners, n_val=128, seed=2026 + seed)

    def eval_loss(self, params: Dict[str, torch.Tensor]) -> float:
        m = self._work_model
        with torch.no_grad():
            for name, p in m.named_parameters():
                p.copy_(params[name])
            _, loss = m(self.val_x, self.val_y)
        return float(loss.item())

    def run(
        self,
        merge: str,
        admission: str,
        gate: str,
        outer_lr: float,
        stale_burst: bool = False,
        identical_shards: bool = False,
        delta_transform: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """``delta_transform(learner_idx, delta)`` (E10) replaces each fresh learner delta before merging; None = identity."""
        global_params = {k: v.clone() for k, v in self.init_state.items()}
        velocity = torch.zeros(flat(global_params).numel())
        round0_params = {k: v.clone() for k, v in global_params.items()}
        pending: Dict[int, torch.Tensor] = {}  # learner -> stale delta carried to next round
        losses = [self.eval_loss(global_params)]
        wall = [0.0]
        admitted_tokens = 0
        total_tokens = 0
        cosines: List[float] = []
        norm_ratios: List[float] = []  # ||RDA(v)|| / ||direct(v)|| per round, whatever merge is applied
        tokens_per_round = self.h * self.batch

        for r in range(self.rounds):
            t_round = self.arrivals[r]
            deltas: Dict[int, torch.Tensor] = {}
            staleness: Dict[int, float] = {}
            fresh_ids: List[int] = []
            for m_idx in range(self.m):
                if m_idx in pending:
                    deltas[m_idx] = pending.pop(m_idx)
                    staleness[m_idx] = 1.0
                    continue
                shard = self.shards[0] if identical_shards else self.shards[m_idx]
                start = round0_params if (stale_burst and r in (4, 8)) else global_params
                if r == 0 and not identical_shards and not stale_burst:
                    deltas[m_idx] = self._round0_deltas[m_idx].clone()
                else:
                    deltas[m_idx] = flat(inner_train(self._work_model, start, shard, r, self.h, self.batch, self.inner_lr))
                if delta_transform is not None:
                    deltas[m_idx] = delta_transform(m_idx, deltas[m_idx])
                staleness[m_idx] = 40.0 if (stale_burst and r in (4, 8)) else 0.0
                fresh_ids.append(m_idx)
                total_tokens += tokens_per_round
            cosines.append(mean_pairwise_cosine([deltas[i] for i in fresh_ids]))

            if admission == "all":
                admitted = list(deltas.keys())
                round_wall = max(t_round) + self.sync
            else:
                window = self.adaptive[r] if admission == "quorum_grace" else self.random[r]
                arr = [LearnerArrival(str(i), 0.0 if i in staleness and staleness[i] == 1.0 else t_round[i], 1, tokens_per_round) for i in deltas]
                sel = select_quorum(arr, min_quorum_k=min(self.k, len(arr)), grace_window_s=window)
                admitted = [int(i) for i in sel.admitted_learner_ids]
                for lid in sel.deferred_learner_ids:
                    pending[int(lid)] = deltas[int(lid)]
                round_wall = max(t_round[i] for i in admitted) + self.sync if admitted else self.sync
            admitted_tokens += tokens_per_round * sum(1 for i in admitted if staleness.get(i, 0.0) == 0.0)

            vecs: List[torch.Tensor] = []
            for i in admitted:
                v = deltas[i]
                if gate == "cgad" and staleness[i] > 0.0:
                    _, sigma = self.gate.compute_gate(staleness[i])
                    v = v * sigma
                vecs.append(v)
            direct_vec = direct_weighted_average(vecs)
            rda_vec = radial_directional_average(vecs)
            d_norm = float(direct_vec.norm())
            if d_norm > 0.0:
                norm_ratios.append(float(rda_vec.norm()) / d_norm)
            merged = rda_vec if merge == "rda" else direct_vec
            velocity = self.mu * velocity + merged
            step = self.mu * velocity + merged
            g_flat = flat(global_params) - outer_lr * step
            global_params = unflat(g_flat, global_params)
            losses.append(self.eval_loss(global_params))
            wall.append(wall[-1] + round_wall)

        return {
            "final_loss": losses[-1],
            "loss_curve": [round(x, 5) for x in losses],
            "wall_curve": [round(x, 4) for x in wall],
            "total_wall": wall[-1],
            "admitted_token_fraction": admitted_tokens / max(1, total_tokens),
            "mean_pairwise_cosine": statistics.fmean(cosines),
            "mean_rda_over_direct_norm_ratio": statistics.fmean(norm_ratios) if norm_ratios else float("nan"),
        }


def loss_at_budget(loss_curve: Sequence[float], wall_curve: Sequence[float], budget: float) -> float:
    best = loss_curve[0]
    for loss, w in zip(loss_curve, wall_curve):
        if w <= budget + 1e-9:
            best = loss
    return best


def _run_m_seed_bundle(task: Tuple[int, int, Dict[str, Any]]) -> Tuple[int, int, Dict[Tuple[str, str, str, float], Dict[str, Any]], float, Optional[Tuple[float, float]]]:
    torch.set_num_threads(1)
    m, seed, prereg = task
    f = prereg["factors"]
    tr = ToyRun(m, seed, prereg)
    res: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}
    for merge, adm, gate, lr in itertools.product(f["merge"], f["admission"], f["gate"], f["outer_lr"]):
        if adm == "all" and gate == "cgad" and (merge, "all", "ungated", lr) in res:
            res[(merge, adm, gate, lr)] = copy.deepcopy(res[(merge, "all", "ungated", lr)])
        else:
            res[(merge, adm, gate, lr)] = tr.run(merge, adm, gate, lr)
    burst_loss = tr.run("direct", "all", "ungated", 0.7, stale_burst=True)["final_loss"]
    ident_pair: Optional[Tuple[float, float]] = None
    if m == 4 and seed == 1:
        ident_rda = tr.run("rda", "all", "ungated", 0.7, identical_shards=True)["final_loss"]
        ident_dir = tr.run("direct", "all", "ungated", 0.7, identical_shards=True)["final_loss"]
        ident_pair = (ident_rda, ident_dir)
    return m, seed, res, burst_loss, ident_pair


def main() -> int:
    started = time.perf_counter()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    f = prereg["factors"]
    cells: Dict[str, Dict[str, Any]] = {}
    n_runs = 0
    tasks = [(m, seed, prereg) for m, seed in itertools.product(f["num_learners_m"], f["seeds"])]
    bundle_res: Dict[Tuple[int, int], Dict[Tuple[str, str, str, float], Dict[str, Any]]] = {}
    burst_losses: Dict[Tuple[int, int], float] = {}
    ident_losses: Tuple[float, float] = (0.0, 0.0)
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        for m, seed, res, burst_loss, ident_pair in pool.map(_run_m_seed_bundle, tasks):
            bundle_res[(m, seed)] = res
            burst_losses[(m, seed)] = burst_loss
            if ident_pair is not None:
                ident_losses = ident_pair

    for m, merge, adm, gate, lr in itertools.product(f["num_learners_m"], f["merge"], f["admission"], f["gate"], f["outer_lr"]):
        key = f"M{m}|{merge}|{adm}|{gate}|lr{lr}"
        per_seed = [bundle_res[(m, seed)][(merge, adm, gate, lr)] for seed in f["seeds"]]
        n_runs += len(f["seeds"])
        mean, hw = mean_ci([p["final_loss"] for p in per_seed])
        cells[key] = {
            "M": m, "merge": merge, "admission": adm, "gate": gate, "outer_lr": lr,
            "final_loss_mean": mean, "final_loss_ci95": hw,
            "final_loss_per_seed": [p["final_loss"] for p in per_seed],
            "total_wall_mean": statistics.fmean(p["total_wall"] for p in per_seed),
            "admitted_token_fraction_mean": statistics.fmean(p["admitted_token_fraction"] for p in per_seed),
            "mean_pairwise_cosine": statistics.fmean(p["mean_pairwise_cosine"] for p in per_seed),
            "per_seed": per_seed,
        }
        print(f"[{n_runs:4d}] {key}: loss {mean:.4f} ± {hw:.4f}", flush=True)

    # ---------------- analysis ----------------
    def best_lr(m: int, merge: str, adm: str, gate: str) -> Tuple[float, Dict[str, Any]]:
        cands = [(cells[f"M{m}|{merge}|{adm}|{gate}|lr{lr}"]["final_loss_mean"], lr) for lr in f["outer_lr"]]
        _, lr = min(cands)
        return lr, cells[f"M{m}|{merge}|{adm}|{gate}|lr{lr}"]

    # P2: best lr vs M under direct and rda (admission 'all', ungated: pure merge comparison)
    p2 = {}
    for merge in f["merge"]:
        p2[merge] = {f"M{m}": best_lr(m, merge, "all", "ungated")[0] for m in f["num_learners_m"]}
    cos_all = statistics.fmean(cells[f"M{m}|direct|all|ungated|lr0.7"]["mean_pairwise_cosine"] for m in f["num_learners_m"])
    p2_applicable = abs(cos_all) <= 0.3
    direct_grows = p2["direct"]["M8"] > p2["direct"]["M4"]
    rda_stable = p2["rda"]["M8"] == p2["rda"]["M4"]
    p2_status = "inapplicable (learner deltas not near-orthogonal)" if not p2_applicable else ("supported" if direct_grows and rda_stable else "refuted")

    # P3: adaptive vs matched-random at each cell's best lr, loss at matched wall budget (paired by seed)
    p3_rows = []
    diffs: List[float] = []
    for m in f["num_learners_m"]:
        for gate in f["gate"]:
            for merge in f["merge"]:
                lr_a, ca = best_lr(m, merge, "quorum_grace", gate)
                lr_r, cr = best_lr(m, merge, "quorum_random", gate)
                for s_idx, seed in enumerate(f["seeds"]):
                    pa, pr = ca["per_seed"][s_idx], cr["per_seed"][s_idx]
                    budget = min(pa["total_wall"], pr["total_wall"])
                    la = loss_at_budget(pa["loss_curve"], pa["wall_curve"], budget)
                    lr_ = loss_at_budget(pr["loss_curve"], pr["wall_curve"], budget)
                    diffs.append(lr_ - la)  # positive = adaptive better
                    p3_rows.append({"M": m, "gate": gate, "merge": merge, "seed": seed, "budget": budget, "loss_adaptive": la, "loss_random": lr_})
    p3_mean, p3_hw = mean_ci(diffs) if len(diffs) >= 2 else (diffs[0], 0.0)
    # with n diffs > 3 use t with df=n-1 ~ normal approx is not pre-registered; keep the conservative df=2 critical value
    p3_status = "supported" if p3_mean - p3_hw > 0 else ("refuted" if p3_mean + p3_hw < 0 else "inconclusive (CI includes 0)")

    # P4: gate gain under quorum_grace and under all (negative control: must be 0)
    p4 = {}
    for adm in ("quorum_grace", "all"):
        gains: List[float] = []
        for m in f["num_learners_m"]:
            for merge in f["merge"]:
                lr_u, cu = best_lr(m, merge, adm, "ungated")
                _, cg_same_lr = lr_u, cells[f"M{m}|{merge}|{adm}|cgad|lr{lr_u}"]
                for s_idx in range(len(f["seeds"])):
                    gains.append(cu["per_seed"][s_idx]["final_loss"] - cg_same_lr["per_seed"][s_idx]["final_loss"])
        gm, gh = mean_ci(gains)
        p4[adm] = {"gain_mean": gm, "gain_ci95": gh, "n": len(gains)}
    p4_negative_control_ok = abs(p4["all"]["gain_mean"]) < 1e-9
    p4_status = (
        "supported" if (p4["quorum_grace"]["gain_mean"] - p4["quorum_grace"]["gain_ci95"] > 0)
        else ("refuted" if (p4["quorum_grace"]["gain_mean"] + p4["quorum_grace"]["gain_ci95"] < 0) else "inconclusive (CI includes 0)")
    )

    # ---------------- controls ----------------
    pos_rows = []
    for m in f["num_learners_m"]:
        for seed in f["seeds"]:
            clean = cells[f"M{m}|direct|all|ungated|lr0.7"]["per_seed"][f["seeds"].index(seed)]["final_loss"]
            burst = burst_losses[(m, seed)]
            pos_rows.append({"M": m, "seed": seed, "clean_loss": clean, "stale_burst_loss": burst})
    positive_ok = all(r["stale_burst_loss"] > r["clean_loss"] for r in pos_rows)

    ident_rda, ident_dir = ident_losses
    neg1_ok = abs(ident_rda - ident_dir) <= 1e-5 * max(1.0, abs(ident_dir))

    ctrl0 = AdaptiveGraceWindowController(overlap_steps_tau=2, safety_margin_gamma=0.0, ema_beta=0.5, sync_duration_s=0.1)
    neg2_ok = True
    for t in sample_arrivals(8, int(prereg["fixed"]["outer_rounds"]), 1):
        ordered = sorted(t)
        ctrl0.observe_step_duration(statistics.median(ordered))
        ctrl0.observe_quorum_duration(ordered[3])
        arr = [LearnerArrival(str(i), t[i], 1, 1) for i in range(8)]
        sel = select_quorum(arr, 4, ctrl0.grace_window_seconds())
        neg2_ok &= len(sel.admitted_learner_ids) == 4 and ctrl0.grace_window_seconds() == 0.0

    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E5-toy/v1.0",
        "experiment_id": "E5-toy",
        "prereg_sha256": prereg_sha,
        "num_runs": n_runs,
        "cells": cells,
        "analysis": {
            "P2": {"best_outer_lr": p2, "mean_pairwise_cosine_all_ungated": cos_all, "applicable": p2_applicable, "status": p2_status},
            "P3": {"paired_diff_random_minus_adaptive_mean": p3_mean, "ci95_halfwidth_df2": p3_hw, "n_pairs": len(diffs), "status": p3_status, "rows": p3_rows},
            "P4": {**p4, "negative_control_all_gain_is_zero": p4_negative_control_ok, "status": p4_status},
        },
        "controls": {
            "positive_stale_burst": {"rows": pos_rows, "passed": positive_ok},
            "negative_rda_identity": {"rda_loss": ident_rda, "direct_loss": ident_dir, "passed": neg1_ok},
            "negative_grace_gamma0": {"passed": neg2_ok},
        },
        "controls_all_passed": bool(positive_ok and neg1_ok and neg2_ok),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scope_note": "Toy synthetic LM on one CPU; arrival times from a parametric model; decides RQ2-RQ4 only at this scale.",
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    colors = {"direct": "#e67e22", "rda": "#2980b9"}
    for ax, m in zip(axes, f["num_learners_m"]):
        for merge in f["merge"]:
            means = [cells[f"M{m}|{merge}|all|ungated|lr{lr}"]["final_loss_mean"] for lr in f["outer_lr"]]
            hws = [cells[f"M{m}|{merge}|all|ungated|lr{lr}"]["final_loss_ci95"] for lr in f["outer_lr"]]
            ax.errorbar(f["outer_lr"], means, yerr=hws, marker="o", capsize=3, color=colors[merge], label=f"{merge} merge (3 seeds, 95% CI)")
        ax.set_title(f"E5-toy: held-out loss vs outer LR, M = {m}")
        ax.set_xlabel("outer learning rate $\\eta_{\\mathrm{outer}}$")
        ax.set_ylabel("held-out loss after 12 rounds")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig5_e5_toy_rda_vs_direct.{ext}", dpi=160)
    plt.close(fig)

    print(json.dumps({k: report["analysis"][k]["status"] for k in ("P2", "P3", "P4")} | {"controls_all_passed": report["controls_all_passed"], "runs": n_runs, "elapsed_s": report["elapsed_seconds"], "receipt": report["receipt_sha256"]}, indent=2))
    return 0 if report["controls_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
