"""Figures 9 and 10 of the preprint, drawn only from the receipts in artifacts/experiments/.

CPU only. Run: CUDA_VISIBLE_DEVICES= python3 paper/figures/make_fig9_fig10.py
fig9: retention of the synchronous baseline's improvement per cell (E13, E13c, E14).
fig10: E16 VRAM per tier and pass@1 raw vs loop per tier.
"""
from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts" / "experiments"
OUT = Path(__file__).resolve().parent
T_DF2 = 4.302652729911275


def load(name: str):
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def mean_hw(x):
    m = sum(x) / len(x)
    return m, T_DF2 * st.stdev(x) / len(x) ** 0.5


def retention(l0, s0, cell):
    return [(l0[i] - cell[i]) / (l0[i] - s0[i]) for i in range(len(s0))]


def fig9() -> None:
    e13 = load("E13_swarm_local_report.json")["models"]
    rot = load("E13_swarm_local_report_rotstraggle.json")["models"]["qwen0.5b"]["cells"]
    e14q = load("E14_report_qwen0.5b.json")["models"]["qwen0.5b"]["cells"]
    e14g = load("E14_report_granite1b.json")["models"]["granite1b"]["cells"]
    q = e13["qwen0.5b"]["cells"]
    g = e13["granite1b"]["cells"]
    s0q, l0q = q["S0"]["final_loss_per_seed"], q["S0"]["initial_loss_per_seed"]
    s0g, l0g = g["S0"]["final_loss_per_seed"], g["S0"]["initial_loss_per_seed"]

    left = [
        ("S2\nall on", retention(l0q, s0q, q["S2"]["final_loss_per_seed"]), "///"),
        ("S3\nno codec", retention(l0q, s0q, q["S3"]["final_loss_per_seed"]), "..."),
        ("S4\nno RDA", retention(l0q, s0q, q["S4"]["final_loss_per_seed"]), "xxx"),
        ("S5\nK=M", retention(l0q, s0q, q["S5"]["final_loss_per_seed"]), ""),
        ("E13c\nrotating", retention(l0q, s0q, rot["S2"]["final_loss_per_seed"]), "\\\\\\"),
        ("S6 fair\nquorum", retention(l0q, s0q, e14q["S6"]["final_loss_per_seed"]), "+++"),
        ("S7 deferred\ncarry", retention(l0q, s0q, e14q["S7"]["final_loss_per_seed"]), "ooo"),
    ]
    right = [
        ("S2 E13\n(registered)", retention(l0g, s0g, g["S2"]["final_loss_per_seed"]), "///"),
        ("S2 E14\n(re-run)", retention(l0g, s0g, e14g["S2"]["final_loss_per_seed"]), "///"),
        ("S6 fair\nquorum", retention(l0g, s0g, e14g["S6"]["final_loss_per_seed"]), "+++"),
        ("S7 deferred\ncarry", retention(l0g, s0g, e14g["S7"]["final_loss_per_seed"]), "ooo"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [7, 4]})
    for ax, data, title in (
        (axes[0], left, "Qwen2.5-0.5B (dense), M=4"),
        (axes[1], right, "Granite-3.1-1B-A400M (MoE), M=3"),
    ):
        xs = list(range(len(data)))
        for x, (lab, vals, hatch) in zip(xs, data):
            m, hw = mean_hw(vals)
            ax.bar(x, m, width=0.7, color="0.82", edgecolor="0.1", hatch=hatch, linewidth=0.8)
            ax.errorbar(x, m, yerr=hw, color="0.05", capsize=4, linewidth=1.3)
            ax.scatter([x] * len(vals), vals, s=14, color="0.1", zorder=3)
            if m >= 0:
                ax.text(x, m + hw + 0.04, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(x + 0.38, m, f"{m:.2f}", ha="left", va="center", fontsize=8)
        ax.axhline(0.5, color="0.2", linestyle=":", linewidth=1.2)
        ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="0.0", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([d[0] for d in data], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-1.5, 1.35)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("retention  (L0 - L_cell) / (L0 - L_S0)")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_e13_e14_retention.pdf")
    fig.savefig(OUT / "fig9_e13_e14_retention.png", dpi=150)
    plt.close(fig)


def fig10() -> None:
    r = load("E16_gwaya_lite_t4_report.json")
    probes = {p["key"]: p for p in r["probes"]}
    tiers = ["1.5b", "3b", "7b"]
    labels = ["1.5B", "3B", "7B"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    w = 0.34
    for j, (ctx, hatch) in enumerate(((4096, ""), (8192, "///"))):
        vals = [probes[f"{t}:{ctx}"]["measured_mib"] for t in tiers]
        est = [probes[f"{t}:{ctx}"]["estimator_mib"] for t in tiers]
        xs = [i + (j - 0.5) * w for i in range(3)]
        ax.bar(xs, vals, width=w, color="0.80" if j == 0 else "0.55", edgecolor="0.1", hatch=hatch, label=f"measured, ctx {ctx}")
        ax.scatter(xs, est, marker="D", color="black", s=22, zorder=3, label="PR #21 estimate" if j == 0 else None)
        for x, v in zip(xs, vals):
            ax.text(x, 60, str(v), ha="center", fontsize=8, rotation=90, va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.axhline(6656, color="0.1", linestyle=":", linewidth=1.3)
    ax.text(-0.45, 6720, "6656 MiB threshold (H16a)", fontsize=8, va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_ylim(0, 7600)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_title("VRAM on a Tesla T4 (Q4_K_M, Ollama)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    tiers_acc = r["analysis"]["tiers"]
    raw = [tiers_acc[t]["R4096"] for t in tiers]
    loop = [tiers_acc[t]["L4096"] for t in tiers]
    lad = r["analysis"]["escalation_ladder"]
    w = 0.34
    for j, (series, hatch, name) in enumerate(((raw, "", "raw greedy (R)"), (loop, "///", "GWAYA loop (L)"))):
        xs = [i + (j - 0.5) * w for i in range(3)]
        p = [100 * s["pass_at_1"] for s in series]
        lo = [100 * s["pass_at_1"] - 100 * s["wilson95"][0] for s in series]
        hi = [100 * s["wilson95"][1] - 100 * s["pass_at_1"] for s in series]
        ax.bar(xs, p, width=w, color="0.80" if j == 0 else "0.55", edgecolor="0.1", hatch=hatch, label=name)
        ax.errorbar(xs, p, yerr=[lo, hi], fmt="none", ecolor="0.05", capsize=3)
        for x, v in zip(xs, p):
            ax.text(x, 3, f"{v:.1f}", ha="center", fontsize=8, rotation=90, va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    lp = 100 * lad["ladder_pass_at_1"]
    ax.bar([3 + 0.17], [lp], width=w, color="0.55", edgecolor="0.1", hatch="xxx", label="ladder 1.5B>3B>7B (L)")
    ax.text(3 + 0.17, 3, f"{lp:.1f}", ha="center", fontsize=8, rotation=90, va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels + ["ladder"])
    ax.set_ylim(0, 125)
    ax.set_ylabel("MBPP-sanitized pass@1 (%), n = 257")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Accuracy: raw vs GWAYA loop (Wilson 95%)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_e16_gwaya_lite_tiers.pdf")
    fig.savefig(OUT / "fig10_e16_gwaya_lite_tiers.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig9()
    fig10()
    print("ok")
