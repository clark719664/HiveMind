"""Aggregator for the exploratory E11b KV-lattice sweep (``artifacts/experiments/E11b_kv_lattice_sweep_prereg.json``).

Reads every ``E11b_raw_<model>_<dataset>_w<window>.json`` and the registered ``E11_raw_<model>.json`` in ``--in-dir`` (any
may be missing), reuses ``decide_model`` / ``per_seed_dnll`` / ``mean_ci`` from the E11 runner (not modified), and writes
a report with, per (model, dataset, window): the E11 decision output, the paired differences Z-A2 and Z-E8 for every
transform present (incl. kltc) at each rate, kltc-minus-klt per lattice and rate, and a ranking-agreement table against the
registered E11 run of the same model. Status is exploratory: nothing here changes a prediction status.

    python3 scripts/aggregate_E11b_sweep.py [--in-dir D] [--out F] [--fig-dir D]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_experiment_E11_kv_lattice import LATTICES, MODELS, RATES, decide_model, mean_ci, per_seed_dnll  # noqa: E402

STATUS = "exploratory, decides nothing"
RAW_RE = re.compile(r"^E11b_raw_(?P<model>.+)_(?P<dataset>[^_]+)_w(?P<window>\d+)\.json$")
FIG_NAME = "fig9_e11b_sweep"
CLAMP = 1e-5  # display floor for non-positive dNLL on the log axis


def _diff(raw: Dict[str, Any], a: str, b: str) -> Optional[Dict[str, Any]]:
    """Paired per-seed difference dNLL(a) - dNLL(b) with its df=2 interval, or None if either cell is absent."""
    cells = {r["cell"] for r in raw["runs"]}
    if a not in cells or b not in cells:
        return None
    d = [x - y for x, y in zip(per_seed_dnll(raw, a), per_seed_dnll(raw, b))]
    m, h = mean_ci(d)
    return {"mean": m, "ci95": h, "per_seed": d, "excludes_zero": bool(abs(m) > h)}


def transforms_present(raw: Dict[str, Any]) -> List[str]:
    cells = {r["cell"] for r in raw["runs"]}
    return [t for t in ("none", "klt", "kltc") if any(c.endswith("|" + t) for c in cells)]


def paired_differences(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Keys 'Z-A2@R=r|t', 'Z-E8@R=r|t' and 'kltc-klt:<lattice>@R=r'."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in transforms_present(raw):
        for r in RATES:
            for other in ("A2", "E8"):
                d = _diff(raw, f"Z@{r}|{t}", f"{other}@{r}|{t}")
                if d is not None:
                    out[f"Z-{other}@R={r}|{t}"] = d
    for k in LATTICES:
        for r in RATES:
            d = _diff(raw, f"{k}@{r}|kltc", f"{k}@{r}|klt")
            if d is not None:
                out[f"kltc-klt:{k}@R={r}"] = d
    return out


def _sign(m: float) -> int:
    return (m > 0) - (m < 0)


def agreement(e11b: Dict[str, Dict[str, Any]], e11: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per paired difference present in both runs: same sign of the mean, and whether both CIs exclude 0."""
    rows = {}
    for key, b in e11b.items():
        a = e11.get(key)
        if a is None:
            continue
        rows[key] = {"e11_mean": a["mean"], "e11b_mean": b["mean"], "same_sign": _sign(a["mean"]) == _sign(b["mean"]),
                     "e11_excludes_zero": a["excludes_zero"], "e11b_excludes_zero": b["excludes_zero"],
                     "both_exclude_zero": bool(a["excludes_zero"] and b["excludes_zero"])}
    n = len(rows)
    return {"rows": rows, "n_compared": n, "n_same_sign": sum(r["same_sign"] for r in rows.values()),
            "n_both_exclude_zero": sum(r["both_exclude_zero"] for r in rows.values()),
            "n_same_sign_and_both_exclude_zero": sum(r["same_sign"] and r["both_exclude_zero"] for r in rows.values())}


def _analyse(raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        decision = decide_model(raw)
    except (KeyError, ZeroDivisionError, ValueError, StopIteration) as e:  # incomplete raw file
        decision = {"error": f"decide_model failed: {type(e).__name__}: {e}"}
    return {"decision": decision, "transforms_present": transforms_present(raw), "paired_differences": paired_differences(raw)}


def _load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(in_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (report without receipt, per-entry R=3 dNLL table used by the figure)."""
    registered: Dict[str, Dict[str, Any]] = {}
    for key in MODELS:
        p = in_dir / f"E11_raw_{key}.json"
        if p.exists():
            registered[key] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "raw": _load(p)}
    entries: Dict[str, Any] = {}
    fig_rows: Dict[str, Any] = {}
    for p in sorted(in_dir.glob("E11b_raw_*.json")):
        mt = RAW_RE.match(p.name)
        if not mt:
            continue
        model, dataset, window = mt["model"], mt["dataset"], int(mt["window"])
        raw = _load(p)
        res = _analyse(raw)
        res["raw_file"], res["raw_sha256"] = p.name, hashlib.sha256(p.read_bytes()).hexdigest()
        reg = registered.get(model)
        if reg is None:
            res["agreement_vs_E11"] = {"status": "no registered E11 run for this model"}
        else:
            res["agreement_vs_E11"] = {"e11_raw_file": reg["file"], **agreement(res["paired_differences"], paired_differences(reg["raw"]))}
        entries[f"{model}|{dataset}|w{window}"] = {"model": model, "dataset": dataset, "window": window, **res}
        present = {r["cell"] for r in raw["runs"]}
        fig_rows[f"{model}|{dataset}|w{window}"] = {c: per_seed_dnll_mean(raw, c) for c in sorted(present) if c.endswith("@3") or "@3|" in c}
    report = {
        "schema_version": "hivemind-experiment-E11b-kv-lattice-sweep/v1.0",
        "experiment_id": "E11b-kv-lattice-sweep",
        "status": STATUS,
        "rule": "no prediction status is changed by E11b; a disagreement with E11 is a scope limit of E11's decision",
        "registered_e11_inputs": {k: {"file": v["file"], "sha256": v["sha256"]} for k, v in registered.items()},
        "entries": entries,
    }
    return report, fig_rows


def per_seed_dnll_mean(raw: Dict[str, Any], cell: str) -> float:
    return mean_ci(per_seed_dnll(raw, cell))[0]


def plot(fig_rows: Dict[str, Any], fig_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({k.split("|")[0] for k in fig_rows})
    if not models:
        return []
    codecs = ["int@3"] + [f"{k}@3|{t}" for k in LATTICES for t in ("none", "klt", "kltc")]
    colors = {"int": "#7f8c8d", "Z": "#e67e22", "A2": "#27ae60", "E8": "#2980b9"}
    hatch = {"none": "", "klt": "//", "kltc": "xx"}
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        groups = sorted((k for k in fig_rows if k.startswith(model + "|")), key=lambda k: k.split("|")[1:])
        width = 0.8 / len(codecs)
        for gi, g in enumerate(groups):
            for ci, c in enumerate(codecs):
                if c not in fig_rows[g]:
                    continue
                kind = c.split("@")[0]
                t = c.split("|")[1] if "|" in c else ""
                ax.bar(gi + ci * width, max(fig_rows[g][c], CLAMP), width, color=colors[kind], hatch=hatch.get(t, ""), edgecolor="white",
                       label=c if c not in ax.get_legend_handles_labels()[1] else None)
        ax.set_yscale("log")
        ax.set_xticks([gi + 0.4 - width / 2 for gi in range(len(groups))])
        ax.set_xticklabels([g.split("|", 1)[1].replace("|", "\n") for g in groups], fontsize=8)
        ax.set_ylabel("dNLL vs unquantised (nats/token)")
        ax.set_title(f"E11b {model}, R=3 (non-positive clamped to {CLAMP:g})", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0][-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = fig_dir / f"{FIG_NAME}.{ext}"
        fig.savefig(p, dpi=160)
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    exp = PROJECT_ROOT / "artifacts" / "experiments"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", type=Path, default=exp)
    ap.add_argument("--out", type=Path, default=exp / "E11b_kv_lattice_sweep_report.json")
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "paper" / "figures")
    a = ap.parse_args(argv)
    report, fig_rows = build_report(a.in_dir)
    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(fig_rows, a.fig_dir)
    print(json.dumps({"status": STATUS, "n_entries": len(report["entries"]), "receipt": report["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
