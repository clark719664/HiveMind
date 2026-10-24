"""Registered statuses of E14 (H14a/H14b) and E15 (H15a..H15e), computed from the committed reports by the preregistered rules.

CPU only, no tuning: every threshold below is quoted from the pre-registration files and nothing else is applied.

E14 (``E14_swarm_fairness_prereg.json``, per model): per-seed paired difference ``loss(S2) - loss(cell)`` of the final held-out loss, df = 2 95 %
t-interval (t = 4.302652729911275) = mean +- t * sd / sqrt(3). H14a (cell S6) / H14b (cell S7): supported if the interval is above 0, refuted if it is
below 0, inconclusive if it includes 0 (or fewer than 3 paired seeds). Retention (descriptive H14c) rho = (L_round0 - L_cell) / (L_round0 - L_S0) per seed.

E15 (``E15_gwaya_rft_prereg.json`` + amendment 1), differences in pass@1 percentage points, exact two-sided McNemar p from the discordant counts:
  H15a  C2 - C0 >= +3.0 points and p < 0.05 -> supported; C2 - C0 < 0 and p < 0.05 -> refuted; otherwise inconclusive
  H15b  C1 - C0 >= +5.0 points and p < 0.05 -> supported; C1 - C0 <= 0 -> refuted; otherwise inconclusive
  H15c  descriptive: C3 - C1 and C3 - C0 (no status)
  H15d  every stage peak reserved <= 8.0 GiB (all stages ok) -> supported; an out-of-memory or a peak above 8.0 GiB -> refuted
  H15e  descriptive: HumanEval C2h - C0h (no status)
  run_is_valid_iff all E15 controls pass; otherwise every status is reported as invalid (the computed word is kept next to it).

    python3 scripts/analyze_E14_E15_hypotheses.py [--artifacts DIR] [--out-dir DIR]

Writes E14_hypotheses.json and E15_hypotheses.json (each with a ``receipt_sha256`` over its canonical JSON without that field).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: a GPU service may be running on this host

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments"
T_CRIT_DF2 = 4.302652729911275
E14_MODELS = ("qwen0.5b", "granite1b")
E14_CELLS = {"H14a": "S6", "H14b": "S7"}
H15A_MIN_POINTS, H15B_MIN_POINTS, ALPHA, PEAK_LIMIT_GIB = 3.0, 5.0, 0.05, 8.0


def receipt(doc: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


# ---------------------------------------------------------------------------------------------- E14
def t_interval(values: Sequence[float]) -> Dict[str, Any]:
    """df = n - 1 = 2 95 % t-interval; only n = 3 is registered (the t value is the df = 2 one)."""
    if len(values) != 3:
        return {"n": len(values), "decidable": False}
    m = statistics.fmean(values)
    h = T_CRIT_DF2 * statistics.stdev(values) / math.sqrt(3)
    return {"n": 3, "decidable": True, "mean": m, "half_width": h, "interval": [m - h, m + h]}


def interval_status(iv: Dict[str, Any]) -> str:
    """Registered H14 rule: interval above 0 supported, below 0 refuted, includes 0 inconclusive."""
    if not iv.get("decidable"):
        return "inconclusive (needs exactly 3 paired seeds)"
    lo, hi = iv["interval"]
    return "supported" if lo > 0 else "refuted" if hi < 0 else "inconclusive"


def paired_s2_minus_cell(cells: Dict[str, Any], cell: str) -> Optional[Dict[str, float]]:
    try:
        s2 = dict(zip(cells["S2"]["seeds"], cells["S2"]["final_loss_per_seed"]))
        c = dict(zip(cells[cell]["seeds"], cells[cell]["final_loss_per_seed"]))
    except (KeyError, TypeError):
        return None
    return {str(s): s2[s] - c[s] for s in sorted(c) if s in s2}


def retention(cells: Dict[str, Any], cell: str) -> Optional[Dict[str, float]]:
    """Registered rho_s = (L_round0 - L_cell,s) / (L_round0 - L_S0,s)."""
    try:
        s0, c = cells["S0"], cells[cell]
        l0 = dict(zip(s0["seeds"], s0["initial_loss_per_seed"]))
        f0 = dict(zip(s0["seeds"], s0["final_loss_per_seed"]))
        fc = dict(zip(c["seeds"], c["final_loss_per_seed"]))
    except (KeyError, TypeError):
        return None
    out = {str(s): (l0[s] - fc[s]) / (l0[s] - f0[s]) for s in sorted(fc) if s in l0 and s in f0 and l0[s] != f0[s]}
    return out or None


def e14_model_decision(model_report: Dict[str, Any], twin_bit_identical: Optional[bool] = None) -> Dict[str, Any]:
    cells = model_report.get("cells") or {}
    out: Dict[str, Any] = {}
    reported = ((model_report.get("e14") or {}).get("paired_vs_S2")) or {}
    for h, cell in E14_CELLS.items():
        diffs = paired_s2_minus_cell(cells, cell)
        if diffs is None:
            out[h] = {"cell": cell, "status": "inconclusive (cell or S2 missing)", "decidable": False}
            continue
        iv = t_interval(list(diffs.values()))
        row: Dict[str, Any] = {"cell": cell, "S2_minus_cell_per_seed": diffs, **iv, "status": interval_status(iv)}
        rep = reported.get(cell) or {}
        if iv.get("decidable") and "mean" in rep and "ci95" in rep:
            row["report_check"] = {"report_mean": rep["mean"], "report_ci95": rep["ci95"], "report_direction": rep.get("direction"),
                                   "agrees": abs(rep["mean"] - iv["mean"]) < 1e-9 and abs(rep["ci95"] - iv["half_width"]) < 1e-9}
        out[h] = row
    ret: Dict[str, Any] = {}
    for cell in ("S2", "S6", "S7"):
        rho = retention(cells, cell)
        if rho:
            iv = t_interval(list(rho.values()))
            ret[cell] = {"per_seed": rho, **{k: v for k, v in iv.items() if k in ("mean", "half_width", "interval", "n")}}
    out["retention_vs_S0"] = ret
    out["retention_bar_P13b"] = 0.5
    lat = {c: ((cells.get(c) or {}).get("round_latency_s") or {}).get("p50") for c in ("S0", "S2", "S6", "S7")}
    cost: Dict[str, Any] = {"round_latency_p50_s": lat}
    if lat.get("S2") and lat.get("S6"):
        cost["S6_over_S2_latency"] = lat["S6"] / lat["S2"]
    if lat.get("S2") and lat.get("S7"):
        cost["S7_over_S2_latency"] = lat["S7"] / lat["S2"]
    s6 = (((cells.get("S6") or {}).get("e14") or {}).get("fair")) or {}
    s7 = (((cells.get("S7") or {}).get("e14") or {}).get("carry")) or {}
    cost["S6_fair"] = {k: s6.get(k) for k in ("extra_wait_s_total", "extra_wait_s_mean_per_round", "rounds_with_extra_wait", "forced_admissions_total", "fair_wait_timeouts_total")}
    cost["S7_carry"] = {k: s7.get(k) for k in ("carried_count_total", "carried_tokens_total", "staleness_hist", "too_stale_discarded_total")}
    cost["admitted_fraction_mean"] = {c: (cells.get(c) or {}).get("admitted_fraction_mean") for c in ("S2", "S6", "S7")}
    cost["late_discards_total"] = {c: (cells.get(c) or {}).get("late_discards_total") for c in ("S2", "S6", "S7")}
    out["descriptive_costs"] = cost
    ctrl = model_report.get("controls") or {}
    flags = {k: (v.get("passed") if isinstance(v, dict) else None) for k, v in ctrl.items()}
    evaluated = {k: v for k, v in flags.items() if v is not None}
    out["controls"] = {"flags": flags, "failed": sorted(k for k, v in evaluated.items() if v is False), "not_evaluated": sorted(k for k, v in flags.items() if v is None),
                       "s0_twin_bit_identical": twin_bit_identical, "report_controls_all_passed": model_report.get("controls_all_passed"),
                       "valid": not any(v is False for v in evaluated.values()) and twin_bit_identical is not False,
                       "note": "report_controls_all_passed is false only because controls of cells outside the E14 grid (S1 identity, raw wire bytes) are not evaluated (null)"}
    return out


def twin_from_raw(raw: Optional[Dict[str, Any]]) -> Optional[bool]:
    for run in (raw or {}).get("runs", []):
        nf = run.get("identity_noise_floor")
        if run.get("cell") == "S0" and isinstance(nf, dict) and nf.get("available"):
            return bool(nf.get("bit_identical"))
    return None


def build_e14(art: Path) -> Dict[str, Any]:
    prereg = art / "E14_swarm_fairness_prereg.json"
    pdoc = load(prereg) or {}
    doc: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E14-hypotheses/v1.0", "experiment_id": "E14-swarm-fairness",
        "rules": (pdoc.get("hypotheses_and_rules") or {}), "prereg_sha256": file_sha(prereg), "t_crit_df2": T_CRIT_DF2,
        "scope": pdoc.get("scope"), "status_of_experiment": pdoc.get("status"), "models": {}, "sources": {},
    }
    for m in E14_MODELS:
        p = art / f"E14_report_{m}.json"
        rep = load(p)
        if rep is None:
            continue
        mr = (rep.get("models") or {}).get(m)
        if not isinstance(mr, dict):
            continue
        doc["models"][m] = e14_model_decision(mr, twin_from_raw(load(art / f"E14_raw_{m}.json")))
        doc["sources"][m] = {"file": p.name, "receipt_sha256": rep.get("receipt_sha256"), "file_sha256": file_sha(p)}
    doc["receipt_sha256"] = receipt(doc)
    return doc


# ---------------------------------------------------------------------------------------------- E15
def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Exact two-sided McNemar p (binomial, p = 0.5) from the discordant counts."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def h15a_status(points: float, p: float) -> str:
    if points >= H15A_MIN_POINTS and p < ALPHA:
        return "supported"
    if points < 0 and p < ALPHA:
        return "refuted"
    return "inconclusive"


def h15b_status(points: float, p: float) -> str:
    if points >= H15B_MIN_POINTS and p < ALPHA:
        return "supported"
    if points <= 0:
        return "refuted"
    return "inconclusive"


def h15d_status(stage_status: Dict[str, str], peaks: Dict[str, Optional[float]]) -> str:
    if any("oom" in str(s).lower() or "out of memory" in str(s).lower() for s in stage_status.values()):
        return "refuted"
    vals = [v for v in peaks.values() if v is not None]
    if any(v > PEAK_LIMIT_GIB for v in vals):
        return "refuted"
    if vals and all(str(s) == "ok" for s in stage_status.values()):
        return "supported"
    return "inconclusive (a stage did not finish ok)"


def paired_row(conds: Dict[str, Any], comps: Dict[str, Any], a: str, b: str, key: str) -> Optional[Dict[str, Any]]:
    """Paired difference b - a (points) with the exact McNemar p recomputed from the report's discordant counts."""
    c = comps.get(key)
    if not isinstance(c, dict) or a not in conds or b not in conds:
        return None
    points = 100.0 * (conds[b]["passes"] - conds[a]["passes"]) / conds[a]["n"]
    p = mcnemar_exact(int(c["b_only"]), int(c["a_only"]))
    return {"comparison": key, "n": c["n"], "points": points, "b_only": c["b_only"], "a_only": c["a_only"], "mcnemar_p": p,
            "report_mcnemar_p": c.get("mcnemar_p"), "report_diff_points": 100.0 * c["diff"] if "diff" in c else None,
            "report_agrees": abs(p - c.get("mcnemar_p", p)) < 1e-9 and abs(points - 100.0 * c.get("diff", points / 100.0)) < 1e-9}


def build_e15_from(report: Dict[str, Any]) -> Dict[str, Any]:
    conds, comps = report.get("conditions") or {}, report.get("paired_comparisons") or {}
    ctrl = report.get("controls") or {}
    valid = bool(ctrl.get("all_passed")) and report.get("controls_all_passed", True) is not False
    rows = {"H15a": paired_row(conds, comps, "C0", "C2", "C2-C0"), "H15b": paired_row(conds, comps, "C0", "C1", "C1-C0"),
            "H15c_C3_minus_C1": paired_row(conds, comps, "C1", "C3", "C3-C1"), "H15c_C3_minus_C0": paired_row(conds, comps, "C0", "C3", "C3-C0"),
            "H15e": paired_row(conds, comps, "C0h", "C2h", "C2h-C0h (HumanEval)")}
    stages = report.get("stages") or {}
    stage_status = {k: v.get("status") for k, v in stages.items() if isinstance(v, dict)}
    peaks = {k: (v.get("meter") or {}).get("peak_reserved_gib") for k, v in stages.items() if isinstance(v, dict) and "meter" in v}
    computed: Dict[str, str] = {}
    if rows["H15a"]:
        computed["H15a"] = h15a_status(rows["H15a"]["points"], rows["H15a"]["mcnemar_p"])
    if rows["H15b"]:
        computed["H15b"] = h15b_status(rows["H15b"]["points"], rows["H15b"]["mcnemar_p"])
    computed["H15d"] = h15d_status(stage_status, peaks)
    out: Dict[str, Any] = {"valid": valid, "validity_rule": "run_is_valid_iff all E15 controls pass (report controls.all_passed)"}
    for h, s in computed.items():
        out[h] = {"status": s if valid else f"invalid (controls failed); computed: {s}", "computed_status": s, **({k: v for k, v in rows[h].items()} if rows.get(h) else {})}
    out["H15d"].update({"peak_reserved_gib_per_stage": peaks, "max_peak_reserved_gib": max([v for v in peaks.values() if v is not None], default=None),
                        "limit_gib": PEAK_LIMIT_GIB, "stage_status": stage_status})
    out["H15c_descriptive"] = {"status": "descriptive (no status)", "C3_minus_C1": rows["H15c_C3_minus_C1"], "C3_minus_C0": rows["H15c_C3_minus_C0"]}
    out["H15e_descriptive"] = {"status": "descriptive (no status)", **(rows["H15e"] or {})}
    out["conditions"] = {k: {kk: v.get(kk) for kk in ("n", "passes", "pass_at_1", "wilson95")} for k, v in conds.items() if isinstance(v, dict)}
    out["energy_wh_per_stage"] = report.get("energy_wh_per_stage")
    return out


def build_e15(art: Path) -> Dict[str, Any]:
    rp, pp, ap = art / "E15_gwaya_rft_report.json", art / "E15_gwaya_rft_prereg.json", art / "E15_gwaya_rft_prereg_amendment_1.json"
    rep = load(rp)
    doc: Dict[str, Any] = {"schema_version": "hivemind-experiment-E15-hypotheses/v1.0", "experiment_id": "E15-gwaya-rft",
                           "rules": (load(pp) or {}).get("hypotheses"), "prereg_sha256": file_sha(pp), "amendment_1_sha256": file_sha(ap),
                           "scope": (load(pp) or {}).get("scope"),
                           "source": {"file": rp.name, "receipt_sha256": (rep or {}).get("receipt_sha256"), "file_sha256": file_sha(rp)}}
    doc["hypotheses"] = build_e15_from(rep) if rep else {"available": False}
    doc["receipt_sha256"] = receipt(doc)
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", default=str(EXP))
    ap.add_argument("--out-dir", default=None, help="where to write E14_hypotheses.json / E15_hypotheses.json (default: --artifacts)")
    a = ap.parse_args(argv)
    art = Path(a.artifacts)
    out = Path(a.out_dir) if a.out_dir else art
    out.mkdir(parents=True, exist_ok=True)
    e14, e15 = build_e14(art), build_e15(art)
    (out / "E14_hypotheses.json").write_text(json.dumps(e14, indent=2), encoding="utf-8")
    (out / "E15_hypotheses.json").write_text(json.dumps(e15, indent=2), encoding="utf-8")
    summary = {"E14": {m: {h: v[h]["status"] for h in E14_CELLS} for m, v in e14["models"].items()},
               "E15": {h: e15["hypotheses"][h]["status"] for h in ("H15a", "H15b", "H15d") if h in e15["hypotheses"]}}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
