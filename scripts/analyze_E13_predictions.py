"""Registered E13 predictions P13a-P13d, computed from ``E13_swarm_local_report.json`` (CPU only).

Implements exactly the rules of ``artifacts/experiments/E13_swarm_local_prereg.json`` (+ amendments); nothing is tuned:

  P13a  S1 (and S1t) equals S0 within 1e-6 for every seed                      (control reported by the aggregator)
  P13b  retention rho_s = (L_round0 - L_S2,s) / (L_round0 - L_S0,s); with 3 seeds: supported if mean - half-width >= 0.5,
        refuted if mean + half-width < 0.5, else inconclusive (t = 4.302652729911275, df = 2); fewer seeds: descriptive, no status.
        The retention against the stricter 0.75 bar is reported as well.
  P13c  actual upload bytes of S2 / raw fp32 upload bytes of the same posts <= 0.30 (ideal entropy-coded ratio reported)
  P13d  retention of S3, S4, S5 (each mechanism off), descriptive, qwen0.5b only
A model whose controls did not all pass gets every status 'invalid (controls failed: ...)' (``run_is_valid_iff``).

    python3 scripts/analyze_E13_predictions.py [--report R] [--out O]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
T_CRIT_DF2 = 4.302652729911275
RETENTION_BAR, RETENTION_STRICT, WIRE_BAR = 0.5, 0.75, 0.30


def mean_hw(xs: List[float]) -> Optional[Dict[str, float]]:
    n = len(xs)
    if n < 3:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"mean": m, "half_width": T_CRIT_DF2 * sd / math.sqrt(n) if n == 3 else float("nan")}


def per_seed(cell: Dict[str, Any], key: str) -> Dict[int, float]:
    return {s: v for s, v in zip(cell["seeds"], cell[key])}


def retention(l0: Dict[int, float], l_cell: Dict[int, float], l_s0: Dict[int, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for s in l_cell:
        if s in l_s0 and s in l0 and l0[s] - l_s0[s] != 0.0:
            out[s] = (l0[s] - l_cell[s]) / (l0[s] - l_s0[s])
    return out


def status_p13b(rho: Dict[int, float]) -> Dict[str, Any]:
    vals = [rho[s] for s in sorted(rho)]
    mh = mean_hw(vals)
    if mh is None:
        return {"status": f"descriptive (n={len(vals)})", "per_seed": rho}
    lo, hi = mh["mean"] - mh["half_width"], mh["mean"] + mh["half_width"]
    st = "supported" if lo >= RETENTION_BAR else ("refuted" if hi < RETENTION_BAR else "inconclusive")
    return {"status": st, "per_seed": rho, "mean": mh["mean"], "half_width": mh["half_width"], "interval": [lo, hi],
            "bar": RETENTION_BAR, "meets_strict_bar_0.75": bool(lo >= RETENTION_STRICT)}


def analyse_model(m: Dict[str, Any]) -> Dict[str, Any]:
    cells = m["cells"]
    failed = [k for k, v in m["controls"].items() if isinstance(v, dict) and v.get("passed") is False]
    valid = not failed
    res: Dict[str, Any] = {"controls_failed": failed, "valid": valid, "hosting_fallback": bool(m.get("hosting_fallback_cells"))}
    if "S0" not in cells:
        res["note"] = "no S0 cell"
        return res
    l0, ls0 = per_seed(cells["S0"], "initial_loss_per_seed"), per_seed(cells["S0"], "final_loss_per_seed")
    res["S0_improvement_per_seed"] = {s: l0[s] - ls0[s] for s in ls0}
    # P13a
    ident = m["controls"].get("s1_equals_s0", {})
    res["P13a"] = {"status": "supported" if ident.get("passed") else "refuted", "rows": ident.get("rows")}
    # P13b
    if "S2" in cells:
        res["P13b"] = status_p13b(retention(l0, per_seed(cells["S2"], "final_loss_per_seed"), ls0))
    # P13c
    if "S2" in cells:
        w = cells["S2"]["wire"]
        ra, ri = w.get("compression_ratio_actual"), w.get("compression_ratio_ideal")
        if ra:
            res["P13c"] = {"status": "supported" if 1.0 / ra <= WIRE_BAR else "refuted", "upload_ratio_actual": 1.0 / ra,
                           "upload_ratio_ideal": (1.0 / ri) if ri else None, "bar": WIRE_BAR, "int16_posts_total": w.get("int16_posts_total")}
    # P13d (descriptive)
    d: Dict[str, Any] = {}
    for c in ("S3", "S4", "S5"):
        if c in cells:
            rho = retention(l0, per_seed(cells[c], "final_loss_per_seed"), ls0)
            vals = [rho[s] for s in sorted(rho)]
            d[c] = {"name": cells[c]["name"], "retention_per_seed": rho, "mean_retention": sum(vals) / len(vals) if vals else None,
                    "admitted_fraction_mean": cells[c].get("admitted_fraction_mean"), "late_discards_total": cells[c].get("late_discards_total")}
    if d:
        res["P13d"] = d
    if not valid:
        for k in ("P13a", "P13b", "P13c"):
            if k in res:
                res[k] = {"status": f"invalid (controls failed: {', '.join(failed)})", "computed": res[k]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "artifacts" / "experiments" / "E13_swarm_local_report.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "E13_swarm_local_predictions.json"))
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"schema_version": "hivemind-experiment-E13-predictions/v1.0", "experiment_id": "E13-swarm-local",
                           "source_report": Path(a.report).name, "source_report_receipt": rep.get("receipt_sha256"),
                           "prereg_sha256": hashlib.sha256((ROOT / "artifacts" / "experiments" / "E13_swarm_local_prereg.json").read_bytes()).hexdigest(),
                           "rules": "scripts/analyze_E13_predictions.py docstring = the registered rules", "models": {}}
    for name, m in rep["models"].items():
        out["models"][name] = analyse_model(m)
    payload = json.dumps(out, sort_keys=True).encode("utf-8")
    out["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, r in out["models"].items():
        b = r.get("P13b", {})
        print(name, "valid" if r["valid"] else "INVALID", "| P13a", r.get("P13a", {}).get("status"), "| P13b", b.get("status"),
              f"mean {b['mean']:.3f} +- {b['half_width']:.3f}" if "mean" in b else "", "| P13c", r.get("P13c", {}).get("status"),
              f"{r['P13c']['upload_ratio_actual']:.4f}" if "P13c" in r and "upload_ratio_actual" in r["P13c"] else "")
        for c, v in r.get("P13d", {}).items():
            print("   ", c, v["name"], "mean retention", None if v["mean_retention"] is None else round(v["mean_retention"], 3), "admitted", v["admitted_fraction_mean"], "late", v["late_discards_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
