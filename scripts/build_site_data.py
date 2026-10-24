"""Build ``docs/site/data.js`` (``window.HM_DATA = {...};``) from the experiment artifacts that exist in ``artifacts/experiments/``.

The site's "What the real experiments measured" panel is fed ONLY by this file. The builder copies values and the status strings of the analysis files
verbatim and never computes a statistic, with labelled exceptions: the registered retention formula applied to the rotating-straggler diagnostic
(``derived: true``), the E14 hypothesis word mapped from the report's own ``direction`` by the registered rule (``hypothesis_status`` plus its rule), and
the wording of the two diagnostics, which comes from docs/benchmarks/T4_10h_campaign_report.md (``status_source``) because their JSON has no status. The
E15 report carries numbers only; the H15 and H14 words come from the analysis files E15_hypotheses.json / E14_hypotheses.json (scripts/analyze_E14_E15_hypotheses.py,
registered rules, receipt-tagged) and are copied verbatim into ``registered_hypotheses``; the E13d (six-domain replication), E16 (GWAYA-Lite on the T4) and
GWAYA-Lite live-check blocks (``e13d``, ``e16``, ``gwaya_lite_live``) copy statuses and numbers verbatim from their receipts. The page (docs/site/index.html + app.js)
does not render ``e13d``, ``e16``, ``gwaya_lite_live`` or ``registered_hypotheses`` yet; they are data only. A missing
source gives ``{"available": false, ...}``; a present source gives ``{"available": true, "source": <file name>, "receipt12": <first 12 hex chars of its
receipt_sha256>, "generated_utc": <build time>, ...}``. NaN/Infinity (not valid JSON) become null.

    python3 scripts/build_site_data.py [--artifacts DIR] [--out FILE] [--utc ISO8601]

Sources (all optional): E13_swarm_local_predictions.json, E13_swarm_local_report.json, E13_swarm_local_report_rotstraggle.json,
E13_swarm_local_report_m2proc.json, E14_report_<model>.json (qwen0.5b, granite1b; same schema as an E13 report with an ``e14`` section),
E14_swarm_fairness_prereg.json, E14_hypotheses.json, E15_gwaya_rft_report.json, E15_gwaya_rft_prereg.json, E15_hypotheses.json,
E13_swarm_local_predictions_m6.json + E13_swarm_local_report_m6.json (E13d), E16_gwaya_lite_t4_report.json + E16_gwaya_lite_t4_prereg.json,
gwaya_lite_live_check.json (in the artifacts directory or its parent).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "experiments"
DEFAULT_OUT = ROOT / "docs" / "site" / "data.js"
SCHEMA = "hivemind-site-data/v1"
E14_MODELS = ("qwen0.5b", "granite1b")
E15_REPORT_KEYS = ("status", "smoke", "model", "controls_all_passed", "notes", "written_utc")
# registered E14 rule (E14_swarm_fairness_prereg.json): paired difference loss(S2) - loss(cell), df=2 95% interval above 0 / below 0 / including 0
E14_DIRECTION_TO_STATUS = {"cell better than S2": "supported", "cell worse than S2": "refuted", "within CI of S2": "inconclusive"}


def clean(x: Any) -> Any:
    """Recursively replace NaN/Infinity by None so that the output is strict JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def load(path: Path) -> Optional[Dict[str, Any]]:
    """The parsed JSON object of ``path``, or None when the file is absent or unreadable (the block then becomes ``available: false``)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def tag(path: Path, doc: Dict[str, Any], utc: str) -> Dict[str, Any]:
    """Provenance tag of a block: source file name, first 12 hex chars of its receipt_sha256 (null when it has none) and the build time."""
    rec = doc.get("receipt_sha256")
    out: Dict[str, Any] = {"available": True, "source": path.name, "receipt12": rec[:12] if isinstance(rec, str) else None, "generated_utc": utc}
    if out["receipt12"] is None:
        out["file_sha256_12"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return out


def missing(path: Path, why: str) -> Dict[str, Any]:
    return {"available": False, "source": path.name, "reason": why}


def pick(d: Any, keys: List[str]) -> Dict[str, Any]:
    return {k: d[k] for k in keys if isinstance(d, dict) and k in d}


def cell_summary(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Verbatim per-cell fields of an E13/E14 report cell."""
    out = pick(cell, ["name", "status", "seeds", "failed_seeds", "initial_loss_per_seed", "final_loss_per_seed", "per_domain_final_mean", "admitted_fraction_mean", "late_discards_total"])
    wire = cell.get("wire")
    if isinstance(wire, dict):
        out["wire"] = pick(wire, ["compression_ratio_actual", "compression_ratio_ideal", "int16_posts_total"])
    lat = cell.get("round_latency_s")
    if isinstance(lat, dict):
        out["round_latency_s"] = pick(lat, ["p50", "p95", "p99", "n", "note"])
    if "e14" in cell:
        out["e14"] = cell["e14"]
    return out


def retention_per_seed(cells: Dict[str, Any], cell: str) -> Optional[Dict[str, float]]:
    """Registered retention rho_s = (L_round0 - L_cell,s) / (L_round0 - L_S0,s), computed from the report's own per-seed losses (derived, flagged as such)."""
    try:
        s0, c = cells["S0"], cells[cell]
        l0 = dict(zip(s0["seeds"], s0["initial_loss_per_seed"]))
        f0 = dict(zip(s0["seeds"], s0["final_loss_per_seed"]))
        fc = dict(zip(c["seeds"], c["final_loss_per_seed"]))
    except (KeyError, TypeError):
        return None
    out = {str(s): (l0[s] - fc[s]) / (l0[s] - f0[s]) for s in sorted(fc) if s in l0 and s in f0 and l0[s] - f0[s] != 0.0}
    return out or None


def controls_summary(controls: Dict[str, Any]) -> Dict[str, Any]:
    """control name -> passed (true / false / null = not evaluated in that job)."""
    return {k: (v.get("passed") if isinstance(v, dict) else None) for k, v in controls.items()}


# ---------------------------------------------------------------------------------------------- E13
def e13_registered(art: Path, utc: str, pred_name: str = "E13_swarm_local_predictions.json", rep_name: str = "E13_swarm_local_report.json") -> Dict[str, Any]:
    pred_path, rep_path = art / pred_name, art / rep_name
    pred = load(pred_path)
    if pred is None:
        return missing(pred_path, "predictions file not found")
    rep = load(rep_path)
    block = tag(pred_path, pred, utc)
    block["source_report"] = {"source": pred.get("source_report"), "receipt12": (pred.get("source_report_receipt") or "")[:12] or None}
    if rep is not None:
        block["report"] = {k: v for k, v in tag(rep_path, rep, utc).items() if k != "available"}
    block["rules"] = pred.get("rules")
    models: Dict[str, Any] = {}
    for name, m in (pred.get("models") or {}).items():
        entry = pick(m, ["valid", "controls_failed", "hosting_fallback", "S0_improvement_per_seed", "P13a", "P13b", "P13c", "P13d", "note"])
        rep_model = ((rep or {}).get("models") or {}).get(name)
        if isinstance(rep_model, dict):
            entry["report_cells"] = {c: cell_summary(v) for c, v in (rep_model.get("cells") or {}).items() if isinstance(v, dict) and v.get("seeds")}
            entry["controls_all_passed"] = rep_model.get("controls_all_passed")
        models[name] = entry
    block["models"] = models
    return block


def e13_rotating(art: Path, utc: str) -> Dict[str, Any]:
    path = art / "E13_swarm_local_report_rotstraggle.json"
    doc = load(path)
    if doc is None:
        return missing(path, "report not found")
    block = tag(path, doc, utc)
    block.update(pick(doc, ["controls_all_passed", "smoke"]))
    block["status"] = "exploratory diagnostic, decides nothing"
    block["status_source"] = "docs/benchmarks/T4_10h_campaign_report.md, section E13c (the report JSON itself carries no status word)"
    block["status_note"] = "cell pair S0 and S2 only; the report's own flag controls_all_passed is copied verbatim"
    models: Dict[str, Any] = {}
    for name, m in (doc.get("models") or {}).items():
        cells = m.get("cells") or {}
        entry: Dict[str, Any] = {"cells": {c: cell_summary(v) for c, v in cells.items() if isinstance(v, dict)}, "controls": controls_summary(m.get("controls") or {})}
        entry["controls_not_evaluated"] = sorted(k for k, v in entry["controls"].items() if v is None)
        paired = (m.get("paired_vs_S0") or {}).get("S2")
        if isinstance(paired, dict):
            entry["paired_S2_minus_S0"] = pick(paired, ["final_loss_minus_S0_per_seed", "mean", "ci95", "t_crit_df2", "ci_excludes_0", "direction", "status"])
        rho = retention_per_seed(cells, "S2")
        if rho:
            entry["retention_S2_derived"] = {"derived": True, "formula": "rho_s = (L_round0 - L_S2,s) / (L_round0 - L_S0,s), from this report's per-seed losses", "per_seed": rho,
                                             "mean": sum(rho.values()) / len(rho)}
        models[name] = entry
    block["models"] = models
    return block


def e13_m2(art: Path, utc: str) -> Dict[str, Any]:
    path = art / "E13_swarm_local_report_m2proc.json"
    doc = load(path)
    if doc is None:
        return missing(path, "report not found")
    block = tag(path, doc, utc)
    block.update(pick(doc, ["controls_all_passed", "smoke"]))
    block["status"] = "identity control only"
    block["status_source"] = "docs/benchmarks/T4_10h_campaign_report.md, section 'E13 with M=2 separate OS processes' (the report JSON itself carries no status word)"
    block["status_note"] = "S1 vs S0, 1 seed; the control's own passed flag is copied verbatim"
    models: Dict[str, Any] = {}
    for name, m in (doc.get("models") or {}).items():
        cells = m.get("cells") or {}
        ident = ((m.get("controls") or {}).get("s1_equals_s0")) or {}
        models[name] = {"cells": {c: cell_summary(v) for c, v in cells.items() if isinstance(v, dict)}, "s1_equals_s0": ident, "controls": controls_summary(m.get("controls") or {})}
    block["models"] = models
    return block


# ---------------------------------------------------------------------------------------------- E14
def prereg_block(art: Path, fname: str, keys: List[str]) -> Dict[str, Any]:
    path = art / fname
    doc = load(path)
    if doc is None:
        return {"available": False, "source": path.name}
    return {"available": True, "source": path.name, "file_sha256_12": hashlib.sha256(path.read_bytes()).hexdigest()[:12], **pick(doc, keys)}


def e14_model(doc: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    m = (doc.get("models") or {}).get(model)
    if not isinstance(m, dict):
        return None
    cells = m.get("cells") or {}
    entry: Dict[str, Any] = {"cells": {c: cell_summary(cells[c]) for c in ("S0", "S2", "S6", "S7") if isinstance(cells.get(c), dict)},
                             "controls": controls_summary(m.get("controls") or {}), "controls_all_passed": m.get("controls_all_passed")}
    e14 = m.get("e14")
    if isinstance(e14, dict):
        paired = e14.get("paired_vs_S2") or {}
        hyp: Dict[str, Any] = {}
        for cell, h in (("S6", "H14a"), ("S7", "H14b")):
            p = paired.get(cell)
            if not isinstance(p, dict):
                continue
            row = pick(p, ["S2_minus_cell_final_loss_per_seed", "mean", "ci95", "t_crit_df2", "ci_excludes_0", "direction", "status"])
            if p.get("direction") in E14_DIRECTION_TO_STATUS:
                row["hypothesis_status"] = E14_DIRECTION_TO_STATUS[p["direction"]]
                row["hypothesis_status_rule"] = "registered rule of E14_swarm_fairness_prereg.json applied to the report's own direction field"
            hyp[h] = {"cell": cell, **row}
        entry["e14"] = {"cells": e14.get("cells"), "hypotheses": hyp, "retention_vs_S0": e14.get("retention_vs_S0"), "note": e14.get("note")}
    return entry


def build_e14(art: Path, utc: str) -> Dict[str, Any]:
    prereg = prereg_block(art, "E14_swarm_fairness_prereg.json", ["registered_utc", "status", "hypotheses_and_rules", "scope"])
    models: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []
    for name in E14_MODELS:
        path = art / f"E14_report_{name}.json"
        doc = load(path)
        if doc is None:
            continue
        entry = e14_model(doc, name)
        if entry is not None:
            models[name] = entry
            sources.append(tag(path, doc, utc))
    if not models:
        return {"available": False, "source": "E14_report_<model>.json", "reason": "no E14 report file present yet (registered, not reported)", "prereg": prereg}
    first = sources[0]
    block = {"available": True, "source": [s["source"] for s in sources], "receipt12": [s["receipt12"] for s in sources], "generated_utc": utc,
             "first_source": first["source"], "models": models, "prereg": prereg}
    block["registered_hypotheses"] = registered_hypotheses(art, utc, "E14_hypotheses.json", lambda d: {"models": {
        m: {h: pick(v.get(h), ["cell", "status", "mean", "half_width", "interval", "S2_minus_cell_per_seed"]) for h in ("H14a", "H14b")} for m, v in (d.get("models") or {}).items()}})
    return block


def registered_hypotheses(art: Path, utc: str, fname: str, extract: Any) -> Dict[str, Any]:
    """Statuses of an analysis file (scripts/analyze_E14_E15_hypotheses.py), copied verbatim with the file's receipt; absent file -> available false."""
    path = art / fname
    doc = load(path)
    if doc is None:
        return missing(path, "analysis file not found (run scripts/analyze_E14_E15_hypotheses.py)")
    return {**tag(path, doc, utc), "rules_source": "registered rules of the pre-registration, applied by scripts/analyze_E14_E15_hypotheses.py", **extract(doc)}


# ---------------------------------------------------------------------------------------------- E15
def build_e15(art: Path, utc: str) -> Dict[str, Any]:
    prereg = prereg_block(art, "E15_gwaya_rft_prereg.json", ["registered_utc", "title", "hypotheses", "scope"])
    path = art / "E15_gwaya_rft_report.json"
    doc = load(path)
    if doc is None:
        return {"available": False, "source": path.name, "reason": "no E15 report file present yet (registered, not reported)", "prereg": prereg}
    block = tag(path, doc, utc)
    block["report_keys"] = sorted(doc)
    block["report_excerpt"] = pick(doc, list(E15_REPORT_KEYS))
    block["run_status"] = doc.get("status")  # the runner's own status word ('complete', 'failed', 'aborted by registered fail-fast rule')
    block["smoke"] = doc.get("smoke")
    conds = doc.get("conditions")
    if isinstance(conds, dict):
        block["conditions"] = {k: pick(v, ["label", "n", "passes", "pass_at_1", "wilson95", "accepted_gwaya", "stub_flagged_among_passing"]) for k, v in conds.items() if isinstance(v, dict)}
    comps = doc.get("paired_comparisons")
    if isinstance(comps, dict):
        block["paired_comparisons"] = {k: pick(v, ["n", "diff", "b_only", "a_only", "discordant", "mcnemar_p"]) for k, v in comps.items() if isinstance(v, dict)}
    block["verdicts_note"] = ("the report carries numbers only; the H15 verdicts (supported / refuted / inconclusive) are not computed here, apply the rules registered in the prereg; "
                              "the registered statuses, computed from this report by scripts/analyze_E14_E15_hypotheses.py, are copied from E15_hypotheses.json into registered_hypotheses")

    def e15_extract(d: Dict[str, Any]) -> Dict[str, Any]:
        h = d.get("hypotheses") or {}
        out: Dict[str, Any] = {"valid": h.get("valid"), "hypotheses": {k: pick(h.get(k), ["status", "points", "mcnemar_p", "b_only", "a_only", "max_peak_reserved_gib", "limit_gib"]) for k in ("H15a", "H15b", "H15d")}}
        out["descriptive"] = {"H15c_C3_minus_C1": pick((h.get("H15c_descriptive") or {}).get("C3_minus_C1"), ["points", "mcnemar_p"]),
                              "H15c_C3_minus_C0": pick((h.get("H15c_descriptive") or {}).get("C3_minus_C0"), ["points", "mcnemar_p"]),
                              "H15e": pick(h.get("H15e_descriptive"), ["points", "mcnemar_p"])}
        return out

    block["registered_hypotheses"] = registered_hypotheses(art, utc, "E15_hypotheses.json", e15_extract)
    stages = doc.get("stages")
    if isinstance(stages, dict):
        block["stage_meters"] = {k: pick(v.get("meter"), ["wall_s", "peak_reserved_gib", "energy_wh"]) for k, v in stages.items() if isinstance(v, dict) and isinstance(v.get("meter"), dict)}
        rft = stages.get("rft")
        if isinstance(rft, dict):
            block["rft_yield"] = pick(rft, ["problems", "problems_with_accepted", "mean_yield", "kept_pairs", "unverified_samples"])
    block["prereg"] = prereg
    return block


# ---------------------------------------------------------------------------------------------- E13d, E16, live check
def build_e13d(art: Path, utc: str) -> Dict[str, Any]:
    """E13d: the six-domain, six-worker replication (amendment 6, exploratory); same shape as e13.registered, statuses verbatim from the m6 predictions."""
    blk = e13_registered(art, utc, "E13_swarm_local_predictions_m6.json", "E13_swarm_local_report_m6.json")
    if blk.get("available"):
        blk["status"] = "exploratory replication (E13d, amendment 6); changes no registered E13 result"
    return blk


def build_e16(art: Path, utc: str) -> Dict[str, Any]:
    prereg = prereg_block(art, "E16_gwaya_lite_t4_prereg.json", ["registered_utc", "title", "hypotheses", "scope"])
    path = art / "E16_gwaya_lite_t4_report.json"
    doc = load(path)
    if doc is None:
        return {"available": False, "source": path.name, "reason": "no E16 report file present", "prereg": prereg}
    block = tag(path, doc, utc)
    an = doc.get("analysis") or {}
    block.update({"run_status": doc.get("status"), "smoke": doc.get("smoke"), "run_valid": doc.get("run_valid"), "n_problems": doc.get("n_problems"),
                  "environment": pick(doc.get("environment"), ["ollama_version", "gpu"]), "hypotheses": an.get("H16"), "escalation_ladder": an.get("escalation_ladder")})
    block["tiers"] = {t: {**{k: pick(v.get(k), ["label", "n", "passes", "pass_at_1", "wilson95", "accepted_gwaya", "model_seconds"]) for k in ("R4096", "L4096", "R8192") if k in v},
                          "paired_L_minus_R": pick(v.get("paired_L_minus_R"), ["n", "diff", "b_only", "a_only", "mcnemar_p"]),
                          "p2_gate_calibration": pick(v.get("p2_gate_calibration"), ["precision", "precision_wilson95", "recall", "verified", "status_H16e"]),
                          "determinism": v.get("determinism")} for t, v in (an.get("tiers") or {}).items() if isinstance(v, dict)}
    block["probes"] = [pick(p, ["key", "tier", "ctx", "measured_mib", "estimator_mib", "ratio_measured_over_estimate", "decode_tps_median", "prompt_eval_tps_median", "gpu_pct"])
                       for p in (doc.get("probes") or []) if isinstance(p, dict)]
    block["controls"] = {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in (doc.get("controls") or {}).items()}
    block["notes"] = doc.get("notes")
    block["prereg"] = prereg
    return block


def build_live(art: Path, utc: str) -> Dict[str, Any]:
    for path in (art / "gwaya_lite_live_check.json", art.parent / "gwaya_lite_live_check.json"):
        doc = load(path)
        if doc is not None:
            block = tag(path, doc, utc)
            if isinstance(doc.get("url"), str):  # the scheme is dropped: the static-site tests allow no URL other than the repository's
                block["listen_address"] = doc["url"].split("://", 1)[-1]
            block.update(pick(doc, ["written_utc", "health", "models", "tasks", "untested_request", "sse", "foreign_host_status", "foreign_origin_status", "listening",
                                    "gpu_memory_used_mib_after", "ollama_resident_after", "all_ok"]))
            block["scope"] = "smoke check of three small tasks on loopback, not a benchmark; the sandbox is the E15 sandbox, not GWAYA's own oracle"
            return block
    return {"available": False, "source": "gwaya_lite_live_check.json", "reason": "live-check receipt not found"}


def build(art: Path, utc: str) -> Dict[str, Any]:
    return clean({
        "schema": SCHEMA, "generated_utc": utc,
        "note": "Generated by scripts/build_site_data.py from artifacts/experiments/*.json. Values and status strings are copied verbatim; nothing here is computed except where 'derived' is true.",
        "e13": {"registered": e13_registered(art, utc), "rotating_straggler": e13_rotating(art, utc), "m2_process_identity": e13_m2(art, utc)},
        "e14": build_e14(art, utc),
        "e15": build_e15(art, utc),
        "e13d": build_e13d(art, utc),
        "e16": build_e16(art, utc),
        "gwaya_lite_live": build_live(art, utc),
    })


def render(data: Dict[str, Any]) -> str:
    return "// Generated by scripts/build_site_data.py - do not edit by hand.\nwindow.HM_DATA = " + json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + ";\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--utc", default=None, help="build timestamp to embed (default: now, UTC)")
    a = ap.parse_args(argv)
    utc = a.utc or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = build(Path(a.artifacts), utc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    for blk in ("e13", "e14", "e15", "e13d", "e16", "gwaya_lite_live"):
        sub = data[blk]
        if blk == "e13":
            print(blk, {k: v.get("available") for k, v in sub.items()})
        else:
            print(blk, "available" if sub.get("available") else "absent")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
