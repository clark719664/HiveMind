#!/usr/bin/env python3
"""Assemble the NIGHT-MOE-1 results from the evaluator's ``*_summary.json`` files into one JSON and one Markdown report.

Verdicts follow the preregistered rule (artifacts/experiments/NIGHT_MOE_prereg.json): a subject adapter is VALID only when its
own-pack gain over the base model is significant (exact McNemar p < 0.05, accuracy above base) AND its shuffled-label control shows no
significant gain over base; a control that gains significantly VOIDs the subject. Cells outside the own pack (transfer and interference)
are flagged GAIN/HARM at p < 0.05 and at the Bonferroni level over all such cells. Nothing here re-scores a reply: every number is read
from a summary file the evaluator wrote.
Usage: python scripts/build_night_moe_report.py --night-out DIR [--out-json FILE] [--out-md FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUBJECTS: dict[str, str] = {"maths": "maths", "exoplanet": "exoplanet", "stars": "stars", "cyber": "cyber"}
ALPHA = 0.05


def load_summaries(night_out: Path) -> dict[str, dict[str, Any]]:
    """Every ``<tag>_summary.json`` under base/, adapters/ and interference/, keyed by its tag."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(night_out.glob("*/*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.name.removesuffix("_summary.json")] = data
    return out


def _cell(summary: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    vs = summary.get("vs_baseline") or {}
    p = vs.get("mcnemar_exact_p")
    only_a, only_b = vs.get("only_this_right"), vs.get("only_baseline_right")
    # The paired counts are on the items this summary scored (a --limit run scores a subset), so the delta comes from them, not from the full-pack base accuracy.
    delta = None
    if base is not None:
        delta = round((only_a - only_b) / summary["n"], 4) if only_a is not None else round(summary["accuracy"] - base["accuracy"], 4)
    flag = "BASE"
    if base is not None:
        flag = "NS"
        if p is not None and p < ALPHA and delta:
            flag = "GAIN" if delta > 0 else "HARM"
    return {
        "n": summary["n"], "correct": summary["correct"], "accuracy": summary["accuracy"],
        "base_accuracy_same_items": None if delta is None else round(summary["accuracy"] - delta, 4),
        "wilson95": summary.get("wilson95"), "unparsable": summary.get("unparsable"),
        "only_adapter_right": vs.get("only_this_right"), "only_base_right": vs.get("only_baseline_right"),
        "mcnemar_exact_p": p, "delta_vs_base": delta, "flag": flag,
    }


def subject_verdict(own: dict[str, Any], control: dict[str, Any] | None) -> tuple[str, str]:
    """(verdict, reason) under the preregistered rule."""
    if control is None:
        return "NOT_RUN", "no shuffled-label control was evaluated"
    if control["flag"] == "GAIN":
        return "VOID", f"shuffled-label control gained significantly over base (p={control['mcnemar_exact_p']})"
    if own["flag"] != "GAIN":
        return "NO_GAIN", f"own-pack change {own['delta_vs_base']:+.4f}, p={own['mcnemar_exact_p']}"
    return "VALID", f"own-pack gain {own['delta_vs_base']:+.4f} (p={own['mcnemar_exact_p']}); control {control['flag']}"


def build(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bases = {k.removeprefix("base_"): v for k, v in summaries.items() if k.startswith("base_")}
    subjects: dict[str, Any] = {}
    cells: list[tuple[str, str]] = []
    for subj, own_pack in SUBJECTS.items():
        own_s = summaries.get(f"{subj}_{own_pack}")
        if own_s is None:
            continue
        own = _cell(own_s, bases.get(own_pack))
        ctrl_s = summaries.get(f"{subj}_ctrl_{own_pack}")
        control = _cell(ctrl_s, bases.get(own_pack)) if ctrl_s else None
        verdict, reason = subject_verdict(own, control)
        others: dict[str, Any] = {}
        for tag, s in summaries.items():
            for prefix in (f"{subj}_", f"x_{subj}_"):
                if tag.startswith(prefix) and not tag.startswith(f"{subj}_ctrl_") and tag != f"{subj}_{own_pack}":
                    pack = tag.removeprefix(prefix)
                    if pack in bases:
                        others[pack] = _cell(s, bases[pack]) | {"kind": "transfer" if prefix == f"{subj}_" else "interference"}
                        cells.append((subj, pack))
        note = ""
        if control is not None and control["correct"] == 0:
            note = ("the shuffled control scored 0 correct, which is expected for exact-match numeric answers (a shuffled answer is almost never right), "
                    "so this control cannot gain and the VOID rule is not testable for this subject")
        missing = sorted(set(bases) - {own_pack} - set(others))
        subjects[subj] = {"own_pack": own_pack, "own": own, "control": control, "verdict": verdict,
                          "reason": reason, "control_note": note, "missing_cells": missing, "other_packs": others}
    bonferroni = ALPHA / len(cells) if cells else ALPHA
    for subj, pack in cells:
        c = subjects[subj]["other_packs"][pack]
        p = c["mcnemar_exact_p"]
        c["survives_bonferroni"] = bool(p is not None and p < bonferroni and c["flag"] in ("GAIN", "HARM"))
    return {
        "experiment": "NIGHT-MOE-1", "base_model": "Qwen2.5-Coder-7B-Instruct Q4_K_M (QLoRA r16 a32 adapters)",
        "alpha": ALPHA, "n_other_cells": len(cells), "bonferroni_alpha": round(bonferroni, 5),
        "baselines": {k: {"n": v["n"], "correct": v["correct"], "accuracy": v["accuracy"], "unparsable": v.get("unparsable")} for k, v in bases.items()},
        "subjects": subjects,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# NIGHT-MOE-1 results", "",
             f"Base: {report['base_model']}. Exact paired McNemar vs the base model on identical items; "
             f"alpha {report['alpha']}, Bonferroni alpha {report['bonferroni_alpha']} over {report['n_other_cells']} transfer/interference cells.", "",
             "## Subjects", "", "| Subject | Own pack | Base | Adapter | Delta | p | Control | Control p | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for subj, s in report["subjects"].items():
        o, c = s["own"], s["control"]
        base_acc = report["baselines"][s["own_pack"]]["accuracy"]
        lines.append(f"| {subj} | {s['own_pack']} (n={o['n']}) | {base_acc} | {o['accuracy']} | {o['delta_vs_base']:+} | {o['mcnemar_exact_p']} | "
                     f"{c['accuracy'] if c else '-'} ({c['flag'] if c else '-'}) | {c['mcnemar_exact_p'] if c else '-'} | **{s['verdict']}** |")
    lines += ["", "## Transfer and interference cells", "", "| Adapter | Pack | Kind | n | Base (same items) | Adapter | Delta | p | Flag | Bonferroni | Unparsable |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for subj, s in report["subjects"].items():
        for pack, c in s["other_packs"].items():
            lines.append(f"| {subj} | {pack} | {c['kind']} | {c['n']} | {c['base_accuracy_same_items']} | {c['accuracy']} | {c['delta_vs_base']:+} | "
                         f"{c['mcnemar_exact_p']} | {c['flag']} | {'yes' if c['survives_bonferroni'] else 'no'} | {c['unparsable']} |")
    lines += ["", "## Reasons and gaps", ""]
    for subj, s in report["subjects"].items():
        lines.append(f"- **{subj}**: {s['verdict']}, {s['reason']}")
        if s["control_note"]:
            lines.append(f"  - control: {s['control_note']}")
        if s["missing_cells"]:
            lines.append(f"  - no result for: {', '.join(s['missing_cells'])} (stage timed out or was not run)")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--night-out", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/experiments/NIGHT_MOE_results.json"))
    ap.add_argument("--out-md", type=Path, default=Path("docs/NIGHT_MOE_RESULTS.md"))
    args = ap.parse_args()
    report = build(load_summaries(args.night_out))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    for subj, s in report["subjects"].items():
        print(subj, s["verdict"], s["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
