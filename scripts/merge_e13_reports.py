"""Merge the per-model E13 reports (``E13_swarm_local_report_<model>.json``) into the combined ``E13_swarm_local_report.json``.

Why: the shared default report of the runner is overwritten by a fail-fast abort (the abort path does not merge), which lost the Qwen section
once; per-model reports written with ``--report`` cannot clobber each other. The variants ``*_rotstraggle`` and ``*_m2proc`` are NOT merged
(they have their own reports). Reads only artifacts; CPU.

    python3 scripts/merge_e13_reports.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "artifacts" / "experiments"
MODELS = ("qwen0.5b", "granite1b", "granite3b")


def main() -> int:
    merged = {"schema_version": "hivemind-experiment-E13-swarm-local/merged-v1", "experiment_id": "E13-swarm-local", "models": {}, "sources": {}}
    design = {}
    for m in MODELS:
        p = EXP / f"E13_swarm_local_report_{m}.json"
        if not p.exists():
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        if m not in rep.get("models", {}):
            continue
        merged["models"][m] = rep["models"][m]
        merged["sources"][m] = {"file": p.name, "receipt_sha256": rep.get("receipt_sha256"), "design_sha256": rep.get("design_sha256")}
        design[m] = rep.get("design_sha256")
    merged["controls_all_passed"] = bool(merged["models"]) and all(v.get("controls_all_passed") for v in merged["models"].values())
    merged["receipt_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in merged.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()
    (EXP / "E13_swarm_local_report.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("merged", list(merged["models"]), "controls_all_passed", merged["controls_all_passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
