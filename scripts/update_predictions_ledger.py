"""Write ``artifacts/predictions_ledger.json`` from the experiment artifacts that exist.

The ledger lists every prediction of ``docs/RESEARCH_PROGRAM.md`` with its status
(``untested`` / ``supported`` / ``refuted`` / ``inconclusive`` / ``inapplicable``), the artifact
and receipt that decided it, and the scale at which it was decided. Statuses are copied from
the artifacts, never typed here.

Run: ``PYTHONPATH=src python scripts/update_predictions_ledger.py``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = PROJECT_ROOT / "artifacts" / "predictions_ledger.json"
E5_TOY = PROJECT_ROOT / "artifacts" / "experiments" / "E5_toy_report.json"
E5_HET = PROJECT_ROOT / "artifacts" / "experiments" / "E5_het_report.json"
E10 = PROJECT_ROOT / "artifacts" / "experiments" / "E10_lattice_report.json"

PREDICTIONS: Dict[str, Dict[str, str]] = {
    "P1": {"rq": "RQ1", "claim": "WAN loss with K<M quorum, grace and RDA <= single-host loss with M(1-churn) workers", "decided_by": "E8"},
    "P2": {"rq": "RQ2", "claim": "best outer LR grows with M under direct averaging and is constant under RDA (conditional on near-orthogonal learner deltas)", "decided_by": "E5"},
    "P3": {"rq": "RQ3", "claim": "adaptive grace admission beats matched-random admission at equal loss on wall-clock", "decided_by": "E5"},
    "P4": {"rq": "RQ4", "claim": "PA-CGAD gate helps under quorum with stale deltas and never hurts; zero effect without stale deltas", "decided_by": "E5"},
    "P5": {"rq": "RQ5", "claim": "Skip-Token quality loss is governed by the measured ghost fraction", "decided_by": "E7"},
    "P6": {"rq": "RQ6", "claim": "whitening inner optimizer flattens loss-vs-M slope", "decided_by": "E6"},
    "P7": {"rq": "RQ7", "claim": "learned depth prior makes terminal-KV sharing safe", "decided_by": "E7b"},
    "P8": {"rq": "RQ8", "claim": "Multi-Krum + 2sigma does not stop ALIE at f/n >= 0.2; spot-check replay does", "decided_by": "E9"},
    "P9": {"rq": "RQ2'", "claim": "RDA acts as an outer-LR multiplier: log(best_lr_direct / best_lr_rda) within one grid step of log(mean ||RDA||/||direct||)", "decided_by": "E5-het"},
    "P10a": {"rq": "RQ9", "claim": "E8 blockwise quantisation of outer deltas has lower distortion than Z8 at matched measured rate", "decided_by": "E10-lattice"},
    "P10b": {"rq": "RQ9", "claim": "E8 reaches a lower held-out loss than Z8 at matched rate (headline: 1 bit/dim with error feedback)", "decided_by": "E10-lattice"},
    "P10c": {"rq": "RQ9", "claim": "error feedback recovers the unquantised loss at 1 bit/dim with E8", "decided_by": "E10-lattice"},
}


def _status_from_e5(report: Dict[str, Any], key: str) -> Dict[str, Any]:
    a = report["analysis"][key]
    status = a["status"]
    norm = "inapplicable" if status.startswith("inapplicable") else status.split(" ")[0]
    return {
        "status": norm,
        "status_detail": status,
        "scale": "toy synthetic LM, single host, 3 seeds (E5-toy)",
        "artifact": "artifacts/experiments/E5_toy_report.json",
        "receipt_sha256": report["receipt_sha256"],
        "prereg_sha256": report["prereg_sha256"],
        "controls_all_passed": report["controls_all_passed"],
        "evidence": {k: v for k, v in a.items() if k not in ("rows",)},
    }


def _status_from_e5_het(report: Dict[str, Any], key: str) -> Dict[str, Any]:
    a = report["analysis"][key]
    status = a["status"]
    norm = "inapplicable" if status.startswith("inapplicable") else status.split(" ")[0]
    return {
        "status": norm,
        "status_detail": status,
        "scale": "toy synthetic LM with per-learner teachers (rho = 4, delta cosine < 0.3), single host, 3 seeds (E5-het)",
        "artifact": "artifacts/experiments/E5_het_report.json",
        "receipt_sha256": report["receipt_sha256"],
        "prereg_sha256": report["prereg_sha256"],
        "controls_all_passed": report["controls_all_passed"],
        "manipulation_check": {k: v for k, v in report["manipulation_check"].items() if k != "rda_over_direct_norm_ratio"},
        "evidence": {k: v for k, v in a.items() if k not in ("rows",)},
    }


def main() -> int:
    ledger: Dict[str, Any] = {"schema_version": "hivemind-predictions-ledger/v1.1", "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "predictions": {}}
    e5 = json.loads(E5_TOY.read_text(encoding="utf-8")) if E5_TOY.exists() else None
    e5h = json.loads(E5_HET.read_text(encoding="utf-8")) if E5_HET.exists() else None
    e10 = json.loads(E10.read_text(encoding="utf-8")) if E10.exists() else None
    for pid, meta in PREDICTIONS.items():
        entry: Dict[str, Any] = dict(meta)
        if e5 is not None and pid in ("P2", "P3", "P4"):
            entry.update(_status_from_e5(e5, pid))
            entry["note"] = "decided at toy scale only; the full-scale E5 cell on Qwen2.5-0.5B is still open"
            if e5h is not None:
                het = _status_from_e5_het(e5h, pid)
                entry["decisions"] = [{"experiment": "E5-toy", **{k: entry[k] for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256", "evidence")}}, {"experiment": "E5-het", **{k: het[k] for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256", "evidence")}}]
                # the heterogeneity-controlled run supersedes the IID one whenever it is applicable
                if het["status"] != "inapplicable":
                    for k in ("status", "status_detail", "scale", "artifact", "receipt_sha256", "prereg_sha256", "controls_all_passed"):
                        entry[k] = het[k]
                    entry["evidence"] = het["evidence"]
                    entry["manipulation_check"] = het["manipulation_check"]
                    entry["note"] = "decided at toy scale (E5-het supersedes E5-toy); the full-scale E5 cell on Qwen2.5-0.5B is still open"
        elif e10 is not None and pid.startswith("P10"):
            a = e10["analysis"][pid]
            status = a["status"]
            entry.update({"status": status.split(";")[0].split(" ")[0], "status_detail": status,
                          "scale": "toy LoRA deltas (2048 coords) on the E5-het rho = 4, M = 8 testbed, 3 seeds (E10-lattice)",
                          "artifact": "artifacts/experiments/E10_lattice_report.json", "receipt_sha256": e10["receipt_sha256"], "prereg_sha256": e10["prereg_sha256"],
                          "controls_all_passed": e10["controls_all_passed"], "evidence": {k: v for k, v in a.items() if k != "cells"} | ({"cells": {ck: {kk: vv for kk, vv in cv.items() if kk != "per_seed"} for ck, cv in a["cells"].items()}} if "cells" in a else {}),
                          "note": "decided at toy scale only"})
        elif e5h is not None and pid == "P9":
            entry.update(_status_from_e5_het(e5h, "P9"))
            entry["note"] = "decided at toy scale only"
        else:
            entry.update({"status": "untested", "scale": None, "artifact": None})
        ledger["predictions"][pid] = entry
    counts: Dict[str, int] = {}
    for e in ledger["predictions"].values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    ledger["status_counts"] = counts
    payload = json.dumps({k: v for k, v in ledger.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    ledger["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(json.dumps({"status_counts": counts, "receipt": ledger["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
