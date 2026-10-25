"""Compare a community replication of the swarm concept (E13/E14 runner report) with the registered T4 results.

A replicator runs a few cells on their own GPU with scripts/run_experiment_E13_swarm_local.py (recipe: docs/community/T4_REPLICATION_KIT.md)
and passes the report here. The script recomputes the REGISTERED retention  rho = (L_init - L_cell) / (L_init - L_S0)  per seed from the
report's own losses (it does not trust any status text) and checks two qualitative claims of the T4 campaign:

  C1  (E13, P13b)  all mechanisms on with a quorum K < M and a persistent straggler loses the learning: retention of S2 < 0.5
  C2  (E14, H14a)  bounded exclusion (fair quorum, S6) keeps more of the improvement than S2: retention(S6) > retention(S2)

Fewer than 3 seeds gives a DESCRIPTIVE verdict only (the registered test is a df = 2 t-interval over 3 seeds).  A replication that disagrees
is a result, not an error: report it as such.  Writes a small receipt (sha256 of the input and of the result); nothing is uploaded.

    python3 scripts/compare_replication.py REPORT.json [--model qwen0.5b] [--out receipt.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

T_DF2 = 4.302652729911275
VIABLE = 0.5


def retention(cells: dict, name: str) -> dict:
    """Per-seed registered retention of cell `name` against S0 (needs S0 in the same report)."""
    s0, c = cells.get("S0"), cells.get(name)
    if not s0 or not c:
        return {}
    out = {}
    for i, seed in enumerate(c.get("seeds", [])):
        if seed not in s0.get("seeds", []):
            continue
        j = s0["seeds"].index(seed)
        l0, lc, ls = c["initial_loss_per_seed"][i], c["final_loss_per_seed"][i], s0["final_loss_per_seed"][j]
        if l0 - ls > 0:  # the synchronous baseline must have improved, otherwise the ratio is undefined
            out[seed] = (l0 - lc) / (l0 - ls)
    return out


def summarise(values: dict) -> dict:
    v = list(values.values())
    d = {"per_seed": {str(k): round(x, 4) for k, x in values.items()}, "n": len(v)}
    if v:
        d["mean"] = round(statistics.mean(v), 4)
    if len(v) == 3:
        d["ci95_half_width_df2"] = round(T_DF2 * statistics.stdev(v) / len(v) ** 0.5, 4)
    return d


def compare(report: dict, model: str) -> dict:
    cells = report["models"][model]["cells"]
    ok = {k: v for k, v in cells.items() if v.get("status", "ok") == "ok" and v.get("final_loss_per_seed")}
    r2, r6 = retention(ok, "S2"), retention(ok, "S6")
    res = {"schema_version": "hivemind-replication-compare/v1.0", "model": model, "cells_present": sorted(ok),
           "retention": {"S2": summarise(r2), "S6": summarise(r6)}, "claims": {}}
    n = min(len(r2), len(r6)) if r6 else len(r2)
    registered = n >= 3
    res["evidence_level"] = "registered-design (3 seeds)" if registered else f"descriptive only ({n} seed(s); the registered test needs 3)"
    if r2:
        m2 = statistics.mean(r2.values())
        res["claims"]["C1_quorum_with_straggler_loses_learning"] = {"measured_mean_retention_S2": round(m2, 4), "bar": VIABLE,
                                                                  "replicates": bool(m2 < VIABLE)}
    if r2 and r6:
        common = sorted(set(r2) & set(r6))
        diffs = [r6[s] - r2[s] for s in common]
        res["claims"]["C2_fair_quorum_keeps_more"] = {"mean_retention_gain_S6_minus_S2": round(statistics.mean(diffs), 4), "seeds": len(diffs),
                                                    "replicates": bool(diffs and all(d > 0 for d in diffs))}
    if report.get("smoke"):
        res["evidence_level"] = "SMOKE RUN: NOT EVIDENCE (tiny random model on CPU; checks your install only)"
    if not res["claims"]:
        res["note"] = "report needs cells S0 and S2 (and S6 for C2) completed on the same seeds"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = Path(a.report).read_bytes()
    res = compare(json.loads(raw), a.model)
    res["input_sha256"] = hashlib.sha256(raw).hexdigest()
    res["result_sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    text = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
