"""Tests for scripts/compare_replication.py (CPU, fast)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_replication", ROOT / "scripts" / "compare_replication.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def cell(seeds, init, final):
    return {"status": "ok", "seeds": seeds, "initial_loss_per_seed": init, "final_loss_per_seed": final}


def rep(cells):
    return {"models": {"m": {"cells": cells}}}


def test_matches_registered_t4_report():
    r = json.loads((ROOT / "artifacts" / "experiments" / "E14_report_qwen0.5b.json").read_text())
    out = cr.compare(r, "qwen0.5b")
    assert out["retention"]["S2"]["mean"] == -0.0514 and out["retention"]["S6"]["mean"] == 0.879
    assert out["retention"]["S2"]["ci95_half_width_df2"] == 0.3503
    assert out["claims"]["C1_quorum_with_straggler_loses_learning"]["replicates"] and out["claims"]["C2_fair_quorum_keeps_more"]["replicates"]
    assert out["evidence_level"].startswith("registered")


def test_synthetic_replication_and_disagreement():
    s0 = cell([1], [3.0], [2.0])
    good = cr.compare(rep({"S0": s0, "S2": cell([1], [3.0], [2.9]), "S6": cell([1], [3.0], [2.1])}), "m")
    assert good["claims"]["C1_quorum_with_straggler_loses_learning"]["replicates"]
    assert good["claims"]["C2_fair_quorum_keeps_more"]["replicates"]
    assert good["evidence_level"].startswith("descriptive")  # one seed is never registered evidence
    bad = cr.compare(rep({"S0": s0, "S2": cell([1], [3.0], [2.2]), "S6": cell([1], [3.0], [2.4])}), "m")
    assert not bad["claims"]["C1_quorum_with_straggler_loses_learning"]["replicates"]  # S2 retains 0.8
    assert not bad["claims"]["C2_fair_quorum_keeps_more"]["replicates"]


def test_baseline_that_did_not_improve_gives_no_claim_and_failed_cells_are_ignored():
    flat = cr.compare(rep({"S0": cell([1], [3.0], [3.0]), "S2": cell([1], [3.0], [2.9])}), "m")
    assert flat["claims"] == {} and "note" in flat
    failed = dict(cell([1], [3.0], [2.5]), status="failed")
    out = cr.compare(rep({"S0": cell([1], [3.0], [2.0]), "S2": failed}), "m")
    assert out["claims"] == {}


def test_smoke_report_is_never_evidence():
    r = rep({"S0": cell([1], [3.0], [2.0]), "S2": cell([1], [3.0], [2.9])})
    r["smoke"] = True
    assert "NOT EVIDENCE" in cr.compare(r, "m")["evidence_level"]
