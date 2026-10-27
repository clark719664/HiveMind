# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Solver Bee problem library and its tools (stdlib only; regeneration tests skip without mpmath/scipy)."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "solvers" / "tools"
PROBLEMS = ROOT / "solvers" / "problems"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check = _load("check_problems")


def _docs():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(PROBLEMS.glob("*.json"))}


def test_library_has_at_least_twelve_valid_problems():
    n, problems = check.check_dir(PROBLEMS, 12)
    assert n >= 12
    assert problems == []


def test_library_covers_required_families():
    docs = _docs()
    need = {"harmonic_undamped", "linear_2x2", "linear_5x5", "decay_chain", "robertson", "van_der_pol_mu1",
            "van_der_pol_mu100", "brusselator", "lotka_volterra", "stiff_chain"}
    assert need <= set(docs)
    kinds = {d["spec"]["problem"]["kind"] for d in docs.values()}
    assert kinds == {"named", "linear", "mass_action"}


def test_every_reference_records_method_tool_versions_and_cross_check():
    for pid, d in _docs().items():
        r = d["reference"]
        assert r["tool"] == "solvers/tools/make_references.py", pid
        assert {"python", "mpmath", "scipy", "numpy"} <= set(r["tool_versions"]), pid
        assert r["cross_check"]["method"] and r["cross_check"]["max_abs_diff"] >= 0, pid
        assert r["method"] != r["cross_check"]["method"], pid


def test_reference_is_100x_tighter_than_spec_tolerance():
    for pid, d in _docs().items():
        s, r = d["spec"], d["reference"]
        tol_min = min(s["rtol"] * abs(v) + s["atol"] for row in r["y"] for v in row)
        assert r["max_abs_error_bound"] * 100 <= tol_min, pid


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda d: d.__setitem__("schema", "nope"), "schema"),
        (lambda d: d["spec"].__setitem__("bogus", 1), "unknown key"),
        (lambda d: d["spec"].__setitem__("t_out", [2.0, 1.0]), "t_out"),
        (lambda d: d["spec"].__setitem__("y0", [1.0]), "y0"),
        (lambda d: d["spec"].__setitem__("rtol", 0.0), "rtol"),
        (lambda d: d["spec"].__setitem__("method", "rk4"), "method"),
        (lambda d: d["reference"].__setitem__("t", [9.0]), "reference.t"),
        (lambda d: d["reference"].__setitem__("y", [[float("nan")] * 2] * 5), "reference.y"),
        (lambda d: d["reference"].__setitem__("max_abs_error_bound", 1e-3), "100x"),
        (lambda d: d["reference"].pop("cross_check"), "cross_check"),
        (lambda d: d["tolerance_policy"].__setitem__("max_rel_err_factor", -1), "max_rel_err_factor"),
        (lambda d: d.__setitem__("extra", 1), "unknown top-level"),
        (lambda d: d["invariants"][0].__setitem__("value", 5.0), "invariant"),
    ],
)
def test_checker_rejects_corrupted_problem(mutate, fragment):
    d = copy.deepcopy(_docs()["harmonic_undamped"])
    mutate(d)
    errs = check.validate_problem(d, "harmonic_undamped")
    assert any(fragment in e for e in errs), errs


def test_checker_rejects_mass_action_problems():
    d = copy.deepcopy(_docs()["decay_chain"])
    d["spec"]["problem"]["reactions"][0]["reactants"] = {"Z": 1}
    assert any("unknown species" in e for e in check.validate_problem(d, "decay_chain"))
    d = copy.deepcopy(_docs()["decay_chain"])
    d["spec"]["problem"]["reactions"][0]["reactants"] = {"A": 0}
    assert any("coefficient" in e for e in check.validate_problem(d, "decay_chain"))
    d = copy.deepcopy(_docs()["decay_chain"])
    d["spec"]["problem"]["reactions"] = d["spec"]["problem"]["reactions"] * 20
    assert any("reactions" in e for e in check.validate_problem(d, "decay_chain"))


def test_checker_main_reports_too_few_problems(tmp_path):
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    n, problems = check.check_dir(tmp_path, 12)
    assert n == 1 and problems


def test_run_problems_evaluate_flags_out_of_policy():
    run = _load("run_problems")
    d = _docs()["harmonic_undamped"]
    good = {"status": "ok", "y": d["reference"]["y"], "stats": {"steps": 1, "rhs_evals": 1},
            "solver": {"method": "bdf"}}
    assert run.evaluate(d, good)["ok"]
    bad = copy.deepcopy(good)
    bad["y"][0][0] += 1e-3
    assert not run.evaluate(d, bad)["ok"]
    assert not run.evaluate(d, {"status": "error", "error": "x"})["ok"]


def test_tools_contain_no_dynamic_code_execution():
    for p in TOOLS.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for bad in ("eval(", "exec(", "pickle", "shell=True", "os.system"):
            assert bad not in text.replace("evaluate(", "").replace("eval_invariant(", ""), (p.name, bad)


# ---------------------------------------------------------------- regeneration (needs mpmath + scipy + numpy)

FAST_IDS = ["harmonic_undamped", "harmonic_damped", "linear_2x2", "linear_5x5", "decay_chain",
            "reversible_network", "stiff_chain"]


def _strip_versions(doc):
    # Drop the fields that legitimately depend on the installed numpy/scipy versions (the cross-check
    # disagreement is float noise of the second method); the exact reference values must still match.
    doc = copy.deepcopy(doc)
    for k in ("tool_versions", "cross_check", "max_abs_error_bound"):
        doc["reference"].pop(k)
    return doc


def test_regenerating_exact_references_reproduces_committed_files(tmp_path):
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    mk = _load("make_references")
    assert mk.main(["--ids", *FAST_IDS, "--out", str(tmp_path)]) == 0
    docs = _docs()
    for pid in FAST_IDS:
        new = json.loads((tmp_path / f"{pid}.json").read_text(encoding="utf-8"))
        assert _strip_versions(new) == _strip_versions(docs[pid]), pid
    # and the regenerated files pass the checker
    # (only these ids exist in tmp_path, so use the per-document validator)
    for pid in FAST_IDS:
        new = json.loads((tmp_path / f"{pid}.json").read_text(encoding="utf-8"))
        assert check.validate_problem(new, pid) == []


def test_make_references_lists_every_committed_problem():
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    mk = _load("make_references")
    assert sorted(e["id"] for e in mk.problems()) == sorted(_docs())
    for e in mk.problems():
        assert e["spec"] == _docs()[e["id"]]["spec"], e["id"]
