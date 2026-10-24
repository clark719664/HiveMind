"""Level B gate: the exact paired test, fail-closed handling of UNVERIFIED, and the evaluator that feeds it."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest
from hypothesis import given, settings, strategies as st

from hivemind.swarm.verified_update_gate import (
    ACCEPT,
    FAIL,
    INSUFFICIENT,
    PASS,
    REJECT,
    UNVERIFIED,
    HeldOutTask,
    UpdatePolicy,
    decide,
    evaluate_tasks,
    exact_paired_p,
    extract_code,
    gate_update,
)

ROOT = Path(__file__).resolve().parent.parent


def vec(prefix: str, outcomes: List[str]) -> Dict[str, str]:
    return {f"{prefix}{i}": o for i, o in enumerate(outcomes)}


def pair(base: List[str], cand: List[str]) -> tuple[Dict[str, str], Dict[str, str]]:
    return vec("t", base), vec("t", cand)


# ------------------------------------------------------------------ the exact test
def test_exact_paired_p_matches_closed_forms() -> None:
    assert exact_paired_p(0, 0) == 1.0
    assert exact_paired_p(0, 5) == pytest.approx(2 / 32)
    assert exact_paired_p(3, 3) == 1.0
    assert exact_paired_p(1, 9) == pytest.approx(2 * (1 + 10) / 1024)
    assert exact_paired_p(9, 1) == exact_paired_p(1, 9)


# ------------------------------------------------------------------ controls
def test_clearly_better_candidate_is_accepted_even_when_improvement_is_required() -> None:
    base, cand = pair([PASS] * 30 + [FAIL] * 30, [PASS] * 50 + [FAIL] * 10)
    for policy in (UpdatePolicy(), UpdatePolicy(require_improvement=True)):
        r = decide(base, cand, policy)
        assert r.decision == ACCEPT and r.accepted
        assert (r.b, r.c, r.n) == (0, 20, 60)
        assert r.p_improvement < 1e-5 and r.p_two_sided < 1e-5


def test_regressed_candidate_is_rejected_with_the_numbers_in_the_receipt() -> None:
    base, cand = pair([PASS] * 40 + [FAIL] * 20, [PASS] * 25 + [FAIL] * 35)
    r = decide(base, cand)
    assert r.decision == REJECT and not r.accepted
    assert (r.b, r.c) == (15, 0)
    assert r.p_regression < 1e-4
    assert any("net loss" in x for x in r.reasons)


def test_identical_candidate_follows_the_policy() -> None:
    base, cand = pair([PASS] * 20 + [FAIL] * 20, [PASS] * 20 + [FAIL] * 20)
    assert decide(base, cand).decision == ACCEPT  # non-regression policy
    strict = decide(base, cand, UpdatePolicy(require_improvement=True))
    assert strict.decision == REJECT and any("improvement required" in x for x in strict.reasons)


def test_a_small_net_loss_is_rejected_unless_tolerated() -> None:
    base, cand = pair([PASS] * 31 + [FAIL] * 9, [PASS] * 30 + [FAIL] * 10)  # one task lost: not significant, still a loss
    r = decide(base, cand)
    assert r.decision == REJECT and r.p_regression > 0.05 and (r.b, r.c) == (1, 0)
    assert decide(base, cand, UpdatePolicy(tolerated_net_loss=1)).decision == ACCEPT


# ------------------------------------------------------------------ fail-closed behaviour
def test_unverified_heavy_results_block_the_decision_even_for_a_better_candidate() -> None:
    base = vec("t", [FAIL] * 40)
    cand = vec("t", [PASS] * 30 + [UNVERIFIED] * 10)
    r = decide(base, cand)
    assert r.decision == INSUFFICIENT and not r.accepted
    assert r.unverified_candidate == 10 and r.unverified_fraction == pytest.approx(0.25)
    assert decide(base, cand, UpdatePolicy(max_unverified_fraction=0.3)).decision == ACCEPT  # the knob is real


def test_unverified_never_counts_as_a_pass_for_the_candidate() -> None:
    base = vec("t", [PASS] * 58 + [PASS] * 2)
    cand = vec("t", [PASS] * 58 + [UNVERIFIED] * 2)
    r = decide(base, cand, UpdatePolicy(max_unverified_fraction=0.1))
    assert (r.b, r.c, r.candidate_pass) == (2, 0, 58)  # scored as failures
    assert r.decision == REJECT


def test_unverified_baseline_is_scored_as_a_pass_so_it_cannot_manufacture_a_gain() -> None:
    base = vec("t", [PASS] * 58 + [UNVERIFIED] * 2)
    cand = vec("t", [PASS] * 58 + [FAIL] * 2)
    r = decide(base, cand, UpdatePolicy(max_unverified_fraction=0.1, require_improvement=True))
    assert (r.b, r.c) == (2, 0) and r.baseline_pass == 60
    assert r.decision == REJECT


def test_too_few_tasks_mismatched_task_sets_and_bad_outcomes_are_refused() -> None:
    small_b, small_c = pair([PASS] * 10, [PASS] * 10)
    assert decide(small_b, small_c).decision == INSUFFICIENT
    base = vec("a", [PASS] * 40)
    cand = vec("b", [PASS] * 40)
    mism = decide(base, cand)
    assert mism.decision == INSUFFICIENT and mism.n == 0 and any("differ" in x for x in mism.reasons)
    with pytest.raises(ValueError, match="outside"):
        decide({"t": "maybe"}, {"t": PASS})
    with pytest.raises(ValueError):
        UpdatePolicy(alpha=1.5)


def test_receipt_hash_is_deterministic_and_content_sensitive() -> None:
    base, cand = pair([PASS] * 30 + [FAIL] * 10, [PASS] * 33 + [FAIL] * 7)
    r1, r2 = decide(base, cand), decide(base, cand)
    assert r1.receipt_sha256 == r2.receipt_sha256 and len(r1.receipt_sha256) == 64
    assert decide(base, cand, UpdatePolicy(alpha=0.01)).receipt_sha256 != r1.receipt_sha256


# ------------------------------------------------------------------ properties
OUT = st.sampled_from([PASS, FAIL])


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(OUT, OUT), min_size=30, max_size=80))
def test_swapping_baseline_and_candidate_swaps_the_discordant_counts(pairs: List[tuple[str, str]]) -> None:
    base, cand = vec("t", [p[0] for p in pairs]), vec("t", [p[1] for p in pairs])
    fwd, rev = decide(base, cand), decide(cand, base)
    assert (fwd.b, fwd.c) == (rev.c, rev.b)
    assert fwd.p_two_sided == rev.p_two_sided
    assert 0.0 <= fwd.p_regression <= 1.0 and 0.0 <= fwd.p_improvement <= 1.0
    assert fwd.both_pass + fwd.both_fail + fwd.b + fwd.c == fwd.n


@settings(max_examples=150, deadline=None)
@given(st.lists(OUT, min_size=30, max_size=80), st.lists(st.booleans(), min_size=80, max_size=80))
def test_a_candidate_that_keeps_every_baseline_pass_is_never_rejected(base_out: List[str], flips: List[bool]) -> None:
    cand_out = [PASS if (o == PASS or flips[i]) else FAIL for i, o in enumerate(base_out)]
    r = decide(vec("t", base_out), vec("t", cand_out))
    assert r.b == 0 and r.decision == ACCEPT


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(OUT, OUT), min_size=30, max_size=80))
def test_accept_implies_no_net_loss_and_no_significant_regression(pairs: List[tuple[str, str]]) -> None:
    r = decide(vec("t", [p[0] for p in pairs]), vec("t", [p[1] for p in pairs]))
    if r.decision == ACCEPT:
        assert r.b - r.c <= 0
        assert not (r.b > r.c and r.p_regression < 0.05)


# ------------------------------------------------------------------ evaluator (the sandbox is the only mocked I/O)
def task(key: str) -> HeldOutTask:
    return HeldOutTask(key, f"write {key}", ("assert True",))


def fake_gate(verdicts: Dict[str, str]):
    def gate(code: str, tests, setup: str):
        if code == "RAISE":
            raise RuntimeError("sandbox crashed")
        return SimpleNamespace(verdict=verdicts[code])
    return gate


def test_evaluator_maps_gate_verdicts_and_generator_failures() -> None:
    tasks = [task(k) for k in ("ok", "bad", "unv", "boom", "empty", "gate_raises")]
    replies = {"ok": "```python\nok\n```", "bad": "bad", "unv": "unv", "empty": "   ", "gate_raises": "RAISE"}

    def generate(t: HeldOutTask) -> str:
        if t.key == "boom":
            raise ConnectionError("generator down")
        return replies[t.key]

    out = evaluate_tasks(tasks, generate, fake_gate({"ok": "ACCEPTED", "bad": "REJECTED_TEST", "unv": "UNVERIFIED"}))
    assert out == {"ok": PASS, "bad": FAIL, "unv": UNVERIFIED, "boom": UNVERIFIED, "empty": FAIL, "gate_raises": UNVERIFIED}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_tasks([task("x"), task("x")], generate, fake_gate({}))


def test_extract_code_takes_the_last_fenced_block_or_the_whole_reply() -> None:
    assert extract_code("a\n```python\nx = 1\n```\nthen\n```py\ny = 2\n```") == "y = 2"
    assert extract_code("  z = 3  ") == "z = 3"


def test_gate_update_end_to_end_with_better_and_worse_generators() -> None:
    tasks = [task(f"k{i}") for i in range(40)]
    gate = fake_gate({"good": "ACCEPTED", "bad": "REJECTED_TEST"})
    base_gen = lambda t: "good" if int(t.key[1:]) < 20 else "bad"  # noqa: E731
    better = lambda t: "good" if int(t.key[1:]) < 35 else "bad"  # noqa: E731
    worse = lambda t: "good" if int(t.key[1:]) < 5 else "bad"  # noqa: E731
    up = gate_update(tasks, base_gen, better, gate)
    down = gate_update(tasks, base_gen, worse, gate)
    assert up.decision == ACCEPT and (up.b, up.c) == (0, 15)
    assert down.decision == REJECT and (down.b, down.c) == (15, 0)


def test_real_sandbox_judges_a_correct_and_a_wrong_solution() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    e15_gate = pytest.importorskip("e15_gate")
    if not e15_gate.gate_available():
        pytest.skip(f"sandbox unavailable on this host: {e15_gate.availability_reason()}")
    tasks = [HeldOutTask("add-right", "add", ("assert add(2, 3) == 5",)), HeldOutTask("add-wrong", "add", ("assert add(2, 3) == 5",))]
    replies = {"add-right": "def add(a, b):\n    return a + b\n", "add-wrong": "def add(a, b):\n    return a - b\n"}
    out = evaluate_tasks(tasks, lambda t: replies[t.key], lambda code, tests, setup: e15_gate.gate(code, tests, setup))
    assert out == {"add-right": PASS, "add-wrong": FAIL}
