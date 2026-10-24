"""Verified mode: the repair loop's policy (fake model and judge), the CLI's exit codes, and the real local sandbox."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from hypothesis import given, settings, strategies as st

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_verified.py"
_spec = importlib.util.spec_from_file_location("hivemind_verified_under_test", _PATH)
assert _spec is not None and _spec.loader is not None
V = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = V  # dataclasses resolve string annotations through sys.modules
_spec.loader.exec_module(V)

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
MSGS = [{"role": "user", "content": "write add"}]


def judge(code: str) -> object:
    ok = "a + b" in code
    return V.Judgement(ok, "ACCEPTED" if ok else "REJECTED_TEST", "" if ok else "add(1,2) gave -1, expected 3")


class Script:
    """A fake model that replays replies and records the conversations it was shown."""

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.seen = list(replies), []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.seen.append([dict(m) for m in messages])
        return self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]


def test_a_first_attempt_that_passes_is_verified_after_one_call() -> None:
    ask = Script([GOOD])
    result = V.verified_solve(ask, MSGS, judge)
    assert result.verified and result.status == "VERIFIED" and result.rounds == 1 and len(ask.seen) == 1
    assert "a + b" in result.code


def test_a_failed_attempt_is_repaired_with_the_sandbox_feedback_in_the_next_prompt() -> None:
    ask = Script([BAD, GOOD])
    result = V.verified_solve(ask, MSGS, judge)
    assert result.verified and result.rounds == 2 and [a.verdict for a in result.attempts] == ["REJECTED_TEST", "ACCEPTED"]
    assert "add(1,2) gave -1, expected 3" in ask.seen[1][-1]["content"]  # the model is told exactly what failed
    assert ask.seen[1][-2]["role"] == "assistant"


def test_when_nothing_passes_the_first_attempt_comes_back_flagged_unverified() -> None:
    first = "```python\ndef add(a, b):\n    return a * b  # first\n```"
    ask = Script([first, BAD, BAD, BAD])
    result = V.verified_solve(ask, MSGS, judge, max_repairs=3)
    assert result.status == "UNVERIFIED_FALLBACK" and not result.verified and result.rounds == 4 and len(ask.seen) == 4
    assert "# first" in result.code


def test_an_unavailable_sandbox_blocks_at_once_and_never_looks_verified() -> None:
    ask = Script([GOOD])
    blocked = lambda code: V.Judgement(False, "UNVERIFIED", "sandbox unavailable", blocked=True)  # noqa: E731
    result = V.verified_solve(ask, MSGS, blocked)
    assert result.status == "BLOCKED" and not result.verified and result.code == "" and len(ask.seen) == 1


def test_prose_is_never_run_and_a_missing_code_block_is_asked_for_again() -> None:
    ask = Script(["Sure! You could add the numbers.", "still no code", GOOD])
    result = V.verified_solve(ask, MSGS, judge)
    assert result.verified and result.rounds == 3 and result.attempts[0].verdict == "NO_CODE"
    none = V.verified_solve(Script(["no code at all"]), MSGS, judge, max_repairs=1)
    assert none.status == "NO_CODE" and none.code == "" and none.rounds == 2
    assert V.extract_code("```python\n   \n```") is None and V.extract_code("x = 1") is None
    with pytest.raises(ValueError):
        V.verified_solve(Script([GOOD]), MSGS, judge, max_repairs=-1)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.booleans(), min_size=1, max_size=8), st.integers(min_value=0, max_value=5))
def test_property_verified_iff_some_attempt_within_budget_passes_and_calls_are_bounded(passes: list[bool], repairs: int) -> None:
    replies = [(GOOD if ok else BAD).replace("add", f"add{i}") for i, ok in enumerate(passes)]
    ask = Script(replies)
    result = V.verified_solve(ask, MSGS, judge, max_repairs=repairs)
    window = passes[: repairs + 1]
    assert result.verified == any(window)
    assert 1 <= len(ask.seen) <= repairs + 1
    if result.verified:
        assert len(ask.seen) == window.index(True) + 1 and f"add{window.index(True)}" in result.code  # the FIRST passing candidate wins
    else:
        assert "add0" in result.code  # fallback is the first attempt


# ---------------------------------------------------------------- CLI

def _client(reply: str, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer hm-test"
        return httpx.Response(status, json={"choices": [{"message": {"content": reply}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_cli_exit_codes_and_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test")
    argv = ["--task", "write add", "--test", "assert add(1,2)==3"]
    assert V.run(argv, client=_client(GOOD), judge=judge) == 0
    assert "a + b" in capsys.readouterr().out
    assert V.run([*argv, "--max-repairs", "0"], client=_client(BAD), judge=judge) == 1
    assert capsys.readouterr().out == ""  # an unverified answer is not printed by default
    assert V.run([*argv, "--max-repairs", "0", "--allow-unverified"], client=_client(BAD), judge=judge) == 1
    assert "a - b" in capsys.readouterr().out
    assert V.run([*argv, "--json"], client=_client(GOOD), judge=judge) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert V.run(argv, client=_client("oops", status=500), judge=judge) == 2


def test_cli_refuses_without_tests_or_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test")
    assert V.run(["--task", "x"], client=_client(GOOD), judge=judge) == 2  # nothing to verify against
    monkeypatch.delenv("HIVEMIND_API_KEY")
    assert V.run(["--task", "x", "--test", "assert 1"], client=_client(GOOD), judge=judge) == 2


# ---------------------------------------------------------------- the real local sandbox

def test_the_real_sandbox_accepts_correct_code_and_rejects_wrong_code() -> None:
    sys_path_gate = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(sys_path_gate))
    import e15_gate

    if not e15_gate.gate_available():
        pytest.skip("local sandbox unavailable: " + e15_gate.availability_reason())
    real = V.sandbox_judge(["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    assert real("def add(a, b):\n    return a + b\n").accepted is True
    wrong = real("def add(a, b):\n    return a - b\n")
    assert wrong.accepted is False and wrong.blocked is False and wrong.feedback
    assert real("def add(a, b):\n    pass\n").accepted is False  # a stub is refused even if no test were reached
