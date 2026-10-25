"""Checkers of the live monitor: each must accept the right answer and reject near-misses, or the probes prove nothing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

_SPEC = importlib.util.spec_from_file_location("live_user_monitor", Path(__file__).resolve().parents[1] / "scripts" / "live_user_monitor.py")
assert _SPEC is not None and _SPEC.loader is not None
monitor = importlib.util.module_from_spec(_SPEC)
sys.modules["live_user_monitor"] = monitor
_SPEC.loader.exec_module(monitor)


def test_checker_controls_all_hold_for_the_probe_item() -> None:
    controls = monitor.checker_controls("Class: M")
    assert set(controls) >= {"bee_rejects_wrong", "coder_rejects_minus", "arith_rejects_381"}
    assert all(controls.values()), controls


def test_coder_check_reads_the_syntax_tree_not_keywords() -> None:
    assert monitor.check_coder("```python\ndef add(a, b):\n    return b + a\n```")
    assert not monitor.check_coder("```python\ndef add(a, b):\n    return a * b\n```")
    assert not monitor.check_coder("```python\ndef add(x, y):\n    return x + y\n```")  # the prompt names a and b
    assert not monitor.check_coder("```python\ndef add(a, b):\n    return a +\n```")  # syntax error
    assert not monitor.check_coder("def add(a, b): return a + b  # and print(1)\nprint(")


def test_bee_check_is_exact() -> None:
    assert monitor.check_bee(" Class: M\n", "Class: M")
    assert not monitor.check_bee("Class: M because it is red", "Class: M")
    assert not monitor.check_bee("Class: K", "Class: M")


@given(st.lists(st.floats(0.0, 1000.0, allow_nan=False), min_size=1, max_size=50))
def test_percentiles_are_ordered_and_inside_the_sample(values: list[float]) -> None:
    p50 = monitor._percentile(values, 0.5)
    p95 = monitor._percentile(values, 0.95)
    assert min(values) <= p50 <= p95 <= max(values)
    assert p50 in values and p95 in values


def test_percentile_of_nothing_is_none() -> None:
    assert monitor._percentile([], 0.5) is None
    assert monitor._percentile([3.0], 0.95) == 3.0
