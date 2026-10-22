"""README headline numbers must be measured, labelled, or absent; counts must not go stale."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
SPEC = (REPO_ROOT / "lean4" / "HiveMindSpec.lean").read_text(encoding="utf-8")

STALE = (
    "302%2B%20Passing",
    "98%25%20Coverage",
    "64.96x",
    "2,809 fps",
    "27-Theorem Audit",
    "`48` theorems",
    "(`48/48`)",
    "`500x` lower communication frequency",
)


def test_stale_or_unmeasured_figures_are_gone() -> None:
    for text in STALE:
        assert text not in README, text


def test_badges_do_not_hardcode_counts_that_drift() -> None:
    badges = [line for line in README.splitlines() if "img.shields.io" in line]
    tests_badge = next(b for b in badges if "Test%20Suite" in b or "Tests-" in b)
    assert "coverage%20gate" in tests_badge
    lean_badge = next(b for b in badges if "Lean%204" in b)
    assert not re.search(r"\d+%20Theorems", lean_badge)
    assert "0%20sorry" in lean_badge
    paper_badge = next(b for b in badges if "Foundation%20Paper" in b)
    assert "self--assessed" in paper_badge
    strategy_badge = next(b for b in badges if "Strategy%20Engine" in b)
    assert "synthetic" in strategy_badge


def test_lean_spec_still_meets_the_ci_minimum_the_badge_relies_on() -> None:
    assert len(re.findall(r"^theorem ", SPEC, flags=re.M)) >= 48
    assert "sorry" not in re.sub(r"--.*|/-.*?-/", "", SPEC, flags=re.S).split("#print axioms")[0]


def test_diloco_cadence_claim_matches_the_default_config() -> None:
    config = (REPO_ROOT / "src" / "hivemind" / "config.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*inner_steps_H: int = 500$", config, flags=re.M)
    assert "`H=500` inner steps per synchronisation by default" in README
