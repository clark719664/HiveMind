"""The Lean error-feedback model (Theorems 65-67) stays in the spec and machine-checked."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "lean4" / "HiveMindSpec.lean"
BOUND_THEOREMS = (
    "ef_residual_stays_bounded_under_decay",
    "ucb1_well_sampled_suboptimal_arm_is_not_selected",
    "ucb1_sample_threshold_bounds_bonus",
)
MODEL_THEOREMS = (
    "ef_conservation_any_compressor",
    "sparsify_residual_energy_le",
    "sparsify_energy_split",
    "sparsify_strict_contraction",
    "ef_conservation_with_carry_map",
    "topk_contraction_rate",
)


def test_model_theorems_are_declared_and_printed() -> None:
    source = SPEC.read_text(encoding="utf-8")
    assert "namespace Season6_ErrorFeedbackModel" in source
    for name in MODEL_THEOREMS:
        assert re.search(rf"^theorem {name}\b", source, flags=re.M), name
        assert f"#print axioms HiveMindSpec.Season6_ErrorFeedbackModel.{name}" in source
    assert "namespace Season6_CoreOnlyBounds" in source
    for name in BOUND_THEOREMS:
        assert re.search(rf"^theorem {name}\b", source, flags=re.M), name
        assert f"#print axioms HiveMindSpec.Season6_CoreOnlyBounds.{name}" in source


def test_theorem_49_docstring_does_not_claim_to_model_the_update() -> None:
    source = SPEC.read_text(encoding="utf-8")
    doc = source[source.index("/-- Theorem 49") : source.index("theorem ef21_error_feedback_residual_contraction")]
    assert "does not model the error-feedback update" in doc
    assert "\t" not in source[source.index("/-- Theorem 51") : source.index("theorem ucb1_bandit")]


@pytest.mark.skipif(not os.environ.get("LEAN_BIN"), reason="LEAN_BIN not set; CI compiles the spec separately")
def test_model_theorems_are_machine_checked_with_standard_axioms_only() -> None:
    from paper.autoevolve_peer_review_pipeline import require_machine_checked, verify_lean4_specification

    result = verify_lean4_specification()
    require_machine_checked(result)
    allowed = {"propext", "Quot.sound", "Classical.choice"}
    for name in MODEL_THEOREMS:
        axioms = result["axioms_by_theorem"][f"HiveMindSpec.Season6_ErrorFeedbackModel.{name}"]
        assert set(axioms) <= allowed, (name, axioms)
    for name in BOUND_THEOREMS:
        axioms = result["axioms_by_theorem"][f"HiveMindSpec.Season6_CoreOnlyBounds.{name}"]
        assert set(axioms) <= allowed, (name, axioms)
