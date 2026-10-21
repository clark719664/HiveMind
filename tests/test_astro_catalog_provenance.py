"""The 24-cycle astrophysics discovery catalog must say what it is: illustrative constants, not run output."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hivemind.boinc.astrophysics_24h_campaign import (
    CATALOG_MEASUREMENT_KIND,
    COSMOLOGY_PARAMETERS_SOURCE,
    CONFIDENCE_BASIS_ILLUSTRATIVE,
    _compute_cosmology_discoveries_and_catalog,
    annotate_campaign_report_provenance,
    annotate_discovery_catalog_provenance,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "hivemind" / "boinc" / "astrophysics_24h_campaign.py"
ARTIFACT = REPO / "artifacts" / "astrophysics_24h_gpu_campaign_report.json"
README = REPO / "README.md"
DOC_REPORT = REPO / "docs" / "ASTROPHYSICS_24H_GPU_DISCOVERY_REPORT.md"

ADDED_CATALOG_KEYS = {"measurement_kind", "catalog_provenance"}


def _strip_labels(catalog: dict) -> dict:
    """Remove every key the labelling adds, so the remainder can be compared with the unlabelled original."""
    out = {k: copy.deepcopy(v) for k, v in catalog.items() if k not in ADDED_CATALOG_KEYS}
    for entry in out.get("exoplanet_biosignature_candidates", []):
        entry.pop("confidence_basis", None)
    out.get("cosmology_dark_energy_reconstruction", {}).pop("parameters_source", None)
    return out


def test_catalog_carries_provenance_fields() -> None:
    catalog = _compute_cosmology_discoveries_and_catalog()
    keys = list(catalog)
    assert keys[:2] == ["measurement_kind", "catalog_provenance"], "the notice must lead the block"
    assert catalog["measurement_kind"] == CATALOG_MEASUREMENT_KIND == "illustrative_hardcoded"
    notice = catalog["catalog_provenance"].lower()
    assert "illustrative" in notice and "not inferred from the run" in notice
    for entry in catalog["exoplanet_biosignature_candidates"]:
        entry_keys = list(entry)
        i = entry_keys.index("moe_biosignature_confidence")
        assert entry_keys[i + 1] == "confidence_basis"
        assert entry["confidence_basis"] == CONFIDENCE_BASIS_ILLUSTRATIVE == "illustrative constant"
    cosmo = catalog["cosmology_dark_energy_reconstruction"]
    assert cosmo["parameters_source"] == COSMOLOGY_PARAMETERS_SOURCE == "fixed input constants, not a fit"
    assert cosmo["hubble_constant_h0_kms_mpc"] == 68.42, "labelling must not touch the values"


def test_labelling_adds_keys_only_and_is_idempotent() -> None:
    catalog = _compute_cosmology_discoveries_and_catalog()
    bare = _strip_labels(catalog)
    assert "confidence_basis" not in bare["exoplanet_biosignature_candidates"][0]
    relabelled = annotate_discovery_catalog_provenance(bare)
    assert relabelled == catalog
    assert annotate_discovery_catalog_provenance(copy.deepcopy(catalog)) == catalog
    assert _strip_labels(relabelled) == bare


def test_report_labelling_preserves_measured_fields() -> None:
    report = {
        "campaign": "Project HiveMind — 24-Cycle Full-Day Local RTX 2070 Astrophysics & Cosmology BOINC Campaign",
        "elapsed_wall_seconds": 197.64,
        "compute_summary": {"total_boinc_outer_cycles": 24, "inner_steps_per_worker": 12},
        "scientific_discoveries": _strip_labels(_compute_cosmology_discoveries_and_catalog()),
    }
    labelled = annotate_campaign_report_provenance(copy.deepcopy(report))
    assert list(labelled)[:2] == ["campaign", "schedule_basis"]
    assert labelled["campaign"] == "Project HiveMind — 24-Cycle Local RTX 2070 Astrophysics & Cosmology BOINC Campaign"
    sb = labelled["schedule_basis"]
    assert sb["measured_elapsed_wall_seconds"] == 197.64
    assert sb["total_boinc_outer_cycles"] == 24 and sb["inner_steps_per_worker"] == 12
    assert sb["cycle_labels_are_simulated"] is True
    assert "not elapsed hours" in sb["note"]
    assert labelled["elapsed_wall_seconds"] == 197.64
    assert labelled["compute_summary"] == report["compute_summary"]
    assert annotate_campaign_report_provenance(copy.deepcopy(labelled)) == labelled


@pytest.mark.skipif(not ARTIFACT.exists(), reason="campaign artifact not present")
def test_committed_artifact_is_labelled_and_measured_numbers_untouched() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert report["elapsed_wall_seconds"] == 197.64
    assert report["schedule_basis"]["measured_elapsed_wall_seconds"] == 197.64
    assert report["scientific_discoveries"]["measurement_kind"] == "illustrative_hardcoded"
    assert "Full-Day" not in report["campaign"]
    assert annotate_campaign_report_provenance(copy.deepcopy(report)) == report, "artifact must already be fully labelled"
    first_cycle = report["hourly_cycle_log"][0]
    assert first_cycle["simulated_schedule_hour"] == "01:00"


def _readme_showcase_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## 🌌 0. 24-Cycle")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_readme_showcase_carries_the_notice() -> None:
    section = _readme_showcase_section()
    assert "Provenance notice: illustrative example catalog, not inferred from the run." in section
    assert "197.64 s" in section
    assert "illustrative_hardcoded" in section
    for banned in ("Full-Day", "24-Hour", "Identified **", "Reconstructed $", "MoE confidence"):
        assert banned not in section, banned


def test_markdown_report_carries_the_notice() -> None:
    text = DOC_REPORT.read_text(encoding="utf-8")
    assert "Provenance notice: illustrative example catalog, not inferred from the run." in text
    assert "197.64 s" in text
    assert "Illustrative Example Catalog (not inferred from the run)" in text
    assert "Confidence (illustrative constant)" in text
    for banned in ("Full-Day", "24-Hour", "Key Scientific Discoveries", "| MoE Confidence |"):
        assert banned not in text, banned


def test_source_no_longer_claims_moe_verification() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "verified by the 6-Expert MoE" not in source
    assert "24-Cycle Full-Day" not in source.replace('replace("24-Cycle Full-Day"', "")
    assert 'measurement_kind' in source
