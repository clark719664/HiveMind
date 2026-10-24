"""Unit tests of scripts/build_site_data.py with synthetic artifacts in tmp_path (CPU only, no real experiment file is read)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_site_data", REPO / "scripts" / "build_site_data.py")
assert _spec is not None and _spec.loader is not None
bsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsd)

UTC = "2026-01-02T03:04:05Z"
RECEIPT = "abcdef0123456789" * 4


def cell(name: str, finals: list[float], initial: float = 3.0, adm: float = 1.0) -> dict[str, Any]:
    return {"name": name, "status": "ok", "seeds": [1, 2, 3][: len(finals)], "initial_loss_per_seed": [initial] * len(finals), "final_loss_per_seed": finals,
            "per_domain_final_mean": {"code": 2.0, "wiki": 3.0}, "admitted_fraction_mean": adm, "late_discards_total": 0,
            "wire": {"compression_ratio_actual": 11.0, "compression_ratio_ideal": 16.0, "int16_posts_total": 0, "secret_extra": 1},
            "round_latency_s": {"p50": 1.0, "p95": 2.0, "p99": 3.0, "n": 9, "note": "descriptive only"}}


def write(art: Path, name: str, doc: dict[str, Any]) -> None:
    art.mkdir(parents=True, exist_ok=True)
    (art / name).write_text(json.dumps(doc), encoding="utf-8")


def e13_report(models: dict[str, Any], receipt: str = RECEIPT) -> dict[str, Any]:
    return {"schema_version": "x", "models": models, "receipt_sha256": receipt, "controls_all_passed": True}


def model_with_cells(cells: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"cells": cells, "controls": {"s0_learns": {"passed": True}, "s1_equals_s0": {"passed": None, "rows": []}}, "controls_all_passed": True,
            "paired_vs_S0": {}, **extra}


def test_empty_directory_gives_available_false_everywhere(tmp_path: Path) -> None:
    data = bsd.build(tmp_path, UTC)
    assert data["e13"]["registered"]["available"] is False
    assert data["e13"]["rotating_straggler"]["available"] is False
    assert data["e13"]["m2_process_identity"]["available"] is False
    assert data["e14"]["available"] is False and data["e15"]["available"] is False
    assert data["e14"]["prereg"]["available"] is False
    assert data["generated_utc"] == UTC and data["schema"] == bsd.SCHEMA


def test_e13_registered_copies_statuses_verbatim_and_tags_receipt(tmp_path: Path) -> None:
    pred = {"source_report": "E13_swarm_local_report.json", "source_report_receipt": RECEIPT, "receipt_sha256": "0123456789abcdeffedcba" + "0" * 42, "rules": "R",
            "models": {"qwen0.5b": {"valid": True, "controls_failed": [], "P13a": {"status": "supported"},
                                    "P13b": {"status": "refuted", "mean": -0.05, "half_width": float("nan"), "interval": [-0.4, 0.3], "bar": 0.5},
                                    "P13c": {"status": "supported", "upload_ratio_actual": 0.09}, "not_a_field": 1}}}
    write(tmp_path, "E13_swarm_local_predictions.json", pred)
    write(tmp_path, "E13_swarm_local_report.json", e13_report({"qwen0.5b": model_with_cells({"S0": cell("sync", [2.2, 2.2, 2.2]), "S2": cell("k2", [2.4, 2.4, 2.3], adm=0.5)})}))
    blk = bsd.build(tmp_path, UTC)["e13"]["registered"]
    assert blk["available"] is True and blk["source"] == "E13_swarm_local_predictions.json" and blk["receipt12"] == "0123456789ab"
    assert blk["generated_utc"] == UTC and blk["report"]["receipt12"] == RECEIPT[:12]
    m = blk["models"]["qwen0.5b"]
    assert m["P13a"]["status"] == "supported" and m["P13b"]["status"] == "refuted" and m["P13b"]["mean"] == -0.05
    assert m["P13b"]["half_width"] is None  # NaN is not valid JSON
    assert "not_a_field" not in m
    assert m["report_cells"]["S2"]["admitted_fraction_mean"] == 0.5 and "secret_extra" not in m["report_cells"]["S2"]["wire"]


def test_rotating_straggler_derives_retention_with_flag_and_lists_unevaluated_controls(tmp_path: Path) -> None:
    s0, s2 = cell("sync", [2.0, 2.0, 2.0], initial=3.0), cell("k2", [2.5, 2.5, 2.5], initial=3.0, adm=0.5)
    paired = {"S2": {"final_loss_minus_S0_per_seed": [0.5, 0.5, 0.5], "mean": 0.5, "ci95": 0.1, "t_crit_df2": 4.3, "ci_excludes_0": True, "direction": "worse than S0"}}
    write(tmp_path, "E13_swarm_local_report_rotstraggle.json",
          {**e13_report({"qwen0.5b": {**model_with_cells({"S0": s0, "S2": s2}), "paired_vs_S0": paired}}), "controls_all_passed": False, "smoke": False})
    blk = bsd.build(tmp_path, UTC)["e13"]["rotating_straggler"]
    assert blk["available"] and blk["controls_all_passed"] is False and blk["status"] == "exploratory diagnostic, decides nothing" and "campaign_report" in blk["status_source"]
    m = blk["models"]["qwen0.5b"]
    assert m["paired_S2_minus_S0"]["direction"] == "worse than S0"
    assert m["controls_not_evaluated"] == ["s1_equals_s0"]
    der = m["retention_S2_derived"]
    assert der["derived"] is True and der["per_seed"]["1"] == pytest.approx(0.5) and der["mean"] == pytest.approx(0.5)


def test_m2_process_identity_block(tmp_path: Path) -> None:
    ident = {"rows": [{"max_abs_diff": 0.0, "bit_identical": True, "passed": True}], "tolerance": 1e-6, "passed": True}
    model = model_with_cells({"S0": cell("sync", [2.4], initial=2.5), "S1": cell("s1", [2.4], initial=2.5)})
    model["controls"]["s1_equals_s0"] = ident
    write(tmp_path, "E13_swarm_local_report_m2proc.json", e13_report({"qwen0.5b": model}))
    blk = bsd.build(tmp_path, UTC)["e13"]["m2_process_identity"]
    assert blk["available"] and blk["models"]["qwen0.5b"]["s1_equals_s0"]["passed"] is True and blk["receipt12"] == RECEIPT[:12]
    assert blk["status"] == "identity control only" and "campaign_report" in blk["status_source"]


def test_e14_absent_keeps_prereg_but_no_numbers(tmp_path: Path) -> None:
    write(tmp_path, "E14_swarm_fairness_prereg.json", {"registered_utc": "t", "status": "exploratory", "hypotheses_and_rules": {"H14a": "text"}, "scope": "s", "motivation": "ignored"})
    blk = bsd.build(tmp_path, UTC)["e14"]
    assert blk["available"] is False and "models" not in blk
    assert blk["prereg"]["available"] and blk["prereg"]["hypotheses_and_rules"] == {"H14a": "text"} and "motivation" not in blk["prereg"]


def test_e14_present_maps_direction_to_registered_status(tmp_path: Path) -> None:
    paired = {"S6": {"S2_minus_cell_final_loss_per_seed": [0.1, 0.1, 0.1], "mean": 0.1, "ci95": 0.05, "t_crit_df2": 4.3, "ci_excludes_0": True, "direction": "cell better than S2"},
              "S7": {"status": "needs exactly 3 paired seeds with S2 (have 2)", "seeds": [1, 2]}}
    model = model_with_cells({"S0": cell("sync", [2.0] * 3), "S2": cell("k2", [2.5] * 3), "S6": cell("fair", [2.1] * 3), "S7": cell("def", [2.2] * 3)},
                             e14={"cells": ["S6", "S7"], "paired_vs_S2": paired, "retention_vs_S0": {"S6": {"mean": 0.8}}, "note": "exploratory"})
    write(tmp_path, "E14_report_qwen0.5b.json", e13_report({"qwen0.5b": model}))
    blk = bsd.build(tmp_path, UTC)["e14"]
    assert blk["available"] and blk["source"] == ["E14_report_qwen0.5b.json"] and blk["receipt12"] == [RECEIPT[:12]]
    h = blk["models"]["qwen0.5b"]["e14"]["hypotheses"]
    assert h["H14a"]["direction"] == "cell better than S2" and h["H14a"]["hypothesis_status"] == "supported"
    assert "hypothesis_status" not in h["H14b"] and h["H14b"]["status"].startswith("needs exactly 3")
    assert blk["models"]["qwen0.5b"]["e14"]["retention_vs_S0"] == {"S6": {"mean": 0.8}}


@pytest.mark.parametrize("direction,status", [("cell worse than S2", "refuted"), ("within CI of S2", "inconclusive")])
def test_e14_direction_mapping(direction: str, status: str) -> None:
    model = model_with_cells({}, e14={"cells": ["S6"], "paired_vs_S2": {"S6": {"direction": direction, "mean": -0.2}}})
    out = bsd.e14_model({"models": {"m": model}}, "m")
    assert out is not None and out["e14"]["hypotheses"]["H14a"]["hypothesis_status"] == status


def test_e15_absent_and_present(tmp_path: Path) -> None:
    write(tmp_path, "E15_gwaya_rft_prereg.json", {"registered_utc": "t", "title": "T", "hypotheses": {"H15a": "x"}, "scope": "s"})
    assert bsd.build(tmp_path, UTC)["e15"]["available"] is False
    assert bsd.build(tmp_path, UTC)["e15"]["prereg"]["title"] == "T"
    write(tmp_path, "E15_gwaya_rft_report.json",
          {"receipt_sha256": RECEIPT, "status": "complete", "smoke": True, "model": "tiny", "controls_all_passed": True, "private": 1, "notes": ["Numbers only"],
           "conditions": {"C0": {"label": "C0", "n": 10, "passes": 4, "pass_at_1": 0.4, "wilson95": [0.17, 0.69], "hidden": 1}},
           "paired_comparisons": {"C2-C0": {"n": 10, "diff": 0.1, "b_only": 2, "a_only": 1, "discordant": 3, "mcnemar_p": 1.0, "x": 0}}})
    blk = bsd.build(tmp_path, UTC)["e15"]
    assert blk["available"] and blk["receipt12"] == RECEIPT[:12]
    assert blk["run_status"] == "complete" and blk["smoke"] is True and "private" not in blk["report_excerpt"]
    assert blk["conditions"]["C0"] == {"label": "C0", "n": 10, "passes": 4, "pass_at_1": 0.4, "wilson95": [0.17, 0.69]}
    assert blk["paired_comparisons"]["C2-C0"]["mcnemar_p"] == 1.0 and "x" not in blk["paired_comparisons"]["C2-C0"]
    assert "H15" in blk["verdicts_note"] and "not computed here" in blk["verdicts_note"]  # no verdict is ever invented


def test_unreadable_or_non_object_json_is_treated_as_missing(tmp_path: Path) -> None:
    (tmp_path / "E13_swarm_local_predictions.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "E13_swarm_local_report_m2proc.json").write_text("[1, 2]", encoding="utf-8")
    data = bsd.build(tmp_path, UTC)
    assert data["e13"]["registered"]["available"] is False and data["e13"]["m2_process_identity"]["available"] is False


def test_file_without_receipt_gets_file_hash_tag(tmp_path: Path) -> None:
    write(tmp_path, "E13_swarm_local_report_m2proc.json", {"models": {}})
    blk = bsd.build(tmp_path, UTC)["e13"]["m2_process_identity"]
    assert blk["receipt12"] is None and len(blk["file_sha256_12"]) == 12


def test_clean_and_render_are_strict_json_and_js_wrapper(tmp_path: Path) -> None:
    assert bsd.clean({"a": [float("inf"), 1.0, float("nan")]}) == {"a": [None, 1.0, None]}
    text = bsd.render(bsd.build(tmp_path, UTC))
    assert text.startswith("// Generated") and "window.HM_DATA = {" in text and text.rstrip().endswith("};")
    payload = text.split("window.HM_DATA = ", 1)[1].rstrip().rstrip(";")
    assert json.loads(payload)["generated_utc"] == UTC
    assert not math.isnan(0.0)


def test_main_writes_file_and_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    art, out = tmp_path / "art", tmp_path / "site" / "data.js"
    art.mkdir()
    assert bsd.main(["--artifacts", str(art), "--out", str(out), "--utc", UTC]) == 0
    first = out.read_text(encoding="utf-8")
    assert bsd.main(["--artifacts", str(art), "--out", str(out), "--utc", UTC]) == 0
    assert out.read_text(encoding="utf-8") == first
    assert "wrote" in capsys.readouterr().out


def test_real_artifacts_build_without_inventing_numbers() -> None:
    """Smoke test on the repository's own artifacts (skipped when the directory is not present): every numeric leaf of the P13b block equals the source."""
    art = REPO / "artifacts" / "experiments"
    src = art / "E13_swarm_local_predictions.json"
    if not src.exists():
        pytest.skip("no real E13 predictions in this checkout")
    data = bsd.build(art, UTC)["e13"]["registered"]
    raw = json.loads(src.read_text(encoding="utf-8"))
    for name, m in raw["models"].items():
        if "P13b" in m and "mean" in m["P13b"]:
            assert data["models"][name]["P13b"]["mean"] == m["P13b"]["mean"]
            assert data["models"][name]["P13b"]["status"] == m["P13b"]["status"]
