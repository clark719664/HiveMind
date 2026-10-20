"""Regression guards: the astrophysics swarm demo must not report unmeasured figures as results."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from hivemind.cli import demo_hf_domain_swarm as demo
from hivemind.data.hf_real_trainer import CURATED_SEASON3_DOMAIN_CORPORA

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "artifacts" / "hf_astrophysics_swarm_report.json"


def _legacy_report() -> dict:
    return {
        "experiments": [
            {
                "pre_lora_perplexity": 20.5965,
                "post_lora_perplexity": 1.8956,
                "worker_loss_trajectories": {"w": {"initial_loss": 3.0, "final_loss": 0.6, "spot_check_max_diff": 0.0}},
                "inference_settlement": {"verified_tokens": 45, "rtt_speedup": 5.0, "fee_paid_hive": 0.225},
            }
        ],
        "regional_swarm_speculative_generations": {
            "Kepler": {"latency_reduction_factor": 5.33, "effective_tokens_per_sec": 4.56}
        },
    }


def test_labels_replace_unmeasured_figures_and_keep_measured_ones() -> None:
    report = demo.label_astrophysics_report_evidence(_legacy_report())
    exp = report["experiments"][0]
    assert exp["inference_settlement"]["rtt_speedup"] is None
    assert exp["inference_settlement"]["rtt_speedup_status"].startswith("not_measured")
    assert exp["classification_accuracy"] is None
    assert exp["heldout_perplexity"] is None
    assert "not held out" in exp["perplexity_eval_rows"]
    assert exp["data_provenance"] == {"source_status": "not_recorded_by_this_run"}
    assert (exp["pre_lora_perplexity"], exp["post_lora_perplexity"]) == (20.5965, 1.8956)
    assert exp["worker_loss_trajectories"]["w"]["final_loss"] == 0.6

    gen = report["regional_swarm_speculative_generations"]["Kepler"]
    assert gen["latency_reduction_factor"] is None
    assert gen["tokens_per_verification_pass"] == 5.33
    assert gen["effective_tokens_per_sec"] == 4.56
    assert "simulated" in gen["effective_tokens_per_sec_basis"]

    scope = report["evidence_scope"]
    assert scope["classification_accuracy_measured"] is False
    assert scope["heldout_evaluation"] is False
    assert scope["network_timing_measured"] is False


def test_labelling_is_idempotent() -> None:
    once = demo.label_astrophysics_report_evidence(_legacy_report())
    twice = demo.label_astrophysics_report_evidence(copy.deepcopy(once))
    assert twice == once


def test_data_provenance_counts_curated_fallback_rows() -> None:
    fallback = CURATED_SEASON3_DOMAIN_CORPORA["Kepler_Exoplanet_Transit_Classifier"]
    assert demo.describe_data_provenance(fallback[:2] * 2) == {
        "rows": 4,
        "distinct_rows": 2,
        "curated_fallback_rows": 4,
        "source_status": "all_rows_are_curated_fallback",
    }
    mixed = demo.describe_data_provenance([fallback[0], "a fetched dataset row"])
    assert mixed["source_status"] == "mixed_fetched_and_curated_fallback"
    assert mixed["curated_fallback_rows"] == 1
    assert demo.describe_data_provenance(["row"])["source_status"] == "no_curated_fallback_rows_detected"
    assert demo.describe_data_provenance([])["source_status"] == "no_rows"
    generic = demo._GENERIC_FALLBACK_SAMPLE
    assert demo.describe_data_provenance([generic])["curated_fallback_rows"] == 1


def test_committed_report_carries_no_unmeasured_speed_or_accuracy_figures() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["evidence_scope"]["classification_accuracy_measured"] is False
    for exp in report["experiments"]:
        assert exp["inference_settlement"]["rtt_speedup"] is None
        assert exp["classification_accuracy"] is None
        assert "not held out" in exp["perplexity_eval_rows"]
    for gen in report["regional_swarm_speculative_generations"].values():
        assert gen["latency_reduction_factor"] is None
        assert gen["tokens_per_verification_pass"] > 0
    assert report == demo.label_astrophysics_report_evidence(copy.deepcopy(report))


def test_source_no_longer_hardcodes_claims() -> None:
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "rtt_speedup=5.0" not in source
    assert "PASS (0.0e+00)" not in source
    cli = (REPO_ROOT / "src" / "hivemind" / "cli" / "main.py").read_text(encoding="utf-8")
    assert "Classification QLoRA Swarm experiments on real Hugging Face datasets" not in cli
    assert "not classification accuracy" in cli
