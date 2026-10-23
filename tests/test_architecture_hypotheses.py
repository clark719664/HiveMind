"""Unit & integration tests for the 10-Hypothesis Architecture Verifier (`H1`..`H10`)."""

from __future__ import annotations

from pathlib import Path

from hivemind.swarm.architecture_hypothesis_verifier import (
    SCHEMA_VERSION,
    ArchitectureHypothesisVerifier,
)


def test_all_ten_architecture_hypotheses_pass_and_emit_signed_receipt(tmp_path: Path) -> None:
    verifier = ArchitectureHypothesisVerifier(seed=2026)
    out_path = tmp_path / "arch_hypotheses.json"
    report = verifier.run_all_verifications(output_path=out_path)

    assert out_path.exists()
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["total_hypotheses"] == 10
    assert report["passed_hypotheses"] == 10
    assert report["all_passed"] is True
    assert len(report["receipt_sha256"]) == 64

    by_id = {h["hypothesis_id"]: h for h in report["hypotheses"]}
    assert set(by_id.keys()) == {f"H{i}" for i in range(1, 11)}
    for h_id, rec in by_id.items():
        assert rec["passed"] is True, f"{h_id} failed: {rec}"

    # Spot-check key physical invariants across the 10 hypotheses
    assert by_id["H1"]["key_metrics"]["extra_gpu_vram_bytes"] == 0
    assert by_id["H1"]["key_metrics"]["max_nesterov_equivalence_diff"] <= 1e-6
    assert by_id["H2"]["key_metrics"]["bias_within_theoretical_bound"] is True
    assert by_id["H2"]["key_metrics"]["stale_fragment_v_buffer_polluted"] is False
    assert by_id["H3"]["key_metrics"]["gating_flop_reduction_x"] == 8.0
    assert by_id["H3"]["key_metrics"]["fomoe_wan_payload_reduction_x"] > 3.0
    assert by_id["H4"]["key_metrics"]["saved_weight_bytes"] > 0
    assert by_id["H4"]["key_metrics"]["early_exit_fraction"] > 0.0
    assert by_id["H5"]["key_metrics"]["exposed_swap_latency_ms_simulated"] == 0.0
    assert abs(by_id["H6"]["key_metrics"]["iso_cts_condition_number_after"] - 1.0) <= 1e-5
    assert by_id["H7"]["key_metrics"]["dct_sparse_2bit_package_compression_ratio_x"] >= 16.0
    assert by_id["H8"]["key_metrics"]["outer_lr_at_h64_doubled"] < by_id["H8"]["key_metrics"]["outer_lr_at_h32_ref"]
    assert by_id["H9"]["key_metrics"]["validator_spot_check_max_replay_diff"] == 0.0
    assert by_id["H10"]["key_metrics"]["multistep_telescoping_conservation_error_l2"] <= 1e-5
