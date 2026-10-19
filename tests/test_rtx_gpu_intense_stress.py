"""Intense RTX GPU Stress & Torture Tests + v1.0.0 Free Steam Tool Deployment Verification."""

from __future__ import annotations

import json
from pathlib import Path

from hivemind.game.rtx_steam_tool_deployer import (
    deploy_free_steam_rtx_tool_v1,
    run_intense_rtx_stress_suite,
)


def test_intense_rtx_gpu_stress_suite_and_free_steam_tool_v1_deploy(tmp_path: Path) -> None:
    """Execute the 6-Stage Intense RTX GPU Stress Suite and verify v1.0.0 Free Steam Tool packaging."""
    report = run_intense_rtx_stress_suite(
        vram_cycles=3,
        udp_burst_packets=100,
        num_hotplug_experts=12,
        steam_concurrent_accounts=50,
    )
    assert report["all_stages_passed"] is True
    assert report["release_version"] == "1.0.0"
    assert report["certification_hmac"].startswith("rtx_steam_cert_")

    # Stage 1 assertions
    s1 = report["stage1_vram_and_preemption_torture"]
    assert s1["cycles_executed"] == 3
    assert s1["all_preemptions_under_1500ms"] is True
    assert s1["zero_vram_leak_verified"] is True

    # Stage 2 assertions
    s2 = report["stage2_moe12_hotplug_and_dma"]
    assert s2["final_hotplugged_experts"] == 12
    assert s2["lru_eviction_verified"] is True
    assert s2["double_buffer_slots_used"] == ["slot_A", "slot_B"]

    # Stage 3 assertions
    s3 = report["stage3_byzantine_and_spotcheck"]
    assert s3["honest_spotcheck_passed"] is True
    assert s3["tampered_cheater_caught"] is True
    assert s3["all_byzantine_poisoners_rejected"] is True
    assert len(s3["accepted_workers"]) == 7

    # Stage 4 assertions
    s4 = report["stage4_p2p_swarm_fabric_burst"]
    assert s4["udp_0rtt_packets_verified"] == 100
    assert s4["gossipsub_within_25ms_sla"] is True
    assert s4["token_tree_accepted_length"] == 7
    assert s4["merkle_dispute_fraud_layer_detected"] == 7
    assert s4["merkle_dispute_slashed_and_refunded"] is True

    # Stage 5 assertions
    s5 = report["stage5_steamworks_50_accounts"]
    assert s5["sybil_gate_100pct_accurate"] is True
    assert s5["non_limited_verified"] == 35
    assert s5["limited_sybil_bots_blocked"] == 10
    assert s5["vac_banned_blocked"] == 5

    # Stage 6 assertions
    s6 = report["stage6_gateway_and_hud_load"]
    assert s6["all_endpoints_200_ok"] is True
    assert s6["openai_sse_done_verified"] is True

    # Verify v1.0.0 Free Steam Tool deployment package
    deploy_pkg = deploy_free_steam_rtx_tool_v1(
        stress_report=report,
        output_root=tmp_path / "HiveMind_Steam_FreeEdition",
    )
    assert deploy_pkg["release_version"] == "1.0.0"
    assert deploy_pkg["all_stress_stages_passed"] is True
    assert Path(deploy_pkg["steam_tool_vdf"]).exists()
    assert Path(deploy_pkg["windows_rtx_tool_launcher_bat"]).exists()
    assert Path(deploy_pkg["rtx_deployment_cert_json"]).exists()

    manifest = json.loads(Path(deploy_pkg["steam_store_manifest"]).read_text(encoding="utf-8"))
    assert manifest["release_version"] == "1.0.0"
    assert manifest["app_type"] == "Free Steam Tool & Game"
