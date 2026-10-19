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

    # Stage 7 assertions (RTX TensorCore FP16 GEMM + 16-Expert Fused Top-2 MoE FP8)
    s7 = report["stage7_rtx_tensorcore_fp16_fp8_benchmark"]
    assert s7["num_fused_experts"] == 16
    assert s7["router_dtype"] == "float32"
    assert s7["sqnr_above_20db"] is True
    assert s7["fp16_gemm_gflops"] > 0.0

    # Stage 8 assertions (WAN STUN UDP Hole-Punching & RFC 9001 0-RTT Anti-Replay)
    s8 = report["stage8_wan_nat_stun_hole_punching_and_replay_defense"]
    assert s8["restricted_cone_direct_udp_verified"] is True
    assert s8["symmetric_nat_turn_fallback_verified"] is True
    assert s8["anti_replay_100pct_accurate"] is True

    # Stage 9 assertions (Season 3 Hack-Miner Forge & CAS Expert Vault)
    s9 = report["stage9_season3_hack_miner_and_16_expert_cas"]
    assert len(s9["season3_contracts_forged"]) == 3
    assert s9["season3_all_spotchecks_passed"] is True
    assert s9["cas_roundtrip_verified"] is True

    # Stage 10 assertions (Lean 4 Formal Audit)
    s10 = report["stage10_lean4_21_theorems_formal_audit"]
    assert s10["theorem_count"] >= 27
    assert s10["sorry_count"] == 0
    assert s10["user_axiom_count"] == 0
    assert s10["season3_theorems_present"] is True
    assert s10["formal_spec_verified"] is True

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


def test_season3_rtx_fused_moe_fp8_nat_hole_punch_and_21_lean4_theorems() -> None:
    """Verify Season 3 (`HM-S3-01`..`HM-S3-05`) RTX Fused MoE INT4+FP8, STUN UDP Hole-Punching, and 27 Lean 4 theorems."""
    import torch
    from hivemind.game.hf_hack_miner import (
        EXTENDED_HF_BOUNTY_CONTRACTS,
        FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS,
        SEASON_3_HF_BOUNTY_CONTRACTS,
    )
    from hivemind.swarm.quic_transport import AsyncQUICGeoRingTransport
    from hivemind.swarm.swarm_fabric import (
        fused_int4_fp8_moe_route_and_project,
        fused_top2_moe_fp8_dispatch,
    )
    from paper.autoevolve_peer_review_pipeline import verify_lean4_specification

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 1. HM-S3-02: 16-Expert Fused Top-2 Sparse MoE FP8 & INT4+FP8 Dispatch on RTX GPU
    hs = torch.randn(2, 16, 64, device=device, dtype=torch.float16)
    wg = torch.randn(16, 64, device=device, dtype=torch.float32) * 0.05
    experts = [torch.randn(64, 64, device=device, dtype=torch.float16) * 0.02 for _ in range(16)]
    fused = fused_top2_moe_fp8_dispatch(hs, wg, experts, top_k=2)
    assert fused["output"].shape == (2, 16, 64)
    assert fused["top_k_indices"].shape == (2, 16, 2)
    assert fused["sqnr_db"] >= 20.0
    assert fused["router_dtype"] == "float32"

    fused_int4 = fused_int4_fp8_moe_route_and_project(hs, wg, experts, top_k=2)
    assert fused_int4["output"].shape == (2, 16, 64)
    assert fused_int4["cosine_similarity"] >= 0.99
    assert fused_int4["vram_compression_ratio"] >= 3.2

    # 2. HM-S3-01: STUN UDP Hole-Punching & RFC 9001 0-RTT Anti-Replay
    transport = AsyncQUICGeoRingTransport(bind_host="127.0.0.1")
    punch = transport.perform_udp_hole_punch("node-a", "node-b", nat_type="restricted_cone")
    assert punch["hole_punch_succeeded"] is True
    assert punch["handshake_verified"] is True
    wan_punch = transport.execute_udp_hole_punch("node-a", "node-b")
    assert wan_punch["direct_hole_punch_succeeded"] is True
    assert wan_punch["sub_5ms_verified"] is True
    ok1, _ = transport.verify_0rtt_ticket_anti_replay("node-a", nonce=10)
    dup1, reason = transport.verify_0rtt_ticket_anti_replay("node-a", nonce=10)
    assert ok1 is True
    assert dup1 is False
    assert "replay_nonce_detected" in reason

    # 3. HM-S3-03 & HM-S3-04: Season 3 Bounty Contracts (Missions 9-12 + Bonus Formal Contracts)
    assert len(SEASON_3_HF_BOUNTY_CONTRACTS) == 4
    assert len(FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS) == 10
    assert len(EXTENDED_HF_BOUNTY_CONTRACTS) == 12

    # 4. HM-S3-05: Machine-Checked Lean 4 Theorems (0 sorry, 0 user axioms)
    lean_res = verify_lean4_specification()
    assert lean_res["theorem_count"] >= 27
    assert lean_res["sorry_count"] == 0
    assert lean_res["user_axiom_count"] == 0


