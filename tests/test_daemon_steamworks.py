"""Unit tests for Win32 Zero-Intrusion Daemon, Steamworks Sybil Defense, and Swarm Inference."""

from __future__ import annotations

from pathlib import Path

import torch

from hivemind.client.daemon import ZeroIntrusionDaemon, get_windows_idle_seconds
from hivemind.client.steamworks import SteamworksSDKBridge
from hivemind.core.btm_upcycle import DEFAULT_EXPERT_DOMAINS, upcycle_branches_to_sparse_moe
from hivemind.core.diloco_inner import run_diloco_inner_loop
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.inference_ring import RegionalDHTSwarmRing, SwarmPeerNode


def test_windows_idle_hook_and_game_launch_preemption(tmp_path: Path) -> None:
    idle_sec = get_windows_idle_seconds()
    assert idle_sec >= 0.0

    # Simulate another Steam game launching mid-training -> must preempt & dump checkpoint in <1.5s
    daemon = ZeroIntrusionDaemon(
        idle_threshold_seconds=30.0,
        enforce_os_idle_hook=True,
        simulated_idle_seconds=120.0,
        simulated_active_game="cyberpunk2077.exe",
    )
    telemetry = daemon.poll()
    assert telemetry.should_yield_vram is True
    assert telemetry.active_game_detected == "cyberpunk2077.exe"

    base_model, _, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    tokens = torch.randint(0, 256, (4, 16), dtype=torch.long)

    res = run_diloco_inner_loop(
        model=bundle.model,
        token_sequences=tokens,
        worker_id="gamer-node-01",
        branch_name="Code",
        outer_step=1,
        inner_steps_H=10,
        checkpoint_interval_s=5,
        should_preempt_fn=daemon.should_preempt,
        emergency_checkpoint_dir=tmp_path,
    )
    assert res.preempted_by_daemon is True
    assert res.emergency_dump_path is not None and res.emergency_dump_path.exists()
    assert res.emergency_dump_latency_sec < 1.5


def test_steamworks_sybil_resistance_and_leaderboard_credits() -> None:
    bridge = SteamworksSDKBridge()
    # Legitimate Non-Limited Steam account ($50 spent)
    valid_ticket = bridge.get_auth_session_ticket("76561198000000101", account_spend_usd=50.0)
    ok, _ = bridge.verify_sybil_resistance(valid_ticket)
    assert ok is True

    # Cheap botnet Limited Steam account ($0 spent < $5 requirement)
    bot_ticket = bridge.get_auth_session_ticket("76561198000000999", account_spend_usd=0.0)
    ok_bot, reason = bridge.verify_sybil_resistance(bot_ticket)
    assert ok_bot is False
    assert "Limited" in reason

    telemetry = bridge.record_validated_workunit(
        steam_id64="76561198000000101",
        persona_name="CommanderShepard",
        wu_id="wu-code-t1",
        tflops_contributed=12.5,
        credits_minted=15.0,
    )
    assert telemetry.api_credits_balance == 15.0
    assert "ACH_FIRST_SYNAPSE" in telemetry.unlocked_achievements
    assert "ACH_10_TFLOPS_SWARM" in telemetry.unlocked_achievements


def test_phase2_regional_dht_swarm_speculative_decoding() -> None:
    device = torch.device("cpu")
    base_model, _, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        device=device,
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    init_lora = extract_lora_state_dict(bundle.model)
    branch_states = {dom: init_lora for dom in DEFAULT_EXPERT_DOMAINS[:4]}

    moe_model, _ = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=branch_states,
        lora_rank_r=4,
        lora_alpha=8,
        top_k=2,
        target_device=device,
    )

    ring = RegionalDHTSwarmRing(region="EU-West-Paris", max_ping_ms=25.0)
    assert ring.register_peer(
        SwarmPeerNode("p1", "7656119801", "EU-West-Paris", 14.2, 12.0, "Code", (0, 2))
    )
    # High-latency peer (>25 ms) must be rejected from the low-latency regional ring
    assert not ring.register_peer(
        SwarmPeerNode("p2", "7656119802", "EU-West-Paris", 85.0, 12.0, "Math", (0, 2))
    )

    prompt = torch.randint(0, 256, (1, 8), dtype=torch.long)
    res = ring.run_speculative_inference(
        moe_target_model=moe_model,
        draft_model=bundle.model,
        prompt_ids=prompt,
        max_new_tokens=12,
        draft_gamma=5,
        simulated_network_rtt_ms=15.0,
    )
    assert len(res.generated_tokens) == 12
    assert res.latency_reduction_factor >= 2.0
