"""Unit and Integration Tests for HiveMind v2 ('SETI-Swarm') 4-Pillar Architecture.

Covers:
1. ContentAddressedSwarmStore (Kademlia DHT + CAS + Merkle Byte-Range Proof-of-Storage)
2. HiveSwarmLedger ($HIVE PoUW Block Mining, Halving, Byzantine Slashing & Inference Settlement)
3. Autonomous 3-State SETI Agent Lifecycle (Gaming/Seed-Only, Idle Miner, Swarm Consumer)
4. Dynamic Domain Expert Hot-Plug (BTM MoE v2 E -> E+1 Router Expansion)
"""

from __future__ import annotations

from pathlib import Path

import torch

from hivemind.client.daemon import SETIAgentState, ZeroIntrusionDaemon
from hivemind.core.btm_upcycle import upcycle_branches_to_sparse_moe
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.ledger import HiveSwarmLedger, WorkReceipt


def test_cas_vault_and_pouw_ledger_tokenomics(tmp_path: Path) -> None:
    """Verify CAS shard storage, Merkle Proof-of-Storage, PoUW Block #1 mining, and fee settlement."""
    store = ContentAddressedSwarmStore(tmp_path / "swarm_store")
    ledger_path = tmp_path / "swarm_store" / "hive_pouw_ledger.json"
    ledger = HiveSwarmLedger(ledger_path=ledger_path)

    # Check Bitcoin-style halving schedule
    assert ledger.block_reward_at_height(1) == 50.0
    assert ledger.block_reward_at_height(10_000) == 25.0
    assert ledger.block_reward_at_height(20_000) == 12.5

    nodes = ["seti-node-paris-01", "seti-node-lyon-02", "seti-node-nice-03"]
    storage_receipts = []
    work_receipts = []

    for idx, nid in enumerate(nodes):
        samples = [f"def fn_{idx}_{j}(x): return x + {j}" for j in range(12)]
        cid, path, nbytes = store.store_hf_dataset_shard(
            node_id=nid,
            domain_name="Python_Systems_Code",
            shard_id=idx,
            samples=samples,
        )
        assert cid.startswith("cid://sha256/")
        assert nbytes == store.SHARD_BLOCK_ALIGNMENT_BYTES

        s_receipt = store.verify_proof_of_storage(
            node_id=nid,
            shard_cid=cid,
            shard_path=path,
            prev_block_hash=ledger.chain[-1].block_hash,
        )
        assert s_receipt.verified is True
        storage_receipts.append(s_receipt)

        work_receipts.append(
            WorkReceipt(
                node_id=nid,
                steam_id64=f"7656119800000000{idx + 1}",
                branch_name="Python_Systems_Code",
                flops_contributed=3.84e11,
                cosine_alignment=0.98,
                spot_check_passed=True,
                multikrum_accepted=True,
                delta_sha256=f"honest_delta_{idx}",
            )
        )

    # Inject 1 Byzantine Poisoner Node
    work_receipts.append(
        WorkReceipt(
            node_id="byzantine-botnet-99",
            steam_id64="76561198999999999",
            branch_name="Python_Systems_Code",
            flops_contributed=3.84e11,
            cosine_alignment=-0.45,
            spot_check_passed=True,
            multikrum_accepted=False,
            delta_sha256="poison_delta_99",
        )
    )

    dummy_lora = {"layers.0.mlp.gate_proj.lora_A.weight": torch.randn(8, 32)}
    expert_cid, expert_path = store.publish_lora_expert_checkpoint(
        domain_name="Python_Systems_Code",
        lora_state_dict=dummy_lora,
        metadata={"dataset": "sahil2801/CodeAlpaca-20k", "outer_step": 1},
    )
    assert expert_path.exists()
    loaded_cid, loaded_meta, loaded_state = store.load_lora_expert_checkpoint(expert_cid)
    assert loaded_cid == expert_cid
    assert loaded_meta["dataset"] == "sahil2801/CodeAlpaca-20k"
    assert "layers.0.mlp.gate_proj.lora_A.weight" in loaded_state

    block = ledger.mine_pouw_block(
        outer_step=1,
        expert_cid=expert_cid,
        work_receipts=work_receipts,
        storage_receipts=storage_receipts,
        validator_node_ids=["validator-node-eu-01"],
    )
    assert block.block_height == 1
    assert block.total_block_reward == 50.0
    assert "byzantine-botnet-99" in block.slashed_nodes
    assert ledger.balances["seti-node-paris-01"] == 15.0
    assert ledger.balances["seti-node-lyon-02"] == 15.0
    assert ledger.balances["seti-node-nice-03"] == 15.0
    assert ledger.balances["validator-node-eu-01"] == 5.0
    assert ledger.balances["byzantine-botnet-99"] == 0.0

    settlement = ledger.settle_swarm_inference(
        consumer_node_id="seti-node-paris-01",
        provider_node_ids=["seti-node-lyon-02", "seti-node-nice-03"],
        domain_expert="Python_Systems_Code",
        verified_tokens=45,
        rtt_speedup=5.0,
        rate_per_token=0.005,
    )
    assert settlement.fee_paid_hive == 0.225
    assert ledger.balances["seti-node-paris-01"] == 14.775
    assert ledger.balances["seti-node-lyon-02"] == 15.1125
    assert ledger.balances["seti-node-nice-03"] == 15.1125
    assert ledger.verify_chain_integrity() is True
    assert ledger_path.exists()


def test_autonomous_3state_seti_daemon_and_dynamic_moe_hotplug() -> None:
    """Verify 3-state SETI daemon switching and dynamic BTM MoE v2 expert hot-plugging (E -> E+1)."""
    daemon = ZeroIntrusionDaemon(
        idle_threshold_seconds=30.0,
        enforce_os_idle_hook=True,
        simulated_idle_seconds=120.0,
        simulated_active_game=None,
    )
    assert daemon.resolve_seti_agent_state() == SETIAgentState.STATE_2_IDLE_SETI_MINER
    assert daemon.resolve_seti_agent_state(consumer_query_active=True) == SETIAgentState.STATE_3_SWARM_CONSUMER

    daemon.simulated_active_game = "cyberpunk2077.exe"
    assert daemon.resolve_seti_agent_state(consumer_query_active=False) == SETIAgentState.STATE_1_GAMING_SEED_ONLY

    # Test Dynamic Domain Expert Hot-Plug (E=2 -> E=3)
    base_model, _, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=8, lora_alpha=16, gradient_checkpointing=False)
    init_lora = extract_lora_state_dict(bundle.model)

    branch_states = {
        "Code": {k: v.clone() for k, v in init_lora.items()},
        "Mathematics": {k: v.clone() + 0.01 for k, v in init_lora.items()},
    }
    moe_model, report = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=branch_states,
        lora_rank_r=8,
        lora_alpha=16,
        top_k=2,
        in_place=True,
    )
    assert moe_model.num_experts == 2
    assert report.num_experts == 2

    new_domain_lora = {k: v.clone() - 0.01 for k, v in init_lora.items()}
    new_count = moe_model.hotplug_domain_expert(
        domain_name="Python_Systems_Code",
        lora_state_dict=new_domain_lora,
        lora_rank_r=8,
        lora_alpha=16,
    )
    assert new_count == 3
    assert moe_model.expert_domains == ["Code", "Mathematics", "Python_Systems_Code"]
    for blk in moe_model.get_moe_blocks():
        assert blk.router.wg.out_features == 3

    x = torch.randint(0, 256, (2, 12), dtype=torch.long)
    out = moe_model(input_ids=x, labels=x)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_three_hf_datasets_streaming_and_3block_circular_economy(tmp_path: Path) -> None:
    """Verify multi-dataset Hugging Face streaming (CodeAlpaca, PubMedQA, GSM8K) and 3-block symmetric circular economy."""
    from hivemind.cli.demo_hf_domain_swarm import fetch_hf_domain_subset

    code_samples = fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=6)
    pubmed_samples = fetch_hf_domain_subset("qiaojin/PubMedQA", num_samples=6)
    gsm8k_samples = fetch_hf_domain_subset("openai/gsm8k", num_samples=6)

    assert len(code_samples) == 6
    assert len(pubmed_samples) == 6
    assert len(gsm8k_samples) == 6

    store = ContentAddressedSwarmStore(tmp_path / "swarm_store")
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "swarm_store" / "hive_pouw_ledger.json")
    nodes = ["seti-node-paris-01", "seti-node-lyon-02", "seti-node-nice-03"]

    for exp_idx, dom in enumerate(
        ["Python_Systems_Code", "Biomedical_Clinical_QA", "Mathematical_Reasoning"], start=1
    ):
        s_receipts = []
        w_receipts = []
        for idx, nid in enumerate(nodes):
            cid, path, _ = store.store_hf_dataset_shard(nid, dom, idx, [f"{dom}_{idx}"])
            s_receipts.append(
                store.verify_proof_of_storage(nid, cid, path, ledger.chain[-1].block_hash)
            )
            w_receipts.append(
                WorkReceipt(
                    node_id=nid,
                    steam_id64=f"7656119800000000{idx + 1}",
                    branch_name=dom,
                    flops_contributed=1.0e11,
                    cosine_alignment=0.98,
                    spot_check_passed=True,
                    multikrum_accepted=True,
                    delta_sha256=f"delta_{exp_idx}_{idx}",
                )
            )
        ledger.mine_pouw_block(
            outer_step=exp_idx,
            expert_cid=f"cid://sha256/expert-{dom.lower()}",
            work_receipts=w_receipts,
            storage_receipts=s_receipts,
            validator_node_ids=["validator-node-eu-01"],
        )
        consumer = nodes[exp_idx - 1]
        providers = [n for n in nodes if n != consumer]
        ledger.settle_swarm_inference(
            consumer_node_id=consumer,
            provider_node_ids=providers,
            domain_expert=dom,
            verified_tokens=45,
            rtt_speedup=5.0,
            rate_per_token=0.005,
        )

    assert len(ledger.chain) == 4  # Genesis + 3 PoUW blocks
    assert ledger.verify_chain_integrity() is True
    # Symmetric circular economy: each node mined 45 HIVE, spent 0.225 HIVE, and earned 2 * 0.1125 HIVE = 45.0 HIVE
    for nid in nodes:
        assert ledger.balances[nid] == 45.0
    assert ledger.balances["validator-node-eu-01"] == 15.0


def test_hf_hack_miner_gamification_and_free_steam_edition(tmp_path: Path) -> None:
    """Verify Hugging Face Hack-Miner data scanner, gamer scoring, Steam achievements, and Free Steam Edition packaging."""
    from fastapi.testclient import TestClient
    from hivemind.game.hf_hack_miner import (
        GamerMinerProfile,
        HackMiningGameEngine,
        HFDataReconScanner,
        execute_interactive_bounty_probe,
        execute_interactive_hf_shard_scan,
    )
    from hivemind.game.steam_edition import (
        STEAM_APP_ID,
        build_free_steam_edition_package,
        create_hack_miner_fastapi_app,
    )

    raw_rows = [
        "### Instruction:\nWrite binary_search in Python.\n### Response:\ndef binary_search(arr, target):\n    while low <= high: return mid",
        "### Instruction:\nSort list.\n### Response:\nreturn sorted(arr)",
        "### Instruction:\nBinary search helper.\n### Response:\ndef helper(low, high): return (low + high) // 2",
    ]
    curated, sig_score = HFDataReconScanner.score_and_curate_shard(
        raw_samples=raw_rows,
        gamer_idx=0,
        num_gamers=1,
        target_keywords=["def", "low", "high", "while", "mid", "return"],
    )
    assert len(curated) == 3
    assert 0.50 <= sig_score <= 1.0

    engine = HackMiningGameEngine(store_dir=tmp_path / "engine_store")
    audit = engine.verify_all_steam_accounts()
    assert len(audit["verified_gamers"]) == 3
    assert all(g["verified"] is True for g in audit["verified_gamers"])
    assert len(audit["blocked_sybil_bots"]) == 2
    assert all(b["verified"] is False for b in audit["blocked_sybil_bots"])

    gamer = GamerMinerProfile(
        gamer_tag="CyberValkyrie_FR",
        steam_id64="76561198000000001",
        node_id="seti-node-paris-01",
        region="EU-West-Paris",
        specialty_focus="Code & Algorithmic Breaching",
        total_points=8500,
        shards_mined=3,
        bounties_solved=3,
        boss_raids_cleared=1,
        swarm_tokens_served=45,
        best_loss_reduction_pct=78.5,
    )
    gamer.update_rank_and_achievements(is_top_miner=True)
    assert "Singularity Core" in gamer.rank_title
    assert "TOP_1_APEX_MINER" in gamer.achievements_unlocked
    assert "SWARM_ARCHITECT_E3" in gamer.achievements_unlocked
    assert "MOE_BOSS_SLAYER" in gamer.achievements_unlocked

    receipt_hmac = engine.steamworks.upload_hack_miner_score(
        steam_id64=gamer.steam_id64,
        persona_name=gamer.gamer_tag,
        total_points=gamer.total_points,
        swarm_contribution_points=2500,
        hive_balance=45.0,
        achievements=gamer.achievements_unlocked,
    )
    assert receipt_hmac.startswith("steam_lb_hmac_")

    pkg = build_free_steam_edition_package(
        game_report={
            "game_title": "HiveMind: Neural Breach",
            "hardware": {"gpu_name": "NVIDIA GeForce RTX 2070", "vram_total_gb": 8.0},
            "vram_telemetry": {"peak_allocated_mb": 1326.0, "free_vram_gb": 5.59},
            "leaderboard": [],
            "missions": [],
            "moe_boss_fusion": {
                "num_experts": 3,
                "expert_domains": ["Code", "Bio", "Math"],
                "router_initial_loss": 1.8,
                "router_final_loss": 1.3,
                "expert_utilization": {"Code": 0.33, "Bio": 0.33, "Math": 0.34},
            },
        },
        output_root=tmp_path / "HiveMind_Steam_FreeEdition",
    )
    assert pkg["steam_app_id"] == STEAM_APP_ID
    assert pkg["is_free_to_play"] is True
    assert Path(pkg["steam_appid_file"]).read_text(encoding="utf-8").strip() == str(STEAM_APP_ID)
    assert Path(pkg["app_build_vdf"]).exists()
    assert Path(pkg["depot_build_vdf"]).exists()
    assert Path(pkg["steamworks_partner_schema"]).exists()
    assert Path(pkg["steam_cloud_save_file"]).exists()
    assert Path(pkg["windows_launcher_bat"]).exists()
    assert Path(pkg["windows_hud_launcher_bat"]).exists()
    assert Path(pkg["player_readme_md"]).exists()
    assert Path(pkg["interactive_hud_html"]).exists()

    # Verify interactive shard scan & bounty probe helpers and FastAPI routes
    scan_res = execute_interactive_hf_shard_scan(
        gamer_tag="CyberValkyrie_FR",
        dataset_id="sahil2801/CodeAlpaca-20k",
        custom_rows=raw_rows,
    )
    assert scan_res["merkle_pos_verified"] is True
    assert scan_res["data_discovery_pts_awarded"] >= 200

    bounty_res = execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="Write a Python function to calculate HIVE minted across 3 blocks.",
    )
    assert len(bounty_res["top2_active_experts"]) == 2
    assert bounty_res["total_pts_awarded"] > 700

    app = create_hack_miner_fastapi_app()
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/leaderboard").status_code == 200
    assert client.get("/api/missions").status_code == 200
    assert client.get("/api/steam-manifest").status_code == 200
    assert client.get("/api/steam-cloud").status_code == 200
    r_bounty = client.post(
        "/api/solve-bounty",
        json={"gamer_tag": "CyberValkyrie_FR", "prompt": "Mitochondria ATP synthesis in Python"},
    )
    assert r_bounty.status_code == 200
    assert r_bounty.json()["total_pts_awarded"] > 0


def test_phase2_swarm_fabric_m1_to_m4_and_openai_gateway(tmp_path: Path) -> None:
    """Verify Swarm Fabric M1 (Geo-Ring), M2 (Pinned RAM DMA + Macro/Micro Router), M3 (FP8 + Tree Speculation gamma=6), and M4 (OPoI + L2 State Channel + /v1/chat/completions)."""
    from fastapi.testclient import TestClient
    from hivemind.boinc.coordinator_api import BOINCCoordinatorService, create_coordinator_fastapi_app
    from hivemind.config import HiveMindConfig
    from hivemind.swarm.swarm_fabric import (
        GeoRingPeer,
        HiveLayer2StateChannel,
        OptimisticProofOfInferenceAuditor,
        PinnedRAMExpertDMAStreamer,
        RegionalGeoRingTransport,
        TokenTreeSpeculativeVerifier,
        TwoLevelMacroMicroRouter,
        compress_hidden_states_fp8,
        decompress_hidden_states_fp8,
    )

    # M1: Sub-20ms Kademlia Geo-Ring
    from hivemind.swarm.quic_transport import AsyncQUICGeoRingTransport

    ring = RegionalGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    assert ring.register_peer(GeoRingPeer("peer-paris-01", "EU-West-Paris", 11.2)) is True
    assert ring.register_peer(GeoRingPeer("peer-slow-99", "EU-West-Paris", 42.0)) is False
    closest = ring.find_closest_peers("Python_Systems_Code", k=1)
    assert len(closest) == 1 and closest[0].peer_id == "peer-paris-01"

    quic_ring = AsyncQUICGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    udp_bench = quic_ring.benchmark_loopback_0rtt(torch.randn(1, 8, 32, dtype=torch.float16), num_frames=120)
    assert udp_bench["frames_per_sec"] >= 100.0
    assert udp_bench["serialization_overhead_ms"] < 1.5
    ws_bench = quic_ring.benchmark_loopback_0rtt(
        torch.randn(1, 8, 32, dtype=torch.float16), num_frames=20, simulate_udp_blocked=True
    )
    assert ws_bench["protocol"] == "libp2p_websocket_tls"

    # M2: 2-Level Macro/Micro Router & Pinned Host RAM Async DMA Streamer
    router = TwoLevelMacroMicroRouter(
        hidden_size=32,
        macro_clusters={
            "Code_Math": ["Python_Systems_Code", "Mathematical_Reasoning"],
            "Bio_Clinical": ["Biomedical_Clinical_QA", "Legal_Medical"],
        },
        top_k_micro=2,
    )
    hidden = torch.randn(2, 6, 32)
    route_out = router(hidden)
    assert route_out["selected_macro_cluster"] in ["Code_Math", "Bio_Clinical"]
    assert len(route_out["selected_micro_experts"]) == 2

    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    nbytes = streamer.stage_expert_in_pinned_ram(
        "Python_Systems_Code",
        {"lora_A": torch.randn(8, 32), "lora_B": torch.randn(32, 8)},
    )
    assert nbytes > 0
    dma_info = streamer.async_dma_prefetch_to_vram(
        "Python_Systems_Code", simulated_attention_compute_ms=5.0
    )
    assert dma_info["exposed_swap_latency_ms"] == 0.0

    # M3: FP8 Hidden-State Wire Compression & Token Tree Speculation (gamma = 6)
    pkt = compress_hidden_states_fp8(hidden.to(torch.float16))
    assert pkt.compression_ratio >= 1.8
    restored = decompress_hidden_states_fp8(pkt)
    assert restored.shape == hidden.shape

    tree_verifier = TokenTreeSpeculativeVerifier(gamma=6, branching_factor=2)
    tree_res = tree_verifier.verify_token_tree(
        draft_tree_candidates=[[10, 20, 30, 99, 100, 101], [10, 20, 30, 40, 50, 60]],
        target_reference_tokens=[10, 20, 30, 40, 50, 60, 70],
    )
    assert tree_res["tree_depth_gamma"] == 6
    assert tree_res["accepted_length"] == 7  # 6 draft tokens + 1 bonus token

    # M4: OPoI 5% Activation Auditor, Layer-2 $HIVE State Channel & OpenAI /v1/chat/completions
    auditor = OptimisticProofOfInferenceAuditor(audit_probability=0.05)
    audit_pass = auditor.verify_activation_commitment("peer-paris-01", hidden, hidden)
    assert audit_pass["opoi_passed"] is True
    audit_fail = auditor.verify_activation_commitment("peer-poison-99", hidden, -hidden)
    assert audit_fail["opoi_passed"] is False and audit_fail["slashed"] is True

    ledger = HiveSwarmLedger(ledger_path=tmp_path / "l2_ledger.json")
    ledger.balances["seti-node-paris-01"] = 10.0
    l2 = HiveLayer2StateChannel(
        consumer_id="seti-node-paris-01",
        provider_ids=["seti-node-lyon-02", "seti-node-nice-03"],
        escrow_hive=5.0,
    )
    v1 = l2.sign_micro_voucher(20)
    v2 = l2.sign_micro_voucher(25)
    assert v2.nonce == 2 and v2.cumulative_tokens == 45
    settled = l2.settle_to_l1_ledger(ledger)
    assert settled["settled_hive"] == 0.225
    assert ledger.balances["seti-node-paris-01"] == 9.775

    cfg = HiveMindConfig.from_yaml()
    coord = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state={"lora_A": torch.zeros(4, 8)},
        base_model_sha256="sha256_test",
        shards_dir=tmp_path / "shards",
    )
    app = create_coordinator_fastapi_app(coord)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "hivemind-qwen2.5-sparse-moe-top2",
            "messages": [{"role": "user", "content": "Write a Python function to compute 3 blocks * 50 HIVE."}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["hivemind_swarm_fabric"]["tree_speculation_gamma"] == 6
    assert body["hivemind_swarm_fabric"]["opoi_audit"]["opoi_passed"] is True


def test_lean4_specification_and_foundation_article_autoevolve_pipeline() -> None:
    """Verify Lean 4 formal specification (0 sorry, 0 axiom), foundation .tex/.pdf article, and AutoevolveAI peer review."""
    from paper.autoevolve_peer_review_pipeline import (
        AUTOEVOLVE_MANIFEST_JSON,
        PDF_PATH,
        PEER_REVIEW_JSON,
        TEX_PATH,
        run_autoevolve_peer_review,
        verify_lean4_specification,
    )

    lean_res = verify_lean4_specification()
    assert lean_res["sorry_count"] == 0
    assert lean_res["user_axiom_count"] == 0
    assert lean_res["theorem_count"] >= 14
    assert lean_res["tier_a_machine_checked"] is True

    assert TEX_PATH.exists() and TEX_PATH.stat().st_size > 10_000
    assert PDF_PATH.exists() and PDF_PATH.stat().st_size > 100_000

    review_report = run_autoevolve_peer_review()
    assert review_report["meta_review"]["decision"] == "ACCEPT_AS_FOUNDATION_ARTICLE"
    assert review_report["meta_review"]["aggregate_score"] >= 9.0
    assert PEER_REVIEW_JSON.exists()
    assert AUTOEVOLVE_MANIFEST_JSON.exists()
