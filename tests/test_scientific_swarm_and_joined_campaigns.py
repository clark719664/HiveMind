"""Scientific Swarm Evaluation, Peer Knowledge Transfer & Joined Campaign Tests."""

from __future__ import annotations

from pathlib import Path

import torch

from hivemind.game.hf_hack_miner import execute_interactive_joined_campaign
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.scientific_swarm_eval import (
    PROPOSED_JOINED_CAMPAIGNS,
    SCIENTIFIC_BENCHMARK_SUITES,
    JoinedSwarmCampaignEngine,
    SwarmKnowledgeTransferEngine,
    ties_merge_lora_adapters,
)


def _sample_ref_lora(rank: int = 16, dim: int = 64) -> dict[str, torch.Tensor]:
    return {
        "layer0.lora_A": torch.zeros((rank, dim), dtype=torch.float32),
        "layer0.lora_B": torch.zeros((dim, rank), dtype=torch.float32),
        "layer1.lora_A": torch.zeros((rank, dim), dtype=torch.float32),
        "layer1.lora_B": torch.zeros((dim, rank), dtype=torch.float32),
    }


def test_peer_agents_leverage_retrained_adapter_on_scientific_holdout(tmp_path: Path) -> None:
    """
    Verifies that after `CyberValkyrie_FR` retrains an adapter on `Rust_Systems_Security`,
    all other swarm agents (`NeuralGhost_DE`, `ZeroCool_UK`, `SakuraNode_JP`) fetch and hot-swap
    the CAS checkpoint with 0 local training steps and inherit >= 95% of the holdout loss reduction.
    """
    store = ContentAddressedSwarmStore(tmp_path / "swarm_store")
    engine = SwarmKnowledgeTransferEngine(store=store)
    ref_lora = _sample_ref_lora()

    audit = engine.verify_swarm_peer_knowledge_transfer(
        domain_name="Rust_Systems_Security",
        trainer_agent_id="CyberValkyrie_FR",
        reference_lora_state=ref_lora,
        inner_steps=16,
        topk_ratio=0.05,
        royalty_per_peer_hive=5.0,
    )

    assert audit.expert_cid.startswith("cid://sha256/")
    assert audit.trainer_final_loss < audit.trainer_initial_loss
    assert audit.trainer_loss_reduction_pct > 80.0
    assert len(audit.peer_evaluations) == 3
    assert audit.all_peers_leveraged_successfully is True
    assert audit.knowledge_transfer_efficiency_ratio >= 0.95
    assert audit.mean_peer_post_transfer_loss < audit.mean_peer_pre_transfer_loss
    assert audit.total_peer_gpu_steps_saved == 3 * 16
    assert audit.royalties_paid_to_trainer_hive == 15.0
    assert engine.peer_reuse_counts["CyberValkyrie_FR"] == 3


def test_collective_4domain_swarm_cross_pollination_matrix(tmp_path: Path) -> None:
    """
    Verifies that when 4 swarm agents each train 1 domain and broadcast via CAS, every agent in
    the swarm masters all 4 scientific benchmark domains (`compute_leverage_multiplier == 4.0`).
    """
    store = ContentAddressedSwarmStore(tmp_path / "swarm_store_4d")
    transfer_engine = SwarmKnowledgeTransferEngine(store=store)
    campaign_engine = JoinedSwarmCampaignEngine(transfer_engine=transfer_engine)
    ref_lora = _sample_ref_lora()

    full_report = campaign_engine.execute_full_scientific_swarm_campaign(
        reference_lora_state=ref_lora,
        moe_model=None,
        inner_steps=16,
    )

    matrix = full_report["collective_4domain_agent_matrix"]
    assert len(matrix) == 4
    for agent_id, stats in matrix.items():
        assert stats["domains_mastered"] == 4
        assert stats["domains_trained_locally"] == 1
        assert stats["domains_leveraged_from_swarm_peers"] == 3
        assert stats["compute_leverage_multiplier"] == 4.0
        assert stats["mean_4domain_holdout_loss"] < 0.55, f"{agent_id} holdout loss too high"




def test_joined_coop_campaign_mines_faster_and_better_than_solo(tmp_path: Path) -> None:
    """
    Verifies that a 4-agent Joined Campaign (`CAMPAIGN-COOP-01-DILOCO-BLITZ`) using TIES Sign-Consensus
    Merging achieves both parallel speedup (`>= 3.5x`) and lower holdout loss than a solo miner.
    """
    store = ContentAddressedSwarmStore(tmp_path / "swarm_store_coop")
    transfer_engine = SwarmKnowledgeTransferEngine(store=store)
    campaign_engine = JoinedSwarmCampaignEngine(transfer_engine=transfer_engine)
    ref_lora = _sample_ref_lora()

    res = campaign_engine.execute_joined_shard_parallel_campaign(
        domain_name="Rust_Systems_Security",
        reference_lora_state=ref_lora,
        participating_agents=[
            "CyberValkyrie_FR",
            "NeuralGhost_DE",
            "ZeroCool_UK",
            "SakuraNode_JP",
        ],
        total_inner_steps=16,
        topk_ratio=0.05,
    )

    assert res["effective_parallel_speedup"] >= 3.5
    assert res["coop_beats_solo_miner"] is True
    assert res["joined_coop_holdout_loss"] < res["solo_miner_same_time_holdout_loss"]
    assert res["quality_improvement_over_solo_pct"] > 0.0
    assert res["published_coop_expert_cid"].startswith("cid://sha256/")


def test_dual_miner_and_contributor_leaderboard_and_hf_badges() -> None:
    """
    Verifies the interactive Joined Campaign entrypoint and Dual Leaderboard (Best Miner + Best Contributor).
    """
    report = execute_interactive_joined_campaign(
        lead_gamer_tag="CyberValkyrie_FR",
        campaign_id="CAMPAIGN-COOP-02-POLYMATH-FUSION",
        inner_steps=12,
    )
    assert len(report["proposed_joined_campaigns"]) == len(PROPOSED_JOINED_CAMPAIGNS) == 4
    dual = report["dual_leaderboard"]
    assert dual["best_miner_champion"]["gamer_tag"] == "CyberValkyrie_FR"
    assert dual["best_miner_champion"]["peers_leveraged_count"] >= 3
    assert dual["best_contributor_champion"]["contributor_handle"] == "@xaviercallens"
    contrib_handles = [c["contributor_handle"] for c in dual["contributor_standings"]]
    assert "@clark719664" in contrib_handles
    assert "@pedromineo" in contrib_handles
    assert len(dual["miner_standings"]) == 4
    assert len(dual["contributor_standings"]) >= 3
    assert "HF_SWARM_KNOWLEDGE_DONOR" in dual["hf_gamification_badges_catalog"]


def test_hf_community_hub_bundle_sharing_and_swarm_guilds(tmp_path: Path) -> None:
    """
    Verifies the v1.5.0 Hugging Face Community Sharing Hub & Swarm Mining Guilds:
    - Exporting a multi-domain community LoRA bundle (`r=8`, `r=16`, `r=32`, `r=64` normalized to `r=16` via truncated SVD)
    - Generating the Hugging Face Hub `README.md` Model Card, `adapter_config.json`, and `hivemind_swarm_manifest.json`
    - Zero-retraining peer import with SHA-256 verification and automatic `$HIVE` + `PTS` reuse royalty settlement
    - Swarm Mining Guild synergy multipliers (`1.30x`–`1.45x`) and Guild Alliance Leaderboard
    """
    from hivemind.swarm.hf_community_hub import (
        CommunityAdapterBundleManager,
        SwarmMiningGuildEngine,
    )

    store = ContentAddressedSwarmStore(tmp_path / "community_cas")
    manager = CommunityAdapterBundleManager(store=store)

    def _make_adapter(rank_r: int, seed: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        return {
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight": torch.randn(
                rank_r, 64, generator=gen
            ),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight": torch.randn(
                128, rank_r, generator=gen
            ),
        }

    bundle_dir = tmp_path / "hf_bundle_s4_guild02"
    pub = manager.create_and_publish_bundle(
        bundle_id="HIVEMIND-S4-POLYGLOT-BUNDLE-V1",
        bundle_title="HiveMind Season 4 Multi-Vendor & ZK-OPoI Community Bundle",
        guild_id="GUILD-02-GLOBAL-ZK-GUARDIANS",
        domain_adapters={
            "Rust_Systems_Security": {
                "lora_state_dict": _make_adapter(8, 101),
                "dataset_id": "anon/rust-code-instructions",
                "author_gamer_tag": "CyberValkyrie_FR",
                "author_contributor_handle": "@xaviercallens",
                "baseline_loss": 6.45,
                "trained_loss": 0.35,
            },
            "ZK_OPoI_Prover": {
                "lora_state_dict": _make_adapter(32, 202),
                "dataset_id": "hivemind/opoi-merkle-activations",
                "author_gamer_tag": "ZeroCool_UK",
                "author_contributor_handle": "@clark719664",
                "baseline_loss": 6.12,
                "trained_loss": 0.41,
            },
            "Triton_Kernel_Synth": {
                "lora_state_dict": _make_adapter(64, 303),
                "dataset_id": "hivemind/triton-int4-fp8-kernels",
                "author_gamer_tag": "NeuralGhost_DE",
                "author_contributor_handle": "@pedromineo",
                "baseline_loss": 5.98,
                "trained_loss": 0.38,
            },
        },
        output_dir=bundle_dir,
        canonical_rank_r=16,
    )

    assert (bundle_dir / "README.md").exists()
    assert (bundle_dir / "adapter_config.json").exists()
    assert (bundle_dir / "hivemind_swarm_manifest.json").exists()
    readme_text = (bundle_dir / "README.md").read_text(encoding="utf-8")
    assert "library_name: peft" in readme_text
    assert "@clark719664" in readme_text and "@pedromineo" in readme_text
    assert pub["export_info"]["num_experts_exported"] == 3

    import_res = manager.import_community_bundle(
        bundle_dir=bundle_dir,
        importer_agent_id="seti-node-tokyo-04",
    )
    assert import_res["status"] == "verified_and_hotplugged"
    assert import_res["sha256_verified"] is True
    assert import_res["royalties_settled_hive"] == 45.0
    assert import_res["royalties_settled_pts"] == 750
    assert manager.ledger.get_balance("CyberValkyrie_FR") >= 15.0

    guild_engine = SwarmMiningGuildEngine()
    act = guild_engine.record_guild_bundle_activity(
        guild_id="GUILD-02-GLOBAL-ZK-GUARDIANS",
        base_pts_earned=2000,
        base_hive_earned=100.0,
        peer_imports_count=3,
    )
    assert act["boosted_pts_added"] == int(2000 * 1.45)
    lb = guild_engine.build_guild_leaderboard()
    assert lb["num_guilds"] == 3
    assert lb["top_guild_champion"]["guild_id"] == "GUILD-02-GLOBAL-ZK-GUARDIANS"

