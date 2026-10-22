"""HM-S6-02: 3-tier router, token-tree expert prefetch, and the Season-5 hot-plug regressions."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.btm_moe import (
    HierarchicalMoERouter,
    ThreeTierHierarchicalMoERouter,
    hotplug_hierarchical_experts,
    hotplug_three_tier_experts,
)
from hivemind.core.btm_upcycle import initialize_domain_branches, upcycle_branches_to_sparse_moe
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.swarm_fabric import PinnedRAMExpertDMAStreamer


@pytest.fixture
def tiny_moe():
    base_model, _, _ = build_local_qwen2_backbone(
        seed=42, vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model=base_model, rank_r=2, lora_alpha=4,
                                   gradient_checkpointing=False, seed=42)
    init_lora = extract_lora_state_dict(bundle.model)
    branches = initialize_domain_branches(init_lora, expert_domains=["DomA", "DomB", "DomC"])
    moe, _ = upcycle_branches_to_sparse_moe(
        bundle.model, {k: v.lora_state for k, v in branches.items()},
        lora_rank_r=2, lora_alpha=4, in_place=False,
    )
    moe.moe_blocks = moe.get_moe_blocks()  # type: ignore[attr-defined]
    moe.hidden_size = 16  # type: ignore[attr-defined]
    return moe, init_lora


def _forward_ok(moe) -> None:
    ids = torch.randint(0, 128, (2, 8))
    out = moe(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss)


def test_season5_router_survives_hotplug_and_keeps_indices_valid() -> None:
    torch.manual_seed(0)
    router = HierarchicalMoERouter(hidden_size=8, num_macro_domains=2, experts_per_domain=2, top_k=2)
    router.expand_for_new_expert()
    assert router.num_experts == 5 and router.cluster_sizes == [2, 3]
    _, selected, _ = router(torch.randn(64, 8))
    assert int(selected.min()) >= 0 and int(selected.max()) <= 4

    with torch.no_grad():
        router.macro_gate.weight.zero_()
        router.macro_gate.weight[1, 0] = 10.0
        router.micro_gates[1].weight.zero_()
        router.micro_gates[1].weight[2, 0] = 10.0
    x = torch.zeros(4, 8)
    x[:, 0] = 1.0
    _, selected, _ = router(x)
    assert int(selected[:, 0].unique()) == 4, "new expert lives at offset 2 + micro index 2"


def test_season5_router_rejects_direct_num_experts_writes() -> None:
    router = HierarchicalMoERouter(hidden_size=8, num_macro_domains=2, experts_per_domain=2)
    with pytest.raises(ValueError):
        router.num_experts = 9
    router.num_experts = 4


def test_three_tier_default_layout_counts_16_vs_128() -> None:
    torch.manual_seed(0)
    router = ThreeTierHierarchicalMoERouter(hidden_size=32)
    assert router.num_experts == 128
    assert router.gating_inner_products_per_token == 16
    assert router.flat_gating_inner_products_per_token == 128
    assert router.gating_reduction_factor == 8.0

    weights, selected, aux = router(torch.randn(50, 32))
    assert weights.shape == (50, 2) and selected.shape == (50, 2)
    assert int(selected.min()) >= 0 and int(selected.max()) < 128
    assert torch.allclose(weights.sum(dim=-1), torch.ones(50), atol=1e-5)
    assert (selected[:, 0] != selected[:, 1]).all()
    assert float(aux) > 0.0
    assert (selected // 8 == selected[:, :1] // 8).all(), "both experts come from one micro cluster"


@pytest.mark.parametrize(
    "kwargs", [{"num_super_clusters": 0}, {"macro_per_super": 0}, {"micro_per_macro": 0},
               {"micro_cluster_sizes": [1, 2]}, {"micro_cluster_sizes": [0] * 16}]
)
def test_three_tier_rejects_bad_layouts(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ThreeTierHierarchicalMoERouter(hidden_size=8, **kwargs)


def test_three_tier_ragged_clusters_and_hotplug_reach_new_expert() -> None:
    torch.manual_seed(0)
    router = ThreeTierHierarchicalMoERouter(
        hidden_size=8, num_super_clusters=2, macro_per_super=2, top_k=2,
        micro_cluster_sizes=[2, 3, 2, 2],
    )
    assert router.num_experts == 9 and router.micro_cluster_offsets == [0, 2, 5, 7]
    router.expand_for_new_expert()
    assert router.num_experts == 10 and router.micro_cluster_sizes == [2, 3, 2, 3]
    _, selected, _ = router(torch.randn(64, 8))
    assert int(selected.max()) <= 9

    with torch.no_grad():
        router.super_gate.weight.zero_(); router.super_gate.weight[1, 0] = 10.0
        router.macro_gates[1].weight.zero_(); router.macro_gates[1].weight[1, 0] = 10.0
        router.micro_gates[3].weight.zero_(); router.micro_gates[3].weight[2, 0] = 10.0
    x = torch.zeros(3, 8); x[:, 0] = 1.0
    _, selected, _ = router(x)
    assert int(selected[:, 0].unique()) == 9


def test_three_tier_predict_candidate_experts_is_a_subdistribution_on_the_winning_path() -> None:
    torch.manual_seed(0)
    router = ThreeTierHierarchicalMoERouter(hidden_size=16, num_super_clusters=2, macro_per_super=2,
                                            micro_per_macro=4, top_k=2)
    x = torch.randn(3, 5, 16)
    idx, probs = router.predict_candidate_experts(x, top_n=4)
    assert idx.shape == (15, 4) and probs.shape == (15, 4)
    assert (probs >= 0).all() and (probs.sum(dim=-1) <= 1.0 + 1e-5).all()
    assert (idx // 4 == idx[:, :1] // 4).all()
    _, selected, _ = router(x.reshape(-1, 16))
    assert (selected[:, 0] == idx[:, 0]).all(), "top-1 forward choice equals top-1 candidate"


def test_token_tree_prefetch_weights_candidates_by_branch_probability() -> None:
    torch.manual_seed(0)
    router = ThreeTierHierarchicalMoERouter(hidden_size=8, num_super_clusters=1, macro_per_super=1,
                                            micro_per_macro=4, top_k=2)
    with torch.no_grad():
        router.micro_gates[0].weight.zero_()
        router.micro_gates[0].weight[0, 0] = 6.0
        router.micro_gates[0].weight[1, 1] = 6.0
        router.micro_gates[0].weight[2, 2] = 6.0
    names = ["E0", "E1", "E2", "E3"]
    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    for n in ("E0", "E1", "E3"):
        streamer.stage_expert_in_pinned_ram(n, {"w": torch.zeros(16)})

    def branch(dim: int) -> torch.Tensor:
        h = torch.zeros(1, 2, 8); h[..., dim] = 1.0
        return h

    rep = streamer.prefetch_token_tree_lookahead(
        router, [branch(0), branch(1), branch(2)], [0.6, 0.3, 0.1], names, layer_idx=0, max_prefetch_experts=2,
    )
    demand = list(rep["expected_expert_demand"])
    assert demand[:3] == ["E0", "E1", "E2"]
    assert rep["prefetched_experts"] == ["E0", "E1"]
    assert rep["lookahead_dma_slot"] == "slot_B" and set(streamer.double_buffer_slots["slot_B"]) == {"E0", "E1"}
    assert "simulated" in rep["latency_basis"] and "exposed_swap_latency_ms_simulated" in rep
    assert "exposed_swap_latency_ms" not in rep

    rep2 = streamer.prefetch_token_tree_lookahead(router, [branch(2)], [1.0], names, layer_idx=1, max_prefetch_experts=1)
    assert rep2["prefetched_experts"] == [] and rep2["skipped_not_staged"] == ["E2"]
    with pytest.raises(ValueError):
        streamer.prefetch_token_tree_lookahead(router, [branch(0)], [0.5, 0.5], names)
    with pytest.raises(ValueError):
        streamer.prefetch_token_tree_lookahead(router, [branch(0)], [0.0], names)
    with pytest.raises(TypeError):
        streamer.prefetch_token_tree_lookahead(object(), [branch(0)], [1.0], names)


def test_season5_hotplug_entry_point_runs_without_shims(tiny_moe) -> None:
    moe, init_lora = tiny_moe
    report = hotplug_hierarchical_experts(moe, {"DomD": init_lora}, canonical_rank_r=2, num_macro_domains=2)
    assert report["router_type"] == "HierarchicalMoERouter"
    assert report["total_experts"] == 4 and sum(report["cluster_sizes"]) == 4
    assert report["fits_under_vram_ceiling"] is None and "not_measured" in report["vram_measurement"]
    assert "not a measured latency" in report["gating_reduction_basis"]
    _forward_ok(moe)
    moe.hotplug_domain_expert("DomE", init_lora, lora_rank_r=2, lora_alpha=4)
    assert moe.moe_blocks[0].router.num_experts == moe.num_experts == 5
    _forward_ok(moe)
    hotplug_hierarchical_experts(moe, {}, canonical_rank_r=2, num_macro_domains=2)
    _forward_ok(moe)


def test_three_tier_hotplug_entry_point_sizes_router_to_model(tiny_moe) -> None:
    moe, init_lora = tiny_moe
    report = hotplug_three_tier_experts(
        moe, {"DomD": init_lora, "DomE": init_lora}, canonical_rank_r=2,
        num_super_clusters=4, macro_per_super=4, micro_per_macro=8,
    )
    assert report["router_type"] == "ThreeTierHierarchicalMoERouter"
    assert report["total_experts"] == 5 == sum(report["micro_cluster_sizes"])
    assert report["num_super_clusters"] * report["macro_per_super"] == len(report["micro_cluster_sizes"])
    assert report["gating_flops_reduction_factor"] == round(5 / report["gating_inner_products_per_token"], 2)
    assert report["vram_measured_gb"] is None and report["parameter_footprint_gb"] > 0
    _forward_ok(moe)
    moe.hotplug_domain_expert("DomF", init_lora, lora_rank_r=2, lora_alpha=4)
    assert moe.moe_blocks[0].router.num_experts == moe.num_experts == 6
    _forward_ok(moe)
