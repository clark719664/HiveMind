"""HM-S6-02 follow-up: token-tree expert prefetch wired into the speculative inference ring.

CPU-only. The hit rate asserted here is an exact count on the tiny MoE; nothing about VRAM or
latency is measured or claimed.
"""

from __future__ import annotations

import math

import pytest
import torch

from hivemind.core.btm_moe import hotplug_three_tier_experts
from hivemind.core.btm_upcycle import initialize_domain_branches, upcycle_branches_to_sparse_moe
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.inference_ring import (
    RegionalDHTSwarmRing,
    SpeculativeSwarmResult,
    prefix_trie_branch_probabilities,
)
from hivemind.swarm.swarm_fabric import PinnedRAMExpertDMAStreamer

DOMAINS = ["DomA", "DomB", "DomC"]


@pytest.fixture
def two_layer_three_tier_moe():
    base_model, _, _ = build_local_qwen2_backbone(
        seed=42, vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model=base_model, rank_r=2, lora_alpha=4,
                                   gradient_checkpointing=False, seed=42)
    init_lora = extract_lora_state_dict(bundle.model)
    branches = initialize_domain_branches(init_lora, expert_domains=DOMAINS)
    moe, _ = upcycle_branches_to_sparse_moe(
        bundle.model, {k: v.lora_state for k, v in branches.items()},
        lora_rank_r=2, lora_alpha=4, in_place=False,
    )
    moe.moe_blocks = moe.get_moe_blocks()  # type: ignore[attr-defined]
    moe.hidden_size = 16  # type: ignore[attr-defined]
    hotplug_three_tier_experts(moe, {}, canonical_rank_r=2, num_super_clusters=1, macro_per_super=1)
    assert len(moe.get_moe_blocks()) == 2
    return moe


def _staged_streamer(names) -> PinnedRAMExpertDMAStreamer:
    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    for n in names:
        streamer.stage_expert_in_pinned_ram(n, {"w": torch.zeros(8)})
    return streamer


def _run(moe, streamer=None, **kw) -> SpeculativeSwarmResult:
    torch.manual_seed(0)
    ring = RegionalDHTSwarmRing(region="EU-West-Paris")
    return ring.run_speculative_inference(
        moe_target_model=moe, draft_model=moe, prompt_ids=torch.tensor([1, 2, 3, 4]),
        max_new_tokens=6, draft_gamma=3, expert_streamer=streamer, **kw,
    )


def test_prefix_trie_probabilities_sum_to_one_over_the_trie() -> None:
    nodes, probs = prefix_trie_branch_probabilities([[7, 8, 9]], [[0.5, 0.5, 0.2]])
    assert nodes == [(), (7,), (7, 8), (7, 8, 9)]
    raw = [1.0, 0.5, 0.25, 0.05]
    assert probs == pytest.approx([r / sum(raw) for r in raw])
    assert math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-12)

    # Shared prefixes are deduplicated by the trie; each node keeps its product once.
    nodes2, probs2 = prefix_trie_branch_probabilities([[7, 8], [7, 9]], [[0.5, 0.4], [0.5, 0.1]])
    assert nodes2 == [(), (7,), (7, 8), (7, 9)]
    assert math.isclose(sum(probs2), 1.0, abs_tol=1e-12)
    assert probs2[2] > probs2[3]
    with pytest.raises(ValueError):
        prefix_trie_branch_probabilities([[1, 2]], [[0.5]])


def test_prefetch_hit_rate_is_an_exact_count_on_the_verified_path(two_layer_three_tier_moe) -> None:
    moe = two_layer_three_tier_moe
    streamer = _staged_streamer(DOMAINS)
    res = _run(moe, streamer, max_prefetch_experts=2)

    assert res.expert_prefetch_enabled is True
    assert res.expert_prefetch_calls == res.swarm_verification_passes  # one next-layer prefetch per pass
    assert len(res.expert_prefetch_log) == res.expert_prefetch_calls
    hits = uses = 0
    for entry in res.expert_prefetch_log:
        assert entry["prefetched_layer"] == 1
        assert math.isclose(sum(entry["branch_probabilities"]), 1.0, abs_tol=1e-9)
        assert len(entry["branch_probabilities"]) == len(entry["branch_prefixes"]) == res.draft_gamma + 1
        assert len(entry["prefetched_experts"]) <= 2
        assert set(entry["prefetched_experts"]) <= set(DOMAINS)
        # Recompute the hit count from the raw per-position router picks recorded in the entry.
        recomputed_hits = sum(
            1 for picks in entry["used_experts_by_position"] for e in picks if e in entry["prefetched_experts"]
        )
        recomputed_uses = sum(len(picks) for picks in entry["used_experts_by_position"])
        assert entry["hits"] == recomputed_hits and entry["uses"] == recomputed_uses
        assert entry["uses"] == len(entry["used_experts_by_position"]) * moe.top_k
        assert "simulated" in entry["streamer_report"]["latency_basis"]
        hits += entry["hits"]
        uses += entry["uses"]
    assert uses > 0
    assert res.expert_prefetch_hits == hits and res.expert_prefetch_expert_uses == uses
    assert res.expert_prefetch_hit_rate == pytest.approx(hits / uses)
    assert 0.0 <= res.expert_prefetch_hit_rate <= 1.0
    assert "measured" in res.expert_prefetch_basis and "simulated" in res.expert_prefetch_basis
    # Experts prefetched into the lookahead DMA slot really landed there.
    last = res.expert_prefetch_log[-1]
    assert set(streamer.double_buffer_slots[last["streamer_report"]["lookahead_dma_slot"]]) == set(
        last["prefetched_experts"]
    )


def test_prefetching_one_expert_per_layer_bounds_the_hit_rate_by_half(two_layer_three_tier_moe) -> None:
    moe = two_layer_three_tier_moe
    res = _run(moe, _staged_streamer(DOMAINS), max_prefetch_experts=1)
    assert all(len(entry["prefetched_experts"]) == 1 for entry in res.expert_prefetch_log)
    assert moe.top_k == 2
    assert 0.0 < res.expert_prefetch_hit_rate <= 0.5


def test_unstaged_experts_are_skipped_and_count_as_misses(two_layer_three_tier_moe) -> None:
    moe = two_layer_three_tier_moe
    res = _run(moe, _staged_streamer([]), max_prefetch_experts=3)
    assert res.expert_prefetch_calls > 0
    assert all(entry["prefetched_experts"] == [] for entry in res.expert_prefetch_log)
    assert all(len(entry["streamer_report"]["skipped_not_staged"]) > 0 for entry in res.expert_prefetch_log)
    assert res.expert_prefetch_hits == 0 and res.expert_prefetch_expert_uses > 0
    assert res.expert_prefetch_hit_rate == 0.0


def test_disabled_streamer_reproduces_the_old_result_exactly(two_layer_three_tier_moe) -> None:
    moe = two_layer_three_tier_moe
    off = _run(moe, None)
    on = _run(moe, _staged_streamer(DOMAINS))
    assert off.expert_prefetch_enabled is False
    assert off.expert_prefetch_calls == 0 and off.expert_prefetch_log == []
    assert off.expert_prefetch_hit_rate is None and off.expert_prefetch_basis == "disabled"
    assert on.generated_tokens == off.generated_tokens
    assert on.swarm_verification_passes == off.swarm_verification_passes
    assert on.mean_acceptance_rate == off.mean_acceptance_rate
    assert on.expert_routing_share == off.expert_routing_share
    # Prefetch hooks must not leave anything behind on the model.
    for blk in moe.get_moe_blocks():
        assert not blk.router._forward_hooks
    for layer in moe.backbone.model.layers:
        assert not layer._forward_pre_hooks


def test_prefetch_requires_a_candidate_predicting_router_and_matching_names(two_layer_three_tier_moe) -> None:
    moe = two_layer_three_tier_moe
    with pytest.raises(ValueError, match="expert_names"):
        _run(moe, _staged_streamer(DOMAINS), expert_names=["only-one"])

    base_model, _, _ = build_local_qwen2_backbone(
        seed=1, vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model=base_model, rank_r=2, lora_alpha=4,
                                   gradient_checkpointing=False, seed=1)
    branches = initialize_domain_branches(extract_lora_state_dict(bundle.model), expert_domains=DOMAINS)
    flat_moe, _ = upcycle_branches_to_sparse_moe(
        bundle.model, {k: v.lora_state for k, v in branches.items()}, lora_rank_r=2, lora_alpha=4,
    )
    with pytest.raises(TypeError, match="predict_candidate_experts"):
        _run(flat_moe, _staged_streamer(DOMAINS))


def test_single_moe_layer_has_no_next_layer_to_prefetch() -> None:
    base_model, _, _ = build_local_qwen2_backbone(
        seed=3, vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model=base_model, rank_r=2, lora_alpha=4,
                                   gradient_checkpointing=False, seed=3)
    branches = initialize_domain_branches(extract_lora_state_dict(bundle.model), expert_domains=DOMAINS)
    moe, _ = upcycle_branches_to_sparse_moe(
        bundle.model, {k: v.lora_state for k, v in branches.items()}, lora_rank_r=2, lora_alpha=4,
    )
    moe.moe_blocks = moe.get_moe_blocks()  # type: ignore[attr-defined]
    moe.hidden_size = 16  # type: ignore[attr-defined]
    hotplug_three_tier_experts(moe, {}, canonical_rank_r=2, num_super_clusters=1, macro_per_super=1)
    res = _run(moe, _staged_streamer(DOMAINS))
    assert res.expert_prefetch_enabled is True
    assert res.expert_prefetch_calls == 0 and res.expert_prefetch_hit_rate is None
    assert "no next MoE layer" in res.expert_prefetch_basis
