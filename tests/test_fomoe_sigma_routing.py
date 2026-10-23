"""Tests for FoMoE sigma-MoE routing with ghost-expert Skip-Token accounting (`arXiv:2606.19025`)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import torch

from hivemind.core.fomoe_sigma_routing import (
    sigmoid_independent_topk_routing,
    skip_token_flops_per_token,
)


def test_sigma_routing_is_independent_sigmoid_topk_and_skips_ghosts() -> None:
    logits = torch.tensor(
        [
            [4.0, 3.0, -5.0, -5.0],  # top-2 = experts 0, 1
            [-5.0, 2.0, 3.0, -5.0],  # top-2 = experts 2, 1
        ]
    )
    resident = torch.tensor([True, True, False, False])
    out = sigmoid_independent_topk_routing(logits, top_k=2, resident_mask=resident)
    assert out.expert_indices.tolist() == [[0, 1], [2, 1]]
    # Gates are raw sigmoids (no softmax renormalisation across the selected experts).
    assert out.gate_values[0].tolist() == pytest.approx(torch.sigmoid(torch.tensor([4.0, 3.0])).tolist(), rel=1e-6)
    # Expert 2 is a ghost for this worker: the slot is skipped and its gate zeroed.
    assert out.skipped_slots.tolist() == [[False, False], [True, False]]
    assert float(out.gate_values[1, 0]) == 0.0
    assert out.ghost_assignment_fraction == pytest.approx(1 / 4)
    assert out.ideal_skip_token_scaling == pytest.approx(0.5)
    assert out.empirical_skip_token_scaling == pytest.approx(0.75)


def test_sigma_routing_full_replica_has_no_ghosts_and_validates_inputs() -> None:
    logits = torch.randn(16, 8)
    out = sigmoid_independent_topk_routing(logits, top_k=2)
    assert out.ghost_assignment_fraction == 0.0
    assert out.empirical_skip_token_scaling == 1.0
    assert bool((out.gate_values > 0).all())
    with pytest.raises(ValueError):
        sigmoid_independent_topk_routing(logits, top_k=9)
    with pytest.raises(ValueError):
        sigmoid_independent_topk_routing(logits, top_k=2, resident_mask=torch.zeros(8, dtype=torch.bool))
    with pytest.raises(ValueError):
        sigmoid_independent_topk_routing(logits, top_k=2, resident_mask=torch.ones(8))


@settings(max_examples=60, deadline=None)
@given(
    tokens=st.integers(min_value=1, max_value=64),
    experts=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=5000),
)
def test_sigma_routing_ghost_fraction_is_exact_bookkeeping(tokens: int, experts: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=g)
    k = min(2, experts)
    resident = torch.zeros(experts, dtype=torch.bool)
    resident[: max(1, experts // 2)] = True
    out = sigmoid_independent_topk_routing(logits, top_k=k, resident_mask=resident)
    ghosts = int((~resident[out.expert_indices]).sum())
    assert out.ghost_assignment_fraction == pytest.approx(ghosts / (tokens * k))
    assert out.empirical_skip_token_scaling == pytest.approx(1.0 - ghosts / (tokens * k))
    assert bool((out.gate_values[out.skipped_slots] == 0).all())
    assert bool((out.gate_values[~out.skipped_slots] > 0).all())
    assert 0.0 <= out.ghost_assignment_fraction <= 1.0


def test_skip_token_flops_matches_fomoe_eq3_ideal_and_empirical() -> None:
    # Two layers, 128 experts, 32 resident per worker (O_e/M = 1/4), top-2, unpartitioned router.
    res = skip_token_flops_per_token(
        dense_flops=1000.0,
        ffn_flops_per_layer=[800.0, 800.0],
        gate_flops_per_layer=[16.0, 16.0],
        top_k=2,
        local_experts_per_layer=[32, 32],
        total_experts_per_layer=[128, 128],
    )
    # Eq. 3: F = F_dense + sum_l [ (32/128) * min(2,32)/2 * 800 + 1 * 16 ] = 1000 + 2*(200 + 16)
    assert res["partial_replica_flops_per_token"] == pytest.approx(1000.0 + 2 * (200.0 + 16.0))
    assert res["full_replica_flops_per_token"] == pytest.approx(1000.0 + 2 * (800.0 + 16.0))
    assert res["scaling_source"] == "ideal_N_le_over_N_e"

    # A measured imbalance (fewer ghost hits than ideal) must raise the applied scaling.
    emp = skip_token_flops_per_token(
        dense_flops=1000.0,
        ffn_flops_per_layer=[800.0, 800.0],
        gate_flops_per_layer=[16.0, 16.0],
        top_k=2,
        local_experts_per_layer=[32, 32],
        total_experts_per_layer=[128, 128],
        empirical_ghost_fraction_per_layer=[0.5, 0.5],
    )
    assert emp["partial_replica_flops_per_token"] == pytest.approx(1000.0 + 2 * (0.5 * 800.0 + 16.0))
    assert emp["partial_replica_flops_per_token"] > res["partial_replica_flops_per_token"]
    assert emp["per_layer"][0]["applied_skip_token_scaling"] == pytest.approx(0.5)

    # Partitioned router scales the gate by N_le / N_e as in the paper's non-ghost variant.
    part = skip_token_flops_per_token(
        dense_flops=0.0,
        ffn_flops_per_layer=[0.0],
        gate_flops_per_layer=[64.0],
        top_k=2,
        local_experts_per_layer=[16],
        total_experts_per_layer=[64],
        router_partitioned=True,
    )
    assert part["partial_replica_flops_per_token"] == pytest.approx(16.0)


def test_skip_token_flops_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError):
        skip_token_flops_per_token(1.0, [1.0], [1.0, 1.0], 2, [1], [2])
    with pytest.raises(ValueError):
        skip_token_flops_per_token(1.0, [1.0], [1.0], 2, [3], [2])
    with pytest.raises(ValueError):
        skip_token_flops_per_token(1.0, [1.0], [1.0], 2, [1], [2], empirical_ghost_fraction_per_layer=[1.5])


def test_sparse_moe_mlp_block_sigma_moe_routing_and_measured_ghost_fraction() -> None:
    """HM-S9-02: Verify SparseMoEMLPBlock and UpcycledSparseMoEModel support sigma_moe routing and feed skip_token_flops_per_token."""
    from hivemind.core.btm_upcycle import SparseMoEMLPBlock, UpcycledSparseMoEModel, UpcycledSwiGLUExpert
    import torch.nn as nn

    g = torch.Generator().manual_seed(42)

    def _build_block() -> SparseMoEMLPBlock:
        experts = [
            UpcycledSwiGLUExpert(
                hidden_size=8,
                intermediate_size=16,
                gate_weight=torch.randn(16, 8, generator=g),
                up_weight=torch.randn(16, 8, generator=g),
                down_weight=torch.randn(8, 16, generator=g),
            )
            for _ in range(4)
        ]
        return SparseMoEMLPBlock(
            hidden_size=8,
            experts=experts,
            expert_domains=["A", "B", "C", "D"],
            top_k=2,
            routing_mode="sigma_moe",
        )

    backbone = nn.Sequential(_build_block(), _build_block())
    moe = UpcycledSparseMoEModel(upcycled_backbone=backbone, expert_domains=["A", "B", "C", "D"], top_k=2)
    moe.set_routing_mode("sigma_moe")
    moe.set_local_expert_subset([0, 1])

    x = torch.randn(2, 6, 8, generator=g)
    out = backbone(x)
    assert out.shape == (2, 6, 8)

    measured_ghosts = moe.measured_ghost_fraction_per_layer()
    assert len(measured_ghosts) == 2
    for g_frac, blk in zip(measured_ghosts, moe.get_moe_blocks()):
        assert 0.0 <= g_frac <= 1.0
        assert blk.total_routed_assignments == 2 * 6 * 2
        assert g_frac == pytest.approx(blk.ghost_skipped_assignments / blk.total_routed_assignments)

    flops = skip_token_flops_per_token(
        dense_flops=500.0,
        ffn_flops_per_layer=[400.0, 400.0],
        gate_flops_per_layer=[8.0, 8.0],
        top_k=2,
        local_experts_per_layer=[2, 2],
        total_experts_per_layer=[4, 4],
        empirical_ghost_fraction_per_layer=measured_ghosts,
    )
    assert flops["scaling_source"] == "empirical_ghost_fraction"
    assert flops["per_layer"][0]["applied_skip_token_scaling"] == pytest.approx(1.0 - measured_ghosts[0])

