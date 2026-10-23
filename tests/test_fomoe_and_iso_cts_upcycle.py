"""Season 8 (`HM-S8-03` / `HM-S8-06`) upcycling and FoMoE routing tests."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from hivemind.core.btm_upcycle import (
    LoopedExpertBackbone,
    SparseMoEMLPBlock,
    UpcycledSwiGLUExpert,
    initialize_domain_branches,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)


class _CountingExpert(nn.Module):
    """Wraps an expert and records how many token rows reached its FFN."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.forward_rows = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_rows += int(x.shape[0])
        return self.inner(x)


class _FixedRouter(nn.Module):
    """Deterministic stand-in for `TopKMoERouter` in FoMoE routing tests."""

    def __init__(self, selected: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        self.selected = selected
        self.weights = weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = int(hidden_states.shape[0])
        selected = self.selected[:num_tokens].to(hidden_states.device)
        weights = self.weights[:num_tokens].to(device=hidden_states.device, dtype=hidden_states.dtype)
        return weights, selected, torch.tensor(0.0, device=hidden_states.device)


def _make_block(seed: int = 7) -> tuple[SparseMoEMLPBlock, list[_CountingExpert]]:
    generator = torch.Generator().manual_seed(seed)
    experts: list[_CountingExpert] = []
    for _ in range(4):
        inner = UpcycledSwiGLUExpert(
            hidden_size=4,
            intermediate_size=8,
            gate_weight=torch.randn(8, 4, generator=generator),
            up_weight=torch.randn(8, 4, generator=generator),
            down_weight=torch.randn(4, 8, generator=generator),
        )
        experts.append(_CountingExpert(inner))
    block = SparseMoEMLPBlock(
        hidden_size=4,
        experts=list(experts),
        expert_domains=["A", "B", "C", "D"],
        top_k=2,
    )
    selected = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=torch.float32)
    block.router = _FixedRouter(selected, weights)
    return block, experts


def _build_tiny_branches() -> tuple[nn.Module, dict[str, dict[str, torch.Tensor]]]:
    base_model, _, _ = build_local_qwen2_backbone(
        seed=11,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=4,
        lora_alpha=8,
        gradient_checkpointing=False,
        seed=11,
    )
    initial = extract_lora_state_dict(bundle.model)
    branches = initialize_domain_branches(initial, expert_domains=["Code", "Math"])

    branch_states: dict[str, dict[str, torch.Tensor]] = {}
    for branch_idx, (domain, branch) in enumerate(branches.items()):
        specialized: dict[str, torch.Tensor] = {}
        for key, value in branch.lora_state.items():
            if "lora_B" in key:
                ramp = torch.linspace(
                    -1.0, 1.0, value.numel(), dtype=value.dtype
                ).reshape(value.shape)
                specialized[key] = value + (branch_idx + 1) * 0.05 * ramp
            else:
                specialized[key] = value.clone()
        branch_states[domain] = specialized
    return bundle.model, branch_states


def _q_proj_weight(moe_model: nn.Module, layer_idx: int = 0) -> torch.Tensor:
    return moe_model.backbone.model.layers[layer_idx].self_attn.q_proj.weight.detach().clone()


def test_attention_merge_modes_and_reputation_weighting() -> None:
    base_model, branch_states = _build_tiny_branches()
    common = dict(
        peft_or_base_model=base_model,
        branch_lora_states=branch_states,
        lora_rank_r=4,
        lora_alpha=8,
        top_k=2,
        target_device=torch.device("cpu"),
        target_dtype=torch.float32,
    )

    moe_uniform, report_uniform = upcycle_branches_to_sparse_moe(
        **common, attention_merge_mode="uniform"
    )
    moe_iso, report_iso = upcycle_branches_to_sparse_moe(
        **common,
        attention_merge_mode="iso_cts",
        branch_reputations={"Code": 1.0, "Math": 3.0},
    )
    moe_tsv, report_tsv = upcycle_branches_to_sparse_moe(
        **common,
        attention_merge_mode="tsv",
        branch_reputations={"Code": 1.0, "Math": 3.0},
    )

    assert report_uniform.attention_merge_mode == "uniform"
    assert report_iso.attention_merge_mode == "iso_cts"
    assert report_tsv.attention_merge_mode == "tsv"
    assert report_uniform.attention_layers_merged == 8
    assert report_iso.attention_layers_merged == 8
    assert report_tsv.attention_layers_merged == 8

    q_uniform = _q_proj_weight(moe_uniform)
    q_iso = _q_proj_weight(moe_iso)
    q_tsv = _q_proj_weight(moe_tsv)
    assert torch.isfinite(q_iso).all()
    assert torch.isfinite(q_tsv).all()
    assert not torch.allclose(q_uniform, q_iso, atol=1e-7)
    assert not torch.allclose(q_uniform, q_tsv, atol=1e-7)
    assert not torch.allclose(q_iso, q_tsv, atol=1e-7)

    # Reputations must actually steer the ISO-CTS/TSV merges.
    moe_iso_reversed, _ = upcycle_branches_to_sparse_moe(
        **common,
        attention_merge_mode="iso_cts",
        branch_reputations={"Code": 3.0, "Math": 1.0},
    )
    assert not torch.allclose(q_iso, _q_proj_weight(moe_iso_reversed), atol=1e-7)

    # Model-level FoMoE residency propagates to every block and drops ghost tokens.
    moe_iso.set_local_expert_subset([0])
    assert moe_iso.local_expert_indices == (0,)
    assert all(blk.local_expert_indices == (0,) for blk in moe_iso.get_moe_blocks())
    moe_iso.reset_routing_telemetry()
    with torch.no_grad():
        _ = moe_iso(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert moe_iso.ghost_skipped_assignments > 0
    assert moe_iso.ghost_skipped_assignments == sum(
        blk.ghost_skipped_assignments for blk in moe_iso.get_moe_blocks()
    )
    moe_iso.reset_routing_telemetry()
    assert moe_iso.ghost_skipped_assignments == 0
    moe_iso.set_local_expert_subset(None)
    assert all(blk.local_expert_indices is None for blk in moe_iso.get_moe_blocks())


def test_attention_merge_validation() -> None:
    base_model, branch_states = _build_tiny_branches()
    common = dict(
        peft_or_base_model=base_model,
        branch_lora_states=branch_states,
        lora_rank_r=4,
        lora_alpha=8,
    )
    with pytest.raises(ValueError, match="attention_merge_mode"):
        upcycle_branches_to_sparse_moe(**common, attention_merge_mode="bogus")
    with pytest.raises(ValueError, match="unknown domains"):
        upcycle_branches_to_sparse_moe(
            **common,
            attention_merge_mode="iso_cts",
            branch_reputations={"Nope": 1.0},
        )
    with pytest.raises(ValueError, match="non-negative"):
        upcycle_branches_to_sparse_moe(
            **common,
            attention_merge_mode="tsv",
            branch_reputations={"Code": -1.0},
        )
    with pytest.raises(ValueError, match="finite"):
        upcycle_branches_to_sparse_moe(
            **common,
            attention_merge_mode="iso_cts",
            branch_reputations={"Code": math.nan},
        )
    with pytest.raises(ValueError, match="positive weight"):
        upcycle_branches_to_sparse_moe(
            **common,
            attention_merge_mode="iso_cts",
            branch_reputations={"Code": 0.0, "Math": 0.0},
        )


def test_fomoe_skip_token_executes_only_local_experts() -> None:
    block, experts = _make_block()
    block.set_local_expert_subset([0, 2])
    x = torch.tensor(
        [
            [1.0, 0.0, -1.0, 2.0],
            [0.5, 0.5, 0.0, 1.0],
            [-1.0, 2.0, 1.0, 0.0],
        ]
    )

    out = block(x)

    # Three of the six top-2 assignments target ghost experts 1 or 3.
    assert block.ghost_skipped_assignments == 3
    assert experts[1].forward_rows == 0
    assert experts[3].forward_rows == 0
    assert experts[0].forward_rows == 1
    assert experts[2].forward_rows == 2

    weights = block.router.weights  # type: ignore[attr-defined]
    expected = torch.stack(
        [
            weights[0, 0] * experts[0].inner(x[0]),
            weights[1, 0] * experts[2].inner(x[1]),
            weights[2, 1] * experts[2].inner(x[2]),
        ]
    )
    assert torch.allclose(out, expected, atol=1e-6)

    # Restoring full replication resumes ghost-expert activation without resetting telemetry.
    for expert in experts:
        expert.forward_rows = 0
    block.set_local_expert_subset(None)
    _ = block(x)
    assert block.ghost_skipped_assignments == 3
    assert experts[1].forward_rows == 2
    assert experts[3].forward_rows == 1


def test_fomoe_local_expert_subset_validation() -> None:
    block, _ = _make_block()
    with pytest.raises(ValueError, match="at least one"):
        block.set_local_expert_subset([])
    with pytest.raises(ValueError, match="within"):
        block.set_local_expert_subset([4])
    with pytest.raises(ValueError, match="within"):
        block.set_local_expert_subset([-1])
    with pytest.raises(ValueError, match="sequence of integers"):
        block.set_local_expert_subset(["bad"])  # type: ignore[list-item]


@pytest.mark.parametrize("attention_merge_mode", ["iso_cts", "tsv"])
@pytest.mark.parametrize("fixed_point_tol,executed", [(0.0, 4), (1e9, 1)])
def test_looped_fomoe_replay_preserves_logical_ghost_counts(
    attention_merge_mode: str, fixed_point_tol: float, executed: int,
) -> None:
    base_model, branch_states = _build_tiny_branches()
    moe, _ = upcycle_branches_to_sparse_moe(
        peft_or_base_model=base_model,
        branch_lora_states=branch_states,
        lora_rank_r=4,
        lora_alpha=8,
        top_k=2,
        target_device=torch.device("cpu"),
        target_dtype=torch.float32,
        attention_merge_mode=attention_merge_mode,
        branch_reputations={"Code": 1.0, "Math": 3.0},
    )
    moe.eval()
    moe.set_local_expert_subset([0])
    looped = LoopedExpertBackbone(
        moe, num_prelude_layers=0, num_coda_layers=0, loop_steps=4,
        per_step_lora_rank=2, fixed_point_tol=fixed_point_tol, truncated_bptt_steps=2,
    )
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        expected = looped(ids, labels=ids)
    expected_aux = looped.last_aux_loss.detach().clone()
    blocks = looped.get_moe_blocks()
    expected_counts = [block.expert_token_counts.clone() for block in blocks]
    # Each of two layers routes five tokens to both experts, one of them a ghost.
    assert moe.ghost_skipped_assignments == 10 * executed
    assert looped.last_mean_effective_steps == executed
    looped.reset_routing_telemetry()
    assert moe.ghost_skipped_assignments == 0

    actual = looped(ids, labels=ids)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(looped.last_aux_loss, expected_aux, rtol=0, atol=0)
    assert looped.last_mean_effective_steps == executed
    assert [block.ghost_skipped_assignments for block in blocks] == [5 * executed] * 2
    assert moe.ghost_skipped_assignments == 10 * executed
    for block, counts in zip(blocks, expected_counts):
        torch.testing.assert_close(block.expert_token_counts, counts, rtol=0, atol=0)
    actual.loss.backward()
    gradients = [adapter.up.weight.grad for adapter in looped.step_adapters]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)

    looped(ids, labels=ids)
    assert [block.ghost_skipped_assignments for block in blocks] == [10 * executed] * 2
    assert moe.ghost_skipped_assignments == 20 * executed
    looped.reset_routing_telemetry()
    assert moe.ghost_skipped_assignments == 0
