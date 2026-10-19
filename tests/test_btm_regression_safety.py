"""Regressions for bare CausalLM upcycling and independent dense hot-plug experts."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from hivemind.config import rocm_supports_native_bf16
from hivemind.core.btm_upcycle import upcycle_branches_to_sparse_moe


def _base():
    torch.manual_seed(19)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1,
    ))


@pytest.mark.parametrize("in_place", [False, True])
def test_bare_causal_lm_keeps_its_decoder_and_language_head(in_place):
    model, report = upcycle_branches_to_sparse_moe(
        _base(), {"first": {}, "second": {}}, lora_rank_r=2, lora_alpha=2,
        in_place=in_place,
    )
    assert report.num_experts == 2
    with torch.no_grad():
        assert model(torch.ones(1, 4, dtype=torch.long)).logits.shape == (1, 4, 32)


def test_dense_hotplug_does_not_inherit_first_expert_adapter():
    base = _base()
    original = {
        projection: getattr(base.model.layers[0].mlp, f"{projection}_proj").weight.detach().clone()
        for projection in ("gate", "up", "down")
    }
    first = {}
    for projection, weight in original.items():
        first[f"model.layers.0.mlp.{projection}_proj.lora_A.weight"] = torch.full((2, weight.shape[1]), 0.1)
        first[f"model.layers.0.mlp.{projection}_proj.lora_B.weight"] = torch.full((weight.shape[0], 2), 0.1)
    zero = {key: torch.zeros_like(value) for key, value in first.items()}
    model, _ = upcycle_branches_to_sparse_moe(
        base, {"first": first, "second": zero}, lora_rank_r=2, lora_alpha=2,
    )
    model.hotplug_domain_expert("third", zero, lora_rank_r=2, lora_alpha=2)
    model.hotplug_domain_expert("fourth", first, lora_rank_r=2, lora_alpha=2)
    block = model.get_moe_blocks()[0]
    for projection, weight in original.items():
        torch.testing.assert_close(getattr(block.experts[2], f"{projection}_proj").weight, weight)
        torch.testing.assert_close(
            getattr(block.experts[3], f"{projection}_proj").weight,
            getattr(block.experts[0], f"{projection}_proj").weight,
        )
    reference_buffers = {name: tensor for name, tensor in block.named_buffers() if name.startswith("reference_")}
    assert len(reference_buffers) == 6
    assert sum(value.numel() for value in reference_buffers.values()) < sum(value.numel() for value in original.values())
    assert "backbone.model.layers.0.mlp.reference_gate_A" in model.state_dict()


@pytest.mark.parametrize("architecture", ["gfx908", "gfx90a", "gfx942", "gfx950", "gfx90a:sramecc+:xnack-"])
def test_cdna_matrix_targets_recognize_bf16(architecture):
    assert rocm_supports_native_bf16(architecture)


@pytest.mark.parametrize("architecture", [None, "gfx900", "gfx906", "gfx1030", "unknown"])
def test_unsupported_or_unknown_targets_do_not_claim_bf16(architecture):
    assert not rocm_supports_native_bf16(architecture)
