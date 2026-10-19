"""NF4 buffers must exist, decode independently and never masquerade as runtime VRAM savings."""
import copy

import pytest
import torch

from hivemind.core.quantization import (
    attach_qlora_adapters, build_local_qwen2_backbone,
    dequantize_tensor_nf4, quantize_tensor_nf4_double_quant,
)


@pytest.mark.parametrize("shape,block_size", [((7, 9), 64), ((3, 5), 3), ((2, 8), 4), ((0, 4), 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_real_packed_state_roundtrips_without_mutating_input(shape, block_size, dtype):
    weight = torch.randn(shape, dtype=dtype)
    before = weight.clone()
    state, reconstructed = quantize_tensor_nf4_double_quant(weight, block_size)
    assert torch.equal(weight, before)
    assert torch.equal(reconstructed, dequantize_tensor_nf4(state, dtype=dtype))
    num_blocks = (weight.numel() + block_size - 1) // block_size
    assert state.packed_4bit.numel() == (num_blocks * block_size + 1) // 2
    assert state.q_absmax_uint8.numel() == num_blocks
    assert state.storage_bytes == state.packed_4bit.nbytes + state.q_absmax_uint8.nbytes + 8
    assert state.packed_4bit.device.type == "cpu"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nf4_weights_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        quantize_tensor_nf4_double_quant(torch.tensor([[value]]))


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_invalid_nf4_block_sizes_are_rejected(block_size):
    with pytest.raises(ValueError, match="block_size"):
        quantize_tensor_nf4_double_quant(torch.ones(2, 2), block_size)


def test_truncated_nf4_state_is_rejected():
    state, _ = quantize_tensor_nf4_double_quant(torch.ones(7, 9))
    broken = copy.deepcopy(state)
    broken.packed_4bit = torch.empty(0, dtype=torch.uint8)
    with pytest.raises(ValueError, match="packed buffers"):
        dequantize_tensor_nf4(broken)


def test_dense_reconstruction_does_not_claim_resident_4bit_weights():
    model, _, is_nf4 = build_local_qwen2_backbone(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, device=torch.device("cpu"), use_4bit_nf4_if_cuda=True,
    )
    assert not is_nf4
    assert model._nf4_storage_mode == "dense_reconstruction"
    bundle = attach_qlora_adapters(model, rank_r=2, lora_alpha=4, gradient_checkpointing=False)
    assert not bundle.is_4bit_nf4
    assert bundle.nf4_storage_mode == "dense_reconstruction"
    assert bundle.nf4_compressed_base_mb > 0


def test_adapter_accounting_counts_actual_parameter_dtype():
    model, _, _ = build_local_qwen2_backbone(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, device=torch.device("cpu"), use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(model, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    actual_bytes = sum(p.numel() * p.element_size() for p in bundle.model.parameters() if p.requires_grad)
    assert bundle.lora_size_mb == round(actual_bytes / 1024**2, 3)
