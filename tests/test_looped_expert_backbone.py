"""Season 7 (`HM-S7-01`): looped `Prelude -> RecurrentCore x T -> Coda` backbone."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hivemind.core.btm_upcycle import (
    DEFAULT_EXPERT_DOMAINS,
    LoopedExpertBackbone,
    initialize_domain_branches,
    train_moe_router_gating,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)

VOCAB = 128
HIDDEN = 32


def _backbone(layers: int = 4, seed: int = 7) -> nn.Module:
    model, _, _ = build_local_qwen2_backbone(
        seed=seed,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    return model.eval()


def _tokens(batch: int = 2, seq: int = 12, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(seed))


def _upcycled_moe(layers: int = 3, domains: int = 4):
    bundle = attach_qlora_adapters(
        base_model=_backbone(layers, seed=3), rank_r=4, lora_alpha=8, gradient_checkpointing=False, seed=3
    )
    branches = initialize_domain_branches(
        extract_lora_state_dict(bundle.model), expert_domains=DEFAULT_EXPERT_DOMAINS[:domains]
    )
    states = {
        dom: {k: v + (i + 1) * 0.01 if "lora_B" in k else v.clone() for k, v in branch.lora_state.items()}
        for i, (dom, branch) in enumerate(branches.items())
    }
    moe_model, _ = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=states,
        lora_rank_r=4,
        lora_alpha=8,
        top_k=2,
        target_device=torch.device("cpu"),
        target_dtype=torch.float32,
    )
    return moe_model


def _logits(model: nn.Module, ids: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        return LoopedExpertBackbone(model, **kwargs)(ids).logits


def test_a_single_plain_pass_reproduces_the_source_model() -> None:
    model = _backbone()
    ids = _tokens()
    looped = LoopedExpertBackbone(model, loop_steps=1, per_step_lora_rank=0, input_injection="none")
    with torch.no_grad():
        torch.testing.assert_close(looped(ids).logits, model(input_ids=ids).logits)
        torch.testing.assert_close(looped(ids, labels=ids).loss, model(input_ids=ids, labels=ids).loss)


def test_padding_mask_matches_the_source_model_on_real_tokens() -> None:
    model = _backbone()
    ids = _tokens()
    attention_mask = torch.ones_like(ids)
    attention_mask[1, 8:] = 0
    looped = LoopedExpertBackbone(model, loop_steps=1, per_step_lora_rank=0, input_injection="none")
    with torch.no_grad():
        ours = looped(ids, attention_mask=attention_mask).logits
        theirs = model(input_ids=ids, attention_mask=attention_mask).logits
        real = attention_mask.bool()
        torch.testing.assert_close(ours[real], theirs[real])
        unpadded = looped(ids[1:, :8]).logits
        torch.testing.assert_close(ours[1, :8], unpadded[0])


@pytest.mark.parametrize("steps", [2, 3, 4])
def test_core_layers_are_reused_not_copied(steps: int) -> None:
    model = _backbone(layers=4)
    looped = LoopedExpertBackbone(model, loop_steps=steps, per_step_lora_rank=0)
    report = looped.report()

    assert all(a is b for a, b in zip(looped.core, model.model.layers[1:3]))
    assert (report.num_prelude_layers, report.num_core_layers, report.num_coda_layers) == (1, 2, 1)
    assert report.effective_depth == 1 + 2 * steps + 1
    assert report.depth_multiplier == round(report.effective_depth / 4, 4)
    assert report.loop_adapter_bytes == 0
    assert report.extra_weight_bytes == 0
    assert report.saved_weight_bytes == report.core_weight_bytes * (steps - 1) > 0


def test_step_adapters_start_as_identity_and_are_the_only_extra_weights() -> None:
    model = _backbone()
    ids = _tokens()
    plain = _logits(model, ids, loop_steps=3, per_step_lora_rank=0)
    adapted = LoopedExpertBackbone(model, loop_steps=3, per_step_lora_rank=4)
    with torch.no_grad():
        torch.testing.assert_close(adapted(ids).logits, plain)

    report = adapted.report()
    assert len(adapted.step_adapters) == 3
    assert report.loop_adapter_bytes == 3 * 2 * HIDDEN * 4 * 4  # steps * (A, B) * hidden * rank * fp32
    assert report.extra_weight_bytes == report.loop_adapter_bytes


def test_loop_depth_and_input_injection_change_the_computation() -> None:
    model = _backbone()
    ids = _tokens()
    once = _logits(model, ids, loop_steps=1, per_step_lora_rank=0, input_injection="none")
    twice = _logits(model, ids, loop_steps=2, per_step_lora_rank=0, input_injection="none")
    injected = _logits(model, ids, loop_steps=2, per_step_lora_rank=0, input_injection="add")

    assert not torch.allclose(once, twice)
    assert not torch.allclose(twice, injected)
    torch.testing.assert_close(_logits(model, ids, loop_steps=1, per_step_lora_rank=0, input_injection="add"), once)


def test_backward_reaches_every_step_adapter_and_the_tied_core() -> None:
    model = _backbone().train()
    for param in model.parameters():
        param.requires_grad_(True)
    looped = LoopedExpertBackbone(model, loop_steps=2, per_step_lora_rank=4)
    ids = _tokens()

    out = looped(ids, labels=ids)
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert all(adapter.up.weight.grad.abs().sum() > 0 for adapter in looped.step_adapters)
    assert looped.core[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_looping_an_upcycled_moe_routes_every_core_pass() -> None:
    moe_model = _upcycled_moe()
    looped = LoopedExpertBackbone(moe_model, loop_steps=3, per_step_lora_rank=2)
    ids = _tokens(batch=1, seq=8)

    looped.reset_routing_telemetry()
    with torch.no_grad():
        out = looped(ids, labels=ids)
        no_aux = LoopedExpertBackbone(moe_model, loop_steps=3, per_step_lora_rank=0, router_aux_loss_coef=0.0)

    counts = [int(blk.expert_token_counts.sum()) for blk in looped.get_moe_blocks()]
    assert counts == [8 * 2, 3 * 8 * 2, 8 * 2]  # prelude, core x 3 passes, coda; top-2 slots per token
    assert looped.router_aux_loss_coef == moe_model.router_aux_loss_coef
    assert float(looped.last_aux_loss) > 0.0
    with torch.no_grad():
        assert float(out.loss) > float(no_aux(ids, labels=ids).loss)


def test_router_gating_training_runs_through_the_loop() -> None:
    moe_model = _upcycled_moe()
    looped = LoopedExpertBackbone(moe_model, loop_steps=2, per_step_lora_rank=0)
    batches = {dom: _tokens(batch=2, seq=8, seed=i) for i, dom in enumerate(moe_model.expert_domains)}

    initial, final, utilization = train_moe_router_gating(looped, batches, steps=8, lr=1e-2)

    assert initial > 0.0 and final > 0.0
    assert set(utilization) == set(moe_model.expert_domains)
    assert abs(sum(utilization.values()) - 1.0) < 1e-2


def test_a_plain_causal_lm_has_no_routing_telemetry() -> None:
    looped = LoopedExpertBackbone(_backbone())
    looped.reset_routing_telemetry()
    assert looped.get_moe_blocks() == []
    assert looped.get_expert_utilization() == {}
    assert looped.router_aux_loss_coef == 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"loop_steps": 0}, "loop_steps"),
        ({"num_prelude_layers": -1}, "non-negative"),
        ({"num_prelude_layers": 2, "num_coda_layers": 2}, "no core"),
        ({"input_injection": "concat"}, "input_injection"),
        ({"per_step_lora_rank": -1}, "per_step_lora_rank"),
    ],
)
def test_invalid_loop_configurations_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LoopedExpertBackbone(_backbone(layers=4), **kwargs)


def test_models_without_a_decoder_stack_are_rejected() -> None:
    with pytest.raises(ValueError, match=".model.layers"):
        LoopedExpertBackbone(nn.Linear(4, 4))
