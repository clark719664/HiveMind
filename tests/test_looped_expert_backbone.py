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


class _ControlledLayer(nn.Module):
    def __init__(self, mode="half", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.calls = []
        self.inputs = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(torch.is_grad_enabled())
        self.inputs.append(hidden_states.detach().clone())
        if self.mode == "identity":
            return (hidden_states,)
        if self.mode == "context":
            return (torch.stack((hidden_states[:, 0], hidden_states[:, 1] + hidden_states[:, 0]), dim=1),)
        return (self.dropout(hidden_states * self.scale),)


def _controlled(*, steps=4, injection="none", mode="half", rank=0, dropout=0.0, **kwargs):
    source = nn.Module()
    source.model = nn.Module()
    source.model.embed_tokens = nn.Embedding(4, 2)
    with torch.no_grad():
        source.model.embed_tokens.weight.copy_(torch.tensor([[2., 0.], [0., 2.], [0., 0.], [1., 1.]]))
    core = _ControlledLayer(mode, dropout)
    source.model.layers = nn.ModuleList([core])
    source.model.norm = nn.Identity()
    source.lm_head = nn.Identity()
    return LoopedExpertBackbone(
        source, num_prelude_layers=0, num_coda_layers=0, loop_steps=steps,
        per_step_lora_rank=rank, input_injection=injection, **kwargs,
    ), core


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


@pytest.mark.parametrize("tol", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_fixed_point_tolerances(tol: float) -> None:
    with pytest.raises(ValueError, match="fixed_point_tol"):
        _controlled(fixed_point_tol=tol)


@pytest.mark.parametrize("k", [0, -1, 5, 1.5, True, False, "2"])
def test_invalid_truncation_windows(k) -> None:
    with pytest.raises(ValueError, match="truncated_bptt_steps"):
        _controlled(truncated_bptt_steps=k)


def test_telemetry_is_fresh_and_padding_is_excluded() -> None:
    looped, core = _controlled()
    fields = ("last_mean_effective_steps", "last_early_exit_fraction", "last_fixed_point_residual")
    assert all(getattr(looped, field) is None for field in fields)
    assert all(getattr(looped.report(), field) is None for field in fields)
    ids = torch.tensor([[0, 2]])
    for mask, expected in [(torch.tensor([[1, 0]]), (4.0, 0.0, 0.5)),
                           (torch.zeros_like(ids), (0.0, 0.0, 0.0)),
                           (torch.ones_like(ids), (4.0, 0.0, 0.25))]:
        looped(ids, attention_mask=mask)
        assert tuple(getattr(looped, field) for field in fields) == expected
        assert tuple(getattr(looped.report(), field) for field in fields) == expected
    assert len(core.calls) == 12


def test_halted_token_is_frozen_but_remains_attention_context() -> None:
    looped, core = _controlled(mode="context", fixed_point_tol=0.0)
    out = looped(torch.tensor([[0, 1]]))
    # First token is fixed at [2,0]; second accumulates that context on all four passes.
    torch.testing.assert_close(out.logits, torch.tensor([[[2., 0.], [8., 2.]]]))
    assert len(core.calls) == 4
    assert looped.last_mean_effective_steps == 2.5
    assert looped.last_early_exit_fraction == 0.5
    assert looped.last_fixed_point_residual == pytest.approx(1 / (2 * 10 ** 0.5))


def test_frozen_context_is_not_reinjected_or_adapted() -> None:
    looped, core = _controlled(mode="context", injection="add", rank=1, fixed_point_tol=0.0)
    with torch.no_grad():
        for adapter in looped.step_adapters[1:]:
            adapter.up.weight.fill_(1.)
            adapter.down.weight.fill_(1.)
    looped(torch.tensor([[0, 1]]))
    for state in core.inputs:
        torch.testing.assert_close(state[:, 0], torch.tensor([[2., 0.]]))


@pytest.mark.parametrize("mask", [None, torch.tensor([[1, 0]]), torch.tensor([[0, 0]])])
def test_whole_batch_exit_and_empty_padding(mask) -> None:
    looped, core = _controlled(mode="identity", fixed_point_tol=0.)
    looped(torch.tensor([[0, 1]]), attention_mask=mask)
    empty = mask is not None and not mask.any()
    assert len(core.calls) == (0 if empty else 1)
    assert looped.last_mean_effective_steps == (0. if empty else 1.)
    assert looped.last_early_exit_fraction == (0. if empty else 1.)
    assert looped.last_fixed_point_residual == 0.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relative_drift_uses_pre_injection_state_and_fp32_zero_norm_floor(dtype) -> None:
    looped, core = _controlled(injection="add", fixed_point_tol=0.3)
    looped.to(dtype=dtype)
    with torch.no_grad():
        core.scale.fill_(0.25)
    looped(torch.tensor([[0]]))
    assert len(core.calls) == 2
    assert looped.last_fixed_point_residual == 0.25
    # A zero recurrent state acquiring a unit vector has drift 1 / 1e-6.
    looped, _ = _controlled(mode="context", fixed_point_tol=0.)
    looped.to(dtype=dtype)
    looped(torch.tensor([[0, 2]]))
    assert looped.last_fixed_point_residual == pytest.approx(1 / 6)
    looped, _ = _controlled(steps=1, mode="context")
    looped.to(dtype=dtype)
    looped(torch.tensor([[0, 2]]))
    assert looped.last_fixed_point_residual == 1_000_000.


@pytest.mark.parametrize("k", [None, 4])
def test_default_and_full_window_match_legacy_recurrence_and_gradients(k) -> None:
    looped, core = _controlled(injection="add", truncated_bptt_steps=k)
    out = looped(torch.tensor([[0]]))
    torch.testing.assert_close(out.logits, torch.tensor([[[1.875, 0.]]]), rtol=0, atol=0)
    out.logits.sum().backward()
    # h4 = x*(s+s^2+s^3+s^4), derivative at s=.5 is x*3.25.
    assert core.scale.grad.item() == 6.5
    torch.testing.assert_close(looped.embed_tokens.weight.grad[0], torch.full((2,), 0.9375))
    assert core.calls == [True] * 4


def test_truncation_tracks_only_suffix_and_detaches_injected_prelude() -> None:
    looped, core = _controlled(rank=1, injection="add", truncated_bptt_steps=2)
    out = looped(torch.tensor([[0]]))
    out.logits.sum().backward()
    assert core.calls == [False, False, True, True]
    assert looped.embed_tokens.weight.grad is None
    assert [a.up.weight.grad is not None for a in looped.step_adapters] == [False, False, True, True]
    assert core.scale.grad.item() == 5.5
    torch.testing.assert_close(out.logits, torch.tensor([[[1.875, 0.]]]))


@pytest.mark.parametrize("tol, executed, k", [(1., 1, 2), (0.3, 2, 1), (0.3, 2, 3), (0., 4, 2)])
def test_combined_adaptation_replays_last_executed_steps(tol, executed, k) -> None:
    looped, core = _controlled(rank=1, injection="add", fixed_point_tol=tol, truncated_bptt_steps=k)
    with torch.no_grad():
        core.scale.fill_(0.25)
    out = looped(torch.tensor([[0]]))
    out.logits.sum().backward()
    assert looped.last_mean_effective_steps == executed
    assert core.calls == [False] * executed + [True] * min(k, executed)
    assert core.scale.grad.abs().item() > 0
    assert looped.embed_tokens.weight.grad is None
    assert [a.up.weight.grad is not None for a in looped.step_adapters] == [
        max(0, executed - k) <= i < executed for i in range(4)
    ]


def test_combined_inference_never_replays_or_enables_gradients() -> None:
    looped, core = _controlled(fixed_point_tol=1., truncated_bptt_steps=2)
    with torch.no_grad():
        looped(torch.tensor([[0]]))
    assert core.calls == [False]


def test_stochastic_replay_preserves_outputs_rng_and_suffix_gradient() -> None:
    looped, core = _controlled(steps=4, dropout=0.25, fixed_point_tol=0., truncated_bptt_steps=2)
    reference, ref_core = _controlled(steps=4, dropout=0.25, truncated_bptt_steps=2)
    ids = torch.tensor([[0, 1, 3]])
    torch.manual_seed(918)
    expected = reference(ids).logits
    expected.sum().backward()
    rng = torch.get_rng_state()
    torch.manual_seed(918)
    actual = looped(ids).logits
    actual.sum().backward()
    assert core.calls == [False] * 4 + [True] * 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(core.scale.grad, ref_core.scale.grad, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("k", [None, 3])
def test_full_window_qwen_has_exact_default_outputs_and_gradients(k) -> None:
    models = [_backbone(layers=3), _backbone(layers=3)]
    for model in models:
        for param in model.parameters():
            param.requires_grad_(True)
    baseline = LoopedExpertBackbone(models[0], loop_steps=3, per_step_lora_rank=0)
    candidate = LoopedExpertBackbone(models[1], loop_steps=3, per_step_lora_rank=0,
                                    fixed_point_tol=None, truncated_bptt_steps=k)
    ids = _tokens(batch=1, seq=5)
    a, b = baseline(ids, labels=ids), candidate(ids, labels=ids)
    torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    a.loss.backward()
    b.loss.backward()
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_real_moe_combined_backward_counts_logical_routes_and_aux_once() -> None:
    looped = LoopedExpertBackbone(_upcycled_moe(), loop_steps=4, per_step_lora_rank=2,
                                 fixed_point_tol=1e9, truncated_bptt_steps=2)
    reference = LoopedExpertBackbone(_upcycled_moe(), loop_steps=1, per_step_lora_rank=0)
    ids = _tokens(batch=1, seq=6)
    looped.reset_routing_telemetry()
    actual, expected = looped(ids, labels=ids), reference(ids, labels=ids)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(looped.last_aux_loss, reference.last_aux_loss)
    assert looped.last_aux_loss.requires_grad
    assert [int(b.expert_token_counts.sum()) for b in looped.get_moe_blocks()] == [12, 12, 12]
    actual.loss.backward()
    assert looped.step_adapters[0].up.weight.grad.abs().sum() > 0
    assert all(a.up.weight.grad is None for a in looped.step_adapters[1:])
    assert looped.get_moe_blocks()[1].router.wg.weight.grad.abs().sum() > 0


def test_multi_pass_moe_replay_matches_suffix_aux_values_and_gradients() -> None:
    reference = LoopedExpertBackbone(_upcycled_moe(), loop_steps=4, per_step_lora_rank=2,
                                    truncated_bptt_steps=2)
    replayed = LoopedExpertBackbone(_upcycled_moe(), loop_steps=4, per_step_lora_rank=2,
                                   fixed_point_tol=0., truncated_bptt_steps=2)
    replayed.load_state_dict(reference.state_dict())
    ids = _tokens(batch=1, seq=5)
    a, b = reference(ids, labels=ids), replayed(ids, labels=ids)
    assert replayed.last_mean_effective_steps == 4.
    torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    torch.testing.assert_close(reference.last_aux_loss, replayed.last_aux_loss, rtol=0, atol=0)
    a.loss.backward()
    b.loss.backward()
    for expected, actual in zip(reference.parameters(), replayed.parameters()):
        torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)
    assert [int(b.expert_token_counts.sum()) for b in replayed.get_moe_blocks()] == [10, 40, 10]


def test_padding_cannot_keep_a_batch_alive_or_leave_stale_adaptive_telemetry() -> None:
    looped, core = _controlled(mode="context", fixed_point_tol=0.)
    ids = torch.tensor([[0, 1]])
    looped(ids, attention_mask=torch.tensor([[1, 0]]))
    assert len(core.calls) == 1
    assert looped.last_mean_effective_steps == looped.last_early_exit_fraction == 1.
    assert looped.last_fixed_point_residual == 0.
    looped(ids, attention_mask=torch.zeros_like(ids))
    assert len(core.calls) == 1
    assert looped.last_mean_effective_steps == looped.last_early_exit_fraction == 0.
    looped(ids)
    assert len(core.calls) == 5
    assert looped.last_mean_effective_steps == 2.5
