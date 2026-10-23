"""Unit tests for `HM-S8-09`: inner-loop Muon Newton-Schulz LoRA preconditioning (`MuLoCo`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import BOINCWorkunit, create_signed_parquet_shard
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.core.diloco_inner import (
    _apply_muon_lora_preconditioning,
    replay_inner_segment,
    run_diloco_inner_loop,
)
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from hivemind.swarm.diloco_nextgen import MuonNewtonSchulzLoRAPreconditioner


@pytest.fixture(scope="module")
def tiny_qwen_bundle():
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=2,
        lora_alpha=4,
        gradient_checkpointing=False,
        seed=42,
    )
    return bundle, extract_lora_state_dict(bundle.model), base_sha


def _tokens() -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(2026)
    return torch.randint(0, 128, (8, 16), generator=gen, dtype=torch.long)


def _inner_kwargs(bundle, tokens: torch.Tensor, seed: int = 7) -> dict:
    return {
        "model": bundle.model,
        "token_sequences": tokens,
        "worker_id": "worker-muon",
        "branch_name": "Code",
        "outer_step": 1,
        "inner_steps_H": 3,
        "checkpoint_interval_s": 1,
        "lr": 1e-3,
        "rng_seed": seed,
        "micro_batch_size": 2,
    }


def test_muon_preconditioner_alias_fast_path_and_validation() -> None:
    torch.manual_seed(23)
    u = torch.eye(16, 8)
    singular_values = torch.tensor([50.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    v = torch.eye(8, 16)
    ill_conditioned = u @ torch.diag(singular_values) @ v

    preconditioner = MuonNewtonSchulzLoRAPreconditioner(ns_steps=5)
    assert preconditioner.steps == 5
    assert preconditioner.ns_steps == 5

    fast = preconditioner.precondition_gradient(ill_conditioned)
    slow, stats = preconditioner.orthogonalize_2d(ill_conditioned)
    assert torch.allclose(fast, slow, atol=1e-6)
    assert fast.shape == ill_conditioned.shape
    assert torch.isfinite(fast).all()
    assert stats["newton_schulz_steps"] == 5.0

    after = torch.linalg.svdvals(fast)
    active = after[after > 1e-3 * after.max().clamp_min(1e-8)]
    condition_after = float(active.max().item() / active.min().clamp_min(1e-8).item())
    assert condition_after < 3.0

    # Transposed (tall) matrices take the same fast path and keep their original shape.
    tall = torch.randn(24, 8)
    assert preconditioner.precondition_gradient(tall).shape == tall.shape

    with pytest.raises(ValueError):
        MuonNewtonSchulzLoRAPreconditioner(ns_steps=0)
    with pytest.raises(ValueError):
        preconditioner.precondition_gradient(torch.randn(8))


def test_apply_muon_only_touches_2d_lora_gradients() -> None:
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_matrix = nn.Parameter(torch.randn(8, 8))
            self.lora_vector = nn.Parameter(torch.randn(8))
            self.base_weight = nn.Parameter(torch.randn(8, 8))

    model = ToyModel()
    model.lora_matrix.grad = torch.randn(8, 8)
    model.lora_vector.grad = torch.randn(8)
    model.base_weight.grad = torch.randn(8, 8)
    original_matrix_grad = model.lora_matrix.grad.clone()
    original_vector_grad = model.lora_vector.grad.clone()
    original_base_grad = model.base_weight.grad.clone()

    preconditioner = MuonNewtonSchulzLoRAPreconditioner(steps=3)
    updated = _apply_muon_lora_preconditioning(model, preconditioner)

    assert updated == 1
    assert not torch.allclose(model.lora_matrix.grad, original_matrix_grad)
    assert torch.equal(model.lora_vector.grad, original_vector_grad)
    assert torch.equal(model.base_weight.grad, original_base_grad)

    # No gradients at all: the helper reports zero preconditioned tensors.
    assert _apply_muon_lora_preconditioning(ToyModel(), preconditioner) == 0


def test_muon_inner_loop_is_deterministic_and_telemetered(tiny_qwen_bundle) -> None:
    bundle, initial_lora, _ = tiny_qwen_bundle
    tokens = _tokens()
    kwargs = _inner_kwargs(bundle, tokens)

    load_lora_state_dict(bundle.model, initial_lora)
    first = run_diloco_inner_loop(
        **kwargs,
        use_muon_lora_preconditioner=True,
        muon_ns_steps=3,
    )
    assert first.steps_completed == 3
    assert first.muon_preconditioned_steps == 3
    assert first.use_muon_lora_preconditioner is True
    assert first.muon_ns_steps == 3
    assert torch.isfinite(torch.tensor(first.final_loss))

    load_lora_state_dict(bundle.model, initial_lora)
    replay_run = run_diloco_inner_loop(
        **kwargs,
        use_muon_lora_preconditioner=True,
        muon_ns_steps=3,
    )
    assert all(
        torch.equal(first.theta_final[name], replay_run.theta_final[name])
        for name in first.theta_final
    )

    load_lora_state_dict(bundle.model, initial_lora)
    plain = run_diloco_inner_loop(**kwargs)
    assert plain.muon_preconditioned_steps == 0
    assert plain.use_muon_lora_preconditioner is False
    assert plain.muon_ns_steps == 5
    max_difference = max(
        float((first.theta_final[name] - plain.theta_final[name]).abs().max().item())
        for name in first.theta_final
    )
    assert max_difference > 1e-6


def test_muon_training_replays_exactly_for_spot_checking(tiny_qwen_bundle) -> None:
    bundle, initial_lora, _ = tiny_qwen_bundle
    tokens = _tokens()
    kwargs = _inner_kwargs(bundle, tokens, seed=11)
    kwargs["inner_steps_H"] = 4
    kwargs["checkpoint_interval_s"] = 2

    load_lora_state_dict(bundle.model, initial_lora)
    submission = run_diloco_inner_loop(
        **kwargs,
        use_muon_lora_preconditioner=True,
        muon_ns_steps=3,
    )
    assert submission.muon_preconditioned_steps == 4

    replayed = replay_inner_segment(
        model=bundle.model,
        start_lora_state=submission.micro_checkpoints[0].state_snapshot,
        token_sequences=tokens,
        start_step=0,
        end_step=4,
        lr=kwargs["lr"],
        rng_seed=kwargs["rng_seed"],
        micro_batch_size=kwargs["micro_batch_size"],
        use_muon_lora_preconditioner=True,
        muon_ns_steps=3,
    )
    for name, expected in submission.micro_checkpoints[4].state_snapshot.items():
        assert torch.equal(replayed[name], expected)

    workunit = BOINCWorkunit(
        wu_id="wu-muon-t1",
        branch_name="Code",
        outer_step=1,
        base_model_id="base-sha",
        lora_checkpoint_url="local",
        data_shard_parquet="unused",
        shard_sha256="0" * 64,
        shard_hmac_signature="0" * 64,
        rng_seed=kwargs["rng_seed"],
        inner_steps_H=4,
        spot_check_interval_s=2,
        seq_len=tokens.shape[1],
        num_sequences=tokens.shape[0],
    )
    verdict = DeterministicSpotCheckValidator(tolerance_eps=1e-4).verify_submission(
        validator_model=bundle.model,
        workunit=workunit,
        submission=submission,
        token_sequences=tokens,
        steam_id64="76561198000000001",
        lr=kwargs["lr"],
        micro_batch_size=kwargs["micro_batch_size"],
    )
    assert verdict.passed is True
    assert verdict.hash_matched is True
    assert verdict.max_abs_divergence <= 1e-4

    with pytest.raises(ValueError, match="muon_ns_steps"):
        replay_inner_segment(
            model=bundle.model,
            start_lora_state=submission.micro_checkpoints[0].state_snapshot,
            token_sequences=tokens,
            start_step=0,
            end_step=1,
            muon_ns_steps=0,
        )
    with pytest.raises(ValueError, match="muon_ns_steps"):
        run_diloco_inner_loop(**kwargs, muon_ns_steps=0)


def test_worker_forwards_muon_flags_to_the_inner_loop(
    tiny_qwen_bundle, tmp_path: Path
) -> None:
    bundle, initial_lora, base_sha = tiny_qwen_bundle
    tokens = _tokens()
    shard_path, shard_sha256, shard_signature = create_signed_parquet_shard(
        shard_path=tmp_path / "muon.parquet",
        branch_name="Code",
        rng_seed=11,
        num_sequences=4,
        seq_len=8,
        vocab_size=128,
    )
    workunit = BOINCWorkunit(
        wu_id="wu-muon-worker",
        branch_name="Code",
        outer_step=1,
        base_model_id=base_sha,
        lora_checkpoint_url="local",
        data_shard_parquet=str(shard_path),
        shard_sha256=shard_sha256,
        shard_hmac_signature=shard_signature,
        rng_seed=11,
        inner_steps_H=2,
        spot_check_interval_s=1,
        seq_len=8,
        num_sequences=4,
    )
    worker = HiveMindWorkerNode(
        worker_id="worker-muon",
        steam_id64="76561198000005601",
        persona_name="worker-muon",
        model=bundle.model,
    )

    load_lora_state_dict(bundle.model, initial_lora)
    workunit_result = worker.execute_workunit(
        workunit,
        initial_lora,
        topk_ratio=0.10,
        use_muon_lora_preconditioner=True,
        muon_ns_steps=2,
    )
    assert workunit_result.muon_preconditioned_steps == 2
    assert workunit_result.use_muon_lora_preconditioner is True
    assert workunit_result.muon_ns_steps == 2

    load_lora_state_dict(bundle.model, initial_lora)
    cas_result = worker.execute_cas_shard(
        domain_name="Code",
        token_sequences=tokens,
        global_lora_state=initial_lora,
        outer_step=2,
        inner_steps_H=2,
        spot_check_interval_s=1,
        rng_seed=12,
        micro_batch_size=2,
        topk_ratio=0.10,
        use_muon_lora_preconditioner=True,
        muon_ns_steps=3,
    )
    assert cas_result.muon_preconditioned_steps == 2
    assert cas_result.muon_ns_steps == 3
