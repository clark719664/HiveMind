"""Adversarial tests for checkpoint replay and the uploaded compressed payload."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import BOINCWorkunit
from hivemind.core.compression import compress_lora_delta, compress_tensor_topk_fp8, decompress_tensor_topk_fp8
from hivemind.core.diloco_inner import replay_inner_segment, run_diloco_inner_loop
from hivemind.core.quantization import compute_state_dict_sha256


class TinyTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)

    def forward(self, input_ids, labels):
        target = input_ids.float().mean() / 100
        return SimpleNamespace(loss=((self.lora_weight - target) ** 2).mean())


@pytest.fixture
def training_case():
    model = TinyTrainableModel()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    wu = BOINCWorkunit("wu-test", "test", 1, "base", "local", "local", "hash", "sig", 7,
                       inner_steps_H=4, spot_check_interval_s=2)
    result = run_diloco_inner_loop(model, tokens, "worker", "test", inner_steps_H=4,
                                  checkpoint_interval_s=2, rng_seed=7, lr=1e-3)
    return model, tokens, wu, result


def verify(case, result=None, **kwargs):
    model, tokens, wu, honest = case
    return DeterministicSpotCheckValidator().verify_submission(
        model, wu, result or honest, tokens, "steam-test", lr=1e-3, **kwargs,
    )


def refresh_payload(result):
    result.raw_delta = {name: value - result.theta_final[name] for name, value in result.theta_init.items()}
    result.compressed_delta = compress_lora_delta(result.raw_delta, result.worker_id, result.branch_name,
                                                 result.outer_step, result.steps_completed)


def test_honest_replay_reaches_final_checkpoint_and_reports_actual_cost(training_case):
    verdict = verify(training_case)
    assert verdict.passed and verdict.hash_matched
    assert verdict.segment_end_step == 4
    assert verdict.verification_overhead_pct == 100


def test_self_consistent_forged_tail_is_rejected(training_case):
    result = copy.deepcopy(training_case[3])
    result.theta_final["lora_weight"] += 0.1
    result.micro_checkpoints[4].lora_sha256 = compute_state_dict_sha256(result.theta_final)
    refresh_payload(result)
    verdict = verify(training_case, result)
    assert not verdict.passed
    assert verdict.max_abs_divergence > 0.01


def test_forged_middle_checkpoint_with_valid_hash_is_rejected(training_case):
    result = copy.deepcopy(training_case[3])
    result.micro_checkpoints[2].state_snapshot["lora_weight"] += 0.1
    result.micro_checkpoints[2].lora_sha256 = compute_state_dict_sha256(result.micro_checkpoints[2].state_snapshot)
    assert not verify(training_case, result).passed


@pytest.mark.parametrize("mutation", ["payload", "metadata", "nan", "missing_zero", "missing_final", "shape", "hash"])
def test_malformed_or_unbound_submissions_fail_closed(training_case, mutation):
    result = copy.deepcopy(training_case[3])
    if mutation == "payload":
        result.compressed_delta.tensors["lora_weight"].scale *= 2
    elif mutation == "metadata":
        result.compressed_delta.worker_id = "other-worker"
    elif mutation == "nan":
        result.raw_delta["lora_weight"][0, 0] = float("nan")
    elif mutation == "missing_zero":
        del result.micro_checkpoints[0]
    elif mutation == "missing_final":
        del result.micro_checkpoints[4]
    elif mutation == "shape":
        result.theta_final["lora_weight"] = torch.zeros(100)
    else:
        result.micro_checkpoints[2].lora_sha256 = "not-a-hash"
    verdict = verify(training_case, result)
    assert not verdict.passed and verdict.steam_account_flagged


def test_trusted_initial_checkpoint_cannot_be_selected_by_worker(training_case):
    trusted = {"lora_weight": training_case[3].theta_init["lora_weight"] + 1}
    verdict = verify(training_case, trusted_initial_state=trusted)
    assert not verdict.passed
    assert "coordinator checkpoint" in verdict.reason


def test_preemption_commits_nonperiodic_final_state(training_case):
    model, tokens, wu, _ = training_case
    calls = 0
    def preempt():
        nonlocal calls
        calls += 1
        return calls == 4
    result = run_diloco_inner_loop(model, tokens, "worker", "test", inner_steps_H=4,
                                  checkpoint_interval_s=2, rng_seed=7, lr=1e-3, should_preempt_fn=preempt)
    assert result.steps_completed == 3
    assert 3 in result.micro_checkpoints
    assert verify(training_case, result).passed


def test_nonzero_segment_without_optimizer_history_is_rejected(training_case):
    model, tokens, _, result = training_case
    with pytest.raises(ValueError, match="initial state"):
        replay_inner_segment(model, result.theta_init, tokens, 2, 4)


def test_payload_accounting_uses_actual_buffers():
    values = torch.arange(100, dtype=torch.float32)
    package = compress_lora_delta({"lora_weight": values}, "worker", "test", 1)
    payload = package.tensors["lora_weight"]
    assert payload.compressed_bytes == payload.indices.nbytes + payload.fp8_values.nbytes + 8
    assert package.uncompressed_bytes == values.nbytes


@pytest.mark.parametrize("mutation", ["duplicates", "bounds", "shape", "nan_scale", "negative_scale", "values"])
def test_decompress_rejects_malformed_payloads(mutation):
    payload = compress_tensor_topk_fp8(torch.arange(10, dtype=torch.float32), topk_ratio=0.5)
    if mutation == "duplicates":
        payload.indices[0] = payload.indices[1]
    elif mutation == "bounds":
        payload.indices[0] = 10
    elif mutation == "shape":
        payload.original_shape = (11,)
    elif mutation == "nan_scale":
        payload.scale = float("nan")
    elif mutation == "negative_scale":
        payload.scale = -1
    else:
        payload.fp8_values = torch.zeros(2, dtype=payload.fp8_values.dtype)
    with pytest.raises(ValueError):
        decompress_tensor_topk_fp8(payload)


def test_coordinator_rejects_payload_before_queuing_or_rewarding(tmp_path):
    from hivemind.boinc.coordinator_api import BOINCCoordinatorService
    from hivemind.config import HiveMindConfig
    from hivemind.boinc.workunit import verify_and_load_parquet_shard
    model = TinyTrainableModel()
    service = BOINCCoordinatorService(HiveMindConfig.from_yaml(), {"lora_weight": model.lora_weight.detach().clone()},
                                     "base", shards_dir=tmp_path, vocab_size=64)
    wu, _ = service.assign_workunit("worker", "steam", inner_steps_H=4, spot_check_interval_s=2,
                                    num_sequences=2, seq_len=8, rng_seed=7)
    tokens = verify_and_load_parquet_shard(wu.data_shard_parquet, wu.shard_sha256, wu.shard_hmac_signature, wu.rng_seed)
    result = run_diloco_inner_loop(model, tokens, "worker", wu.branch_name, inner_steps_H=4,
                                  checkpoint_interval_s=2, rng_seed=7, lr=1e-3)
    result.compressed_delta.tensors["lora_weight"].scale *= 2
    verdict = service.validate_and_record_submission(model, wu, result, "steam", lr=1e-3)
    assert not verdict.passed
    assert service.pending_branch_deltas[wu.branch_name] == {}
    assert service.reputation_ledger["steam"].minted_api_credits == 0


@pytest.fixture
def coordinator_case(tmp_path):
    from hivemind.boinc.coordinator_api import BOINCCoordinatorService
    from hivemind.config import HiveMindConfig
    from hivemind.boinc.workunit import verify_and_load_parquet_shard
    model = TinyTrainableModel()
    service = BOINCCoordinatorService(HiveMindConfig.from_yaml(), {"lora_weight": model.lora_weight.detach().clone()},
                                     "base", shards_dir=tmp_path, vocab_size=64)
    wu, initial = service.assign_workunit("worker", "steam", inner_steps_H=4, spot_check_interval_s=2,
                                        num_sequences=2, seq_len=8, rng_seed=7)
    tokens = verify_and_load_parquet_shard(wu.data_shard_parquet, wu.shard_sha256, wu.shard_hmac_signature, wu.rng_seed)
    result = run_diloco_inner_loop(model, tokens, "worker", wu.branch_name, inner_steps_H=4,
                                  checkpoint_interval_s=2, rng_seed=7, lr=1e-3)
    return service, model, wu, initial, result


def test_coordinator_consumes_successful_assignment_only_once(coordinator_case):
    service, model, wu, _, result = coordinator_case
    assert service.validate_and_record_submission(model, wu, result, "steam", lr=1e-3).passed
    assert not service.validate_and_record_submission(model, wu, result, "steam", lr=1e-3).passed
    assert service.reputation_ledger["steam"].validated_workunits == 1
    assert service.reputation_ledger["steam"].minted_api_credits == 10
    assert wu.wu_id not in service.active_workunits


@pytest.mark.parametrize("mutation", ["unknown", "worker", "account", "metadata", "initial", "stale"])
def test_coordinator_rejects_unassigned_or_mutated_work(coordinator_case, mutation):
    service, model, wu, initial, result = coordinator_case
    steam = "steam"
    if mutation == "unknown":
        wu.wu_id = "not-assigned"
    elif mutation == "worker":
        result.worker_id = "other-worker"
    elif mutation == "account":
        steam = "other-steam"
    elif mutation == "metadata":
        wu.data_shard_parquet = "/must-not-be-opened"
    elif mutation == "initial":
        initial["lora_weight"] += 1
        result.theta_init["lora_weight"] += 1
    else:
        service.branches[wu.branch_name].outer_step += 1
    verdict = service.validate_and_record_submission(model, wu, result, steam, lr=1e-3)
    assert not verdict.passed
    assert service.reputation_ledger["steam"].minted_api_credits == 0
    assert service.pending_branch_deltas[wu.branch_name] == {}
    assert "other-steam" not in service.reputation_ledger


def test_coordinator_restores_assignment_after_server_failure(coordinator_case, monkeypatch):
    service, model, wu, _, result = coordinator_case
    real_verify = service.validator.verify_submission
    def fail_replay(*args, **kwargs):
        raise RuntimeError("simulated server outage")
    monkeypatch.setattr(service.validator, "verify_submission", fail_replay)
    with pytest.raises(RuntimeError, match="server outage"):
        service.validate_and_record_submission(model, wu, result, "steam", lr=1e-3)
    assert wu.wu_id in service.active_workunits
    assert service.reputation_ledger["steam"].minted_api_credits == 0
    monkeypatch.setattr(service.validator, "verify_submission", real_verify)
    assert service.validate_and_record_submission(model, wu, result, "steam", lr=1e-3).passed


def test_validator_does_not_classify_server_failure_as_fraud(training_case, monkeypatch):
    import hivemind.boinc.validator as validator_module
    def fail_replay(*args, **kwargs):
        raise RuntimeError("simulated device failure")
    monkeypatch.setattr(validator_module, "replay_inner_segment", fail_replay)
    with pytest.raises(RuntimeError, match="device failure"):
        verify(training_case)
