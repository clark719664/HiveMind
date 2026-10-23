"""Small CPU checks for runtime validation and observable guard state."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from hivemind.data import token_training_eval as tokens
from hivemind.swarm.ledger import HiveSwarmLedger
from hivemind.swarm.scientific_swarm_eval import ties_merge_lora_adapters
from hivemind.swarm.strategy_optimizer import (
    EF21DiLoCoErrorFeedbackBuffer,
    StrategyOptimizerTelemetryEngine,
)


@pytest.mark.parametrize("sample_id,text", [(17, "valid text"), ("sample", None), ("sample", ["text"])])
def test_split_validation_rejects_non_string_examples_without_mutation(sample_id, text):
    fixture = {"train": [{"id": sample_id, "text": text}],
               "holdout": [{"id": "holdout", "text": "Independent holdout"}]}
    before = copy.deepcopy(fixture)
    with pytest.raises(ValueError, match="sample IDs and text must be strings"):
        tokens.validate_splits(fixture)
    assert fixture == before


def test_pinned_fixture_rejects_tampered_bytes_at_the_fixture_boundary(tmp_path, monkeypatch):
    path = tmp_path / "fixed-fixture.json"
    raw = b'{"train": [], "holdout": []}'
    path.write_bytes(raw)
    monkeypatch.setattr(tokens, "FIXTURE_PATH", path)
    with pytest.raises(ValueError, match="pinned smoke fixture digest does not match"):
        tokens.load_fixture(path)
    assert path.read_bytes() == raw


def test_extreme_but_finite_token_loss_reports_perplexity_overflow():
    class BadPredictor(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            assert not torch.is_grad_enabled()
            assert input_ids.shape == attention_mask.shape == (1, 2)
            return SimpleNamespace(logits=torch.tensor([[[0.0, -1000.0], [0.0, 0.0]]]))

    model = BadPredictor()
    batch = {"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones((1, 2), dtype=torch.long),
             "labels": torch.tensor([[-100, 1]])}
    before = {key: value.clone() for key, value in batch.items()}
    result = tokens.evaluate_tokens(model, batch)
    assert result == {
        "status": "measured", "metric": "next_token_cross_entropy_nats", "total_nll_nats": 1000.0,
        "scored_tokens": 1, "mean_nll_nats": 1000.0, "perplexity": None, "perplexity_status": "overflow",
    }
    assert model.training is False
    assert torch.is_grad_enabled()
    assert all(torch.equal(batch[key], value) for key, value in before.items())


def test_token_cli_help_exposes_required_output_and_reproducibility_options(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["token-eval", "--help"])
    with pytest.raises(SystemExit) as error:
        tokens.main()
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert all(option in text for option in ("--output-dir", "--steps", "--seed"))


def test_token_cli_rejects_negative_steps_before_creating_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "new-run"
    monkeypatch.setattr(sys, "argv", ["token-eval", "--output-dir", str(output), "--steps", "-1", "--seed", "7"])
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        tokens.main()
    assert not output.exists()
    assert capsys.readouterr().out == ""


def _delta():
    return {"adapter": torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)}


def test_pulse_compression_carries_decayed_residual_across_skipped_steps():
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.25, stale_decay=0.5)
    first = _delta()
    first_before = first["adapter"].clone()
    reconstruction, initial_stats = buffer.compress_with_pulse_sparseloco_2bit(first, outer_step=1)
    carried = buffer.residuals["adapter"].clone()
    assert torch.linalg.vector_norm(carried).item() > 0
    torch.testing.assert_close(reconstruction["adapter"] + carried, first_before)

    second = {"adapter": torch.full((8, 8), 0.25)}
    corrected = second["adapter"] + 0.25 * carried
    reconstruction, stats = buffer.compress_with_pulse_sparseloco_2bit(second, outer_step=3)
    torch.testing.assert_close(reconstruction["adapter"] + buffer.residuals["adapter"], corrected)
    assert torch.equal(first["adapter"], first_before)
    assert torch.equal(second["adapter"], torch.full((8, 8), 0.25))
    assert buffer.last_outer_step == 3
    assert buffer.clip_events == 0
    assert stats["outer_step"] == 3
    assert stats["conservation_error_l2"] == pytest.approx(0.0, abs=1e-6)
    assert initial_stats["raw_bf16_bytes"] == stats["raw_bf16_bytes"] == 128


@pytest.mark.parametrize("changed_key,step,error_type,message", [
    (True, 2, KeyError, "keys differ"),
    (False, 1, ValueError, "outer_step must increase"),
    (False, 0, ValueError, "outer_step must increase"),
])
def test_pulse_rejected_upload_does_not_mutate_buffer(changed_key, step, error_type, message):
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.25)
    buffer.compress_with_pulse_sparseloco_2bit(_delta(), outer_step=1)
    before = {key: value.clone() for key, value in buffer.residuals.items()}
    upload = {"other-adapter": _delta()["adapter"]} if changed_key else _delta()
    with pytest.raises(error_type, match=message):
        buffer.compress_with_pulse_sparseloco_2bit(upload, outer_step=step)
    assert buffer.last_outer_step == 1
    assert buffer.clip_events == 0
    assert set(buffer.residuals) == set(before)
    assert all(torch.equal(buffer.residuals[key], value) for key, value in before.items())


def test_pulse_residual_norm_cap_and_reset_are_observable_state_transitions():
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.125, max_residual_norm=0.5)
    _, stats = buffer.compress_with_pulse_sparseloco_2bit(_delta(), outer_step=1)
    assert buffer.clip_events == 1
    assert buffer.residual_norm() == pytest.approx(0.5, abs=1e-6)
    assert stats["residual_l2_norm"] == pytest.approx(0.5, abs=1e-6)
    buffer.reset()
    assert buffer.residuals == {}
    assert buffer.last_outer_step is None
    assert buffer.clip_events == 0
    assert buffer.residual_norm() == 0.0
    replacement = {"replacement-adapter": torch.ones((2, 2))}
    reconstructed, _ = buffer.compress_with_pulse_sparseloco_2bit(replacement, outer_step=1)
    assert set(reconstructed) == set(buffer.residuals) == {"replacement-adapter"}


def test_telemetry_constructor_copies_custom_link_profile_into_public_snapshot():
    profile = {"ring_rtt_ms": 55.0, "bbr_pacing_mbps": 20.0, "packet_loss_rate": 0.03, "turn_relay_active": True}
    expected = dict(profile)
    engine = StrategyOptimizerTelemetryEngine(link_profile=profile)
    profile["ring_rtt_ms"] = 999.0
    snapshot = engine.snapshot()
    assert all(snapshot["live_inputs"][key] == value for key, value in expected.items())
    assert engine.link_profile["ring_rtt_ms"] == 55.0
    assert engine.revision == 0


@pytest.mark.parametrize("height,expected", [(-1, 0.0), (0, 50.0), (9999, 50.0), (10000, 25.0), (640000, 0.0)])
def test_block_reward_halving_and_exhaustion_do_not_mutate_balances(height, expected):
    ledger = HiveSwarmLedger()
    before = dict(ledger.balances)
    assert ledger.block_reward_at_height(height) == expected
    assert ledger.balances == before
    assert len(ledger.chain) == 1


def test_block_reward_respects_remaining_supply_using_public_credits():
    ledger = HiveSwarmLedger()
    ledger.credit_reward("consumer", ledger.MAX_SUPPLY_HIVE - 0.01)
    assert ledger.block_reward_at_height(1) == 0.01
    ledger.credit_reward("consumer", 0.01)
    assert ledger.total_circulating_supply() == ledger.MAX_SUPPLY_HIVE
    assert ledger.block_reward_at_height(1) == 0.0


def test_insufficient_inference_balance_leaves_memory_and_persisted_ledger_unchanged(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = HiveSwarmLedger(ledger_path=path)
    ledger.credit_reward("consumer", 0.1)
    persisted = path.read_bytes()
    balances = dict(ledger.balances)
    with pytest.raises(ValueError, match="insufficient"):
        ledger.settle_swarm_inference("consumer", ["provider"], "test-domain", verified_tokens=40, rtt_speedup=None)
    assert ledger.balances == balances
    assert ledger.settlement_history == []
    assert ledger.get_balance("provider") == 0.0
    assert path.read_bytes() == persisted
    assert ledger.verify_chain_integrity() is True


@pytest.mark.parametrize("corruption", ["empty", "payload", "parent-link"])
def test_chain_integrity_rejects_deletion_or_tampering_without_repairing(corruption):
    ledger = HiveSwarmLedger()
    block = ledger.mine_pouw_block(outer_step=1, expert_cid="cid://fixture-expert", work_receipts=[],
                                  storage_receipts=[], validator_node_ids=["validator"])
    assert ledger.verify_chain_integrity() is True
    if corruption == "empty":
        ledger.chain.clear()
    elif corruption == "payload":
        block.expert_cid = "cid://tampered"
    else:
        block.prev_block_hash = "f" * 64
        block.block_hash = block.compute_hash()
    before = [asdict(item) for item in ledger.chain]
    balances = dict(ledger.balances)
    assert ledger.verify_chain_integrity() is False
    assert [asdict(item) for item in ledger.chain] == before
    assert ledger.balances == balances


def test_ties_without_peers_returns_independent_exact_base_copy():
    base = {"adapter": torch.tensor([[1.0, 2.0]], dtype=torch.float16)}
    merged = ties_merge_lora_adapters(base, [])
    assert merged["adapter"].dtype == base["adapter"].dtype
    assert torch.equal(merged["adapter"], base["adapter"])
    merged["adapter"].zero_()
    assert torch.equal(base["adapter"], torch.tensor([[1.0, 2.0]], dtype=torch.float16))


def test_ties_sparse_peer_state_preserves_unshared_adapter_without_aliasing():
    base = {"shared": torch.tensor([1.0, 2.0]), "unshared": torch.tensor([3.0, 4.0], dtype=torch.float16)}
    peer = {"shared": torch.tensor([3.0, 4.0])}
    before = {key: value.clone() for key, value in base.items()}
    merged = ties_merge_lora_adapters(base, [peer], trim_density=1.0, scaling_lambda=0.5)
    torch.testing.assert_close(merged["shared"], torch.tensor([2.0, 3.0]))
    assert merged["unshared"].dtype == torch.float16
    assert torch.equal(merged["unshared"], base["unshared"])
    assert merged["unshared"].data_ptr() != base["unshared"].data_ptr()
    assert all(torch.equal(base[key], value) for key, value in before.items())
    assert torch.equal(peer["shared"], torch.tensor([3.0, 4.0]))
