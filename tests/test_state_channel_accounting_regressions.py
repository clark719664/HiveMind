"""Local simulated ledger regressions; no external financial services are used."""
import pytest

from hivemind.swarm.ledger import HiveSwarmLedger
from hivemind.swarm.swarm_fabric import HiveLayer2StateChannel


def ledger_and_channel(**kwargs):
    ledger = HiveSwarmLedger()
    ledger.balances["consumer"] = 10
    return ledger, HiveLayer2StateChannel("consumer", ["provider"], **kwargs)


def test_repeated_settlement_is_idempotent_and_later_vouchers_pay_only_increment():
    ledger, channel = ledger_and_channel()
    channel.sign_micro_voucher(20)
    first = channel.settle_to_l1_ledger(ledger)
    assert first["settled_tokens"] == 20 and first["settled_hive"] == 0.1
    count = len(ledger.settlement_history)
    assert channel.settle_to_l1_ledger(ledger)["settled_hive"] == 0
    assert len(ledger.settlement_history) == count
    channel.sign_micro_voucher(25)
    second = channel.settle_to_l1_ledger(ledger)
    assert second["settled_tokens"] == 25 and second["settled_hive"] == 0.125
    assert ledger.balances["consumer"] == 9.775
    assert ledger.balances["provider"] == 0.225


def test_failed_settlement_does_not_advance_watermark():
    ledger, channel = ledger_and_channel()
    ledger.balances["consumer"] = 0
    channel.sign_micro_voucher(20)
    with pytest.raises(ValueError, match="insufficient"):
        channel.settle_to_l1_ledger(ledger)
    assert channel.settled_tokens == 0
    ledger.balances["consumer"] = 1
    assert channel.settle_to_l1_ledger(ledger)["settled_tokens"] == 20


def test_refund_cancels_only_unsettled_liability_and_closes_channel():
    ledger, channel = ledger_and_channel()
    channel.sign_micro_voucher(20)
    channel.settle_to_l1_ledger(ledger)
    channel.sign_micro_voucher(10)
    assert channel.refund_consumer_escrow() == 0.05
    assert channel.refund_consumer_escrow() == 0
    assert ledger.balances["consumer"] == 9.9
    assert channel.settle_to_l1_ledger(ledger)["settled_hive"] == 0
    with pytest.raises(ValueError, match="closed"):
        channel.sign_micro_voucher(1)


@pytest.mark.parametrize("tokens", [-1, 0, 1.5, True])
def test_invalid_token_counts_do_not_change_channel(tokens):
    _, channel = ledger_and_channel()
    with pytest.raises(ValueError, match="positive integer"):
        channel.sign_micro_voucher(tokens)
    assert channel.nonce == 0 and channel.cumulative_hive_spent == 0


@pytest.mark.parametrize("parameter,value", [("escrow_hive", -1), ("escrow_hive", float("nan")),
                                             ("rate_per_token", -1), ("rate_per_token", float("inf"))])
def test_invalid_funding_parameters_are_rejected(parameter, value):
    with pytest.raises(ValueError):
        ledger_and_channel(**{parameter: value})


def test_rounding_remains_cumulative_across_settlements():
    ledger, channel = ledger_and_channel(rate_per_token=0.0000006)
    for _ in range(5):
        channel.sign_micro_voucher(1)
        channel.settle_to_l1_ledger(ledger)
    assert channel.settled_hive == 0.000003
    assert ledger.balances["provider"] == 0.000003


def test_settlement_persistence_failure_rolls_back_before_retry(monkeypatch):
    ledger, channel = ledger_and_channel()
    channel.sign_micro_voucher(20)
    before = ledger.balances.copy()
    def fail_persist():
        raise OSError("simulated disk failure")
    monkeypatch.setattr(ledger, "_persist", fail_persist)
    with pytest.raises(OSError, match="disk failure"):
        channel.settle_to_l1_ledger(ledger)
    assert ledger.balances == before
    assert ledger.settlement_history == []
    assert channel.settled_tokens == 0
    monkeypatch.setattr(ledger, "_persist", lambda: None)
    channel.settle_to_l1_ledger(ledger)
    assert ledger.balances["consumer"] == 9.9
    assert len(ledger.settlement_history) == 1


def test_atomic_persistence_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    import hivemind.swarm.ledger as ledger_module
    path = tmp_path / "ledger.json"
    ledger = HiveSwarmLedger(ledger_path=path)
    ledger.balances["consumer"] = 10
    ledger._persist()
    before = path.read_bytes()
    def fail_replace(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(ledger_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        ledger.settle_swarm_inference("consumer", ["provider"], "test", 1, 1)
    assert path.read_bytes() == before
    assert ledger.balances["consumer"] == 10
    assert "provider" not in ledger.balances
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("overrides", [
    {"verified_tokens": -1}, {"verified_tokens": 0}, {"verified_tokens": True},
    {"rate_per_token": -1}, {"rate_per_token": float("nan")},
    {"provider_node_ids": []}, {"provider_node_ids": ["same", "same"]},
    {"provider_node_ids": [""]}, {"rtt_speedup": float("inf")},
])
def test_direct_ledger_rejects_invalid_settlement_without_mutation(overrides):
    ledger, _ = ledger_and_channel()
    before = ledger.balances.copy()
    arguments = dict(consumer_node_id="consumer", provider_node_ids=["provider"],
                     domain_expert="test", verified_tokens=1, rtt_speedup=1)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        ledger.settle_swarm_inference(**arguments)
    assert ledger.balances == before
    assert ledger.settlement_history == []


def test_fractional_provider_split_conserves_debit_exactly():
    ledger, _ = ledger_and_channel()
    providers = ["one", "two", "three"]
    receipt = ledger.settle_swarm_inference("consumer", providers, "test", 1, 1, rate_per_token=0.000002)
    assert round(sum(ledger.balances[pid] for pid in providers), 6) == receipt.fee_paid_hive
    assert ledger.balances["consumer"] == 9.999998
