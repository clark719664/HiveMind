"""Durable ledger reload, legacy migration, and fail-closed corruption handling."""
import json
from dataclasses import asdict

import pytest

from hivemind.swarm.ledger import HivePoUWBlock, HiveSwarmLedger, WorkReceipt


def create_funded_ledger(path):
    ledger = HiveSwarmLedger(path)
    ledger.credit_reward("consumer", 10)
    ledger.balances["fraud"] = 4
    receipts = [
        WorkReceipt("worker", "steam", "test", 1, 1, True, True, "a" * 64),
        WorkReceipt("fraud", "steam-bad", "test", 1, 1, False, False, "b" * 64),
    ]
    ledger.mine_pouw_block(1, "cid-test", receipts, [], ["validator"])
    ledger.settle_swarm_inference("consumer", ["provider"], "test", 3, 1)
    return ledger


def write_snapshot(path, data, *, refresh_checksum=True):
    if refresh_checksum and data.get("schema_version") == 1:
        unsigned = {k: v for k, v in data.items() if k != "snapshot_sha256"}
        data["snapshot_sha256"] = HiveSwarmLedger._snapshot_digest(unsigned)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_restart_preserves_all_balances_chain_reputation_slashing_and_history(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = create_funded_ledger(path)
    previous_bytes = path.read_bytes()
    reloaded = HiveSwarmLedger(path)
    assert reloaded.balances == ledger.balances
    assert reloaded.reputation_scores == ledger.reputation_scores
    assert reloaded.slashed_registry == ledger.slashed_registry == ["fraud"]
    assert [asdict(block) for block in reloaded.chain] == [asdict(block) for block in ledger.chain]
    assert reloaded.settlement_history == ledger.settlement_history
    assert reloaded.verify_chain_integrity()
    assert path.read_bytes() == previous_bytes  # Reading never silently rewrites/migrates a file.
    previous_head = reloaded.chain[-1].block_hash
    reloaded.mine_pouw_block(2, "cid-next", [], [], ["validator"])
    reloaded.settle_swarm_inference("consumer", ["provider"], "test", 1, 1)
    again = HiveSwarmLedger(path)
    assert len(again.chain) == 3
    assert again.chain[-1].prev_block_hash == previous_head
    assert again.balances["consumer"] == 9.98
    assert len(again.settlement_history) == 2


def test_legacy_snapshot_loads_and_migrates_only_on_next_write(tmp_path):
    path = tmp_path / "legacy.json"
    original = create_funded_ledger(path)
    legacy = json.loads(path.read_text())
    legacy.pop("schema_version")
    legacy.pop("snapshot_sha256")
    write_snapshot(path, legacy)
    legacy_bytes = path.read_bytes()
    restored = HiveSwarmLedger(path)
    assert restored.balances == original.balances
    assert restored.chain == original.chain
    assert path.read_bytes() == legacy_bytes
    restored.credit_reward("new-node", 1)
    migrated = json.loads(path.read_text())
    assert migrated["schema_version"] == 1
    assert len(migrated["snapshot_sha256"]) == 64
    assert HiveSwarmLedger(path).balances["new-node"] == 1


@pytest.mark.parametrize("content", ["", "{", "[]", '{"balances": NaN}', '{"chain": [], "chain": []}'])
def test_unreadable_or_malformed_snapshot_is_not_reset_or_overwritten(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid persisted ledger"):
        HiveSwarmLedger(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", [
    "future_version", "boolean_version", "missing_checksum", "bad_checksum", "unknown_field",
    "negative_balance", "nonnumeric_balance", "invalid_reputation", "duplicate_slashing",
    "empty_chain", "invalid_hash", "broken_link", "bad_height", "different_genesis",
    "negative_allocation", "missing_block_field", "negative_fee", "invalid_tokens", "duplicate_provider",
])
def test_invalid_persisted_state_fails_closed_without_modifying_file(tmp_path, mutation):
    path = tmp_path / "ledger.json"
    create_funded_ledger(path)
    data = json.loads(path.read_text())
    refresh_checksum = True
    if mutation == "future_version":
        data["schema_version"] = 2
    elif mutation == "boolean_version":
        data["schema_version"] = True
    elif mutation == "missing_checksum":
        data.pop("snapshot_sha256")
        refresh_checksum = False
    elif mutation == "bad_checksum":
        data["balances"]["consumer"] += 1
        refresh_checksum = False
    elif mutation == "unknown_field":
        data["unrecognized"] = "must not be silently discarded"
    elif mutation == "negative_balance":
        data["balances"]["consumer"] = -1
    elif mutation == "nonnumeric_balance":
        data["balances"]["consumer"] = "10"
    elif mutation == "invalid_reputation":
        data["reputation_scores"]["worker"] = 1.5
    elif mutation == "duplicate_slashing":
        data["slashed_registry"] *= 2
    elif mutation == "empty_chain":
        data["chain"] = []
    elif mutation == "invalid_hash":
        data["chain"][1]["block_hash"] = "f" * 64
    elif mutation in ("broken_link", "bad_height", "negative_allocation"):
        entry = data["chain"][1]
        if mutation == "broken_link":
            entry["prev_block_hash"] = "f" * 64
        elif mutation == "bad_height":
            entry["block_height"] = 99
        else:
            entry["reward_allocations"]["worker"] = -1
        entry["block_hash"] = HivePoUWBlock(**entry).compute_hash()
    elif mutation == "different_genesis":
        data["chain"][0]["expert_cid"] = "alternate-genesis"
        data["chain"][0]["block_hash"] = HivePoUWBlock(**data["chain"][0]).compute_hash()
        data["chain"][1]["prev_block_hash"] = data["chain"][0]["block_hash"]
        data["chain"][1]["block_hash"] = HivePoUWBlock(**data["chain"][1]).compute_hash()
    elif mutation == "missing_block_field":
        data["chain"][1].pop("timestamp")
    elif mutation == "negative_fee":
        data["settlement_history"][0]["fee_paid_hive"] = -1
    elif mutation == "invalid_tokens":
        data["settlement_history"][0]["verified_tokens"] = 0
    else:
        data["settlement_history"][0]["expert_provider_nodes"] *= 2
    write_snapshot(path, data, refresh_checksum=refresh_checksum)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid persisted ledger"):
        HiveSwarmLedger(path)
    assert path.read_bytes() == before


def test_missing_legacy_fields_are_not_assumed_empty(tmp_path):
    path = tmp_path / "legacy.json"
    create_funded_ledger(path)
    data = json.loads(path.read_text())
    for name in ("schema_version", "snapshot_sha256", "settlement_history"):
        data.pop(name)
    write_snapshot(path, data)
    with pytest.raises(ValueError, match="Malformed legacy"):
        HiveSwarmLedger(path)


def test_restart_preserves_negative_reputation_scores(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = HiveSwarmLedger(path)
    ledger.reputation_scores["penalized"] = -50
    ledger._persist()
    assert HiveSwarmLedger(path).reputation_scores["penalized"] == -50


def test_chain_integrity_requires_canonical_genesis_and_contiguous_heights():
    ledger = HiveSwarmLedger()
    ledger.chain[0].expert_cid = "substituted-genesis"
    ledger.chain[0].block_hash = ledger.chain[0].compute_hash()
    assert not ledger.verify_chain_integrity()
