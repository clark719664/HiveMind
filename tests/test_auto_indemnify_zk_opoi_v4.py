"""HM-S7-04: a proven batch fault automatically indemnifies the consumer."""

from __future__ import annotations

import pytest
import torch

from hivemind.swarm.ledger import GuildSlashingInsurancePool, HiveSwarmLedger
from hivemind.swarm.swarm_fabric import OptimisticProofOfInferenceAuditor


def _setup(forge: bool = True, bond: bool = True):
    gen = torch.Generator().manual_seed(5)
    honest = [[torch.randn(2, 16, generator=gen) for _ in range(4)] for _ in range(6)]
    claimed = [list(step) for step in honest]
    if forge:
        claimed[4][2] = honest[4][2] * 3.0
        claimed[1][0] = honest[1][0] * -1.0
    auditor = OptimisticProofOfInferenceAuditor()
    commitment = auditor.commit_speculative_batch("peer-x", claimed)
    challenges = auditor.derive_batch_challenges(
        commitment, commitment.leaf_count, auditor.new_verifier_nonce()
    )
    responses = [
        (auditor.open_speculative_batch(s, l), claimed[s][l], honest[s][l]) for s, l in challenges
    ]
    ledger = HiveSwarmLedger()
    ledger.balances.update({"peer-x": 100.0, "treasury": 100.0, "consumer": 0.0})
    pool = GuildSlashingInsurancePool(ledger)
    if bond:
        pool.bond_stake("GUILD-A", "peer-x", 30.0)
    pool.contribute_reserve("GUILD-B", "treasury", 40.0)
    return auditor, commitment, challenges, responses, ledger, pool


def _supply(ledger, pool) -> float:
    return round(ledger.total_circulating_supply() + pool.total_holdings(), 6)


def test_a_proven_fault_pays_the_consumer_once_and_slashes_the_peer() -> None:
    auditor, commitment, challenges, responses, ledger, pool = _setup()
    before = _supply(ledger, pool)
    report = auditor.audit_batch_and_indemnify(
        commitment, challenges, responses, pool, consumer_node_id="consumer", claimed_hive=45.0
    )
    assert report["batch_passed"] is False
    assert sorted((f["step"], f["layer"]) for f in report["faults"]) == [(1, 0), (4, 2)]
    receipt = report["indemnity"]
    assert report["indemnity_refused"] is None
    assert (receipt["step"], receipt["layer"]) == (
        report["faults"][0]["step"],
        report["faults"][0]["layer"],
    )
    assert (receipt["paid_from_peer_stake"], receipt["paid_from_reserve"]) == (30.0, 15.0)
    assert ledger.balances["consumer"] == 45.0
    assert "peer-x" in ledger.slashed_registry
    assert _supply(ledger, pool) == before

    again = auditor.audit_batch_and_indemnify(
        commitment, challenges, responses, pool, consumer_node_id="consumer", claimed_hive=45.0
    )
    assert again["indemnity"] is None
    assert again["indemnity_refused"]["category"] == "DUPLICATE_FAULT_CLAIM"
    assert ledger.balances["consumer"] == 45.0


def test_an_honest_batch_triggers_no_payment() -> None:
    auditor, commitment, challenges, responses, ledger, pool = _setup(forge=False)
    holdings = pool.total_holdings()
    report = auditor.audit_batch_and_indemnify(
        commitment, challenges, responses, pool, consumer_node_id="consumer", claimed_hive=10.0
    )
    assert report["batch_passed"] is True
    assert report["indemnity"] is None and report["indemnity_refused"] is None
    assert pool.total_holdings() == holdings and ledger.balances["consumer"] == 0.0


@pytest.mark.parametrize(
    "consumer, amount, bond, category",
    [
        ("consumer", 10.0, False, "UNBONDED_PEER"),
        ("peer-x", 10.0, True, "SELF_INDEMNIFICATION_BLOCKED"),
        ("consumer", 5_000.0, True, "CLAIM_EXCEEDS_COVERAGE_CAP"),
    ],
)
def test_a_refused_indemnity_is_reported_with_the_verdict(consumer, amount, bond, category) -> None:
    auditor, commitment, challenges, responses, ledger, pool = _setup(bond=bond)
    balances = dict(ledger.balances)
    report = auditor.audit_batch_and_indemnify(
        commitment, challenges, responses, pool, consumer_node_id=consumer, claimed_hive=amount
    )
    assert report["batch_passed"] is False and report["faults"]
    assert report["indemnity"] is None
    assert report["indemnity_refused"]["rule_id"] == "LEDGER-05"
    assert report["indemnity_refused"]["category"] == category
    assert ledger.balances == balances


def test_verifier_nonces_are_fresh() -> None:
    nonces = {OptimisticProofOfInferenceAuditor.new_verifier_nonce() for _ in range(64)}
    assert len(nonces) == 64 and all(len(n) == 32 for n in nonces)
