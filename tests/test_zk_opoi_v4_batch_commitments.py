"""HM-S6-03: batched speculative activation commitments and the cross-guild insurance pool."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError
from hivemind.swarm.ledger import GuildSlashingInsurancePool, HiveSwarmLedger
from hivemind.swarm.swarm_fabric import (
    OptimisticProofOfInferenceAuditor,
    SpeculativeBatchCommitment,
    SpeculativeBatchOpening,
)


def _activations(num_steps: int, num_layers: int, seed: int = 7) -> list[list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(2, 16, generator=gen) for _ in range(num_layers)] for _ in range(num_steps)
    ]


@pytest.mark.parametrize("num_steps", [6, 7, 8])
@pytest.mark.parametrize("num_layers", [3, 4])
def test_every_leaf_of_a_gamma_batch_opens_under_one_root(num_steps: int, num_layers: int) -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(num_steps, num_layers)
    commitment = auditor.commit_speculative_batch("peer-a", acts)

    assert len(commitment.root) == 64
    assert commitment.leaf_count == num_steps * num_layers
    assert commitment.max_opening_siblings == math.ceil(math.log2(commitment.leaf_count))
    for step in range(num_steps):
        for layer in range(num_layers):
            opening = auditor.open_speculative_batch(step, layer)
            assert len(opening.siblings) <= commitment.max_opening_siblings
            assert auditor.verify_speculative_batch_opening(commitment, opening, acts[step][layer])


def test_single_leaf_batch_opens_with_an_empty_path() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(1, 1)
    commitment = auditor.commit_speculative_batch("peer-a", acts)
    opening = auditor.open_speculative_batch(0, 0)
    assert opening.siblings == ()
    assert auditor.verify_speculative_batch_opening(commitment, opening, acts[0][0])


def test_tampered_activation_does_not_open() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(6, 4)
    commitment = auditor.commit_speculative_batch("peer-a", acts)
    opening = auditor.open_speculative_batch(2, 1)
    assert not auditor.verify_speculative_batch_opening(commitment, opening, acts[2][1] * 1.5)


def test_a_path_for_one_position_cannot_open_another() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(6, 4)
    commitment = auditor.commit_speculative_batch("peer-a", acts)
    opening = auditor.open_speculative_batch(2, 1)
    for step, layer in ((1, 2), (2, 0), (3, 1), (2, 2)):
        moved = dataclasses.replace(opening, step=step, layer=layer)
        assert not auditor.verify_speculative_batch_opening(commitment, moved, acts[2][1])
        assert not auditor.verify_speculative_batch_opening(commitment, moved, acts[step][layer])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s[:-1],
        lambda s: s + (s[0],),
        lambda s: (s[1], s[0]) + s[2:],
        lambda s: ("zz" * 32,) + s[1:],
        lambda s: (s[0][:-2],) + s[1:],
    ],
    ids=["truncated", "extended", "reordered", "not-hex", "short-digest"],
)
def test_malformed_sibling_paths_are_rejected(mutate) -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(7, 3)
    commitment = auditor.commit_speculative_batch("peer-a", acts)
    opening = auditor.open_speculative_batch(4, 2)
    bad = dataclasses.replace(opening, siblings=tuple(mutate(opening.siblings)))
    assert not auditor.verify_speculative_batch_opening(commitment, bad, acts[4][2])


def test_out_of_range_opening_is_rejected_not_raised() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(6, 4)
    commitment = auditor.commit_speculative_batch("peer-a", acts)
    opening = SpeculativeBatchOpening(step=6, layer=0, siblings=())
    assert not auditor.verify_speculative_batch_opening(commitment, opening, acts[0][0])


def test_root_binds_the_peer_and_the_batch_shape() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    flat = [t for step in _activations(3, 4) for t in step]
    c_a = auditor.commit_speculative_batch("peer-a", [flat[i : i + 4] for i in range(0, 12, 4)])
    opening = auditor.open_speculative_batch(1, 1)
    c_b = auditor.commit_speculative_batch("peer-b", [flat[i : i + 4] for i in range(0, 12, 4)])
    c_shape = auditor.commit_speculative_batch("peer-a", [flat[i : i + 6] for i in range(0, 12, 6)])

    assert len({c_a.root, c_b.root, c_shape.root}) == 3
    reattributed = dataclasses.replace(c_a, peer_id="peer-b")
    assert auditor.verify_speculative_batch_opening(c_a, opening, flat[5])
    assert not auditor.verify_speculative_batch_opening(reattributed, opening, flat[5])


def test_commit_and_open_input_validation() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    with pytest.raises(ValueError, match="commit_speculative_batch"):
        auditor.open_speculative_batch(0, 0)
    with pytest.raises(ValueError, match="peer_id"):
        auditor.commit_speculative_batch("  ", _activations(1, 1))
    with pytest.raises(ValueError, match="at least one speculative step"):
        auditor.commit_speculative_batch("peer-a", [])
    with pytest.raises(ValueError, match="at least one layer"):
        auditor.commit_speculative_batch("peer-a", [[]])
    ragged = _activations(2, 3)
    ragged[1].pop()
    with pytest.raises(ValueError, match="same number of layers"):
        auditor.commit_speculative_batch("peer-a", ragged)
    auditor.commit_speculative_batch("peer-a", _activations(6, 4))
    with pytest.raises(IndexError):
        auditor.open_speculative_batch(0, 4)


def test_commitment_and_opening_wire_format_round_trip() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    acts = _activations(6, 4)
    commitment = auditor.commit_speculative_batch("peer-a", acts)
    opening = auditor.open_speculative_batch(5, 3)
    wire = json.loads(json.dumps({"c": commitment.to_dict(), "o": opening.to_dict()}))
    assert wire["c"]["leaf_count"] == 24
    restored = SpeculativeBatchOpening.from_dict(wire["o"])
    assert restored == opening
    assert auditor.verify_speculative_batch_opening(commitment, restored, acts[5][3])


def test_challenges_are_distinct_in_range_and_depend_on_the_nonce() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    commitment = auditor.commit_speculative_batch("peer-a", _activations(8, 4))
    first = auditor.derive_batch_challenges(commitment, 5, "nonce-1")
    assert first == auditor.derive_batch_challenges(commitment, 5, "nonce-1")
    assert first != auditor.derive_batch_challenges(commitment, 5, "nonce-2")
    assert len(set(first)) == 5
    assert all(0 <= s < 8 and 0 <= l < 4 for s, l in first)
    everything = auditor.derive_batch_challenges(commitment, 32, "nonce-1")
    assert sorted(everything) == [(s, l) for s in range(8) for l in range(4)]
    with pytest.raises(ValueError, match="num_challenges"):
        auditor.derive_batch_challenges(commitment, 0, "nonce-1")
    with pytest.raises(ValueError, match="num_challenges"):
        auditor.derive_batch_challenges(commitment, 33, "nonce-1")
    with pytest.raises(ValueError, match="verifier_nonce"):
        auditor.derive_batch_challenges(commitment, 1, "")


def _forged_batch(auditor: OptimisticProofOfInferenceAuditor, peer: str = "peer-bad"):
    honest = _activations(6, 4, seed=11)
    claimed = [list(step) for step in honest]
    claimed[3][1] = honest[3][1] * 3.0
    commitment = auditor.commit_speculative_batch(peer, claimed)
    return honest, claimed, commitment


def test_challenge_verdicts_cover_every_fault_reason() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    honest, claimed, commitment = _forged_batch(auditor)

    ok = auditor.verify_speculative_batch_challenge(
        commitment, auditor.open_speculative_batch(0, 0), claimed[0][0], honest[0][0]
    )
    assert ok["included"] and ok["dual_bound_passed"] and not ok["fault_proven"]
    assert ok["fault_reason"] is None and ok["zero_knowledge"] is False

    forged = auditor.verify_speculative_batch_challenge(
        commitment, auditor.open_speculative_batch(3, 1), claimed[3][1], honest[3][1]
    )
    assert forged["included"] and forged["fault_proven"]
    assert forged["fault_reason"] == "activation_out_of_bounds"
    assert forged["cosine_similarity"] > 0.99

    swapped = auditor.verify_speculative_batch_challenge(
        commitment, auditor.open_speculative_batch(3, 1), honest[3][1], honest[3][1]
    )
    assert swapped["fault_reason"] == "opening_rejected" and swapped["fault_proven"]

    shapes = auditor.verify_speculative_batch_challenge(
        commitment, auditor.open_speculative_batch(0, 0), claimed[0][0], torch.randn(4, 16)
    )
    assert shapes["fault_reason"] == "shape_mismatch" and shapes["fault_proven"]


def test_whole_batch_verification_finds_the_forged_leaf() -> None:
    auditor = OptimisticProofOfInferenceAuditor()
    honest, claimed, commitment = _forged_batch(auditor)
    challenges = auditor.derive_batch_challenges(commitment, commitment.leaf_count, "nonce")
    responses = [
        (auditor.open_speculative_batch(s, l), claimed[s][l], honest[s][l]) for s, l in challenges
    ]
    report = auditor.verify_speculative_batch(commitment, challenges, responses)
    assert report["batch_passed"] is False
    assert [(f["step"], f["layer"]) for f in report["faults"]] == [(3, 1)]
    assert report["positions_checked"] == 24 and report["commitment_bytes"] == 32
    assert report["zero_knowledge"] is False

    clean = auditor.commit_speculative_batch("peer-good", honest)
    picks = auditor.derive_batch_challenges(clean, 4, "nonce")
    answers = [(auditor.open_speculative_batch(s, l), honest[s][l], honest[s][l]) for s, l in picks]
    assert auditor.verify_speculative_batch(clean, picks, answers)["batch_passed"] is True

    with pytest.raises(ValueError, match="challenged"):
        auditor.verify_speculative_batch(clean, picks, list(reversed(answers)))


# ---------------------------------------------------------------------------
# GuildSlashingInsurancePool
# ---------------------------------------------------------------------------


def _funded_pool(stake: float = 40.0, reserve: float = 100.0):
    ledger = HiveSwarmLedger()
    ledger.balances.update({"peer-bad": 50.0, "peer-good": 50.0, "guild-b-treasury": 500.0, "consumer": 1.0})
    pool = GuildSlashingInsurancePool(ledger, max_claim_hive=1_000.0)
    pool.bond_stake("GUILD-A", "peer-bad", stake)
    pool.bond_stake("GUILD-B", "peer-good", 10.0)
    pool.contribute_reserve("GUILD-B", "guild-b-treasury", reserve)
    return ledger, pool


def _supply(ledger: HiveSwarmLedger, pool: GuildSlashingInsurancePool) -> float:
    return round(ledger.total_circulating_supply() + pool.total_holdings(), 6)


def test_bonding_and_reserve_are_transfers_not_minting() -> None:
    ledger = HiveSwarmLedger()
    ledger.balances.update({"peer-bad": 50.0, "guild-b-treasury": 500.0})
    pool = GuildSlashingInsurancePool(ledger)
    before = _supply(ledger, pool)
    assert pool.bond_stake("GUILD-A", "peer-bad", 40.0) == 40.0
    assert pool.bond_stake("GUILD-A", "peer-bad", 5.0) == 45.0
    assert pool.contribute_reserve("GUILD-B", "guild-b-treasury", 100.0) == 100.0
    assert ledger.balances["peer-bad"] == 5.0 and ledger.balances["guild-b-treasury"] == 400.0
    assert pool.total_holdings() == 145.0
    assert _supply(ledger, pool) == before


@pytest.mark.parametrize(
    "call, category",
    [
        (lambda p: p.bond_stake("GUILD-A", "peer-bad", float("nan")), "NON_POSITIVE_OR_NON_FINITE_STAKE"),
        (lambda p: p.bond_stake("GUILD-A", "peer-bad", -1.0), "NON_POSITIVE_OR_NON_FINITE_STAKE"),
        (lambda p: p.bond_stake("GUILD-A", "peer-bad", "lots"), "NON_POSITIVE_OR_NON_FINITE_STAKE"),
        (lambda p: p.bond_stake("GUILD-A", "peer-bad", 1e9), "INSUFFICIENT_BALANCE"),
        (lambda p: p.bond_stake("", "peer-bad", 1.0), "EMPTY_GUILD_ID"),
        (lambda p: p.bond_stake("GUILD-A", "", 1.0), "EMPTY_NODE_ID"),
        (lambda p: p.bond_stake("GUILD-C", "peer-bad", 1.0), "PEER_ALREADY_BONDED_ELSEWHERE"),
        (lambda p: p.bond_stake("GUILD-A", "banned", 1.0), "SLASHED_NODE_BLOCKED"),
        (lambda p: p.contribute_reserve("", "peer-good", 1.0), "EMPTY_GUILD_ID"),
        (lambda p: p.contribute_reserve("GUILD-B", "peer-good", float("inf")), "NON_POSITIVE_OR_NON_FINITE_RESERVE"),
    ],
)
def test_pool_deposits_reject_bad_input(call, category) -> None:
    ledger, pool = _funded_pool()
    ledger.slashed_registry.append("banned")
    ledger.balances["banned"] = 10.0
    before = (dict(ledger.balances), pool.snapshot())
    with pytest.raises(SwarmSecurityViolationError) as err:
        call(pool)
    assert err.value.rule_id == "LEDGER-05" and err.value.category == category
    assert (dict(ledger.balances), pool.snapshot()) == before


def test_invalid_claim_cap_is_rejected() -> None:
    with pytest.raises(SwarmSecurityViolationError, match="INVALID_CLAIM_CAP"):
        GuildSlashingInsurancePool(HiveSwarmLedger(), max_claim_hive=0.0)


def test_stake_backed_indemnity_slashes_the_peer_and_conserves_supply() -> None:
    ledger, pool = _funded_pool(stake=40.0, reserve=100.0)
    auditor = OptimisticProofOfInferenceAuditor()
    honest, claimed, commitment = _forged_batch(auditor)
    before = _supply(ledger, pool)

    receipt = pool.indemnify_proven_fault(
        auditor, commitment, auditor.open_speculative_batch(3, 1), claimed[3][1], honest[3][1],
        consumer_node_id="consumer", claimed_hive=25.0,
    )
    assert receipt.indemnified_hive == 25.0 and receipt.shortfall_hive == 0.0
    assert receipt.paid_from_peer_stake == 25.0 and receipt.paid_from_reserve == 0.0
    assert receipt.stake_surplus_to_reserve == 15.0
    assert receipt.fault_reason == "activation_out_of_bounds"
    assert (receipt.step, receipt.layer, receipt.faulty_guild_id) == (3, 1, "GUILD-A")
    assert receipt.settlement_ms < 10.0
    assert ledger.balances["consumer"] == 26.0
    assert "peer-bad" in ledger.slashed_registry and ledger.reputation_scores["peer-bad"] == 0
    assert "peer-bad" not in pool.peer_stakes and pool.reserve_hive == 115.0
    assert _supply(ledger, pool) == before
    json.dumps(pool.snapshot())


def test_reserve_covers_what_the_stake_cannot_and_reports_any_shortfall() -> None:
    ledger, pool = _funded_pool(stake=10.0, reserve=30.0)
    auditor = OptimisticProofOfInferenceAuditor()
    honest, claimed, commitment = _forged_batch(auditor)
    before = _supply(ledger, pool)
    receipt = pool.indemnify_proven_fault(
        auditor, commitment, auditor.open_speculative_batch(3, 1), claimed[3][1], honest[3][1],
        consumer_node_id="consumer", claimed_hive=55.0,
    )
    assert (receipt.paid_from_peer_stake, receipt.paid_from_reserve) == (10.0, 30.0)
    assert receipt.indemnified_hive == 40.0 and receipt.shortfall_hive == 15.0
    assert pool.reserve_hive == 0.0 and ledger.balances["consumer"] == 41.0
    assert _supply(ledger, pool) == before


def test_indemnity_guards() -> None:
    ledger, pool = _funded_pool()
    auditor = OptimisticProofOfInferenceAuditor()
    honest, claimed, commitment = _forged_batch(auditor)
    opening = auditor.open_speculative_batch(3, 1)

    def claim(consumer="consumer", amount=5.0, leaf=(3, 1), batch=commitment, peer_open=opening):
        s, l = leaf
        return pool.indemnify_proven_fault(
            auditor, batch, peer_open, claimed[s][l], honest[s][l],
            consumer_node_id=consumer, claimed_hive=amount,
        )

    def category(**kwargs) -> str:
        with pytest.raises(SwarmSecurityViolationError) as err:
            claim(**kwargs)
        assert err.value.rule_id == "LEDGER-05"
        return err.value.category

    ledger.slashed_registry.append("banned-consumer")
    assert category(leaf=(0, 0), peer_open=auditor.open_speculative_batch(0, 0)) == "FAULT_NOT_PROVEN"
    assert category(consumer="peer-bad") == "SELF_INDEMNIFICATION_BLOCKED"
    assert category(consumer=" ") == "EMPTY_NODE_ID"
    assert category(consumer="banned-consumer") == "SLASHED_NODE_BLOCKED"
    assert category(amount=1_000.01) == "CLAIM_EXCEEDS_COVERAGE_CAP"
    assert category(amount=float("nan")) == "NON_POSITIVE_OR_NON_FINITE_CLAIM"

    stranger = auditor.commit_speculative_batch("peer-unbonded", claimed)
    assert category(batch=stranger, peer_open=auditor.open_speculative_batch(3, 1)) == "UNBONDED_PEER"

    auditor.commit_speculative_batch("peer-bad", claimed)
    claim()
    assert category() == "DUPLICATE_FAULT_CLAIM"


def test_indemnity_is_persisted_with_the_ledger(tmp_path) -> None:
    ledger, pool = _funded_pool()
    ledger.ledger_path = tmp_path / "ledger.json"
    auditor = OptimisticProofOfInferenceAuditor()
    honest, claimed, commitment = _forged_batch(auditor)
    pool.indemnify_proven_fault(
        auditor, commitment, auditor.open_speculative_batch(3, 1), claimed[3][1], honest[3][1],
        consumer_node_id="consumer", claimed_hive=5.0,
    )
    saved = json.loads(ledger.ledger_path.read_text(encoding="utf-8"))
    assert saved["balances"]["consumer"] == 6.0
    assert "peer-bad" in saved["slashed_registry"]
