# HM-S6-03: batched activation commitments and the guild insurance pool

Issue #58 asks for two pieces: batched activation commitments over several speculative
token-tree steps, and a cross-guild slashing insurance pool that pays a consumer when a
fault is proven. This note says what the code does and what it does not claim.

## Batched commitments (`swarm_fabric.py`)

`OptimisticProofOfInferenceAuditor.commit_speculative_batch(peer_id, step_activations)`
takes `gamma` speculative steps (typically 6 to 8), each with the same `L` layer
activations, and returns one `SpeculativeBatchCommitment`.

- Every activation becomes the existing ZK-OPoI v2 leaf (FP8 bytes, FP16 per-row scale and
  shape metadata), then is tagged with its `(step, layer)` position.
- Leaves, internal nodes and the root use different hash prefixes, so an internal node can
  never be presented as a leaf.
- The root also binds the peer id and the batch shape, so a commitment cannot be
  re-attributed to another peer or reinterpreted as a different `gamma x L` grid.
- A lone node at the end of a level is carried up unchanged instead of duplicated, so two
  different leaf lists cannot share a root.

The commitment is 32 bytes whatever `gamma` is. Opening one position costs at most
`ceil(log2(gamma * L))` sibling hashes. The verifier derives the sibling order and the path
length from the position and the committed shape; it never reads them from the opening.

`derive_batch_challenges(commitment, k, verifier_nonce)` picks `k` distinct positions from
the root and a nonce. The verifier must choose the nonce after it receives the commitment.
Positions taken from the root alone would let a prover grind roots until its bad leaves
are skipped. With `b` bad leaves among `n`, `k` distinct challenges miss every one of them
with probability `C(n - b, k) / C(n, k)`.

`verify_speculative_batch_challenge` reports a fault when the opening does not reproduce
the root, when the claimed activation fails the existing cosine and normalized `L_inf`
bounds against the validator's recomputation, or when the shapes differ.

## Guild insurance pool (`ledger.py`)

`GuildSlashingInsurancePool` holds bonded peer stake and a shared reserve that any guild can
fund. Both are transfers out of `HiveSwarmLedger.balances`, so nothing is minted:
`ledger.total_circulating_supply() + pool.total_holdings()` stays constant.

`indemnify_proven_fault` re-runs the challenge with the auditor itself before it pays. It then
pays the consumer from the faulty peer's stake first and from the reserve second. Any unused
stake moves to the reserve, and the peer joins the ledger's `slashed_registry`. The pool
pays one indemnity per batch root. It refuses self-claims, slashed consumers, unbonded
peers, non-finite amounts and claims above its coverage cap (rule `LEDGER-05`).

## What this is not

- **Not zero-knowledge.** The commitment is binding, not hiding: opening a leaf reveals that
  activation. The `zero_knowledge: False` field in every verdict says so.
- **Not a polynomial commitment.** The issue mentions batch polynomial commitments. This is a
  hash-based Merkle batch with no new cryptographic dependency.
- **Not on-chain.** Settlement is a synchronous update of the local ledger.
  `IndemnityReceipt.settlement_ms` measures that update. The test asserts it stays under
  10 ms, which covers the ledger arithmetic, not any network or consensus step.
- **No peer signatures.** As in the v2 and v3 dispute games, the pool trusts that a
  commitment it receives really came from the named peer.
