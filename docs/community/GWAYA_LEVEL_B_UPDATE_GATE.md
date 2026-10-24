# GWAYA Level B: a sandbox pass rate gates LoRA / DiLoCo updates

Level A (`src/hivemind/core/serving_experts.py`) serves the GWAYA Code expert. Level B decides whether a **trained** Python
expert (for example the E15 LoRA on Qwen2.5-Coder-1.5B, `scripts/e15_gate.py`) may replace the current one.

**Status: the gate is implemented and tested; nothing was trained or evaluated with it.** No LoRA was trained here, no
pass rate is claimed, and no update has been accepted or refused by it.

## What it does

`src/hivemind/swarm/verified_update_gate.py` (torch-free, no I/O of its own):

- `evaluate_tasks(tasks, generate, gate)` runs a model on held-out tasks and judges each reply with a sandbox gate such as
  `e15_gate.gate`. Result per task: `pass`, `fail` or `unverified` (sandbox unavailable or errored, generator crashed).
- `decide(baseline, candidate, policy)` compares two such maps on the **same** task keys with the exact paired (McNemar)
  test on the discordant tasks: `b` = baseline passes and candidate fails, `c` = the reverse.
- `gate_update(...)` does both and returns an `UpdateReceipt` (counts, one- and two-sided exact p-values, decision, reasons,
  the policy used, and a SHA-256 of the receipt).

Decisions: `ACCEPT`, `REJECT`, `INSUFFICIENT`. Only `ACCEPT` admits an update.

## Rules (defaults in `UpdatePolicy`)

1. Same task keys in both maps and at least `min_n` = 30 tasks, else `INSUFFICIENT`.
2. More than `max_unverified_fraction` = 5% of tasks unverified (in either map) is `INSUFFICIENT`. Unverified is never a
   pass: an unverified candidate result scores as a failure, an unverified baseline result scores as a pass, so uncertainty
   can only count against the update.
3. A net loss (`b - c`) above `tolerated_net_loss` = 0 tasks is `REJECT`, and so is a one-sided significant regression
   (exact p < `alpha` = 0.05).
4. With `require_improvement` the candidate must beat the baseline with one-sided exact p < `alpha`.
5. Otherwise `ACCEPT`. An identical candidate is accepted under the default non-regression policy and rejected when an
   improvement is required.

## Limits

- A pass means the sandboxed tests passed, not that the code is correct, and the gate is only as good as the held-out set.
  Keep that set out of any training data (`scripts/e15_data.py` has the decontamination helpers).
- The sandbox is `e15_gate`'s user-namespace sandbox, which is not equivalent to GWAYA's own Bubblewrap oracle (see
  `docs/community/GWAYA_CODING_ASSISTANT.md`).
- Many held-out tasks are needed to see a small difference; with the defaults a loss of one task is refused although it is
  not statistically significant, by design.
- It is not yet wired into the DiLoCo merge path or the E14 admission policies; callers invoke `gate_update` themselves.

## Use

```python
from hivemind.swarm.verified_update_gate import HeldOutTask, UpdatePolicy, gate_update
receipt = gate_update(tasks, baseline_generate, candidate_generate, e15_gate.gate, UpdatePolicy())
if not receipt.accepted:
    print(receipt.decision, receipt.reasons)
```
