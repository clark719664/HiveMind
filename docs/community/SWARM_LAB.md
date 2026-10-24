# HiveMind Swarm Lab

A small, pure-Python toy that lets you explore how a **quorum policy** (how many workers a round waits for, and what happens to late ones),
**stragglers** and **non-IID data** interact in a DiLoCo-style swarm. It runs on a laptop CPU in milliseconds, needs no GPU, no network and no
third-party package, and is fully deterministic.

## What it is, and what it is not

- It **is** an educational, qualitative analogy of an effect we measured for real in experiment E13 (see `docs/RESEARCH_PROGRAM.md`, E8/E13/E14):
  with a quorum smaller than the number of workers and a persistently slow worker that owns its own data domain, that domain is starved and the swarm
  keeps only a fraction of the improvement of the synchronous baseline.
- It is **not** a re-implementation of E13, it uses quadratic losses instead of an LLM, and its defaults were **not** tuned to reproduce E13's numbers.
  Never quote a number from the toy as a result about real training. The toy only shows that the *ordering* of the policies can be explained by a
  very simple mechanism.

## Run it

```bash
python3 scripts/swarm_lab.py presets
python3 scripts/swarm_lab.py simulate --preset persistent-straggler          # table + ASCII sparkline of the global loss
python3 scripts/swarm_lab.py simulate --preset churn --json                  # full result as JSON
python3 scripts/swarm_lab.py simulate --M 6 --K 4 --straggle rotating --policy deferred --seed 7
python3 scripts/swarm_lab.py compare --preset persistent-straggler           # policies x straggle modes table
python3 scripts/swarm_lab.py play --auto                                     # "Quorum Quest" (drop --auto to play with the keyboard)
hivemind swarm-lab simulate --preset churn                                   # same CLI through the main entry point
```

`simulate` has one flag per scenario field (`--M --K --rounds --H --eta --sigma --dim --spread --offset --seed --merge --policy --straggle --penalty
--straggle-every --grace --churn-start --churn-end --outer-lr --outer-momentum`); explicit flags override `--preset`. Tests: `python3 -m pytest tests/test_swarm_lab.py`.

## The model in plain words ("SwarmLab-1")

- The model has 16 parameters. There are M data domains (M up to 6); each domain has a quadratic loss with its own optimum. The optima share a common
  centre but are pushed apart by `spread` in random directions: this is the non-IID part. The global metric is the mean of the domain losses.
- Each round every worker takes H small noisy gradient steps on its own domain from the current global model and reports the difference
  (its "delta"). Arrival times are 1 time unit plus a little jitter; a straggler gets `penalty` extra time units.
- The coordinator waits for the K-th arrival, then for a grace window of 10 % more time, and merges the deltas it has (a plain mean, or `rda`, the
  radial-directional average of the repo, which rescales the mean by sum|D| / |sum D|). The merged delta goes through the same Nesterov outer step
  as E13 (lr 0.7, momentum 0.9).
- What happens to a late delta is the *policy*:
  - `discard`: dropped; the worker re-syncs next round.
  - `fair` (like E14's S6): a worker that was excluded last round **must** be waited for this round (dead workers never block). Costs wall time.
  - `deferred` (like E14's S7): the late delta is kept and merged into the next round with weight 1/(1+staleness); older than 2 rounds is dropped.
- The exact algorithm (PRNG, operation order, a digest to compare ports) is the contract in the docstring of `src/hivemind/lab/swarm_lab.py`.

## How to read retention

`retention = (L0 - L_final) / (L0 - L_sync)`, the same formula as E13. `L0` is the initial global loss, `L_sync` the final loss of the same
scenario run synchronously (K = M, no stragglers, same seed). 1.0 means "kept all of the synchronous improvement", 0 means "learned nothing",
below 0 means worse than the start. It is `n/a` when the synchronous run did not improve on the start. Values slightly above 1.0 are noise. The
toy starts at the shared centre (`offset 0`), so the improvement to be had is exactly the domain-specific part that a starved domain cannot provide.
`total time` is the simulated wall time: the price of fairness shows up there.

## What the toy shows (and does not) at the defaults

Mean retention over seeds 1-20, M=4, 12 rounds, merge `direct`, straggler penalised every round (the default `--straggle-every 1`):

| scenario (K=3 of 4) | discard | fair | deferred |
|---|---|---|---|
| K = M, no straggler (reference) | 1.00 | - | - |
| persistent straggler | 0.59 | 0.98 (time 26 vs 15) | 0.93 |
| rotating straggler | 0.77 | 0.77 | 0.78 |

Qualitatively this has the same ordering as E13 (synchronous best, a persistent straggler on one domain worst, rotating less bad, mitigations
recover part of it). The honest caveats: the toy numbers are not E13's (E13 kept about 0 % with K=2, 94 % with K=M, 40 % rotating); with E13's
exact odd-round straggler pattern (`--straggle-every 2`) the toy shows **no** persistent-vs-rotating ordering (persistent 0.98, rotating 0.86);
`fair` cannot help a rotating straggler because a different worker is late each round; and `--merge rda` makes the toy erratic (retention standard
deviation above 1 across seeds) because the radial rescaling amplifies steps once the deltas of different domains cancel, so the default is `direct`.

## Contribution ideas (small, Python, no GPU)

1. **Add a policy**: e.g. `bounded-wait` (wait for a late worker only up to a cap) or `weighted-stale` with another staleness discount; add it to `POLICIES`, `_run`, tests and a row in `compare`.
2. **Add a straggle pattern**: random (each worker late with probability p), bursty, or "slow region" (two workers slow together); keep it deterministic through the documented PRNG.
3. **Add a domain preset**: a few named `Scenario` sets with unequal domain sizes or very different curvatures, and a short note on what each teaches.
4. **Add a codec-noise option**: add seeded noise to each delta before the merge, scaled to a bits-per-dimension figure, to mimic lossy compression (E8/E10b lattice codecs).
5. **Port to Rust or WASM**: re-implement SwarmLab-1 from the docstring contract and check `contract_digest` and the loss curve to 1e-9 (a JavaScript port is already planned for a web page).
6. **Add a plot**: an optional (not required) matplotlib or SVG export of the loss curves and per-domain losses for a scenario or a `compare` table.

Keep the module dependency-free, deterministic and honest about being a toy; do not tune defaults to chase E13 numbers.

## Notice

$HIVE and PTS are internal, non-monetary, non-transferable accounting units; no money or financial payment is or will ever be involved (see
`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`). The game prints a "learning score" which is just a teaching number, not PTS or $HIVE.

## Limitations

- Quadratic losses, 16 parameters, equal step counts, no real network, no data-loading cost, no error feedback: none of the effects of a real LLM swarm beyond the one analogy.
- Round times are synthetic (1 + jitter + penalty); the fairness wait is uncapped here (E14 caps it).
- Retention depends on noise; use several seeds (`--seed`) before drawing a conclusion.
- Not validated against any real measurement; results from the toy must not be cited as evidence about E13 or about real decentralised training.
