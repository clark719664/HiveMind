# Project HiveMind — Research Program (scientific lead: Claude; engineering coordinator: Antigravity)

Version 1.0, 2026-10-23. This document is the scientific contract of the project. It states the
thesis, the research questions, the predictions each question makes, what observation would
refute each prediction, and the pre-registered experiments that decide them. Engineering tasks
(`HM-S9-xx` in `ROADMAP.md`) exist to produce the measurements named here; a task is finished
when its artifact answers the question it was opened for, not when code merges.

Operating rule carried over from the 2026-10-23 review (`MOE_DILOCO_REVIEW_2026-10-23.md`):
no number enters the paper without an artifact produced by a committed script, every experiment
runs a positive and a negative control, no script emits a score or verdict labelled as a
reviewer's, and every arXiv identifier is opened before it is cited.

## 1. Thesis

Volunteer consumer GPUs (8 GB, residential WAN, churn, untrusted operators) can train a sparse
Mixture-of-Experts whose held-out loss, at matched tokens, lies within a **predictable** distance
of a single-host DiLoCo run with the same number of replicas. "Predictable" means: the gap is the
sum of (a) the replica-count penalty already described by the DiLoCo scaling law of Charles et
al. (`arXiv:2503.09799`) and (b) a **volunteer tax** caused by churn, staleness and Byzantine
filtering, and the mechanisms integrated in HiveMind (Decoupled quorum + grace window + RDA,
PA-CGAD, FoMoE partial replication, spot-check verification) keep (b) bounded by the penalty of
simply having fewer, reliable replicas.

If (b) is *not* bounded that way, the system design is wrong and the paper must say so. Every
question below is a piece of that thesis.

## 2. Research questions and predictions ledger

| RQ | Question | Prediction (falsifiable) | Refuted if | Decides | Experiment |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **RQ1** Volunteer tax | How far is the held-out loss of M real WAN learners with churn from M synchronous single-host workers at matched tokens? | **P1**: with `K < M` quorum, grace window and RDA, the WAN loss is no worse than the single-host loss obtained with `M·(1−churn)` workers. | WAN loss exceeds the "fewer workers" baseline by more than the 95 % seed CI. | whether the quorum design pays for itself | E8 (needs E5 first) |
| **RQ2** RDA and outer-LR retuning | Does Radial-Directional Averaging remove the need to retune the outer learning rate with M? | **P2**: under direct averaging the best `η_outer` grows with M (≈√M when learner deltas are near-orthogonal; report the measured mean pairwise cosine); under RDA the best `η_outer` is the same grid point for M ∈ {4, 8, 16}. | best `η_outer` under RDA moves by ≥ 1 grid step between M = 4 and M = 16, or deltas are not near-orthogonal (then the prediction does not apply and must be re-derived). | RDA as default merge | E5 |
| **RQ3** Grace window vs matched random | Is adaptive grace admission better than admitting the same expected number of extra learners at random? (lesson of WA-DiLoCo, `arXiv:2607.02544`: forecast-free policies must beat matched random) | **P3**: at equal loss, adaptive grace reaches it in fewer wall-clock seconds than matched-random admission with the same admission budget. | matched random ties or wins (then the grace window is only a budget knob and the EMA controller should be deleted). | keep or delete `AdaptiveGraceWindowController` | E5 |
| **RQ4** Staleness gating under quorum | Does PA-CGAD still help once a quorum already excludes late learners? | **P4**: CGAD's terminal-MSE gain over un-gated outer AdamW is large without quorum (E2 reproduced) and shrinks with quorum `K < M`; it never hurts. | CGAD hurts under quorum, or its gain does not shrink. | whether to ship CGAD + quorum together | E5 (factorial) |
| **RQ5** Ghost fraction and quality | Is Skip-Token quality loss governed by the *measured* ghost-assignment fraction rather than the ideal `N_le/N_e`? | **P5**: at a fixed ideal factor, perplexity degrades monotonically with the measured ghost fraction across routing-imbalance conditions. | no monotone relation, or the ideal factor predicts as well as the measured one. | whether to log and gate on the measured fraction (`HM-S9-02`) | E7 |
| **RQ6** Replica scaling and inner optimizer | Does a whitening inner optimizer flatten the loss-vs-M slope (claim of `arXiv:2610.19064`, `arXiv:2505.23725`)? | **P6**: slope of held-out loss vs log M with Muon Newton–Schulz inner preconditioning is smaller than with AdamW inner, at matched compute. | slopes are equal within CI, or Muon is worse at M = 4. | default inner optimizer for the campaign | E6 |
| **RQ7** Looped depth prior | Does a broad, learned depth prior make terminal-KV sharing safe where fixed-depth training does not (Part II, `arXiv:2610.06833`)? | **P7**: accuracy drop from full cache to terminal-KV sharing is smaller with the learned prior than with fixed depth, at equal mean recurrence. | drops are equal within CI. | whether `HM-S7-02` KV sharing is enabled by default | E7b |
| **RQ8** Adaptive Byzantine attacks | Does Multi-Krum + 2σ cosine filtering stop an ALIE-style small-perturbation attack under non-IID shards? | **P8**: it does not for `f/n ≥ 0.2`; the deterministic spot-check replay is what catches it. The paper currently claims only sign-flip/scale resistance and must not claim more until measured. | the filter alone blocks ALIE at `f/n = 0.3` (then the spot-check cost can be cut). | security section of the paper | E9 |

Status column convention (kept in `artifacts/predictions_ledger.json` once E5 starts): `untested`,
`supported`, `refuted`, `inconclusive (CI includes 0)`. A refuted prediction is reported in the
paper as refuted; it is not silently removed.

## 3. Scaling-law prior (how "predictable" is computed)

Before E5, fit the functional form used by Charles et al. (`arXiv:2503.09799`) for the loss as a
function of model size, tokens, replica count M and inner steps H to HiveMind's own single-host
runs at M ∈ {1, 2, 4, 8, 16} on the `Qwen2.5-0.5B` LoRA setting (E1 model, ≥ 3 seeds). We do not
reuse their coefficients, which were fitted on different models and data; we reuse the form. The
fitted curve is the "replica-count penalty" baseline. The volunteer tax of RQ1 is then the residual
between the WAN run and that curve. Report the fit, its residual standard error and the number of
points; if the form does not fit single-host runs (R² < 0.9), say so and fall back to the
"fewer workers" baseline only.

## 4. Pre-registered experiments

Common protocol for every experiment:

- **Seeds**: ≥ 3 seeds per cell; report mean and 95 % t-interval; claim an effect only if the
  interval excludes zero. E1–E4 in paper v3.2.0 use a single seed (2026) and are therefore
  demonstrations, not evidence of an effect; E5 re-runs E1's regimes with 3 seeds as its first cell.
- **Positive control**: a cell where the known effect must reproduce (e.g. un-gated AdamW under
  stale bursts must diverge as in E2). If it does not, the run is invalid, not "surprising".
- **Negative control**: a cell where the mechanism is switched off or the signal is destroyed
  (e.g. RDA with identical learner deltas must equal direct averaging; grace window with
  `γ = 0` must equal pure quorum). If the negative control shows the effect, the measurement is
  an artefact.
- **Matched-random control** for every *policy* (grace window, cadence controller, Mix2 block
  selection): same budget, random decisions.
- **Pre-registration**: the metric, the cells, the seeds and the acceptance rule are written in
  `artifacts/experiments/E<n>_prereg.json` before the run and hashed into the result artifact.
- **Artifacts**: `scripts/run_experiment_E<n>.py` → `artifacts/experiments/E<n>_report.json`
  with `receipt_sha256`; figures regenerated from the JSON; the paper's provenance lock
  (`PROV-*`) extended for every number quoted.

### E5 — Outer-loop factorial under simulated churn (single host; decides RQ2, RQ3, RQ4)

- Model/data: E1 synthetic LM for the full factorial (cheap), then the `Qwen2.5-0.5B` LoRA
  8-domain setting for the two best and two worst cells.
- Factors: merge ∈ {direct, RDA} × quorum ∈ {K = M, K = ⌈M/2⌉} × grace ∈ {off, adaptive,
  matched-random} × staleness gate ∈ {outer AdamW, PA-CGAD} × M ∈ {4, 8, 16}.
- Churn model: learner arrival times drawn from the empirical RTT and dropout distribution
  recorded by `wan_fault_benchmark.py` (not from a Gaussian), replayed identically across cells.
- Outer-LR sweep per cell: `η_outer ∈ {0.3, 0.5, 0.7, 0.9, 1.1}` (RQ2 needs the argmax).
- Metrics: held-out loss at matched tokens; wall-clock to reach the direct/K=M/no-grace loss;
  admitted-token fraction; mean pairwise cosine of learner deltas (reported, conditions P2).
- Controls: positive = E2 divergence reproduces in the {no quorum, AdamW} cell; negative =
  identical-delta learners give RDA ≡ direct; matched random for grace.
- Acceptance: P2, P3, P4 each get `supported/refuted/inconclusive` with the rule in §2.

### E6 — Replica scaling vs inner optimizer (decides RQ6)

- M ∈ {2, 4, 8, 16}, inner ∈ {AdamW, Muon-NS5}, H = 32 and 128, 3 seeds, matched total tokens.
- Metric: slope of held-out loss vs log₂ M by least squares; report both slopes with CI.
- Controls: M = 1 both inner optimizers (positive: Muon ≥ AdamW at M = 1 is *not* required, the
  claim is about slope); negative: shuffled-label run must show no inner-optimizer effect.

### E7 — Ghost fraction sweep (decides RQ5) and E7b looped depth prior (decides RQ7)

- E7: 128 experts, 32 resident, top-2; induce routing imbalance by temperature on the router
  logits so that the measured ghost fraction spans ≈ 0.5–0.9 at a fixed ideal factor 0.25;
  3 seeds; metric: held-out perplexity vs measured ghost fraction (Spearman ρ with CI).
- E7b: `LoopedExpertBackbone` T = 6 trained with fixed depth vs learned categorical prior
  (entropy-regularised, budget-penalised, Part II); evaluate with full cache and with terminal-KV
  sharing; metric: accuracy drop; 3 seeds.

### E8 — Multi-host residential WAN GameDay (decides RQ1; the measurement everything waits on)

- 3 machines first, then 8; real tokens from the 8-domain mixture; `K = ⌈M/2⌉`, adaptive grace,
  RDA, PA-CGAD, FoMoE `O_e = 4` for the 8-node run.
- Baselines on the same hardware: single-host synchronous DiLoCo with M workers and with
  `M·(1−churn)` workers, same seeds, same tokens.
- Metrics: held-out loss at matched tokens; p50/p95/p99 outer-round latency; admitted-token
  fraction per round; bytes on the wire per outer round; goodput (useful tokens / wall-clock).
- Controls: positive = single-host run reproduces E5's best cell; negative = a run with the
  learners' network replaced by localhost must match single-host within CI (if it does not,
  the gap is not the network).

### E9 — Adaptive Byzantine evaluation (decides RQ8)

- Attacks: sign-flip (positive control, must be blocked), scaled, ALIE (small coordinated
  perturbation within the honest variance), label-flip on one shard; `f/n ∈ {0.1, 0.2, 0.3}`;
  non-IID shards from the 8-domain mixture.
- Defences: Multi-Krum + 2σ cosine alone; + 10 % spot-check replay; + both.
- Metrics: attack success (held-out loss increase vs no-attack run) and detection rate per
  defence; negative control = honest run through the same filter must lose < 1 % of updates.

## 5. Decision rules (what each outcome changes)

| Outcome | Roadmap consequence | Paper consequence |
| :-- | :-- | :-- |
| P2 refuted | RDA stays optional; add the `√M` outer-LR rule to the campaign config | §7.2 claims only norm bookkeeping, not "no retuning" |
| P3 refuted | delete `AdaptiveGraceWindowController`, keep a fixed admission budget | remove the EMA controller from §7.2 |
| P4: CGAD hurts under quorum | CGAD off by default when `K < M` | E2 reported as a no-quorum result only |
| P5 supported | ghost fraction becomes a logged, gated metric (`HM-S9-02`) | FoMoE cost model quoted with measured factor |
| P6 refuted | AdamW remains default inner optimizer | §7.5 MuLoCo claim limited to conditioning |
| P1 refuted | stop scaling the swarm; investigate which mechanism fails (E5 cells) | the thesis is reported as not supported |
| P8 supported | spot-check replay rate becomes the security parameter, not the filter | §4 rewritten around replay, not Krum |

## 6. Milestones and gates (owned by the engineering coordinator; gates judged by the scientific lead)

| Gate | Content | Evidence required to pass |
| :-- | :-- | :-- |
| **G1** (single-host science) | E5, E6, E7, E7b complete with 3 seeds | four `E<n>_report.json` with receipts; predictions ledger updated; paper v3.3 → v3.4 numbers through provenance lock |
| **G2** (3-node WAN) | E8 at 3 machines | latency and loss artifacts; negative localhost control passes |
| **G3** (8-node WAN + security) | E8 at 8 machines, E9 | RQ1 and RQ8 decided |
| **G4** (external review) | `HM-S9-09`: human or logged-model review with committed transcripts | review transcript in `paper/review/`; no script-emitted score anywhere |

A gate is not passed by a merged PR. It is passed when the artifact named in the evidence column
exists, its receipt matches, and the scientific lead has recorded the prediction status.

## 7. Handoff protocol between the two roles

1. **Scientific lead → coordinator**: a research direction is handed over as an experiment
   spec (`artifacts/experiments/E<n>_prereg.json` + the §4 text). The spec fixes metric, cells,
   seeds, controls and acceptance rule. Engineering may change *how* it is computed, never *what*.
2. **Coordinator → engineering tasks**: the coordinator splits the spec into `HM-S9-xx` tasks,
   assigns branches, runs the 1-hour orchestrator, reviews PRs for the Definition of Done in
   `ENGINEERING_COORDINATION_BRIEF.md`, and merges.
3. **Engineering → scientific lead**: results come back only as artifacts with receipts plus a
   one-page memo: what was run, what deviated from the spec, raw numbers, no interpretation.
4. **Scientific lead → paper**: interpretation, prediction status, and paper text. Numbers are
   pasted only from artifacts and locked with a `PROV-*` check.
5. **Disagreement**: if the coordinator believes a spec is infeasible or a control is wrong, it
   opens a GitHub Discussion tagged `spec-change`; the spec is amended *before* the run, never
   after seeing results.

## 8. Status log

### 2026-10-23 — E5-toy run as pre-registered; E8 two-host pilot

- **E5-toy** (`artifacts/experiments/E5_toy_report.json`, 360 runs, 3 seeds, all three controls passed): P2 **inapplicable** (mean pairwise cosine of learner deltas 0.909 > 0.3, so RDA = direct by construction and the best outer LR sits at the top of the grid for every cell); P3 **inconclusive** (random − adaptive = −0.0004 ± 0.0022 at matched wall budget, 24 pairs); P4 **refuted at this scale** (gate changes loss by −0.00063 ± 0.00029 under quorum; exactly 0 without stale deltas). Every cell lies within 0.011 of every other while the seed interval is ±0.024: the toy LM is a non-discriminating testbed. Consequence: the G1 cell of E5 must run on the `Qwen2.5-0.5B` 8-domain setting (`HM-S9-01`) and must report the delta cosine before any merge or gate claim.
- **E8-pilot** (two real hosts: this workstation and the GCP TPU VM `gwenlaya-tpu-1`, `us-west4-a`, ICMP RTT 151 ms, HTTP RTT through the SSH reverse tunnel ≈ 320 ms; `artifacts/experiments/E8_pilot_report.json`, 3 seeds per cell): (i) with `K = 2` (wait for both learners) the cross-host run reproduces the single-host baseline loss to within 1e-5 at every seed, which validates the shipping of shards and parameters and the merge arithmetic across machines, and gives the first real WAN round latencies (p50 ≈ 4 s for ≈ 2.7 s of local compute); (ii) with `K = 1` + adaptive grace and 400 inner steps, the faster learner (the 24-CPU VM) wins every quorum and the slower one is never merged, because the pilot learner loop is round-synchronous and discards work that misses the round instead of carrying it to the next merge with its token weight as in Decoupled DiLoCo's Algorithm 1; the run ends ≈ 1.0 nats worse than the baseline. This is a defect of the pilot protocol, not a measurement of Decoupled DiLoCo, and it fixes the first requirement of `HM-S9-01`: learners must keep training across rounds and be merged late with token weights. (iii) With 8 inner steps (RTT ≫ compute) the grace window sized from compute-step durations is shorter than one round trip, so the remote learner contributes only through one-round-late deferral; the loss difference to the baseline is within the seed interval of zero.
- Ledger: `artifacts/predictions_ledger.json` (P2 inapplicable, P3 inconclusive, P4 refuted at toy scale, P1/P5–P8 untested).
- Tooling leveraged from the owner's other projects: GWAYA zero-stub audit (`scripts/gwaya_stub_audit.py`, 74 files clean after fixing one `pass`-body hook), a logged local-model review through AutoevolveAI's `APIExtractor` (`scripts/logged_model_review.py`, transcript under `paper/review/`, no score), and the TPU VM as the second WAN host.
