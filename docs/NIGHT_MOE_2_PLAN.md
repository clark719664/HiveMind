# NIGHT-MOE-2: next training run for the Bee MoE experts (QLoRA, larger models, larger GPU)

Status: **draft, not preregistered.** Freeze it (as `artifacts/experiments/NIGHT_MOE_2_prereg.json`) before the first training step.
Evidence base: NIGHT-MOE-1 (`docs/NIGHT_MOE_RESULTS.md`, `docs/NIGHT_MOE_TRAINING_PLAN.md`), one night on one T4, 2026-10-24/25.
Every number below is from a summary or journal file of that run unless it is marked **estimate** or **unverified**.

## 1. What NIGHT-MOE-1 showed

| Finding | Evidence | Consequence for the next run |
|---|---|---|
| Label tasks with a clean signal train well | stars 0.489 -> 0.893 (n=317, p<1e-4); adapter beats its own shuffled control 193 to 7 | keep QLoRA adapters for classification-style experts |
| The preregistered VOID rule is stricter than the evidence it protects | exoplanet: control also gained (27 vs 12 only-right, p=0.0237), yet adapter beats control 309 to 17 (post-hoc, 600 items) | the shuffled control keeps the label marginals, so it learns the prior and the answer format; preregister the **adapter-vs-control** paired test as the primary decision and keep adapter-vs-base as secondary |
| Supervised fine-tuning on terse gold maths solutions hurt a model that already reasons | maths 0.785 -> 0.715; on the 173 items both parsed, adapter 130 vs base 157 correct (only-adapter 6, only-base 33, p<1e-4); it fixed format on 27 base-unparsable items (13 right) | **the format caveat does not rescue it: the loss is real.** Train maths on the model's own verified solutions (rejection-sampled, as in E15), not on gold text |
| An adapter trained on one task can break another task's format | maths adapter: 155/488 unparsable on physics_transfer (0.541 -> 0.342), 54 and 30 unparsable on astro and cyber subsets | evaluate routed behaviour, not adapters in isolation (section 4); always report unparsable counts |
| A control can be unable to fail | maths shuffled control scored 0/100 by construction (a shuffled numeric answer is never right) | for generative tasks use a **cross-domain adapter control** (same recipe, different domain's data) so the control can score above zero |
| Free-text expert with unverified targets is a null | cyber 0.660 vs 0.650 (p=0.383), control 0.658; adapter vs control 11 vs 10 | do not spend GPU hours on unverified LLM text; use verified targets or drop the subject |
| Interference was only sampled | 17 cells, n=100 each, `eval_maths` timed out (720 s) for three adapters; maths->exoplanet +20, ->stars +14 are uncontrolled | full-size interference sets, longer per-stage caps, a control adapter in every cell that is reported as a finding |
| The night was compute-capped, not epoch-complete | 7 training stages ended `OK_TRUNCATED` at their time cap | size caps from measured step time; cyber ran about 33 s/step on the T4 at 10.23 GiB peak |
| "MoE" meant a set of adapters | no mixture, router or gate was trained or tested | NIGHT-MOE-2 evaluates the routed system |
| Disk is the hard limit | VM data disk 490 of 492 GB used (1.6 GB free) | provision space before training; do not rely on deleting data |

## 2. Hypotheses and decision rules (to preregister)

- **H1 (expert value):** a domain adapter beats a cross-domain-recipe control on its own pack: exact paired McNemar p<0.05 and a positive difference, n>=300 per pack. Primary test.
- **H2 (not just format):** the adapter also beats the base model on the items the base parsed. Secondary; report both unparsable counts.
- **H3 (routing):** the routed system (gate chooses the adapter per prompt) scores at least as well as the best single adapter per pack and does not lose more than a preregistered margin on any other pack. Primary test for "MoE".
- **H4 (scale):** the same recipe on the larger base improves H1 effect sizes. Descriptive only: two models, one seed each.
- Multiplicity: one family for all transfer/interference cells, Bonferroni, fixed before the run. Full pack sizes, no `--limit`.
- Seeds: at least 2 training seeds for any subject that reports a headline gain.

## 3. Models and GPU

| Option | Weights (4-bit, Ollama library sizes in `hivemind.router.catalog`) | QLoRA peak memory | Where it runs |
|---|---|---|---|
| Qwen2.5-Coder-7B (NIGHT-MOE-1) | 4.7 GB | **10.23 GiB measured** | T4 16 GB, the VM we have |
| Qwen2.5-Coder-14B | 9.0 GB | **estimate** 14-15 GiB (7B measurement plus the extra 4.3 GB of weights at equal sequence length and batch) | T4 16 GB barely; L4 24 GB comfortably |
| gpt-oss-20B (MoE, 3.6B active) | 14 GB | **unverified** | 24 GB or larger; training support for its MXFP4 weights must be smoke-tested |
| Qwen3-Coder-30B-A3B (MoE, 3.3B active) | 19 GB | **estimate** 25 GiB or more | 40 GB or larger; QLoRA on fused MoE expert layers is **unverified**, run a 1-step smoke test first |

GCP quota read from `us-central1` on 2026-10-25: `NVIDIA_T4_GPUS` 1, `NVIDIA_L4_GPUS` 1 (preemptible 3), `NVIDIA_A100_GPUS` 0, `NVIDIA_A100_80GB_GPUS` 0, no H100 metric listed. So the next step that needs no new quota is **Qwen2.5-Coder-14B on one L4 (24 GB)**; anything with 40 GB or more needs a quota request that the owner must make.
A T4 has no native bf16; an L4 does. TPU v5e (`gwenlaya-tpu-1`) cannot run bitsandbytes QLoRA (CUDA only), so it is not a QLoRA option.

Recommended sequence: (1) 14B on an L4, same four subjects, new controls and metrics; (2) smoke-test the 20B and 30B MoE bases for 1 training step; (3) only then request A100 40/80 GB quota for the 30B run. Each step is a separate decision for the owner.

## 4. Experiment design changes

1. **Training data.** maths: rejection-sampled self-solutions verified against GSM8K answers (no gold text). exoplanet/stars: keep, split by star, add the classical logistic-regression and temperature-rule baselines to every table. cyber: only targets with a verifier (CTI-Bench style labels, MMLU computer-security MCQ items from the train split), otherwise drop. Add a code subject with executable tests, because the router's customers use code.
2. **Controls.** For every subject: (a) shuffled-label control; (b) cross-domain adapter control; decision uses (b) for generative tasks and the adapter-vs-control test for label tasks.
3. **Evaluation.** Full packs, per-stage caps sized from measured time (maths eval needed more than 720 s), report accuracy on base-parsable items and unparsable counts side by side.
4. **Routed MoE.** Gate options, in order of cost: keyword/classifier router (baseline), embedding router, learned sigmoid gate over adapter outputs (the La Ruche SigmaMoE design). Measure routing accuracy first, then end-to-end accuracy. Serve only adapters that pass H1 and the verified update gate.
5. **Operations.** A fresh data disk sized for 4 subjects x (adapter + GGUF + control) plus the base; drop the producer only inside the orchestrator's guarded window; keep the restore step and the exit trap; keep the monitor cron; record the VM disk before and after.

## 5. Open decisions for the owner

- Provision a new or larger data disk (the current one has 1.6 GB free).
- Which compute: one L4 (no quota change), or request A100 quota for the 30B MoE.
- Approve the cyber change (verified targets only, or drop the subject).
- Which base the router serves for coding in production while the experiments run.

## 6. Not claimed

No adapter from NIGHT-MOE-1 is served by the router. The 14B, 20B and 30B memory figures are extrapolations or unverified. Nothing here has been run.
