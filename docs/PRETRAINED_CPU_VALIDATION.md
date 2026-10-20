# Bounded pretrained CPU LoRA validation

The separate acceptance tool `tools/training_acceptance/pretrained_cpu.py`
loads real pretrained weights and exercises the repository's
`train_tokenized_domain_adapter`, disjoint holdout checks, saved adapter loader,
and a fresh-backbone reload. It does not alter the HUD, CLI, or package defaults:
the user-facing local smoke test still uses its explicitly labeled random tiny
model. This test closes the **small pretrained CPU execution/reload** gap; broad
model quality remains unverified.

## Model, data, and safety

- Official public model: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
  Apache-2.0, ungated when checked on 2026-10-20.
- Exact revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- The tool requires the pinned safetensors SHA256 and records hashes of every
  loaded model/tokenizer asset. It refuses missing or modified weights.
- Downloads are off by default. `--allow-download` allows about 270 MB of public
  model assets. No authentication token is sent, remote model code is disabled,
  and no user data, fixture text, or adapter is uploaded.
- Twelve original Rust snippets are fixed in the script before training. The
  deterministic split produces nine training examples and three heldouts, with
  exact-text, model-visible-token, and strict-prefix leakage rejection.
- This is FP32 CPU LoRA on a bounded 135M model, not the production streaming
  NF4 loader or QLoRA. It is not evidence for the production memory ceilings.
- Default run: 16 steps, seed 42, two CPU threads, maximum 128 tokens, rank 4,
  alpha 8, zero dropout, `q_proj` and `v_proj`, AdamW learning rate 0.001.
  A second independent initialization/training run checks exact reproducibility.
  Models are released between stages. Steps and thread count are bounded.

## Reproduce

Run from the repository root in the normal activated Python environment with
the repository dependencies installed. CPU-only PyTorch is sufficient.

```bash
# Explicit first public model download and complete acceptance run.
python tools/training_acceptance/pretrained_cpu.py --allow-download --out artifacts/pretrained_cpu_validation.json

# Subsequent runs use the cache and never download by default.
python tools/training_acceptance/pretrained_cpu.py --out artifacts/pretrained_cpu_validation_offline.json

# Fast unit/regression coverage; no model download or pretrained weights needed.
python -m pytest -q tests/test_pretrained_cpu_validation.py tests/test_actual_text_training.py tests/test_user_facing_real_training.py
```

Both run commands work unchanged in an activated Windows PowerShell environment.
Use `--cache-dir PATH` for an existing cache and `--threads 1` for less CPU usage.
Missing cache assets or any execution failure produce `ERROR` and a nonzero
exit, with no random-model fallback. A prior success report at the requested
output path is replaced by the current error rather than left as stale evidence.
The saved adapter is next to the JSON with suffix `.adapter.pt`. Keep its JSON:
the raw adapter file alone does not authenticate its backbone or tokenizer.

## Recorded result

Full machine-readable evidence:
[`evidence/pretrained_cpu_validation_2026-10-20.json`](evidence/pretrained_cpu_validation_2026-10-20.json).
It was produced from repository training code at `5890c50d87068e6ccc463591e5e6249899dd0d50`
plus the acceptance tool in this change, using Torch 2.6.0+cpu, Transformers
4.57.6, PEFT 0.17.1, and Python 3.12.14 on Linux. The JSON records the exact
acceptance script SHA256, source model hashes, configuration, and every loss.

- Real trainable LoRA parameters: 230,400; base parameters: 134,515,008.
- Three-example mean heldout causal-LM cross-entropy: **2.628913 → 1.416782**,
  a **46.11% reduction** on this fixture.
- Training/heldout tokens were disjoint; frozen backbone hashes were unchanged.
- Fresh-backbone adapter reload reproduced heldout loss and logits exactly
  (maximum absolute logit difference 0.0).
- Independent seeded replay reproduced adapter tensor hashes and training
  losses exactly on the recorded environment.

`PASS` means the execution, immutability, holdout, checkpoint, and reproducibility
checks passed. Heldout improvement is reported separately; it is not required to
pass plumbing checks. Do not tune repeatedly on these heldouts and then present
the best result as an independent evaluation.

## Limits and remaining acceptance

These short snippets share Rust syntax and concepts. Three holdouts are too few
to support a generalization claim, and whether the pretraining corpus contained
similar code is unknown. Lower cross-entropy does not establish correct code,
useful generation, or production expert quality. A representative, independently
held-out task suite with execution/correctness metrics is still needed for those
claims; it does not intrinsically require the user's PC.

This tool awards no rewards and performs no settlement or validator certification.
It does not validate CUDA/NF4, VRAM, desktop game preemption, WAN peers, Windows
packaging, or Steam integration. Those remain governed by their separate
acceptance runbooks. The pretrained CPU run itself no longer requires a PC
handoff: it has already completed in the cloud.
