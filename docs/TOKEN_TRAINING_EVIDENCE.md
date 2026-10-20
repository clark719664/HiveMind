# Reproducible token-training and local-transfer smoke test

## What this adds, and what it does not

This is one bounded evidence milestone: a real causal-language-model loss over
actual text tokens, before training, after LoRA training, and after importing the
serialized adapter into a fresh model. It is **not** a pretrained-model quality
benchmark, a representative Rust evaluation, a QLoRA/compression result, a GPU
performance test or proof of a deployed swarm.

The model is `hivemind/tiny-qwen2-byte-random-v1`: a local, random-initialized
one-layer Qwen2, hidden size 32, intermediate size 64, four attention heads, two
key/value heads, 259-token vocabulary, no dropout, eager attention, CPU float32.
LoRA rank 4/alpha 8 targets q_proj, v_proj and lm_head. The architecture, seed,
dependency versions, exact serialized base weights and their SHA-256 are recorded.
No Hugging Face model or dataset is downloaded, and no remote code is executed.
The random initialization is disclosed rather than passed off as pretraining.

The six repository-authored Rust snippets come from upstream revision
`84fa03c06664fbf1fa357894f79b0e6e6a2f48d7`. Three fixed documents train; three
separate fixed documents evaluate. These are real text inputs, **not** an external
HF dataset. The fixture has a pinned SHA-256 and rejects duplicate document IDs
or whitespace-normalized text. A fixed UTF-8 byte tokenizer has no fitted
vocabulary. BOS is context only, EOS is scored, padding is ignored, and overlapping
128-token windows score every target once, resetting context at each window.
Token cross-entropy is summed and divided by the actual scored-token count.
Perplexity is exp(mean token NLL), not exp(MSE). Byte-token perplexity must not be
compared directly with another tokenizer's perplexity.

Training runs a fixed 24 full-batch AdamW steps at learning rate 0.01. The holdout
is not used in optimization, scheduling, early stopping or checkpoint selection.
The report keeps signed improvements, including regressions and zero-step
no-improvement outcomes. A passing mechanics test does not require an improvement.

## Run without model downloads

From the repository root, with Python 3.11 or 3.12 and uv installed:

```bash
uv venv .venv-token
uv pip install --python .venv-token/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
uv pip install --python .venv-token/bin/python -r requirements/token-training-eval.txt
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-token/bin/python -m hivemind.data.token_training_eval --output-dir /tmp/hivemind-token-run-1
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-token/bin/python -m hivemind.data.token_training_eval --output-dir /tmp/hivemind-token-run-2
```

Use new empty output directories. Each run saves `base_model.safetensors`,
`adapter.safetensors`, the two local CAS directories and `report.json`. No pickle
is used by the new transfer path. Import verifies the full adapter SHA-256 before
safe decoding, then requires exact parameter names, shapes, dtypes and finite
values. The base weights are checked for immutability during training. The
recipient reloads those exact saved base weights and performs no optimization.

The two CAS stores and two models live in the **same process**. Copying their
bytes is a local storage/transfer test, not networking. It intentionally does not
claim that the legacy `.pt` checkpoint loader verifies its stored CID; that is a
separate known limitation, not silently fixed by this new path.

Compare the two reports' base/model/data digests, `holdout`, `transfer`, and
training step losses. Those should match in the pinned environment. Measured wall
time and platform strings may differ. Do not claim cross-device or cross-version
bitwise reproducibility without running it.

## Claim-to-evidence contract

| Claim | Evidence / status |
| --- | --- |
| The model trains on text content | Actual byte IDs/next-token targets, not Gaussian targets; regression test changes input text and observes different IDs |
| The holdout is separate | Fixed document-level split, pinned fixture hash, duplicate rejection, no holdout-driven optimizer step |
| Loss/perplexity are real token metrics | Shifted, non-padding token cross-entropy with exact target counts; hand-calculated metric regression test |
| Adapter learning occurred | Initial/final adapter digests and train-loss history; no claim that every run improves holdout |
| Local import preserves the adapter | Actual serialized size/hash, fresh same-base recipient, exact parameter digest and holdout loss comparison |
| Training did not alter the base | Frozen-parameter hashes before and after |
| Parallel speedup | Unavailable: no concurrent, equal-work timing experiment |
| GPU/hardware validation | Not run: this path is CPU only |
| Multi-process/multi-host/WAN transfer | Not run |
| Rust mastery, domain generalization or frontier ability | Not established by six snippets and a random tiny model |

## Legacy output compatibility

The names `hf_real_trainer`, `SwarmKnowledgeTransferEngine`, and historical JSON
keys remain to avoid gratuitous call-site breakage. Their synthetic path now
states `measurement_kind: synthetic_regression`, MSE, and that text is unused.
Undefined perplexity is null. Unmeasured GPU steps saved, mastery, compute
leverage and parallel/equal-time quality claims are null rather than fabricated.
Consumers must handle nulls; the generated HUD is updated accordingly.

Simulation counters are named separately. A historical trainer result mixes
92% raw trained weights with 8% reconstructed weights, so its compression
accounting is explicitly not a compression-only/wire-transfer result. Existing
historical JSON artifacts have not been retroactively rebranded as measurements.

Next evidence milestones should use a vetted pretrained checkpoint and a larger
fixed task/holdout, then genuinely separate processes and hosts. Each needs its
own configuration, measured metrics, actual bytes and failure reports. Neither
is established by this smoke test.

## Recorded reproduction (October 20, 2026)

[Raw reports from two independent processes](../artifacts/token_training_smoke_reproduction.json)
were generated in a fresh virtual environment using the setup recipe above.
Both runs had identical base/data/implementation hashes, step losses, held-out
metrics and adapter bytes. Timing is measured separately and is not a speedup.

- 27,564 total parameters, 1,612 trainable LoRA parameters.
- 556 training targets and 436 held-out targets.
- Held-out next-token NLL: **5.5764014568 → 3.7811688064 nats**.
- Byte-token perplexity: **264.1194484 → 43.8672841**.
- Fresh recipient NLL: **3.7811688064 nats**, with **0** recipient training steps.
- Actual adapter safetensors size: **7,192 bytes**, including format overhead.
- Adapter SHA-256: `c9d4bb8349d54629943bcf3acbfaabbed8c5844230cf8b65e6d66e816f9b0975`.
- Serialized base size: **105,272 bytes**; base SHA-256:
  `a20fe964f665dd254732deb608e704fb0e43d3dddae5ebd191f791d0bbfda415`.

This observed improvement on three tiny held-out snippets is not evidence of
Rust correctness, representative generalization or useful pretrained performance.
It establishes that the bounded text-training and local adapter-import path can
be reproduced, rather than requiring an impressive result by construction.
