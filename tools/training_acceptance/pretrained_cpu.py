#!/usr/bin/env python
"""Opt-in, pinned pretrained CPU LoRA validation; never a product-quality certificate.

Downloads are disabled unless --allow-download is supplied. Only public, fixed
SmolLM2 safetensors and tokenizer assets are used; no remote model code executes.
The small FP32 CPU model here is a bounded test, not the production NF4 loader.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
ASSETS = ["config.json", "generation_config.json", "model.safetensors",
          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "vocab.json", "merges.txt"]
# Original, public test snippets. These are a tiny fixture, not a benchmark or
# proof of domain correctness. The ordered split is fixed before any training.
CORPUS = [
    'pub fn checked_sum(a: u64, b: u64) -> Option<u64> { a.checked_add(b) }',
    'pub fn first_byte(data: &[u8]) -> Option<u8> { data.first().copied() }',
    'pub fn last_byte(data: &[u8]) -> Option<u8> { data.last().copied() }',
    'pub fn checked_product(a: u64, b: u64) -> Option<u64> { a.checked_mul(b) }',
    'pub fn checked_difference(a: u64, b: u64) -> Option<u64> { a.checked_sub(b) }',
    'pub fn read_byte(data: &[u8], index: usize) -> Option<u8> { data.get(index).copied() }',
    'pub fn prefix(data: &[u8], count: usize) -> Option<&[u8]> { data.get(..count) }',
    'pub fn suffix(data: &[u8], start: usize) -> Option<&[u8]> { data.get(start..) }',
    'pub fn valid_index(data: &[u8], index: usize) -> bool { index < data.len() }',
    'pub fn checked_quotient(a: u64, b: u64) -> Option<u64> { a.checked_div(b) }',
    'pub fn middle(data: &[u8], start: usize, end: usize) -> Option<&[u8]> { data.get(start..end) }',
    'pub fn checked_remainder(a: u64, b: u64) -> Option<u64> { a.checked_rem(b) }',
]
LIMITATIONS = [
    "Twelve hand-written Rust snippets with three held-out examples are not a representative benchmark.",
    "Train/holdout texts and token sequences are disjoint, but share syntax and concepts; pretraining contamination is unknown.",
    "Cross-entropy on this fixture does not establish code correctness, useful generation, or broad domain generalization.",
    "CPU FP32 LoRA does not validate CUDA, NF4/QLoRA, VRAM limits, game preemption, WAN peers, or Steam integration.",
    "No validator certification, reward, settlement, or production-readiness claim is made.",
    "Model weights are downloaded to a local cache; no training text or adapter is uploaded.",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot(cache_dir: Path, allow_download: bool) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir),
        allow_patterns=ASSETS, local_files_only=not allow_download,
        token=False, max_workers=2,
    ))
    missing = [name for name in ASSETS if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete pinned model cache: {', '.join(missing)}")
    if file_sha256(path / "model.safetensors") != MODEL_WEIGHTS_SHA256:
        raise ValueError("Pinned public model safetensors SHA256 mismatch")
    return path


def build_model(path: Path, seed: int):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from hivemind.config import configure_deterministic_runtime

    configure_deterministic_runtime(seed=seed, warn_only=False)
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cpu")
    base.config.use_cache = False
    return get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))


def evaluate(model, tokenizer, texts, max_length: int):
    import torch

    model.eval()
    losses, logits = [], []
    with torch.no_grad():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            result = model(**batch, labels=batch["input_ids"])
            losses.append(float(result.loss))
            logits.append(result.logits.detach().cpu().clone())
    return sum(losses) / len(losses), logits


def run_validation(*, cache_dir: Path, allow_download: bool, output: Path,
                   steps: int = 16, seed: int = 42, threads: int = 2) -> dict:
    import peft
    import torch
    import transformers
    from transformers import AutoTokenizer
    from hivemind.data.hf_real_trainer import (
        load_tokenized_adapter_state, split_tokenized_corpus, train_tokenized_domain_adapter,
    )

    if not 1 <= steps <= 64 or not 1 <= threads <= 8:
        raise ValueError("steps must be 1..64 and threads 1..8 for this bounded validation")
    started = time.monotonic()
    torch.set_num_threads(threads)
    path = snapshot(cache_dir, allow_download)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    train, heldout = split_tokenized_corpus(tokenizer, CORPUS, max_length=128, heldout_fraction=0.25)
    model = build_model(path, seed)
    frozen_before = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_path = output.with_suffix(".adapter.pt")
    state, training = train_tokenized_domain_adapter(
        model, tokenizer, train, heldout_samples=heldout, steps=steps,
        max_length=128, learning_rate=1e-3, adapter_save_path=adapter_path,
    )
    frozen_after = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    trained_digest = tensor_digest(state.items())
    trained_loss, trained_logits = evaluate(model, tokenizer, heldout, 128)
    del model
    gc.collect()
    fresh = build_model(path, seed)
    load_tokenized_adapter_state(fresh, adapter_path)
    reload_loss, reload_logits = evaluate(fresh, tokenizer, heldout, 128)
    reload_max_abs = max(float((left - right).abs().max()) for left, right in zip(trained_logits, reload_logits))
    fresh_frozen = tensor_digest((n, p) for n, p in fresh.named_parameters() if not p.requires_grad)
    del fresh, trained_logits, reload_logits
    gc.collect()
    replay = build_model(path, seed)
    replay_state, replay_training = train_tokenized_domain_adapter(
        replay, tokenizer, train, heldout_samples=heldout, steps=steps,
        max_length=128, learning_rate=1e-3,
    )
    deterministic_replay = (
        trained_digest == tensor_digest(replay_state.items())
        and training["training_losses"] == replay_training["training_losses"]
        and training["heldout_final_loss"] == replay_training["heldout_final_loss"]
    )
    checks = {
        "adapter_parameters_changed": training["adapter_parameters_changed"],
        "heldout_disjoint_verified": training["heldout_disjoint_verified"],
        "checkpoint_round_trip_verified": training["adapter_reloaded_and_verified"],
        "frozen_backbone_unchanged": frozen_before == frozen_after == fresh_frozen,
        "fresh_model_reload_verified": reload_max_abs <= 1e-6 and math.isclose(
            reload_loss, training["heldout_final_loss"], rel_tol=1e-6, abs_tol=1e-6),
        "deterministic_replay_verified": deterministic_replay,
        "finite_heldout_metrics": all(math.isfinite(x) for x in [
            training["heldout_initial_loss"], training["heldout_final_loss"], trained_loss, reload_loss]),
    }
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_script_sha256": file_sha256(Path(__file__)),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "bounded_pretrained_cpu_lora_fixture_validation",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        "model_assets_sha256": {name: file_sha256(path / name) for name in ASSETS},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": peft.__version__, "device": "cpu", "dtype": "float32", "threads": threads},
        "configuration": {"seed": seed, "steps": steps, "max_length": 128,
                          "learning_rate": 1e-3, "lora_rank": 4, "lora_alpha": 8,
                          "lora_targets": ["q_proj", "v_proj"], "lora_dropout": 0.0},
        "parameter_count": parameter_count, "trainable_parameter_count": trainable_count,
        "training": training, "checks": checks,
        "fresh_reload_heldout_loss": reload_loss, "fresh_reload_max_abs_logit_difference": reload_max_abs,
        "adapter_tensor_sha256": trained_digest, "frozen_backbone_tensor_sha256": frozen_before,
        "fixture_quality_result": "heldout_loss_improved" if training["heldout_improved"] else "no_heldout_loss_improvement",
        "production_quality_verified": False, "runtime_validator_certified": False,
        "settlement_executed": False, "rewards_awarded": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3), "limitations": LIMITATIONS,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow public pinned model download (about 270 MB)")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache" / "pretrained-validation")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "pretrained_cpu_validation.json")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = run_validation(cache_dir=args.cache_dir, allow_download=args.allow_download,
                                output=args.out, steps=args.steps, seed=args.seed, threads=args.threads)
    except Exception as exc:
        # A missing model or failed run is never replaced by random/simulated training.
        report = {"schema_version": 1, "status": "ERROR", "model_id": MODEL_ID,
                  "model_revision": MODEL_REVISION, "error": f"{type(exc).__name__}: {exc}",
                  "production_quality_verified": False, "limitations": LIMITATIONS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
