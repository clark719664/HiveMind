"""Server-side creation of QLoRA MoE workunits: tokenised, signed data shards (BOINC ``create_work`` for HiveMind)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hivemind.boinc.workunit import sign_shard_bytes

DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"


def _wikitext_dir() -> Path:
    snaps = Path(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots"))
    return next(snaps.glob("*/wikitext-2-raw-v1"))


def _shard_ids(tokenizer, split: str, lines: slice) -> list[int]:
    import pyarrow.parquet as pq

    root = _wikitext_dir()
    text = "\n".join(t for t in pq.read_table(root / f"{split}-00000-of-00001.parquet").column("text").to_pylist()[lines] if t.strip())
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def build_train_spec(
    wu_id: str,
    shard_index: int = 0,
    model_id: str = DEFAULT_MODEL,
    inner_steps: int = 40,
    segment_steps: int = 10,
    seq_len: int = 128,
    n_seq: int = 64,
    n_eval: int = 8,
    micro_batch: int = 2,
    lr: float = 2e-4,
    rank: int = 8,
    alpha: int = 16,
    rng_seed: int = 7,
    init_seed: int = 42,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from hivemind.boinc.real_client.moe_app import HF_CACHE

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    ids = _shard_ids(tok, "train", slice(shard_index * 3000, (shard_index + 1) * 3000))
    train = [ids[i * seq_len : (i + 1) * seq_len] for i in range(min(n_seq, len(ids) // seq_len))]
    ev_ids = _shard_ids(tok, "validation", slice(0, 2000))
    ev = [ev_ids[i * seq_len : (i + 1) * seq_len] for i in range(min(n_eval, len(ev_ids) // seq_len))]
    spec = {
        "wu_id": wu_id, "model_id": model_id, "inner_steps": inner_steps, "segment_steps": segment_steps,
        "micro_batch": micro_batch, "lr": lr, "weight_decay": 0.01, "rank": rank, "alpha": alpha,
        "rng_seed": rng_seed, "init_seed": init_seed, "shard_index": shard_index,
        "train_tokens": train, "eval_tokens": ev,
    }
    sha, sig = sign_shard_bytes(json.dumps(train, separators=(",", ":")).encode(), rng_seed)
    spec["shard_sha256"], spec["shard_hmac_signature"] = sha, sig
    return spec


def build_infer_spec(wu_id: str, prompts: list[str], model_id: str = DEFAULT_MODEL, max_new_tokens: int = 24, rank: int = 8, alpha: int = 16, init_seed: int = 42) -> dict[str, Any]:
    return {"wu_id": wu_id, "model_id": model_id, "prompts": prompts, "max_new_tokens": max_new_tokens, "rank": rank, "alpha": alpha, "init_seed": init_seed}
