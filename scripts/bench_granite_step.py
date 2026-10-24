"""Microbenchmark (GPU, ~5 min): seconds per inner step of the E13 training step on the Granite MoE models for each experts
implementation and with/without gradient checkpointing. It decides the E13 Granite sizing (prereg: the lead measures step times
on the first S0 run and amends the sizing if the projection exceeds the budget). Random tokens: timing only, not evidence.

    python3 scripts/bench_granite_step.py [--out artifacts/experiments/E13_granite_step_bench.json]
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import e13_swarm_lib as lib  # noqa: E402

CONFIGS = [
    ("granite1b", "eager", True),
    ("granite1b", "eager", False),
    ("granite1b", "grouped_mm", False),
    ("granite1b", "default", False),
    ("granite3b", "eager", False),
    ("granite3b", "eager", True),
]
MICRO_BATCH, SEQ, WARMUP, TIMED = 4, 256, 2, 6


def run(key: str, impl: str, ckpt: bool) -> dict:
    lib.configure_determinism()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model, _tok, params = lib.build_model(key, "cuda", 1, grad_ckpt=ckpt, experts_impl=None if impl == "default" else impl)
    load_s = time.perf_counter() - t_load
    opt = lib.make_opt(params)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    times = []
    for i in range(WARMUP + TIMED):
        b = torch.randint(0, vocab, (MICRO_BATCH, SEQ), generator=g).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(input_ids=b, labels=b, use_cache=False).loss
        opt.zero_grad(set_to_none=True)
        (loss * lib.LOSS_SCALE_FP16).backward()
        for p in params:
            if p.grad is not None:
                p.grad.div_(lib.LOSS_SCALE_FP16)
        torch.nn.utils.clip_grad_norm_(params, lib.GRAD_CLIP)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    timed = times[WARMUP:]
    res = {"model": key, "experts_impl": impl, "grad_ckpt": ckpt, "sec_per_step_median": statistics.median(timed), "sec_per_step_all": [round(x, 3) for x in times],
           "load_s": round(load_s, 1), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2), "finite_loss": bool(torch.isfinite(loss)),
           "resolved_experts_impl": getattr(model.config, "_experts_implementation", None)}
    del model, opt, params, loss
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "artifacts" / "experiments" / "E13_granite_step_bench.json"))
    a = ap.parse_args()
    results = []
    for key, impl, ckpt in CONFIGS:
        try:
            r = run(key, impl, ckpt)
        except Exception as exc:  # one configuration failing (OOM, unsupported kwarg) must not stop the others
            r = {"model": key, "experts_impl": impl, "grad_ckpt": ckpt, "error": repr(exc)[:300], "traceback_tail": traceback.format_exc().splitlines()[-3:]}
            gc.collect()
            torch.cuda.empty_cache()
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "sec_per_step_all"}), flush=True)
        Path(a.out).write_text(json.dumps({"purpose": "timing only (random tokens), decides the E13 Granite sizing", "micro_batch": MICRO_BATCH, "seq_len": SEQ,
                                           "gpu": torch.cuda.get_device_name(0), "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
