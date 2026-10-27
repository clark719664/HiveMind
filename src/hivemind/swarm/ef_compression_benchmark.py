# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S6-01 synthetic benchmark: Top-k + FP8 outer-loop compression with and without error feedback.

This is a small, deterministic CPU experiment on a synthetic multi-worker least-squares
problem. It measures how compression of DiLoCo-style pseudo-gradients changes holdout loss
for a fixed number of outer rounds, and how many bytes each scheme would put on the wire.
It does not train a language model, use real network links, or measure wall-clock speed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from hivemind.core.compression import compress_lora_delta, decompress_lora_delta
from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer

PARAM_NAME = "proj.weight"
SCHEMES = ("dense", "top5_fp16", "top1_fp8", "top1_fp8_ef")


@dataclass(frozen=True)
class SyntheticSwarmProblem:
    w_true: torch.Tensor
    worker_x: List[torch.Tensor]
    worker_y: List[torch.Tensor]
    holdout_x: torch.Tensor
    holdout_y: torch.Tensor


def build_problem(
    seed: int = 0,
    num_workers: int = 4,
    samples_per_worker: int = 256,
    d_in: int = 64,
    d_out: int = 16,
    noise_std: float = 0.1,
) -> SyntheticSwarmProblem:
    """Shared linear target; each worker sees inputs with its own feature scaling (non-IID)."""
    gen = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_out, d_in, generator=gen) / d_in**0.5
    worker_x: List[torch.Tensor] = []
    worker_y: List[torch.Tensor] = []
    for _ in range(num_workers):
        feature_scale = torch.rand(d_in, generator=gen) * 1.5 + 0.25
        x = torch.randn(samples_per_worker, d_in, generator=gen) * feature_scale
        y = x @ w_true.T + noise_std * torch.randn(samples_per_worker, d_out, generator=gen)
        worker_x.append(x)
        worker_y.append(y)
    holdout_x = torch.cat([torch.randn(256, d_in, generator=gen) for _ in range(num_workers)])
    holdout_y = holdout_x @ w_true.T + noise_std * torch.randn(holdout_x.shape[0], d_out, generator=gen)
    return SyntheticSwarmProblem(w_true, worker_x, worker_y, holdout_x, holdout_y)


def _mse(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(((x @ w.T - y) ** 2).mean().item())


def _local_steps(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    w = w.clone()
    n = x.shape[0]
    for _ in range(steps):
        grad = 2.0 / n * (x @ w.T - y).T @ x / y.shape[1]
        w = w - lr * grad
    return w


def _topk_fp16(tensor: torch.Tensor, topk_ratio: float) -> tuple[torch.Tensor, int]:
    """Top-k sparsification with FP16 values and uint16 indices (no FP8, no error feedback)."""
    flat = tensor.reshape(-1)
    k = max(1, int(round(flat.numel() * topk_ratio)))
    _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=True)
    out = torch.zeros_like(flat)
    out[idx] = flat[idx].to(torch.float16).to(torch.float32)
    return out.reshape(tensor.shape), k * 2 + k * 2 + 8


def run_benchmark(
    seed: int = 0,
    num_workers: int = 4,
    outer_rounds: int = 200,
    inner_steps: int = 8,
    inner_lr: float = 0.05,
    outer_lr: float = 1.0,
    schemes: Sequence[str] = SCHEMES,
    log_every: int = 25,
) -> Dict[str, Any]:
    """Run every scheme on the same problem and starting point; return losses and wire bytes."""
    unknown = set(schemes) - set(SCHEMES)
    if unknown:
        raise ValueError(f"unknown schemes: {sorted(unknown)}")
    problem = build_problem(seed=seed, num_workers=num_workers)
    w0 = torch.zeros_like(problem.w_true)
    results: Dict[str, Any] = {}

    for scheme in schemes:
        w = w0.clone()
        buffers = [EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.01) for _ in range(num_workers)]
        wire_bytes = 0
        curve = []
        for step in range(1, outer_rounds + 1):
            sent = []
            for k in range(num_workers):
                w_local = _local_steps(w, problem.worker_x[k], problem.worker_y[k], inner_steps, inner_lr)
                delta = {PARAM_NAME: w - w_local}
                if scheme == "dense":
                    sent.append(delta[PARAM_NAME])
                    wire_bytes += delta[PARAM_NAME].numel() * 2
                elif scheme == "top5_fp16":
                    approx, nbytes = _topk_fp16(delta[PARAM_NAME], 0.05)
                    sent.append(approx)
                    wire_bytes += nbytes
                elif scheme == "top1_fp8":
                    pkg = compress_lora_delta(delta, f"w{k}", "bench", step, inner_steps, topk_ratio=0.01)
                    sent.append(decompress_lora_delta(pkg)[PARAM_NAME])
                    wire_bytes += pkg.compressed_bytes
                else:
                    pkg = buffers[k].compress(delta, f"w{k}", "bench", step, inner_steps)
                    sent.append(decompress_lora_delta(pkg)[PARAM_NAME])
                    wire_bytes += pkg.compressed_bytes
            w = w - outer_lr * torch.stack(sent).mean(dim=0)
            if step % log_every == 0 or step == outer_rounds:
                curve.append({"outer_round": step, "holdout_mse": round(_mse(w, problem.holdout_x, problem.holdout_y), 6)})
        results[scheme] = {
            "final_holdout_mse": round(_mse(w, problem.holdout_x, problem.holdout_y), 6),
            "wire_bytes_all_workers": int(wire_bytes),
            "curve": curve,
        }

    report: Dict[str, Any] = {
        "benchmark": "HM-S6-01 synthetic error-feedback compression",
        "scope": (
            "Synthetic linear least-squares on CPU; non-IID feature scaling per worker. "
            "Not a language model, not a real network, no timing."
        ),
        "config": {
            "seed": seed,
            "num_workers": num_workers,
            "outer_rounds": outer_rounds,
            "inner_steps": inner_steps,
            "inner_lr": inner_lr,
            "outer_lr": outer_lr,
            "parameters": int(w0.numel()),
        },
        "initial_holdout_mse": round(_mse(w0, problem.holdout_x, problem.holdout_y), 6),
        "noise_floor_mse": round(_mse(problem.w_true, problem.holdout_x, problem.holdout_y), 6),
        "results": results,
    }
    if "top5_fp16" in results and "top1_fp8_ef" in results:
        ref = results["top5_fp16"]
        ef = results["top1_fp8_ef"]
        report["comparison_top1_fp8_ef_vs_top5_fp16"] = {
            "holdout_mse_gap": round(ef["final_holdout_mse"] - ref["final_holdout_mse"], 6),
            "wire_bytes_reduction_x": round(ref["wire_bytes_all_workers"] / ef["wire_bytes_all_workers"], 3),
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-rounds", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_benchmark(seed=args.seed, outer_rounds=args.outer_rounds)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0
