"""Experiment E12: spot-check replay divergence across hardware (RQ11, HM-S10-08; external review of v3.5.0, W1).

Pre-registered in ``artifacts/experiments/E12_replay_divergence_prereg.json``. The honest worker and every validator
are the repository's own ``hivemind.core.diloco_inner.replay_inner_segment`` (steps 0 -> 50) on Qwen2.5-0.5B with
rank-8 LoRA on q_proj / v_proj. Per-step snapshots, the fp64 grid-snapping proxy and the positive-control tamper
are applied through PyTorch's global optimizer post-step hook, so the replay code under test is unchanged.

    python3 scripts/run_experiment_E12_replay_divergence.py            # all cells on this host
    python3 scripts/run_experiment_E12_replay_divergence.py --worker --device cuda --dtype float32 --out f.pt
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivemind.core.diloco_inner import replay_inner_segment  # noqa: E402
from hivemind.core.quantization import extract_lora_state_dict  # noqa: E402

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = EXP_DIR / "E12_replay_divergence_prereg.json"
REPORT_PATH = EXP_DIR / "E12_replay_divergence_report.json"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
RNG_SEEDS = (101, 202, 303)
INIT_SEEDS = (1, 2, 3)
STEPS = 50
SNAP_EVERY = 10
EPS = 1e-4
RANK, ALPHA = 8, 16.0
N_SEQ, SEQ_LEN = 64, 128
SNAP_GRID = 2.0**-20
TAMPER_STEP = 25


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, gen: torch.Generator) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        a = torch.randn(r, base.in_features, generator=gen) / math.sqrt(r)
        self.lora_A = nn.Parameter(a.to(base.weight.dtype).to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def build_model(device: str, dtype: torch.dtype, init_seed: int) -> nn.Module:
    from transformers import AutoModelForCausalLM

    path = sorted(glob.glob(str(HUB / ("models--" + REPO.replace("/", "--")) / "snapshots" / "*")))[0]
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(init_seed)
    for layer in m.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, RANK, ALPHA, gen)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, RANK, ALPHA, gen)
    return m


def token_sequences() -> torch.Tensor:
    import pandas as pd
    from transformers import AutoTokenizer

    path = sorted(glob.glob(str(HUB / ("models--" + REPO.replace("/", "--")) / "snapshots" / "*")))[0]
    tok = AutoTokenizer.from_pretrained(path)
    pq = sorted(glob.glob(str(HUB / "datasets--Salesforce--wikitext" / "snapshots" / "*" / "wikitext-2-raw-v1" / "train-*.parquet")))[0]
    text = "\n\n".join(t for t in pd.read_parquet(pq)["text"].tolist() if t.strip())
    ids = tok(text[:400000], return_tensors="pt", add_special_tokens=False).input_ids[0][: N_SEQ * SEQ_LEN]
    return ids.reshape(N_SEQ, SEQ_LEN)


def run_trajectories(device: str, dtype_name: str, snap: bool, tamper: bool) -> Tuple[Dict[str, Dict[int, Dict[str, torch.Tensor]]], List[str]]:
    dtype = getattr(torch, dtype_name)
    seqs = token_sequences()
    out: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    caught: List[str] = []
    for init_seed in INIT_SEEDS:
        model = build_model(device, dtype, init_seed)
        state0 = extract_lora_state_dict(model)
        for rng_seed in RNG_SEEDS:
            snaps: Dict[int, Dict[str, torch.Tensor]] = {}
            counter = {"step": 0}
            lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]

            def post_hook(opt, args, kwargs):  # runs after every optimizer.step() inside replay_inner_segment
                counter["step"] += 1
                with torch.no_grad():
                    if snap:
                        for _, p in lora_params:
                            p.copy_(torch.round(p / SNAP_GRID) * SNAP_GRID)
                    if tamper and counter["step"] == TAMPER_STEP:
                        _, p0 = next((n, p) for n, p in lora_params if n.endswith("lora_A"))
                        flat = p0.view(-1)
                        i = int(flat.abs().argmax())
                        flat[i] = -flat[i]
                if counter["step"] % SNAP_EVERY == 0:
                    snaps[counter["step"]] = {n: p.detach().to("cpu", torch.float64).clone() for n, p in lora_params}

            from torch.optim.optimizer import register_optimizer_step_post_hook

            handle = register_optimizer_step_post_hook(post_hook)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    replay_inner_segment(model, start_lora_state=state0, token_sequences=seqs, start_step=0, end_step=STEPS, rng_seed=rng_seed)
                caught.extend(sorted({str(x.message)[:300] for x in w}))
            finally:
                handle.remove()
            out[f"{rng_seed}|{init_seed}"] = snaps
            print(f"[{device} {dtype_name}{' snap' if snap else ''}{' tamper' if tamper else ''}] rng {rng_seed} init {init_seed} done", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, sorted(set(caught))


def linf(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    m, num, den = 0.0, 0.0, 0.0
    for k in a:
        d = (a[k] - b[k]).abs()
        m = max(m, float(d.max()))
        num += float((d**2).sum())
        den += float((a[k] ** 2).sum())
    return m, math.sqrt(num / den) if den > 0 else 0.0


def compare(x: Dict[str, Any], y: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for seg in x:
        curve = {str(s): linf(x[seg][s], y[seg][s])[0] for s in sorted(x[seg])}
        l, rel = linf(x[seg][STEPS], y[seg][STEPS])
        rows.append({"segment": seg, "linf_at_50": l, "rel_l2_at_50": rel, "linf_curve": curve})
    return rows


def worker_subprocess(device: str, dtype: str, snap: bool, tamper: bool, tmp: Path) -> Tuple[Dict[str, Any], List[str]]:
    out = tmp / f"e12_{device}_{dtype}_{int(snap)}_{int(tamper)}.pt"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--device", device, "--dtype", dtype, "--out", str(out)]
    if snap:
        cmd.append("--snap")
    if tamper:
        cmd.append("--tamper")
    subprocess.run(cmd, check=True)
    blob = torch.load(out, weights_only=False)
    return blob["traj"], blob["warnings"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--tamper", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--tmp", default=str(Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp"))
    a = ap.parse_args()
    if a.worker:
        traj, warns = run_trajectories(a.device, a.dtype, a.snap, a.tamper)
        torch.save({"traj": traj, "warnings": warns}, a.out)
        return 0

    started = time.perf_counter()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    t4_a, w_a = run_trajectories("cuda", "float32", False, False)
    t4_a2, _ = run_trajectories("cuda", "float32", False, False)
    t4_b, w_b = worker_subprocess("cuda", "float32", False, False, tmp)
    cpu_c, w_c = worker_subprocess("cpu", "float32", False, False, tmp)
    t4_d, w_d = worker_subprocess("cuda", "float64", True, False, tmp)
    cpu_d, _ = worker_subprocess("cpu", "float64", True, False, tmp)
    t4_e, _ = worker_subprocess("cuda", "float32", False, True, tmp)

    cells = {
        "A_same_process_T4": compare(t4_a, t4_a2),
        "B_separate_process_T4": compare(t4_a, t4_b),
        "C_cross_T4_vs_cpu": compare(t4_a, cpu_c),
        "D_proxy_fp64_snap_T4_vs_cpu": compare(t4_d, cpu_d),
        "E_tamper": compare(t4_a, t4_e),
    }
    summary = {k: {"max_linf": max(r["linf_at_50"] for r in v), "min_linf": min(r["linf_at_50"] for r in v),
                   "n_above_eps": sum(r["linf_at_50"] > EPS for r in v), "n": len(v)} for k, v in cells.items()}
    a_ok = summary["A_same_process_T4"]["max_linf"] == 0.0
    e_ok = summary["E_tamper"]["n_above_eps"] == summary["E_tamper"]["n"]
    b = summary["B_separate_process_T4"]
    c = summary["C_cross_T4_vs_cpu"]
    p_same = "supported" if b["max_linf"] <= EPS else "refuted"
    p_cross = "supported" if c["n_above_eps"] == c["n"] else ("refuted" if c["n_above_eps"] == 0 else f"inconclusive ({c['n_above_eps']}/{c['n']} above eps)")
    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E12-replay/v1.0",
        "experiment_id": "E12-replay-divergence",
        "prereg_sha256": hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads()},
        "eps": EPS,
        "cells": cells,
        "summary": summary,
        "analysis": {"P12-same": {"status": p_same, "basis": "cell B (same T4, separate process) - a proxy for same class, not a second T4"},
                     "P12-cross": {"status": p_cross, "basis": "cell C (T4 vs this host's x86 CPU, float32 both)"},
                     "P12-fixed-point": {"status": "not decided (no fixed-point kernel); fp64 + 2^-20 snapping proxy reported in cell D"}},
        "controls": {"negative_same_process_exact_zero": {"passed": a_ok, "max_linf": summary["A_same_process_T4"]["max_linf"]},
                     "positive_tamper_exceeds_eps": {"passed": e_ok, "min_linf": summary["E_tamper"]["min_linf"]}},
        "controls_all_passed": bool(a_ok and e_ok),
        "determinism_warnings": {"cuda_fp32": w_a, "cuda_fp32_subprocess": w_b, "cpu_fp32": w_c, "cuda_fp64_snap": w_d},
        "not_run": ["RTX 2070 (sm_75, other host)", "TPU-VM CPU", "a second T4"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "analysis": report["analysis"], "controls_all_passed": report["controls_all_passed"]}, indent=2))
    return 0 if report["controls_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
