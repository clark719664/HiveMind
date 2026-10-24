"""Experiment E13: a SWARM SIMULATION on one local GPU (E8's "localhost negative control"; docs/RESEARCH_PROGRAM.md E8 + amendment, ROADMAP HM-S9-07).

M worker OS processes plus a coordinator exchange LoRA deltas over REAL localhost HTTP (quorum K of M with an adaptive grace window, RDA or direct
token-weighted merge, E8-lattice compression with error feedback, stragglers, churn) on a real (MoE-capable) language model and real Hugging Face
tokens, to validate the concept end to end. Worker i trains only on domain DOMAINS[i] (non-IID); the metric is the cross-entropy of the global LoRA on a
fixed mixture of held-out sequences of all domains.

THIS CAN NOT SUPPORT ANY WAN / VOLUNTEER CLAIM: every process shares one GPU and one host, so latencies are time-sharing artefacts and descriptive only.

Cells (run_cell(model, cell, seed)):
  S0 sync_baseline  one process, M learners simulated sequentially, K=M, direct, no codec (matched-token reference; carries the fail-fast rules)
  S1 swarm_identity real M-process HTTP swarm, K=M, direct, no codec, no stragglers/churn. CONTROL: final global LoRA must equal S0's (max|diff| <= 1e-6)
  S1t = S1 with --peer-hosting thread (smoke default; one frozen model, M peer threads over the same HTTP protocol; identity control applies as well)
  S2 swarm_all_on   K=ceil(M/2), adaptive grace, RDA, E8@2+EF, stragglers and churn ON
  S3 no_codec       S2 with codec none          S4 no_rda  S2 with direct merge          S5 no_quorum  S2 with K=M

NOT INCLUDED in this local swarm (the E8 amendment lists them for the WAN factorial): NF4, Skip-Token, Skip-Attention, PA-CGAD, fixed-point exits.

    python3 scripts/run_experiment_E13_swarm_local.py --estimate --model qwen0.5b --workers 4 --rounds 6 --inner 20
    python3 scripts/run_experiment_E13_swarm_local.py --smoke --out /tmp/e13_smoke.json     # tiny CPU plumbing check (not evidence)
    python3 scripts/run_experiment_E13_swarm_local.py --model qwen0.5b                      # full run (needs the prereg file)
    python3 scripts/run_experiment_E13_swarm_local.py --aggregate --out raw.json --report report.json   # CPU only

Numerics: the base is fp16 on CUDA with torch.autocast(float16) and fp32 LoRA parameters / optimizer / outer state (static loss scale
LOSS_SCALE_FP16, gradients divided back before clipping, a non-finite step is skipped and counted); the tiny CPU model is fp32.
Byte counts are len() of the real HTTP request / response bodies (headers and the small JSON control replies are excluded / counted separately).
"""

from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

import e13_swarm_lib as lib  # noqa: E402
from e13_swarm_lib import (DOMAINS, E8_CODEWORD_SCALE, EVAL_BATCH, EVAL_SEQS_PER_DOMAIN, GRACE_GAMMA, GRACE_TAU, GRAD_CLIP, INNER_BETAS, INNER_LR, INNER_WD, LORA_ALPHA, LORA_RANK,  # noqa: E402,F401
                           LORA_TARGETS, LOSS_SCALE_FP16, MICRO_BATCH, OUTER_LR, OUTER_MOMENTUM, SEQ_LEN, RunCfg)

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = EXP_DIR / "E13_swarm_local_prereg.json"
RAW_PATH = EXP_DIR / "E13_swarm_local_raw.json"
REPORT_PATH = EXP_DIR / "E13_swarm_local_report.json"
T_CRIT_DF2 = 4.302652729911275  # two-sided 95% t quantile, df = 2 (three seeds)
SEEDS = (1, 2, 3)
ALL_CELLS = ("S0", "S1", "S2", "S3", "S4", "S5")
EXTRA_CELLS = ("S1t",)  # S1 with thread peer-hosting (always run by the smoke; opt-in via --cells otherwise)
REPORT_CELLS = ALL_CELLS + EXTRA_CELLS
CODEC_ON = "E8@2+EF"  # E8 lattice, 2 bits/dim measured, error feedback
IDENTITY_TOL = 1e-6
RATE_TOL = 0.05  # |mean measured rate - target| for the codec control
DEFAULT_STRAGGLE = "-1:1,3,5:2.0"  # worker M-1 sleeps 2 x its median compute time before posting in rounds 1, 3, 5
DEFAULT_CHURN = "1:2:3"  # worker 1 SIGKILLed after round 2 closes, re-spawned (fresh optimizer + EF carry) once round 3 closes, i.e. before round 4
SMOKE_STRAGGLE, SMOKE_CHURN = "-1:1:2.0", "1:1:1"  # 2 rounds only: restart worker 1 across the round-1/round-2 boundary
LEARNING_GATE_NATS = 0.05  # fail-fast: S0 mixture held-out loss must drop by this much between round 0 and round FAILFAST_ROUND
SMOKE_LEARNING_GATE_NATS = -1e9  # disabled in the smoke (random tiny model, 3 steps); tests patch it to exercise the abort path
FAILFAST_ROUND = 3
FAILFAST_STEPS, FAILFAST_SEC_FACTOR = 20, 1.3  # abort if sec/inner-step over the first 20 steps exceeds the --estimate assumption by > 30%
EXIT_FAILFAST = 3
# time model (assumptions, to be measured): per-model sec/step lives in lib.MODEL_META
SEC_PER_EVAL, SEC_PROC_START, SEC_S0_LOAD, SEC_DATA_PREP = 3.0, 20.0, 10.0, 120.0
NOT_INCLUDED = ["NF4", "Skip-Token", "Skip-Attention", "PA-CGAD", "fixed-point exits"]


@dataclass(frozen=True)
class CellSpec:
    name: str
    k: str  # "M" or "half" (= ceil(M/2))
    merge: str
    grace: bool
    codec: str
    chaos: bool  # stragglers + churn
    hosting: Optional[str] = None  # None = ctx.hosting


CELL_SPECS: Dict[str, CellSpec] = {
    "S1": CellSpec("swarm_identity", "M", "direct", False, "none", False),
    "S1t": CellSpec("swarm_identity_thread", "M", "direct", False, "none", False, "thread"),
    "S2": CellSpec("swarm_all_on", "half", "rda", True, CODEC_ON, True),
    "S3": CellSpec("no_codec", "half", "rda", True, "none", True),
    "S4": CellSpec("no_rda", "half", "direct", True, CODEC_ON, True),
    "S5": CellSpec("no_quorum", "M", "rda", True, CODEC_ON, True),
}
CELL_NAMES = {"S0": "sync_baseline", **{k: v.name for k, v in CELL_SPECS.items()}}

FULL = RunCfg(rounds=6, workers=4, inner=20)
SMOKE = RunCfg(rounds=2, workers=2, inner=3, seq_len=16, micro_batch=2, eval_seqs=4, eval_batch=4, smoke=True)


def quorum_k(spec: CellSpec, workers: int) -> int:
    return workers if spec.k == "M" else math.ceil(workers / 2)


# ---------------------------------------------------------------- estimate
def estimate(model_key: str, cfg: RunCfg, seeds: Sequence[int] = SEEDS, cells: Sequence[str] = ALL_CELLS) -> Dict[str, Any]:
    """Planned steps / rounds / wall time (no compute). The swarm shares one GPU, so its inner-step work is the same R*M*H steps as S0's."""
    sps = lib.MODEL_META[model_key]["sec_per_step"]
    if sps is None:
        return {"model": model_key, "note": "no SEC_PER_STEP assumption for this model (smoke / placeholder)", "cells": list(cells), "seeds": list(seeds), "cfg": cfg.as_dict()}
    steps_run = cfg.rounds * cfg.workers * cfg.inner
    runs = len(cells) * len(seeds)
    parts: Dict[str, float] = {"inner_steps_s": runs * steps_run * sps, "eval_s": runs * (cfg.rounds + 1) * SEC_PER_EVAL,
                               "process_start_s": len(seeds) * (sum(1 for c in cells if c != "S0") * SEC_PROC_START + ("S0" in cells) * SEC_S0_LOAD), "data_prep_once_s": SEC_DATA_PREP}
    # K=M cells wait for the straggler: factor x its median compute, which under GPU time-sharing is ~ a whole round (M*H*sps)
    wait = 0.0
    n_str = sum(1 for r in lib.parse_straggle(DEFAULT_STRAGGLE, cfg.workers) if r[1] <= cfg.rounds)
    for c in cells:
        if c in CELL_SPECS and CELL_SPECS[c].chaos and CELL_SPECS[c].k == "M":
            wait += len(seeds) * n_str * 2.0 * cfg.workers * cfg.inner * sps
    parts["straggler_wait_K_eq_M_s"] = wait
    total = sum(parts.values())
    return {"model": model_key, "cells": list(cells), "seeds": list(seeds), "cfg": cfg.as_dict(), "runs": runs, "inner_steps_per_run": steps_run, "inner_steps_total": runs * steps_run,
            "rounds_per_run": cfg.rounds, "tokens_per_worker_per_run": cfg.rounds * cfg.inner * cfg.micro_batch * cfg.seq_len,
            "assumed_sec_per_inner_step": sps, "assumed_sec_per_eval": SEC_PER_EVAL, "assumed_sec_process_start": SEC_PROC_START,
            "breakdown_s": {k: round(v, 1) for k, v in parts.items()}, "total_s": round(total, 1), "total_hours": round(total / 3600.0, 2)}


# ---------------------------------------------------------------- run context
@dataclass
class Ctx:
    cfg: RunCfg = FULL
    hosting: str = "process"  # peer hosting: 'process' (one OS process per peer) | 'thread' (one host process, M peer threads)
    grad_ckpt: bool = False
    scratch: Path = Path("scratch_e13")
    device: str = "cuda"
    straggle: str = "default"  # "default" | "off" | spec
    churn: str = "default"
    cell_timeout_s: float = 6 * 3600.0
    threads: int = 2
    results: Dict[Tuple[str, str, int], Dict[str, Any]] = field(default_factory=dict)  # (model, cell, seed) -> record, for the S1-vs-S0 identity check
    s0_done: bool = False  # the first S0 of the invocation carries the fail-fast rules
    _tokens: Dict[str, Tuple[Path, Dict[str, Any]]] = field(default_factory=dict)

    def ensure_tokens(self, model_key: str) -> Tuple[Path, Dict[str, Any]]:
        if model_key not in self._tokens:
            path = self.scratch / model_key / f"tokens_M{self.cfg.workers}_L{self.cfg.seq_len}_E{self.cfg.eval_seqs}.pt"
            self._tokens[model_key] = (path, lib.build_token_file(model_key, self.cfg, path))
        return self._tokens[model_key]

    def chaos_specs(self) -> Tuple[Dict[Tuple[int, int], float], Optional[Tuple[int, int, int]]]:
        smoke = self.cfg.smoke
        s = (SMOKE_STRAGGLE if smoke else DEFAULT_STRAGGLE) if self.straggle == "default" else self.straggle
        c = (SMOKE_CHURN if smoke else DEFAULT_CHURN) if self.churn == "default" else self.churn
        churn = lib.parse_churn(c, self.cfg.workers)
        if churn is not None and churn[1] >= self.cfg.rounds:
            churn = None  # a kill after the last round is meaningless
        return {k: v for k, v in lib.parse_straggle(s, self.cfg.workers).items() if k[1] <= self.cfg.rounds}, churn


class FailFast(Exception):
    def __init__(self, reason: str, details: Dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason, self.details = reason, details


def check_learning_gate(curve: Dict[str, float], rnd: int, gate: float) -> Optional[str]:
    """Fail-fast rule 1: the mixture loss must have dropped by >= gate nats from round 0 by round ``rnd``. Returns the reason if tripped."""
    drop = curve["0"] - curve[str(rnd)]
    if drop < gate:
        return f"S0 mixture held-out loss dropped {drop:.4f} nats by round {rnd} (< {gate}): the sequential reference does not learn"
    return None


def check_step_time(sps: float, model_key: str) -> Optional[str]:
    """Fail-fast rule 2: measured sec/inner-step over the first FAILFAST_STEPS steps vs the --estimate assumption (+30%)."""
    assumed = lib.MODEL_META[model_key]["sec_per_step"]
    if assumed is not None and sps > assumed * FAILFAST_SEC_FACTOR:
        return f"measured {sps:.3f} s per inner step over the first {FAILFAST_STEPS} steps exceeds {assumed * FAILFAST_SEC_FACTOR:.3f} s ({FAILFAST_SEC_FACTOR} x the assumed {assumed})"
    return None


# ---------------------------------------------------------------- S0
def run_s0(model_key: str, seed: int, ctx: Ctx) -> Dict[str, Any]:
    cfg, device = ctx.cfg, ctx.device
    amp = lib.use_amp(model_key, device)
    t_all = time.perf_counter()
    tok_path, prov = ctx.ensure_tokens(model_key)
    model, _, params = lib.build_model(model_key, device, seed, ctx.grad_ckpt)
    train, held = lib.load_token_file(str(tok_path))
    doms = DOMAINS[: cfg.workers]
    held_m = {d: held[d] for d in doms}
    perms = [lib.worker_perm(seed, i, train[doms[i]].shape[0]) for i in range(cfg.workers)]
    opts = [lib.make_opt(params) for _ in range(cfg.workers)]  # one persistent AdamW state per learner
    theta = lib.e10b.get_flat(params).cpu()
    buf = torch.zeros_like(theta)
    init_sha = lib.tensor_sha(theta)
    per0 = lib.eval_domains(model, params, theta, held_m, cfg.eval_batch, device, amp)
    curve: Dict[str, float] = {"0": lib.mixture(per0)}
    per_curve: Dict[str, Dict[str, float]] = {"0": per0}
    round_wall: List[float] = []
    skipped = 0
    failfast = not ctx.s0_done
    ctx.s0_done = True
    step_times: Optional[List[float]] = [] if failfast else None
    gate = SMOKE_LEARNING_GATE_NATS if cfg.smoke else LEARNING_GATE_NATS
    for rnd in range(1, cfg.rounds + 1):
        t0 = time.perf_counter()
        outs = []
        for i in range(cfg.workers):
            outs.append(lib.local_train(model, params, opts[i], theta, train[doms[i]], perms[i], rnd, cfg, device, amp, step_times if (failfast and rnd == 1) else None))
            if step_times is not None and rnd == 1 and len(step_times) - 1 >= FAILFAST_STEPS:
                msg = check_step_time((step_times[FAILFAST_STEPS] - step_times[0]) / FAILFAST_STEPS, model_key)
                step_times = None
                if msg:
                    raise FailFast(msg, {"loss_curve": dict(curve)})
        skipped += sum(o["skipped_steps"] for o in outs)
        g = lib.merge_deltas([o["delta"] for o in outs], [o["tokens"] for o in outs], [o["steps"] for o in outs], "direct")
        theta, buf = lib.outer_nesterov(theta, buf, g)
        round_wall.append(time.perf_counter() - t0)
        per = lib.eval_domains(model, params, theta, held_m, cfg.eval_batch, device, amp)
        curve[str(rnd)] = lib.mixture(per)
        per_curve[str(rnd)] = per
        if not math.isfinite(curve[str(rnd)]):
            raise FailFast(f"S0 diverged (non-finite held-out loss at round {rnd})", {"loss_curve": dict(curve)})
        if failfast and rnd == min(FAILFAST_ROUND, cfg.rounds):
            msg = check_learning_gate(curve, rnd, gate)
            if msg:
                raise FailFast(msg, {"loss_curve": dict(curve)})
    out_dir = ctx.scratch / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    theta_path = out_dir / f"S0_seed{seed}_final.pt"
    torch.save(theta, theta_path)
    routing = lib.routing_stats(model, params, theta, held_m, cfg.eval_batch, device, amp)
    del model, params, opts  # free the GPU before the swarm hosts load their own copies
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"model": model_key, "cell": "S0", "cell_name": CELL_NAMES["S0"], "seed": seed, "status": "ok", "K": cfg.workers, "merge": "direct", "codec": "none", "n_coords": int(theta.numel()),
            "rounds": cfg.rounds, "loss_curve": curve, "initial_loss": curve["0"], "final_loss": curve[str(cfg.rounds)], "per_domain_curve": per_curve, "per_domain_final": per_curve[str(cfg.rounds)],
            "round_wall_s": round_wall, "admitted_per_round": [list(range(cfg.workers))] * cfg.rounds, "late_discard_count": 0,
            "wire": {"up_bytes": 0, "down_bytes": 0, "n_posts": 0, "raw_delta_bytes": 0, "ideal_bytes": 0.0}, "init_sha256": init_sha, "skipped_steps": skipped,
            "final_theta_sha256": lib.tensor_sha(theta), "final_theta_path": str(theta_path), "routing": routing, "data_provenance": prov,
            "wall_seconds": round(time.perf_counter() - t_all, 2)}


# ---------------------------------------------------------------- swarm cells
def run_swarm(model_key: str, cell: str, seed: int, ctx: Ctx) -> Dict[str, Any]:
    spec, cfg = CELL_SPECS[cell], ctx.cfg
    hosting = spec.hosting or ctx.hosting
    M, K = cfg.workers, quorum_k(spec, cfg.workers)
    t_all = time.perf_counter()
    tok_path, prov = ctx.ensure_tokens(model_key)
    cell_dir = ctx.scratch / model_key / f"{cell}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    for old in cell_dir.glob("*.log"):
        old.unlink()
    straggle, churn = ctx.chaos_specs() if spec.chaos else ({}, None)
    coord = lib.Coordinator(M, cfg.rounds, cfg.inner, K, spec.merge, spec.grace)
    server = lib.make_server(coord)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    env = dict(os.environ)
    if cfg.smoke or ctx.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = str(ctx.threads)
    procs: Dict[int, Any] = {}
    killed: set = set()
    churn_events: List[Dict[str, Any]] = []

    def spawn(w: int, inc: int) -> None:
        wcfg = {"worker_id": w, "incarnation": inc, "port": port, "model": model_key, "seed": seed, "device": ctx.device, "threads": ctx.threads, "run_cfg": cfg.as_dict(), "hosting": hosting, "grad_ckpt": ctx.grad_ckpt,
                "tokens_file": str(tok_path), "codec": spec.codec, "straggle": [[a, b, f] for (a, b), f in straggle.items()]}
        p = cell_dir / f"worker{w}_inc{inc}.json"
        p.write_text(json.dumps(wcfg), encoding="utf-8")
        procs[w] = lib.spawn_worker(str(Path(__file__).resolve()), p, cell_dir / f"worker{w}_inc{inc}.log", env)  # thread hosting: key -1 = the single host process

    def before_publish(rnd: int) -> None:  # runs under the coordinator lock, between closing round ``rnd`` and opening the next one
        if churn is None:
            return
        w, kill_after, respawn_after = churn
        if hosting == "thread":  # no process to kill: the coordinator declares the peer dead / re-incarnated; the host stops and restarts its thread (state discarded)
            if rnd == kill_after:
                coord.set_alive(w, False)
                churn_events.append({"event": "peer_stopped", "worker": w, "after_round": rnd})
            if rnd == respawn_after:
                coord.set_alive(w, True)
                churn_events.append({"event": "peer_restarted", "worker": w, "after_round": rnd})
            return
        if rnd == kill_after:
            killed.add(procs[w].pid)
            procs[w].kill()
            procs[w].wait()
            coord.set_alive(w, False)
            churn_events.append({"event": "killed", "worker": w, "after_round": rnd})
        if rnd == respawn_after:
            spawn(w, 1)
            coord.set_alive(w, True)
            churn_events.append({"event": "respawned", "worker": w, "after_round": rnd})

    def check_crash() -> None:
        for w, p in procs.items():
            if p.poll() not in (None, 0) and p.pid not in killed:
                last_log = sorted(cell_dir.glob(f"worker{w}_inc*.log"))[-1]
                raise RuntimeError(f"worker {w} exited with code {p.returncode}\n{lib.log_tail(last_log)}")

    try:
        for w in ([-1] if hosting == "thread" else range(M)):
            spawn(w, 0)
        deadline = time.monotonic() + ctx.cell_timeout_s
        while not coord.wait_ready(0.5):
            check_crash()
            if time.monotonic() > deadline:
                raise TimeoutError("workers did not register in time")
        coord.open_first()
        while not coord.done:
            coord.try_close_round(before_publish)
            check_crash()
            if time.monotonic() > deadline:
                raise TimeoutError(f"cell {cell} seed {seed} exceeded {ctx.cell_timeout_s}s at version {coord.version}")
            time.sleep(lib.CLOSE_POLL_S)
        while not coord.wait_evals(0.5):
            check_crash()
            if time.monotonic() > deadline:
                raise TimeoutError("held-out evaluations did not finish in time")
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                p.kill()
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.wait()
        server.shutdown()
        server.server_close()

    theta = coord.theta
    theta_path = ctx.scratch / model_key / f"{cell}_seed{seed}_final.pt"
    torch.save(theta, theta_path)
    ev = coord.evals
    curve = {str(v): ev[v]["mixture"] for v in range(cfg.rounds + 1)}
    per_curve = {str(v): ev[v]["per_domain"] for v in range(cfg.rounds + 1)}
    up, down, final_sync = coord.per_round_bytes()
    posts = coord.posts
    rates = [p["rate"] for p in posts if p.get("rate") is not None and p["accepted"]]
    rels = [p["rel_mse"] for p in posts if p.get("rel_mse") is not None and p["accepted"]]
    shas = {s for v in coord.init_shas.values() for s in v}
    rec: Dict[str, Any] = {
        "model": model_key, "cell": cell, "cell_name": spec.name, "seed": seed, "status": "ok", "K": K, "merge": spec.merge, "grace": spec.grace, "codec": spec.codec, "hosting": hosting, "n_coords": int(theta.numel()),
        "rounds": cfg.rounds, "loss_curve": curve, "initial_loss": curve["0"], "final_loss": curve[str(cfg.rounds)], "per_domain_curve": per_curve, "per_domain_final": per_curve[str(cfg.rounds)],
        "round_wall_s": [r["latency_s"] for r in coord.rounds_log], "admitted_per_round": [r["admitted"] for r in coord.rounds_log],
        "unadmitted_at_close_per_round": [r["unadmitted_at_close"] for r in coord.rounds_log], "effective_k_per_round": [r["effective_k"] for r in coord.rounds_log],
        "late_discard_count": len(coord.late), "late_discards": coord.late,
        "bytes_up_per_round": up, "bytes_down_per_round": down, "bytes_down_final_sync": final_sync, "bytes_eval_traffic": coord.bytes_eval, "bytes_setup": coord.bytes_setup,
        "bytes_control": coord.bytes_control,
        "wire": {"up_bytes": sum(coord.bytes_up.values()), "down_bytes": sum(coord.bytes_down.values()), "n_posts": len(posts), "n_posts_accepted": sum(1 for p in posts if p["accepted"]),
                 "raw_delta_bytes": sum(p["raw_bytes"] for p in posts), "ideal_bytes": sum(p["ideal_bytes"] for p in posts), "mean_measured_rate_bits_per_dim": statistics.fmean(rates) if rates else None,
                 "mean_relative_mse": statistics.fmean(rels) if rels else None},
        "rounds_log": coord.rounds_log, "churn_events": churn_events, "coordinator_events": coord.events, "straggle": [[a, b, f] for (a, b), f in straggle.items()],
        "churn": list(churn) if churn else None, "init_identical": len(shas) == 1, "init_sha256": next(iter(shas)) if len(shas) == 1 else sorted(shas),
        "final_theta_sha256": lib.tensor_sha(theta), "final_theta_path": str(theta_path), "routing": ev[cfg.rounds].get("routing"), "data_provenance": prov,
        "http_requests": coord.requests, "wall_seconds": round(time.perf_counter() - t_all, 2),
    }
    if cell in ("S1", "S1t"):
        rec["identity_vs_s0"] = identity_vs_s0(model_key, seed, theta, ctx)
    return rec


def identity_vs_s0(model_key: str, seed: int, theta: torch.Tensor, ctx: Ctx) -> Dict[str, Any]:
    ref = ctx.results.get((model_key, "S0", seed))
    if ref is None:
        return {"available": False}
    s0 = torch.load(ref["final_theta_path"], map_location="cpu")
    d = float((theta - s0).abs().max())
    return {"available": True, "max_abs_diff": d, "tolerance": IDENTITY_TOL, "passed": d <= IDENTITY_TOL, "bit_identical": bool(torch.equal(theta, s0))}


def run_cell(model: str, cell: str, seed: int, ctx: Optional[Ctx] = None) -> Dict[str, Any]:
    """One (model, cell, seed) result: loss curve per round of the global mixture, per-domain final losses, wall seconds per round, bytes per round up/down, admitted ids,
    late-discard count, final global LoRA sha256 and its saved path under ``ctx.scratch``."""
    ctx = ctx or Ctx()
    if cell == "S0":
        return run_s0(model, seed, ctx)
    if cell in CELL_SPECS:
        return run_swarm(model, cell, seed, ctx)
    raise KeyError(cell)


# ---------------------------------------------------------------- aggregate (CPU only)
def mean_ci(xs: Sequence[float]) -> Tuple[float, float]:
    return statistics.fmean(xs), T_CRIT_DF2 * statistics.stdev(xs) / math.sqrt(len(xs))


def aggregate(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Per model: paired per-seed differences vs S0 (df = 2 t-intervals), S1-vs-S0 identity, wire bytes and compression ratios, admitted fraction, late discards,
    round-latency percentiles (descriptive), controls and a receipt sha256."""
    runs = raw["runs"]
    out: Dict[str, Any] = {}
    for model in sorted({r["model"] for r in runs}):
        mr = [r for r in runs if r["model"] == model]
        by = {(r["cell"], r["seed"]): r for r in mr}
        seeds = sorted({r["seed"] for r in mr})
        cells: Dict[str, Any] = {}
        for cell in REPORT_CELLS:
            rows = [by[(cell, s)] for s in seeds if (cell, s) in by]
            if not rows:
                continue
            lat = [x for r in rows for x in r["round_wall_s"]]
            w = [r["wire"] for r in rows]
            n_workers = len(rows[0]["per_domain_final"])
            adm = [len(a) / n_workers for r in rows for a in r["admitted_per_round"]]
            raw_b, up_b = sum(x["raw_delta_bytes"] for x in w), sum(x["up_bytes"] for x in w)
            ideal_b = sum(x["ideal_bytes"] for x in w)
            cells[cell] = {
                "name": CELL_NAMES[cell], "seeds": [r["seed"] for r in rows], "final_loss_per_seed": [r["final_loss"] for r in rows], "initial_loss_per_seed": [r["initial_loss"] for r in rows],
                "per_domain_final_mean": {d: statistics.fmean(r["per_domain_final"][d] for r in rows) for d in rows[0]["per_domain_final"]},
                "wire": {"up_bytes_mean_per_run": statistics.fmean(x["up_bytes"] for x in w), "down_bytes_mean_per_run": statistics.fmean(x["down_bytes"] for x in w),
                         "raw_fp32_delta_bytes_total": raw_b, "ideal_entropy_coded_bytes_total": ideal_b, "actual_up_bytes_total": up_b,
                         "compression_ratio_actual": (raw_b / up_b) if up_b else None, "compression_ratio_ideal": (raw_b / ideal_b) if (ideal_b and cell != "S0") else None},
                "admitted_fraction_mean": statistics.fmean(adm), "late_discards_total": sum(r["late_discard_count"] for r in rows),
                "round_latency_s": {"p50": lib.percentile(lat, 0.5), "p95": lib.percentile(lat, 0.95), "p99": lib.percentile(lat, 0.99), "n": len(lat), "note": "descriptive only: one GPU, time-shared"},
                "mean_measured_rate_bits_per_dim": statistics.fmean([r["wire"]["mean_measured_rate_bits_per_dim"] for r in rows if r["wire"].get("mean_measured_rate_bits_per_dim") is not None] or [float("nan")]),
            }
        paired: Dict[str, Any] = {}
        for cell in REPORT_CELLS[1:]:
            ss = [s for s in seeds if (cell, s) in by and ("S0", s) in by]
            if len(ss) != 3:
                paired[cell] = {"status": f"needs exactly 3 paired seeds (have {len(ss)})", "seeds": ss}
                continue
            d = [by[(cell, s)]["final_loss"] - by[("S0", s)]["final_loss"] for s in ss]
            m, h = mean_ci(d)
            paired[cell] = {"final_loss_minus_S0_per_seed": d, "mean": m, "ci95": h, "t_crit_df2": T_CRIT_DF2, "ci_excludes_0": bool(abs(m) > h), "direction": "worse than S0" if m > h else ("better than S0" if m < -h else "within CI of S0")}
        ident = [by[(c, s)].get("identity_vs_s0", {"available": False}) for c in ("S1", "S1t") for s in seeds if (c, s) in by]
        ident_ok = bool(ident) and all(i.get("available") and i.get("passed") for i in ident)
        s0 = [by[("S0", s)] for s in seeds if ("S0", s) in by]
        swarm_none = [r for r in mr if r["cell"] in CELL_SPECS and r["codec"] == "none"]
        swarm_e8 = [r for r in mr if r["cell"] in CELL_SPECS and r["codec"] != "none"]
        controls = {
            "s1_equals_s0": {"rows": ident, "tolerance": IDENTITY_TOL, "passed": ident_ok if ident else None},
            "s0_learns": {"rows": [{"seed": r["seed"], "initial": r["initial_loss"], "final": r["final_loss"]} for r in s0], "passed": (all(r["final_loss"] < r["initial_loss"] for r in s0) if s0 else None)},
            "raw_wire_bytes_exact": {"passed": (all(r["wire"]["up_bytes"] == r["wire"]["n_posts"] * 4 * r["n_coords"] for r in swarm_none) if swarm_none else None),
                                     "note": "codec 'none': every POSTed body is exactly 4 x n_coords bytes"},
            "codec_rate_on_target": {"passed": (all(abs(r["wire"]["mean_measured_rate_bits_per_dim"] - 2.0) <= RATE_TOL for r in swarm_e8) if swarm_e8 else None), "tolerance": RATE_TOL},
            "workers_start_identical": {"passed": (all(r["init_identical"] for r in mr if r["cell"] in CELL_SPECS) if any(r["cell"] in CELL_SPECS for r in mr) else None)},
        }
        out[model] = {"cells": cells, "paired_vs_S0": paired, "controls": controls, "controls_all_passed": all(c["passed"] is True for c in controls.values()),
                      "not_included": NOT_INCLUDED, "scope": "localhost negative control (E8): single GPU, time-shared; NOT evidence for any WAN / volunteer claim"}
    res: Dict[str, Any] = {"schema_version": "hivemind-experiment-E13/v1.0", "experiment_id": "E13-swarm-local", "design_sha256": raw.get("design_sha256"), "models": out,
                           "controls_all_passed": bool(out) and all(m["controls_all_passed"] for m in out.values())}
    res = lib.e10b.clean(res)
    res["receipt_sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode("utf-8")).hexdigest()
    return res


# ---------------------------------------------------------------- raw file (incremental, design-hash guarded)
def design_dict(model_key: str, cfg: RunCfg, prereg_sha: str) -> Dict[str, Any]:
    return {"model": model_key, "cfg": cfg.as_dict(), "domains": list(DOMAINS[: cfg.workers]), "cells": {k: v.__dict__ for k, v in CELL_SPECS.items()}, "prereg_sha256": prereg_sha,
            "inner": [INNER_LR, list(INNER_BETAS), INNER_WD, GRAD_CLIP], "outer": [OUTER_LR, OUTER_MOMENTUM], "lora": [LORA_RANK, LORA_ALPHA, list(LORA_TARGETS)],
            "split": [lib.TRAIN_FRAC, lib.HELDOUT_FRAC], "straggle": [DEFAULT_STRAGGLE, SMOKE_STRAGGLE], "churn": [DEFAULT_CHURN, SMOKE_CHURN], "grace": [GRACE_TAU, GRACE_GAMMA, lib.GRACE_EMA],
            "loss_scale": LOSS_SCALE_FP16, "gates": [LEARNING_GATE_NATS, FAILFAST_ROUND, FAILFAST_STEPS, FAILFAST_SEC_FACTOR]}


def write_raw(path: Path, design_sha: str, runs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"design_sha256": design_sha, "runs": lib.e10b.clean(runs)}), encoding="utf-8")
    tmp.replace(path)


def load_raw(path: Path, design_sha: str) -> List[Dict[str, Any]]:
    """Runs of a raw file, only if written under the same design hash (never resume across designs)."""
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    if not isinstance(blob, dict) or blob.get("design_sha256") != design_sha:
        print(f"[E13] ignoring {path.name}: written under a different design", flush=True)
        return []
    return blob["runs"]


def write_aborted(out: Path, ff: FailFast, runs: List[Dict[str, Any]], design_sha: str, smoke: bool, device: str) -> int:
    status = f"aborted by registered fail-fast rule: {ff.reason}"
    report = {"schema_version": "hivemind-experiment-E13/v1.0", "experiment_id": "E13-swarm-local", "smoke": smoke, "status": status, "design_sha256": design_sha,
              "fail_fast_details": ff.details, "completed_runs": [{k: r.get(k) for k in ("model", "cell", "seed", "final_loss")} for r in runs], "controls_all_passed": False,
              "hardware": {"device": device, "torch": torch.__version__}}
    report = lib.e10b.clean(report)
    report["receipt_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(status, flush=True)
    return EXIT_FAILFAST


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen0.5b", choices=sorted(lib.LOADERS))
    ap.add_argument("--cells", default=None, help="default S0..S5 (smoke: plus S1t)")
    ap.add_argument("--seeds", default=None, help="default 1,2,3 (smoke: 1)")
    ap.add_argument("--workers", type=int, default=None, help="M peers; default per model (qwen0.5b 4, granite1b 3, granite3b 4)")
    ap.add_argument("--rounds", type=int, default=FULL.rounds)
    ap.add_argument("--inner", type=int, default=FULL.inner)
    ap.add_argument("--scratch", default=None, help="directory for token files, logs, final LoRA vectors")
    ap.add_argument("--out", default=None, help="raw results file (written incrementally); the report goes to --report")
    ap.add_argument("--report", default=None)
    ap.add_argument("--resume", action="store_true", help="skip (model, cell, seed) already in --out (same design hash only)")
    ap.add_argument("--smoke", action="store_true", help="tiny random Qwen2 on CPU, M=2, 2 rounds x 3 inner steps, seq 16, synthetic tokens, all cells; writes only to --out (not evidence)")
    ap.add_argument("--estimate", action="store_true", help="print planned steps / rounds / time; no compute")
    ap.add_argument("--aggregate", action="store_true", help="recompute the report from --out (CPU only)")
    ap.add_argument("--straggle", default="default", help="'default' (S2-S5 only), 'off', or 'W:r1,r2:factor;...'")
    ap.add_argument("--churn", default="default", help="'default' (S2-S5 only), 'off', or 'W:KILL:RESPAWN'")
    ap.add_argument("--peer-hosting", choices=("process", "thread"), default=None,
                    help="process: one OS process per peer; thread: ONE host process holds the single frozen model, M peer threads (serialised by a lock) talk to the coordinator over the same HTTP protocol. Default per model (granite3b: thread)")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=None, help="gradient checkpointing (default per model: granite1b on)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cell-timeout", type=float, default=6 * 3600.0)
    ap.add_argument("--parallel-cells", type=int, default=0, help="run this many swarm cells concurrently (default 1; smoke: all). Concurrency distorts latencies: never for the real run")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-config", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        return lib.run_worker(a.worker_config)

    cells = [c for c in (a.cells or ",".join(ALL_CELLS + (EXTRA_CELLS if a.smoke else ()))).split(",") if c]
    for c in cells:
        if c not in REPORT_CELLS:
            ap.error(f"unknown cell {c}")
    model_key = "tiny" if a.smoke else a.model
    meta = lib.MODEL_META[model_key]
    cfg = SMOKE if a.smoke else RunCfg(rounds=a.rounds, workers=a.workers or meta["workers"], inner=a.inner)
    hosting = a.peer_hosting or meta["hosting"]
    grad_ckpt = meta["grad_ckpt"] if a.grad_ckpt is None else a.grad_ckpt
    seeds = tuple(int(s) for s in (a.seeds or ("1" if a.smoke else "1,2,3")).split(","))

    if a.estimate:
        print(json.dumps(estimate(model_key, cfg, seeds, cells), indent=2))
        return 0
    if a.aggregate:
        if not a.out:
            ap.error("--aggregate needs --out (the raw file)")
        res = aggregate(json.loads(Path(a.out).read_text(encoding="utf-8")))
        rep = Path(a.report) if a.report else Path(a.out).with_suffix(".report.json")
        rep.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({"controls_all_passed": res["controls_all_passed"], "receipt": res["receipt_sha256"], "report": str(rep)}, indent=2))
        return 0

    art = (PROJECT_ROOT / "artifacts").resolve()
    if a.smoke:
        if not a.out:
            ap.error("--smoke requires --out")
        out = Path(a.out).resolve()
        if art in out.parents or (a.scratch and art in Path(a.scratch).resolve().parents):
            ap.error("--smoke must not write into artifacts/")
        device, prereg_sha = "cpu", "smoke"
        torch.set_num_threads(2)
        report_path = Path(a.report).resolve() if a.report else out.with_suffix(".report.json")
    else:
        if not PREREG_PATH.exists():
            raise SystemExit(f"{PREREG_PATH} not found: the full run needs the registered prereg")
        device = a.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("full run needs a CUDA device")
        out = Path(a.out) if a.out else RAW_PATH
        report_path = Path(a.report) if a.report else REPORT_PATH
        prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    scratch = Path(a.scratch).resolve() if a.scratch else out.parent / (out.stem + "_scratch")
    ctx = Ctx(cfg=cfg, hosting=hosting, grad_ckpt=grad_ckpt, scratch=scratch, device=device, straggle=a.straggle, churn=a.churn, cell_timeout_s=a.cell_timeout, threads=2)
    design_sha = hashlib.sha256(json.dumps(design_dict(model_key, cfg, prereg_sha), sort_keys=True).encode("utf-8")).hexdigest()
    runs: List[Dict[str, Any]] = load_raw(out, design_sha) if a.resume else []
    for r in runs:
        ctx.results[(r["model"], r["cell"], r["seed"])] = r
    done = {(r["model"], r["cell"], r["seed"]) for r in runs}
    ctx.s0_done = any(c == "S0" for _, c, _ in done)  # fail-fast was already passed in the resumed run
    print(f"[E13] {'SMOKE ' if a.smoke else ''}model {model_key} device {device} plan: {json.dumps(estimate(model_key, cfg, seeds, cells))}", flush=True)
    started = time.perf_counter()
    parallel = a.parallel_cells or (len(REPORT_CELLS) if a.smoke else 1)  # smoke only: process start-up (torch + transformers imports) dominates, so the swarm cells overlap

    def finish(rec: Dict[str, Any]) -> None:
        runs.append(rec)
        ctx.results[(rec["model"], rec["cell"], rec["seed"])] = rec
        write_raw(out, design_sha, runs)
        print(f"seed {rec['seed']} {rec['cell']} {rec['cell_name']:14s} final {rec['final_loss']:.4f} (init {rec['initial_loss']:.4f}) admitted/round "
              f"{[len(x) for x in rec['admitted_per_round']]} late {rec['late_discard_count']} up {rec['wire']['up_bytes']}B {rec['wall_seconds']}s", flush=True)

    for seed in seeds:  # seed-major so an interrupted run holds complete seeds
        todo = [c for c in cells if (model_key, c, seed) not in done]
        try:
            if "S0" in todo:  # first: it carries the fail-fast rules and the identity reference
                finish(run_cell(model_key, "S0", seed, ctx))
            swarm = [c for c in todo if c != "S0"]
            if parallel <= 1:
                for c in swarm:
                    finish(run_cell(model_key, c, seed, ctx))
            elif swarm:
                ctx.ensure_tokens(model_key)
                with ThreadPoolExecutor(max_workers=parallel) as ex:
                    futs = [ex.submit(run_cell, model_key, c, seed, ctx) for c in swarm]
                    for f in futs:
                        finish(f.result())
        except FailFast as ff:
            return write_aborted(report_path, ff, runs, design_sha, bool(a.smoke), device)
    raw = json.loads(out.read_text(encoding="utf-8"))
    res = aggregate(raw)
    res.update({"smoke": bool(a.smoke), "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None, "cpu": platform.processor() or platform.machine(),
                                                    "torch": torch.__version__}, "elapsed_seconds": round(time.perf_counter() - started, 2), "estimate": estimate(model_key, cfg, seeds, cells)})
    if a.smoke:
        res["warning"] = "SMOKE RUN: tiny random model, synthetic tokens; plumbing check only, not evidence"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"controls_all_passed": res["controls_all_passed"], "runs": len(runs), "elapsed_s": res["elapsed_seconds"], "receipt": res["receipt_sha256"]}, indent=2))
    return 0 if (a.smoke or res["controls_all_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
