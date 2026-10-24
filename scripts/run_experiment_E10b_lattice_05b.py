"""Experiment E10b: does the E8 lattice codec's distortion gain reach the held-out loss at 0.5B scale? (RQ9, HM-S10-01)

Pre-registered (DRAFT until the scientific lead commits it) in ``artifacts/experiments/E10b_lattice_05b_prereg.json``.
DiLoCo on Qwen2.5-0.5B-Instruct with rank-8 LoRA on q_proj / v_proj, M = 4 learners simulated sequentially on one
GPU, non-IID fineweb-edu shards, H inner AdamW steps, outer Nesterov SGD (lr 0.7, momentum 0.9), R rounds. The
learner deltas are quantised before the merge by {none, sign, Z8, E8} at matched measured rate. The lattice
quantisers run on the device through the bit-identical torch decoders of ``hivemind.compress.kv_lattice`` (the CPU
numpy decoders of ``lattice_quantizer`` would cost seconds per ~0.5M-coordinate delta); the scale is re-calibrated
to the target measured rate for every (learner, round) vector, which removes the rate drift that made the 2-bit
cells of E10 invalid. Reuses LoRALinear / build_model of ``run_experiment_E12_replay_divergence`` (imported lazily; same HF hub cache path).

    python3 scripts/run_experiment_E10b_lattice_05b.py --estimate        # step counts + time estimate, no compute
    python3 scripts/run_experiment_E10b_lattice_05b.py --smoke --out f.json   # tiny CPU plumbing check (not evidence)
    python3 scripts/run_experiment_E10b_lattice_05b.py                   # full run (T4), writes the report + receipt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hivemind.compress.kv_lattice import DECODERS_T, factorised_rate_t  # noqa: E402
from hivemind.compress.lattice_quantizer import DECODERS, QuantizedDelta, SignDeltaQuantizer, brute_force_decoder_check, factorised_rate_bits_per_dim  # noqa: E402

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = EXP_DIR / "E10b_lattice_05b_prereg.json"
PREREG_DRAFT_PATH = EXP_DIR / "E10b_lattice_05b_prereg_DRAFT.json"
AMENDMENT_PATH = EXP_DIR / "E10b_lattice_05b_prereg_amendment_2.json"  # amendment 2 (2026-10-24): init, sizing, arms, fail-fast rule
REPORT_PATH = EXP_DIR / "E10b_lattice_05b_report.json"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
FINEWEB_GLOB = str(HUB / "datasets--HuggingFaceFW--fineweb-edu" / "snapshots" / "*" / "sample" / "10BT" / "000_00000.parquet")
T_CRIT_DF2 = 4.302652729911275  # two-sided 95% t quantile, df = 2 (three seeds)
SEEDS = (1, 2, 3)
RATES = (1.0,)  # amendment 2: R = 2 is no longer run
FINE_RATE = 6.0
MATCH_TOL = 0.1  # Z8 / E8 realised rates must agree within this many bits/dim (as E10)
FINE_TOL = 5e-3  # negative control 1: |loss(E8 @ 6 bit) - loss(none)| per seed (E10 used 1e-3 on a CPU toy; GPU fp32 run-to-run noise is larger)
OUTER_LR, OUTER_MOMENTUM = 0.7, 0.9
INNER_LR, INNER_BETAS, INNER_WD, GRAD_CLIP = 5e-4, (0.9, 0.95), 0.01, 1.0
SHARD_ROW_STRIDE = 10_000  # learner i reads fineweb rows from i * stride; the held-out slice starts at M * stride
# time model, see estimate(). SEC_PER_STEP is measured (first execution: 1385 s / 1920 steps = 0.72 s on the T4); the eval cost is the
# old 1.3 s scaled by the same 6x; the codec call (1.5 s) and setup are still unmeasured assumptions.
SEC_PER_STEP, SEC_PER_CODEC_CALL, SEC_PER_EVAL, SEC_SETUP = 0.72, 1.5, 7.8, 150.0
# registered fail-fast rule (amendment 2), checked inside the first 'none' cell (seed 1)
FAILFAST_STEPS, FAILFAST_SEC_FACTOR, FAILFAST_ROUND = 40, 1.3, 3
EXIT_FAILFAST = 3


@dataclass(frozen=True)
class Cfg:
    rounds: int
    learners: int
    inner_steps: int
    seq_len: int
    micro_batch: int
    eval_seqs: int
    eval_batch: int
    eval_every: int


FULL = Cfg(rounds=6, learners=4, inner_steps=20, seq_len=256, micro_batch=4, eval_seqs=128, eval_batch=4, eval_every=3)
SMOKE = Cfg(rounds=2, learners=2, inner_steps=3, seq_len=16, micro_batch=2, eval_seqs=8, eval_batch=4, eval_every=1)


@dataclass(frozen=True)
class Arm:
    name: str
    codec: Optional[str]  # None | "sign" | "Z" | "E8"
    rate: Optional[float]
    ef: bool
    role: str


ARMS: Tuple[Arm, ...] = (
    Arm("none", None, None, False, "baseline"),
    Arm("Z8@1+EF", "Z", 1.0, True, "arm"),
    Arm("E8@1+EF", "E8", 1.0, True, "arm"),
    Arm("sign", "sign", 1.0, False, "positive control"),
    Arm("E8@6", "E8", FINE_RATE, False, "negative control (fine scale)"),
)


def estimate(cfg: Cfg = FULL, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Planned step counts and wall-clock estimate (no compute)."""
    n_runs = len(ARMS) * len(seeds)
    steps_run = cfg.rounds * cfg.learners * cfg.inner_steps
    steps = n_runs * steps_run
    lattice_arms = sum(1 for a in ARMS if a.codec in ("Z", "E8"))
    codec_calls = lattice_arms * len(seeds) * cfg.rounds * cfg.learners
    evals = n_runs * (cfg.rounds // cfg.eval_every + 1)
    parts = {"inner_steps_s": steps * SEC_PER_STEP, "lattice_codec_s": codec_calls * SEC_PER_CODEC_CALL, "eval_s": evals * SEC_PER_EVAL, "setup_and_controls_s": SEC_SETUP}
    total = sum(parts.values())
    return {"runs": n_runs, "arms": [a.name for a in ARMS], "seeds": list(seeds), "inner_steps_per_run": steps_run, "inner_steps_total": steps,
            "tokens_per_learner_per_run": cfg.rounds * cfg.inner_steps * cfg.micro_batch * cfg.seq_len, "lattice_codec_calls": codec_calls, "eval_passes": evals,
            "assumed_sec_per_inner_step": SEC_PER_STEP, "assumed_sec_per_lattice_codec_call": SEC_PER_CODEC_CALL, "assumed_sec_per_eval": SEC_PER_EVAL,
            "breakdown_s": {k: round(v, 1) for k, v in parts.items()}, "total_s": round(total, 1), "total_hours": round(total / 3600.0, 2)}


def mean_ci(xs: Sequence[float]) -> Tuple[float, float]:
    """Mean and df = 2 two-sided 95% t half-width (three seeds)."""
    if len(xs) != 3:
        raise ValueError("t-interval is fixed for df = 2 (three seeds)")
    return statistics.fmean(xs), T_CRIT_DF2 * statistics.stdev(xs) / math.sqrt(3.0)


def clean(o: Any) -> Any:
    """JSON-safe copy: non-finite floats become null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


# ---------------------------------------------------------------- codec (torch, on device)
class TorchLatticeCodec:
    """Blockwise Z8 / E8 quantiser of a flat delta with EF21 error feedback and per-call rate calibration.

    Same arithmetic as ``LatticeDeltaQuantizer`` (float64 decode, torch ports are bit-identical to the numpy
    decoders), but the scale is found by bisection on log-scale on the vector actually quantised (delta + carry) so
    that its measured factorised rate hits ``rate`` (tol 0.02 bit/dim, same rule as ``calibrate_scale_for_rate``)."""

    def __init__(self, kind: str, rate: float, error_feedback: bool, block: int = 8, tol: float = 0.02, iters: int = 60) -> None:
        self.kind, self.rate, self.error_feedback, self.block, self.tol, self.iters = kind, float(rate), bool(error_feedback), block, tol, iters
        self._carry: Optional[torch.Tensor] = None
        self.last_scale = float("nan")

    def _rate_and_code(self, rows: torch.Tensor, ctx: torch.Tensor, scale: float) -> Tuple[float, torch.Tensor]:
        c = DECODERS_T[self.kind](rows / scale)
        return factorised_rate_t(self.kind, c, ctx), c

    def calibrate(self, rows: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        ctx = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
        sd = float(rows.std(unbiased=False)) or 1.0
        lo, hi = math.log(sd * 1e-3), math.log(sd * 1e3)
        r_lo, _ = self._rate_and_code(rows, ctx, math.exp(lo))
        r_hi, _ = self._rate_and_code(rows, ctx, math.exp(hi))
        if not (r_hi <= self.rate <= r_lo):
            raise ValueError(f"target {self.rate} bits/dim outside the reachable range [{r_hi:.3f}, {r_lo:.3f}]")
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            r, c = self._rate_and_code(rows, ctx, math.exp(mid))
            if abs(r - self.rate) <= self.tol:
                return math.exp(mid), r, c
            if r > self.rate:
                lo = mid
            else:
                hi = mid
        s = math.exp(0.5 * (lo + hi))
        r, c = self._rate_and_code(rows, ctx, s)
        return s, r, c

    @torch.no_grad()
    def quantize(self, vec: torch.Tensor) -> QuantizedDelta:
        v = vec.detach().to(torch.float32).reshape(-1)
        if self.error_feedback and self._carry is not None:
            v = v + self._carry
        n = v.numel()
        pad = (-n) % self.block
        x = torch.cat([v, v.new_zeros(pad)]) if pad else v
        rows = x.to(torch.float64).reshape(-1, self.block)
        scale, rate, c = self.calibrate(rows)
        self.last_scale = scale
        deq = (c * scale).reshape(-1)[:n].to(torch.float32)
        residual = v - deq
        if self.error_feedback:
            self._carry = residual.clone()
        vn = float(torch.dot(v, v))
        rel = float(torch.dot(residual, residual)) / vn if vn > 0.0 else 0.0
        return QuantizedDelta(deq, rate, rel, residual)


def make_codec(arm: Arm) -> Any:
    if arm.codec is None:
        return None
    if arm.codec == "sign":
        return SignDeltaQuantizer(error_feedback=arm.ef)
    return TorchLatticeCodec(arm.codec, arm.rate or 0.0, arm.ef)


# ---------------------------------------------------------------- model, data
def _e12() -> Any:
    import run_experiment_E12_replay_divergence as e12  # noqa: PLC0415 (heavy import kept lazy)

    return e12


def attach_lora(model: torch.nn.Module, init_seed: int) -> None:
    """Same wrapping and init order as E12's ``build_model`` (q_proj then v_proj per layer, A ~ N(0, 1/r), B = 0)."""
    e12 = _e12()
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(init_seed)
    for layer in model.model.layers:
        layer.self_attn.q_proj = e12.LoRALinear(layer.self_attn.q_proj, e12.RANK, e12.ALPHA, gen)
        layer.self_attn.v_proj = e12.LoRALinear(layer.self_attn.v_proj, e12.RANK, e12.ALPHA, gen)


@torch.no_grad()
def reset_lora(model: torch.nn.Module, init_seed: int) -> None:
    """E10b init (amendment 2): A ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (PEFT default kaiming_uniform a=sqrt 5), B = 0, from the seed's
    generator, shared by all learners of a run. Overwrites the E12-style N(0, 1/r) init that ``attach_lora`` / E12's ``build_model``
    draw, which diverged in the first execution; every run starts here."""
    gen = torch.Generator().manual_seed(init_seed)
    for layer in model.model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.v_proj):
            r, fan_in = proj.lora_A.shape
            a = (torch.rand(r, fan_in, generator=gen) * 2.0 - 1.0) / math.sqrt(fan_in)
            proj.lora_A.copy_(a.to(proj.lora_A.dtype).to(proj.lora_A.device))
            proj.lora_B.zero_()


def lora_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def get_flat(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params]).clone()


@torch.no_grad()
def set_flat(params: Sequence[torch.Tensor], flat: torch.Tensor) -> None:
    o = 0
    for p in params:
        p.copy_(flat[o : o + p.numel()].view_as(p))
        o += p.numel()


def tiny_model(device: str) -> torch.nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).to(device)
    m.train(False)
    attach_lora(m, 1)
    return m


def real_model(device: str) -> torch.nn.Module:
    return _e12().build_model(device, torch.float32, SEEDS[0])


def _pack_rows(tok: Any, pq_file: Any, first_row: int, n_seq: int, seq_len: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Tokenise consecutive fineweb rows from ``first_row`` (documents joined by EOS) into ``n_seq`` sequences."""
    need = n_seq * seq_len
    ids: List[int] = []
    rg = first_row // 1000
    skip = first_row - rg * 1000
    rows_used = 0
    while len(ids) < need:
        texts = pq_file.read_row_group(rg, columns=["text"]).column("text").to_pylist()[skip:]
        skip = 0
        for t in texts:
            ids.extend(tok(t, add_special_tokens=False).input_ids + [tok.eos_token_id])
            rows_used += 1
            if len(ids) >= need:
                break
        rg += 1
    out = torch.tensor(ids[:need], dtype=torch.long).reshape(n_seq, seq_len)
    return out, {"first_row": first_row, "rows_used": rows_used, "tokens": need, "sha256": hashlib.sha256(out.numpy().tobytes()).hexdigest()}


def real_data(cfg: Cfg) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sorted(glob.glob(str(HUB / ("models--" + MODEL_REPO.replace("/", "--")) / "snapshots" / "*")))[0])
    pf = pq.ParquetFile(sorted(glob.glob(FINEWEB_GLOB))[0])
    n_train = cfg.rounds * cfg.inner_steps * cfg.micro_batch
    shards, prov = [], {}
    for i in range(cfg.learners):
        s, info = _pack_rows(tok, pf, i * SHARD_ROW_STRIDE, n_train, cfg.seq_len)
        shards.append(s)
        prov[f"learner{i}"] = info
    held, info = _pack_rows(tok, pf, cfg.learners * SHARD_ROW_STRIDE, cfg.eval_seqs, cfg.seq_len)
    prov["heldout"] = info
    return shards, held, prov


def smoke_data(cfg: Cfg) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    g = torch.Generator().manual_seed(7)
    n_train = cfg.rounds * cfg.inner_steps * cfg.micro_batch
    # learner i draws from a different slice of the vocabulary (non-IID in miniature); all ids < 256
    shards = [torch.randint(32 * i, 32 * i + 128, (n_train, cfg.seq_len), generator=g) for i in range(cfg.learners)]
    held = torch.randint(0, 256, (cfg.eval_seqs, cfg.seq_len), generator=g)
    return shards, held, {"synthetic": True}


# ---------------------------------------------------------------- DiLoCo run
@torch.no_grad()
def eval_loss(model: torch.nn.Module, ids: torch.Tensor, bs: int, device: str) -> float:
    tot = 0.0
    for s in range(0, ids.shape[0], bs):
        b = ids[s : s + bs].to(device)
        tot += float(model(input_ids=b, labels=b, use_cache=False).loss) * b.shape[0]
    return tot / ids.shape[0]


class FailFast(Exception):
    """Registered fail-fast rule (amendment 2) tripped inside the first 'none' cell."""

    def __init__(self, reason: str, details: Dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason, self.details = reason, details


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_arm(model: torch.nn.Module, params: List[torch.nn.Parameter], arm: Arm, seed: int, shards: List[torch.Tensor], held: torch.Tensor, cfg: Cfg, device: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    reset_lora(model, seed)  # always: overwrites the E12-style init drawn by attach_lora / build_model
    a_std = statistics.fmean(float(p.std()) for n, p in model.named_parameters() if "lora_A" in n)
    failfast = arm.name == "none" and seed == SEEDS[0]  # the first cell of the run
    n_steps = 0
    t_steps = 0.0
    torch.manual_seed(seed)
    theta = get_flat(params)
    buf = torch.zeros_like(theta)
    opts = [torch.optim.AdamW(params, lr=INNER_LR, betas=INNER_BETAS, weight_decay=INNER_WD) for _ in range(cfg.learners)]  # one persistent state per learner
    orders = [np.random.RandomState(1000 * seed + i).permutation(shards[i].shape[0]) for i in range(cfg.learners)]
    pos = [0] * cfg.learners
    codecs = [make_codec(arm) for _ in range(cfg.learners)]
    curve: Dict[str, float] = {"0": eval_loss(model, held, cfg.eval_batch, device)}
    rel: List[float] = []
    rates: List[float] = []
    scales: List[float] = []
    diverged = False
    inner_last: List[float] = []
    for rnd in range(1, cfg.rounds + 1):
        deltas = []
        for i in range(cfg.learners):
            set_flat(params, theta)
            for _ in range(cfg.inner_steps):
                if failfast and n_steps == 0:
                    _sync(device)
                    t_steps = time.perf_counter()
                idx = orders[i][pos[i] : pos[i] + cfg.micro_batch]
                pos[i] += cfg.micro_batch
                b = shards[i][torch.as_tensor(idx)].to(device)
                loss = model(input_ids=b, labels=b, use_cache=False).loss
                opts[i].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                opts[i].step()
                n_steps += 1
                if failfast and n_steps == FAILFAST_STEPS:
                    _sync(device)
                    sps = (time.perf_counter() - t_steps) / FAILFAST_STEPS
                    if sps > SEC_PER_STEP * FAILFAST_SEC_FACTOR:
                        raise FailFast(f"measured {sps:.3f} s per inner step over the first {FAILFAST_STEPS} steps exceeds {SEC_PER_STEP * FAILFAST_SEC_FACTOR:.3f} s ({FAILFAST_SEC_FACTOR} x the registered {SEC_PER_STEP})",
                                       {"sec_per_inner_step": sps, "loss_curve": dict(curve)})
            inner_last.append(float(loss))
            d = theta - get_flat(params)
            if not bool(torch.isfinite(d).all()) or not math.isfinite(inner_last[-1]):
                diverged = True
                break
            if codecs[i] is not None:
                q = codecs[i].quantize(d)
                rel.append(q.relative_mse)
                rates.append(q.rate_bits_per_dim)
                if hasattr(codecs[i], "last_scale"):
                    scales.append(codecs[i].last_scale)
                d = q.dequantized
            deltas.append(d)
        if diverged:
            break
        g = torch.stack(deltas).mean(dim=0)  # pseudo-gradient theta_global - theta_local, averaged over learners
        buf = OUTER_MOMENTUM * buf + g
        theta = theta - OUTER_LR * (g + OUTER_MOMENTUM * buf)  # Nesterov SGD, as torch.optim.SGD(nesterov=True)
        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            set_flat(params, theta)
            curve[str(rnd)] = eval_loss(model, held, cfg.eval_batch, device)
            if not math.isfinite(curve[str(rnd)]):
                diverged = True
                break
            if failfast and rnd == FAILFAST_ROUND and curve[str(rnd)] >= curve["0"]:
                raise FailFast(f"held-out loss at round {FAILFAST_ROUND} ({curve[str(rnd)]:.4f}) >= round 0 ({curve['0']:.4f}): the 'none' arm does not learn", {"loss_curve": dict(curve)})
    final = curve.get(str(cfg.rounds), float("inf")) if not diverged else float("inf")
    return {"arm": arm.name, "seed": seed, "diverged": diverged, "final_loss": final, "initial_loss": curve["0"], "loss_curve": curve, "lora_A_std": a_std,
            "mean_relative_mse": statistics.fmean(rel) if rel else 0.0, "mean_rate_bits_per_dim": statistics.fmean(rates) if rates else 32.0,
            "scale_min": min(scales) if scales else None, "scale_max": max(scales) if scales else None, "last_inner_loss_mean": statistics.fmean(inner_last) if inner_last else None,
            "wall_seconds": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------- controls and analysis
def torch_port_check(seed: int = 2026, n: int = 2000) -> Dict[str, Any]:
    """Negative control 2b: the torch decoders (and factorised rate with a constant context) reproduce the numpy ones exactly."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    for kind in ("Z", "E8"):
        x = rng.normal(0.0, 1.5, size=(n, 8))
        c_np = DECODERS[kind](x)
        c_t = DECODERS_T[kind](torch.from_numpy(x))
        r_np = factorised_rate_bits_per_dim(kind, c_np)
        r_t = factorised_rate_t(kind, c_t, torch.zeros(n, dtype=torch.int64))
        out[kind] = {"decoder_identical": bool(np.array_equal(c_np, c_t.numpy())), "rate_numpy": r_np, "rate_torch": r_t, "rate_abs_diff": abs(r_np - r_t)}
        out[kind]["passed"] = out[kind]["decoder_identical"] and out[kind]["rate_abs_diff"] < 1e-9
    out["passed"] = all(out[k]["passed"] for k in ("Z", "E8"))
    return out


def analyse(results: List[Dict[str, Any]], seeds: Sequence[int], run_brute_force: bool) -> Dict[str, Any]:
    def get(arm: str, seed: int) -> Dict[str, Any]:
        return next(r for r in results if r["arm"] == arm and r["seed"] == seed)

    def fin(arm: str) -> List[float]:
        return [get(arm, s)["final_loss"] for s in seeds]

    def any_div(*arms: str) -> bool:
        return any(get(a, s)["diverged"] for a in arms for s in seeds)

    cells: Dict[str, Any] = {}
    for a in ARMS:
        rows = [get(a.name, s) for s in seeds]
        ok = not any(r["diverged"] for r in rows)
        lm, lh = mean_ci([r["final_loss"] for r in rows]) if ok else (float("nan"), float("nan"))
        cells[a.name] = {"role": a.role, "final_loss_mean": lm, "final_loss_ci95": lh, "final_loss_per_seed": [r["final_loss"] for r in rows], "diverged_seeds": [r["seed"] for r in rows if r["diverged"]],
                         "mean_relative_mse": statistics.fmean(r["mean_relative_mse"] for r in rows), "mean_rate_bits_per_dim": statistics.fmean(r["mean_rate_bits_per_dim"] for r in rows)}

    # negative control 3 first: an unmatched R invalidates the P10 comparisons at that R; the headline R = 1 must match
    neg3: Dict[str, Any] = {}
    valid: List[float] = []
    for r in RATES:
        rz, re8 = cells[f"Z8@{r:g}+EF"]["mean_rate_bits_per_dim"], cells[f"E8@{r:g}+EF"]["mean_rate_bits_per_dim"]
        ok = abs(rz - re8) <= MATCH_TOL
        neg3[f"R={r}"] = {"z8_rate": rz, "e8_rate": re8, "matched": ok}
        if ok:
            valid.append(r)
    neg3_ok = 1.0 in valid

    def ratio_status() -> Tuple[Dict[str, Any], str]:
        cs: Dict[str, Any] = {}
        a_ok, a_ref = bool(valid), False
        for r in RATES:
            ratios = [get(f"E8@{r:g}+EF", s)["mean_relative_mse"] / get(f"Z8@{r:g}+EF", s)["mean_relative_mse"] for s in seeds]
            m, h = mean_ci(ratios)
            cs[f"R={r}|EF=on"] = {"ratio_mean": m, "ci95": h, "per_seed": ratios, "gaussian_prediction": 0.86, "e10_toy_ratio_R1": 0.81, "rate_matched": r in valid}
            if r in valid:
                a_ok &= (m + h) < 1.0
                a_ref |= (m - h) > 1.0
        bad = [r for r in RATES if r not in valid]
        txt = "supported" if a_ok else ("refuted" if a_ref else "inconclusive (CI includes 1 at some matched rate)")
        return cs, txt + (f"; R in {bad} invalid (matched-rate control failed)" if bad else "")

    if any_div(*[f"{k}@{r:g}+EF" for k in ("Z8", "E8") for r in RATES]):
        p10a_cells, p10a_status = {}, "invalid (a lattice arm diverged)"
    else:
        p10a_cells, p10a_status = ratio_status()

    p10b: Dict[str, Any] = {}
    for r in RATES:
        za, ea = f"Z8@{r:g}+EF", f"E8@{r:g}+EF"
        if any_div(za, ea):
            p10b[f"R={r}|EF=on"] = {"status": "invalid (diverged run)"}
            continue
        diffs = [get(za, s)["final_loss"] - get(ea, s)["final_loss"] for s in seeds]
        m, h = mean_ci(diffs)
        st = "supported" if m - h > 0 else ("refuted" if m + h < 0 else "inconclusive (CI includes 0)")
        p10b[f"R={r}|EF=on"] = {"z8_minus_e8_mean": m, "ci95": h, "per_seed": diffs, "status": st if r in valid else "invalid (matched-rate control failed)"}
    p10b_status = p10b["R=1.0|EF=on"]["status"]

    def vs_none(arm: str) -> Dict[str, Any]:
        if any_div(arm, "none"):
            return {"status": "invalid (diverged run)"}
        diffs = [get(arm, s)["final_loss"] - get("none", s)["final_loss"] for s in seeds]
        m, h = mean_ci(diffs)
        return {"minus_none_mean": m, "ci95": h, "per_seed": diffs, "status": "refuted" if m - h > 0 else "supported"}

    p10c = vs_none("E8@1+EF")
    p10c["context_only"] = {a: vs_none(a) for a in ("Z8@1+EF",)}

    pos_rows = [{"seed": s, "sign_noEF": get("sign", s)["final_loss"], "none": get("none", s)["final_loss"]} for s in seeds]
    pos_ok = all(r["sign_noEF"] > r["none"] for r in pos_rows)  # a diverged sign run (inf) counts as hurting
    neg1_rows = [{"seed": s, "e8_fine": get("E8@6", s)["final_loss"], "none": get("none", s)["final_loss"]} for s in seeds]
    neg1_ok = all(abs(r["e8_fine"] - r["none"]) <= FINE_TOL for r in neg1_rows)
    learns_rows = [{"seed": s, "initial": get("none", s)["initial_loss"], "final": get("none", s)["final_loss"]} for s in seeds]
    learns_ok = all(r["final"] < r["initial"] for r in learns_rows)
    neg2 = {"torch_port_equals_numpy": torch_port_check()}
    if run_brute_force:
        neg2.update({k: brute_force_decoder_check(k, 1000, 2026) for k in ("Z", "E8")})
    else:
        neg2["brute_force"] = "skipped (smoke)"
    neg2_ok = all(v["passed"] for v in neg2.values() if isinstance(v, dict))
    controls = {"positive_sign_hurts": {"rows": pos_rows, "passed": pos_ok},
                "negative_fine_scale_e8_equals_none": {"rows": neg1_rows, "tolerance": FINE_TOL, "passed": neg1_ok},
                "negative_decoders": {**neg2, "passed": neg2_ok},
                "negative_matched_rate": {**neg3, "valid_rates": valid, "passed": neg3_ok},
                "none_arm_learns": {"rows": learns_rows, "passed": learns_ok}}
    all_ok = bool(pos_ok and neg1_ok and neg2_ok and neg3_ok and learns_ok)
    analysis = {"P10a": {"cells": p10a_cells, "status": p10a_status}, "P10b": {"cells": p10b, "status": p10b_status}, "P10c": p10c}
    if not all_ok:  # registered rule 'run_is_valid_iff': report the computed status, but the prediction is invalid
        failed = [k for k, v in controls.items() if not v["passed"]]
        for v in analysis.values():
            v["computed_status"] = v["status"]
            v["status"] = f"invalid (controls failed: {', '.join(failed)})"
    return {"cells": cells, "analysis": analysis, "controls": controls, "controls_all_passed": all_ok}


# ---------------------------------------------------------------- main
def load_partial(path: Path, design_sha: str) -> List[Dict[str, Any]]:
    """Cells of a partial file, only if it was written under the same design hash (never resume across designs)."""
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    if not isinstance(blob, dict) or blob.get("design_sha256") != design_sha:
        print(f"[E10b] ignoring {path.name}: written under a different design", flush=True)
        return []
    runs = blob["runs"]
    for r in runs:  # null -> inf for diverged runs
        r["final_loss"] = float("inf") if r["final_loss"] is None else r["final_loss"]
    return runs


def write_aborted(out: Path, ff: FailFast, results: List[Dict[str, Any]], prereg_sha: str, amend_sha: str, design_sha: str, smoke: bool, device: str) -> int:
    status = f"aborted by registered fail-fast rule: {ff.reason}"
    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E10b/v1.0", "experiment_id": "E10b-lattice-05b", "smoke": smoke, "status": status,
        "prereg_sha256": prereg_sha, "amendment_2_sha256": amend_sha, "design_sha256": design_sha, "fail_fast_details": ff.details,
        "completed_runs": results, "analysis": {k: {"status": "invalid"} for k in ("P10a", "P10b", "P10c")}, "controls_all_passed": False,
        "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None, "torch": torch.__version__},
    }
    report = clean(report)
    report["receipt_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(status, flush=True)
    return EXIT_FAILFAST


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny random Qwen2, CPU, 2 rounds x 2 learners; writes only to --out (plumbing check, not evidence)")
    ap.add_argument("--estimate", action="store_true", help="print planned step counts and time estimate; no compute")
    ap.add_argument("--out", help="report path (required with --smoke; never inside artifacts/)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="skip (arm, seed) cells already in <report>.partial.json")
    a = ap.parse_args(argv)
    cfg = SMOKE if a.smoke else FULL

    if a.estimate:
        print(json.dumps(estimate(FULL), indent=2))
        return 0
    if a.smoke:
        if not a.out:
            ap.error("--smoke requires --out")
        out = Path(a.out).resolve()
        if EXP_DIR.resolve() in out.parents or (PROJECT_ROOT / "artifacts").resolve() in out.parents:
            ap.error("--smoke must not write into artifacts/")
        device = "cpu"
        torch.set_num_threads(2)  # other CPU jobs share this host
    else:
        out, device = REPORT_PATH, a.device
        for need in (PREREG_PATH, AMENDMENT_PATH):
            if not need.exists():
                raise SystemExit(f"{need} not found: the full run needs the registered prereg AND amendment 2 (the _DRAFT is not a registration)")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("full run needs a CUDA device")
    prereg_path = PREREG_PATH if PREREG_PATH.exists() else PREREG_DRAFT_PATH
    prereg_sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    amend_sha = hashlib.sha256(AMENDMENT_PATH.read_bytes()).hexdigest() if AMENDMENT_PATH.exists() else "absent"
    design_sha = hashlib.sha256((prereg_sha + amend_sha).encode("utf-8")).hexdigest()  # identifies the design a partial file belongs to

    started = time.perf_counter()
    shards, held, data_prov = smoke_data(cfg) if a.smoke else real_data(cfg)
    model = tiny_model(device) if a.smoke else real_model(device)
    model.train(False)
    params = lora_params(model)
    n_coords = sum(p.numel() for p in params)
    print(f"[E10b] {'SMOKE ' if a.smoke else ''}device {device}, {n_coords} LoRA coordinates, plan: {json.dumps(estimate(cfg))}", flush=True)

    partial = Path(str(out) + ".partial.json")
    results: List[Dict[str, Any]] = load_partial(partial, design_sha) if a.resume else []
    done = {(r["arm"], r["seed"]) for r in results}
    for seed in SEEDS:  # seed-major so an interrupted run still holds complete seeds
        for arm in ARMS:
            if (arm.name, seed) in done:
                continue
            try:
                r = run_arm(model, params, arm, seed, shards, held, cfg, device)
            except FailFast as ff:
                return write_aborted(out, ff, results, prereg_sha, amend_sha, design_sha, bool(a.smoke), device)
            results.append(r)
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps({"design_sha256": design_sha, "runs": clean(results)}), encoding="utf-8")
            print(f"seed {seed} {arm.name:8s} final {r['final_loss']:.4f} (init {r['initial_loss']:.4f}) relMSE {r['mean_relative_mse']:.4f} rate {r['mean_rate_bits_per_dim']:.3f} "
                  f"{'DIVERGED ' if r['diverged'] else ''}{r['wall_seconds']}s", flush=True)

    res = analyse(results, SEEDS, run_brute_force=not a.smoke)
    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E10b/v1.0", "experiment_id": "E10b-lattice-05b", "smoke": bool(a.smoke),
        "prereg_sha256": prereg_sha, "prereg_file": prereg_path.name, "prereg_is_draft": prereg_path == PREREG_DRAFT_PATH,
        "amendment_2_sha256": amend_sha, "design_sha256": design_sha,
        "design": {"model": "tiny random Qwen2 (smoke)" if a.smoke else MODEL_REPO, "cfg": cfg.__dict__, "seeds": list(SEEDS), "outer_lr": OUTER_LR, "outer_momentum": OUTER_MOMENTUM,
                   "inner": {"lr": INNER_LR, "betas": list(INNER_BETAS), "weight_decay": INNER_WD, "grad_clip": GRAD_CLIP}, "lora_coordinates": n_coords,
                   "lora_init": "A ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)), B = 0 (amendment 2)",
                   "calibration": "per (learner, round) bisection of the scale on delta + carry to the target measured factorised rate, tol 0.02 bit/dim"},
        "data": data_prov,
        "hardware": {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None, "cpu": platform.processor() or platform.machine(), "torch": torch.__version__},
        "num_runs": len(results), "runs": results, **res, "estimate": estimate(cfg), "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    if a.smoke:
        report["warning"] = "SMOKE RUN: tiny random model and synthetic tokens; the statuses below test the analysis plumbing only and are not evidence"
    report = clean(report)
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    an = report["analysis"]
    print(json.dumps({"P10a": an["P10a"]["status"], "P10b": an["P10b"]["status"], "P10c": an["P10c"]["status"], "controls_all_passed": report["controls_all_passed"],
                      "runs": len(results), "elapsed_s": report["elapsed_seconds"], "receipt": report["receipt_sha256"]}, indent=2))
    return 0 if (a.smoke or report["controls_all_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
