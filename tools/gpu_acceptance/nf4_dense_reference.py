#!/usr/bin/env python
"""Compare HiveMind's NF4 packed path against an independent dense reference.

Always runs CPU CORRECTNESS checks. GPU memory and latency are measured ONLY when
torch.cuda is available; otherwise the report says so and no GPU number is emitted.

What this does and does not establish
-------------------------------------
* The repo reconstructs a dense weight from the packed NF4 nibbles and executes
  with that dense tensor (quantization.py: ``_nf4_storage_mode = "dense_reconstruction"``).
  This tool therefore checks decode correctness and error, not fused-NF4 execution.
* A passing correctness run is NOT evidence of memory-residency savings.
* CPU timings are CPU wall-clock and are not comparable to GPU latency.

Usage (from the repository root, with the project's Python):

    python tools/gpu_acceptance/nf4_dense_reference.py                 # CPU checks; CUDA measurements if available
    python tools/gpu_acceptance/nf4_dense_reference.py --out nf4.json
    python tools/gpu_acceptance/nf4_dense_reference.py --require-cuda   # acceptance run; exit 2 if no CUDA

Exit codes: 0 requested checks/measurements passed; 1 a check or measurement failed;
2 CUDA required but unavailable, a non-CUDA backend, or incomplete CUDA measurements.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "src").is_dir() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SCHEMA_VERSION = 2
NO_CUDA_MESSAGE = "CUDA unavailable: GPU memory/latency measurements NOT performed"

# QLoRA / bitsandbytes published 16-level NormalFloat table, typed here so the reference does
# not import the repo constant. main() also asserts that it equals the repo's NF4_CODEBOOK.
REFERENCE_NF4_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]
_CODE = torch.tensor(REFERENCE_NF4_CODEBOOK, dtype=torch.float64)
MAX_CODE_GAP = float((_CODE[1:] - _CODE[:-1]).max())  # largest spacing between adjacent codes
ABSMAX_FLOOR = 1e-12  # the repo clamps block absmax to this before double quantization

# Stated tolerances (all relative to the per-block dequantized absmax "dq", unless noted).
TOLERANCES = {
    "scale_byte_near_tie_window": 1e-4,          # |frac(absmax/super) - 0.5| below this may flip a rounded scale byte
    "index_near_tie_window": 1e-6,               # |dist_a - dist_b| below this may flip a nibble (fp32 vs fp64)
    "repo_vs_reference_abs": "2e-6*dq + sub + finfo(dtype).eps*max(|ref|,|repo|)",
    "roundtrip_abs": "0.5*max_code_gap*dq + max(true_absmax-dq,0) + 2e-6*dq + sub + finfo(dtype).eps*max(|x|,|rec|)",
    "gaussian_rel_l2_max": 0.15,                 # informational gate for N(0,1) data with >= 256 elements
    "int4_row_roundtrip_abs": "(0.5 + 7*2^-10)*scale_fp32 + sub + finfo(dtype).eps*max(|x|,|rec|)",
    "sub": "0.5*finfo(dtype).tiny*finfo(dtype).eps (half the dtype subnormal spacing)",
}


def dtype_supported(dtype: torch.dtype, device: str = "cpu") -> bool:
    try:
        x = torch.ones(8, dtype=dtype, device=device)
        return bool(torch.isfinite((x * 2).float()).all())
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------
# Independent dense reference (float64, brute-force nearest code; shares no code with the repo)
# --------------------------------------------------------------------------------------


def reference_blocks(weight: torch.Tensor, block_size: int):
    x = weight.detach().to(torch.float32).reshape(-1).to(torch.float64)
    numel = x.numel()
    pad = (-numel) % block_size
    if pad:
        x = F.pad(x, (0, pad))
    blocks = x.reshape(-1, block_size)
    true_absmax = blocks.abs().amax(dim=1)
    absmax = true_absmax.clamp(min=ABSMAX_FLOOR)
    super_scale = float(absmax.max()) / 255.0
    ratio = absmax / super_scale
    q = torch.round(ratio).clamp(1, 255).to(torch.uint8)
    return blocks, true_absmax, ratio, q, super_scale


def decode_nibbles(packed: torch.Tensor, padded_numel: int) -> np.ndarray:
    raw = packed.numpy()
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = raw & 0x0F
    out[1::2] = raw >> 4
    return out[:padded_numel]


def nearest_code(normalized: torch.Tensor):
    dist = (normalized.unsqueeze(-1) - _CODE).abs()
    best = dist.argmin(dim=-1)
    return best, dist


# --------------------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------------------


def make_tensor(kind: str, shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    n = math.prod(shape)
    base = torch.randn(n, generator=gen, dtype=torch.float32)
    if kind == "gaussian":
        x = base
    elif kind == "zeros":
        x = torch.zeros(n)
    elif kind == "constant":
        x = torch.full((n,), 3.140625)
    elif kind == "alternating_sign":
        x = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
    elif kind == "single_outlier":
        x = base * 1e-2
        x[n // 2] = 1e3 if dtype != torch.float16 else 4e4
    elif kind == "mixed_block_magnitude":
        x = base * torch.logspace(-3, 3, steps=max(1, n // 64 + 1))[(torch.arange(n) // 64)]
    elif kind == "tiny":
        x = base * (1e-6 if dtype == torch.float16 else 1e-20)
    elif kind == "huge":
        x = base * (1e4 if dtype == torch.float16 else 1e30)
    else:
        raise ValueError(kind)
    return x.reshape(shape).to(dtype)


# --------------------------------------------------------------------------------------
# Repo NF4 path (hivemind.core.quantization) checked against the reference
# --------------------------------------------------------------------------------------


def check_nf4_case(weight: torch.Tensor, block_size: int, kind: str) -> dict:
    from hivemind.core.quantization import dequantize_tensor_nf4, quantize_tensor_nf4_double_quant

    dtype = weight.dtype
    eps = float(torch.finfo(dtype).eps)
    sub = 0.5 * float(torch.finfo(dtype).tiny) * eps  # half the spacing of the dtype's subnormals
    before = weight.clone()
    state, reconstructed = quantize_tensor_nf4_double_quant(weight, block_size)
    checks: dict[str, bool] = {}
    metrics: dict[str, float | int] = {}

    checks["input_not_mutated"] = bool(torch.equal(weight, before))
    numel = weight.numel()
    if numel == 0:
        checks["empty_roundtrip_shape"] = tuple(reconstructed.shape) == tuple(weight.shape)
        return {"checks": checks, "metrics": metrics}

    num_blocks = -(-numel // block_size)
    padded = num_blocks * block_size
    checks["packed_size"] = state.packed_4bit.numel() == (padded + 1) // 2
    checks["scale_size"] = state.q_absmax_uint8.numel() == num_blocks
    checks["storage_bytes_formula"] = state.storage_bytes == state.packed_4bit.nbytes + state.q_absmax_uint8.nbytes + 8
    checks["dequant_equals_returned_dense"] = bool(
        torch.equal(dequantize_tensor_nf4(state, dtype=dtype), reconstructed)
    )

    blocks, true_absmax, ratio, q_ref, super_ref = reference_blocks(weight, block_size)

    # Stage 1: double-quantized scales. A scale byte may differ only on a rounding tie.
    q_repo = state.q_absmax_uint8.to(torch.int64)
    q_mismatch = q_repo != q_ref.to(torch.int64)
    near_tie = (ratio - torch.floor(ratio) - 0.5).abs() < TOLERANCES["scale_byte_near_tie_window"]
    metrics["scale_byte_mismatches"] = int(q_mismatch.sum())
    checks["scale_bytes_match_except_ties"] = bool((~q_mismatch | near_tie).all())
    checks["super_scale_matches"] = math.isclose(state.superblock_scale, super_ref, rel_tol=1e-6)

    # Stage 2: nibbles, decoded independently and compared to brute-force nearest-code
    # using the repo's own (already verified) scales so a tie in stage 1 cannot cascade.
    dq = q_repo.to(torch.float64) * state.superblock_scale
    normalized = (blocks / dq[:, None]).clamp(-1.0, 1.0)
    ref_idx, dist = nearest_code(normalized)
    repo_idx = torch.from_numpy(decode_nibbles(state.packed_4bit, padded).astype(np.int64)).reshape(num_blocks, block_size)
    idx_mismatch = repo_idx != ref_idx
    repo_dist = dist.gather(-1, repo_idx.unsqueeze(-1)).squeeze(-1)
    ref_dist = dist.gather(-1, ref_idx.unsqueeze(-1)).squeeze(-1)
    idx_tie = (repo_dist - ref_dist).abs() <= TOLERANCES["index_near_tie_window"]
    metrics["nibble_mismatches"] = int(idx_mismatch.sum())
    metrics["nibble_mismatches_that_are_near_ties"] = int((idx_mismatch & idx_tie).sum())
    checks["nibbles_match_except_ties"] = bool((~idx_mismatch | idx_tie).all())

    # Stage 3: dense reconstruction, repo vs reference, excluding flagged tie flips.
    ref_dense64 = (_CODE[ref_idx] * dq[:, None]).reshape(-1)[:numel]
    ref_dense = ref_dense64.to(torch.float32).to(dtype).to(torch.float64)
    repo_dense = reconstructed.detach().reshape(-1).to(torch.float64)
    keep = ~(idx_mismatch.reshape(-1)[:numel])
    block_dq = dq.repeat_interleave(block_size)[:numel]
    diff = (repo_dense - ref_dense).abs()
    tol = 2e-6 * block_dq + sub + eps * torch.maximum(ref_dense.abs(), repo_dense.abs())
    metrics["repo_vs_reference_max_abs"] = float(diff[keep].max()) if bool(keep.any()) else 0.0
    nz = ref_dense.abs() > 0
    sel = keep & nz
    metrics["repo_vs_reference_max_rel"] = float((diff[sel] / ref_dense.abs()[sel]).max()) if bool(sel.any()) else 0.0
    checks["dense_matches_reference_within_tolerance"] = bool((diff[keep] <= tol[keep]).all())

    # Stage 4: analytic round-trip bound against the ORIGINAL values.
    orig = weight.detach().to(torch.float32).reshape(-1).to(torch.float64)
    err = (orig - repo_dense).abs()
    true_absmax_e = true_absmax.repeat_interleave(block_size)[:numel]
    bound = (
        0.5 * MAX_CODE_GAP * block_dq
        + torch.clamp(true_absmax_e - block_dq, min=0.0)
        + 2e-6 * block_dq
        + sub
        + eps * torch.maximum(orig.abs(), repo_dense.abs())
    )
    metrics["roundtrip_max_abs_err"] = float(err.max())
    metrics["roundtrip_max_err_over_block_scale"] = float((err / block_dq).max())
    norm = float(orig.norm())
    metrics["roundtrip_rel_l2"] = float((orig - repo_dense).norm() / norm) if norm > 0 else 0.0
    metrics["roundtrip_bound_slack_min"] = float((bound - err).min())
    checks["roundtrip_within_analytic_bound"] = bool((err <= bound).all())

    # Informational (not gated): double-quant dynamic-range losses.
    nonzero_blocks = true_absmax > 0
    block_recon = repo_dense.reshape(-1)
    pad = padded - numel
    if pad:
        block_recon = F.pad(block_recon, (0, pad))
    recon_blocks = block_recon.reshape(num_blocks, block_size)
    collapsed = nonzero_blocks & (recon_blocks.abs().amax(dim=1) == 0)
    metrics["info_blocks_collapsed_to_zero"] = int(collapsed.sum())
    metrics["info_blocks_total"] = int(num_blocks)
    metrics["info_blocks_scale_floor_hit"] = int(((q_repo == 1) & (ratio < 0.5)).sum())

    if kind == "gaussian" and numel >= 256:
        checks["gaussian_rel_l2_within_gate"] = metrics["roundtrip_rel_l2"] <= TOLERANCES["gaussian_rel_l2_max"]

    return {"checks": checks, "metrics": metrics}


def check_nonfinite_and_invalid() -> list[dict]:
    from hivemind.core.quantization import quantize_tensor_nf4_double_quant

    out = []
    for label, value in (("nan", float("nan")), ("inf", float("inf")), ("-inf", -float("inf"))):
        try:
            quantize_tensor_nf4_double_quant(torch.tensor([[1.0, value]]))
            ok = False
        except ValueError:
            ok = True
        out.append({"case": f"nonfinite_{label}_rejected", "passed": ok})
    for label, block in (("zero", 0), ("negative", -4), ("float", 1.5), ("bool", True)):
        try:
            quantize_tensor_nf4_double_quant(torch.ones(2, 2), block)
            ok = False
        except ValueError:
            ok = True
        out.append({"case": f"invalid_block_size_{label}_rejected", "passed": ok})
    return out


# --------------------------------------------------------------------------------------
# Secondary: swarm_fabric.pack_int4_nf4_weights (uniform signed int4 per row, NOT NF4 codebook)
# --------------------------------------------------------------------------------------


def check_int4_row_case(weight: torch.Tensor, kind: str) -> dict:
    from hivemind.swarm.swarm_fabric import pack_int4_nf4_weights, unpack_int4_nf4_weights

    dtype = weight.dtype
    eps = float(torch.finfo(dtype).eps)
    sub = 0.5 * float(torch.finfo(dtype).tiny) * eps
    w32 = weight.detach().to(torch.float32)
    packed, scale16 = pack_int4_nf4_weights(w32)
    rec = unpack_int4_nf4_weights(packed, scale16, original_in_dim=w32.shape[-1], target_dtype=torch.float32)
    scale32 = (w32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 7.0).to(torch.float64)
    err = (w32.to(torch.float64) - rec.to(torch.float64)).abs()
    bound = (0.5 + 7 * 2**-10) * scale32 + sub + eps * torch.maximum(w32.abs().to(torch.float64), rec.abs().to(torch.float64))
    finite = bool(torch.isfinite(rec).all())
    checks = {"packed_shape": tuple(packed.shape) == (w32.shape[0], (w32.shape[1] + 1) // 2)}
    metrics = {
        "roundtrip_max_abs_err": float(err.max()) if finite else None,
        "scale_fp16_overflow_rows": int((~torch.isfinite(scale16.float())).sum()),
    }
    if kind == "huge":
        # Documented finding, not a gate: row absmax/7 above 65504 overflows the fp16 scale.
        metrics["finding_decoded_nonfinite"] = not finite
    else:
        checks["decoded_finite"] = finite
        checks["roundtrip_within_bound"] = bool((err <= bound).all()) if finite else False
    return {"checks": checks, "metrics": metrics}


# --------------------------------------------------------------------------------------
# CPU wall-clock (labelled; not comparable to GPU)
# --------------------------------------------------------------------------------------


def cpu_wall_clock(reps: int = 5) -> dict:
    from hivemind.core.quantization import dequantize_tensor_nf4, quantize_tensor_nf4_double_quant

    out = {}
    for dtype in (torch.float32, torch.float16):
        w = torch.randn(1024, 1024).to(dtype)
        q_times, d_times = [], []
        state = None
        for _ in range(reps):
            t = time.perf_counter()
            state, _ = quantize_tensor_nf4_double_quant(w)
            q_times.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter()
            dequantize_tensor_nf4(state, dtype=dtype)
            d_times.append((time.perf_counter() - t) * 1e3)
        out[str(dtype).replace("torch.", "")] = {
            "shape": [1024, 1024],
            "quantize_cpu_wall_clock_ms_median": round(statistics.median(q_times), 3),
            "dequantize_cpu_wall_clock_ms_median": round(statistics.median(d_times), 3),
        }
    out["note"] = "CPU wall-clock via time.perf_counter; not comparable to GPU latency and not a CUDA-event measurement."
    return out


# --------------------------------------------------------------------------------------
# CUDA measurements (only reached when torch.cuda.is_available())
# --------------------------------------------------------------------------------------


def _mem_snapshot(dev) -> dict:
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(dev)),
        "reserved_bytes": int(torch.cuda.memory_reserved(dev)),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _cuda_time_ms(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {
        "unit": "ms (torch.cuda.Event)",
        "warmup_iters": warmup,
        "iters": iters,
        "median": round(statistics.median(times), 4),
        "p95": round(times[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)], 4),
        "min": round(times[0], 4),
    }


def _measure_one_cuda(shape, dtype, warmup: int, iters: int, dev) -> dict:
    from hivemind.core.quantization import dequantize_tensor_nf4, quantize_tensor_nf4_double_quant

    rec: dict = {"shape": list(shape), "dtype": str(dtype).replace("torch.", "")}
    gc.collect()
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    base = _mem_snapshot(dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    w = torch.randn(shape, generator=gen, device=dev, dtype=torch.float32).to(dtype)
    torch.cuda.synchronize(dev)
    dense_resident = _mem_snapshot(dev)

    torch.cuda.reset_peak_memory_stats(dev)
    state, recon = quantize_tensor_nf4_double_quant(w)
    torch.cuda.synchronize(dev)
    rec["quantize_peak_allocated_over_baseline_bytes"] = int(torch.cuda.max_memory_allocated(dev)) - base["allocated_bytes"]
    del recon

    gpu_state = dataclasses.replace(
        state, packed_4bit=state.packed_4bit.to(dev), q_absmax_uint8=state.q_absmax_uint8.to(dev)
    )
    torch.cuda.synchronize(dev)
    packed_resident = _mem_snapshot(dev)

    torch.cuda.reset_peak_memory_stats(dev)
    pre = int(torch.cuda.memory_allocated(dev))
    dense_out = dequantize_tensor_nf4(gpu_state, dtype=dtype, device=dev)
    torch.cuda.synchronize(dev)
    rec["dequantize_peak_allocated_over_pre_call_bytes"] = int(torch.cuda.max_memory_allocated(dev)) - pre
    rec["dense_output_bytes"] = int(dense_out.nbytes)
    after_dense = _mem_snapshot(dev)
    rec["memory_bytes_torch_allocator"] = {
        "baseline": base,
        "original_dense_weight_resident": dense_resident,
        "plus_packed_buffers_on_gpu": packed_resident,
        "plus_dequantized_dense_output": after_dense,
        "dense_weight_alone_bytes": dense_resident["allocated_bytes"] - base["allocated_bytes"],
        "packed_buffers_alone_bytes": int(gpu_state.packed_4bit.nbytes + gpu_state.q_absmax_uint8.nbytes),
    }
    del dense_out

    latency = {
        "dequantize_gpu_resident_packed": _cuda_time_ms(
            lambda: dequantize_tensor_nf4(gpu_state, dtype=dtype, device=dev), warmup, iters),
        "dequantize_cpu_resident_packed_includes_h2d_copy": _cuda_time_ms(
            lambda: dequantize_tensor_nf4(state, dtype=dtype, device=dev), warmup, iters),
    }
    if len(shape) == 2:
        x = torch.randn(16, shape[1], generator=gen, device=dev, dtype=torch.float32).to(dtype)
        latency["dense_linear_reference"] = _cuda_time_ms(lambda: F.linear(x, w), warmup, iters)
        latency["dequantize_then_linear_gpu_resident"] = _cuda_time_ms(
            lambda: F.linear(x, dequantize_tensor_nf4(gpu_state, dtype=dtype, device=dev)), warmup, iters)
    rec["latency"] = latency
    rec["status"] = "MEASURED"
    return rec


def measure_cuda(shapes, dtypes, warmup: int, iters: int, device: str = "cuda:0") -> list[dict]:
    dev = torch.device(device)
    results = []
    for shape in shapes:
        for dtype in dtypes:
            row = {"shape": list(shape), "dtype": str(dtype).replace("torch.", "")}
            # Capability skips are explicit. Probe/allocation failures are errors, not
            # unsupported dtypes silently removed from the requested matrix.
            try:
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    row.update(status="SKIPPED_UNSUPPORTED", skipped="bfloat16 unsupported on this device")
                else:
                    row = _measure_one_cuda(shape, dtype, warmup, iters, dev)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            results.append(row)
    return results


def _triton_fp8_no_fallback(sf, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the actual kernel directly; the production wrapper can silently fall back."""
    n_rows, n_cols = x.shape
    quantized = torch.empty_like(x)
    scales = torch.empty((n_rows, 1), device=x.device, dtype=torch.float32)
    sf._triton_fp8_quant_kernel[(n_rows,)](
        x, quantized, scales, n_cols, BLOCK_SIZE=sf.triton.next_power_of_2(n_cols)
    )
    torch.cuda.synchronize(x.device)  # surface asynchronous launch failures before claiming it ran
    return quantized.to(torch.float8_e4m3fn), scales.to(torch.float16)


def cuda_triton_parity() -> dict:
    """On CUDA only: actual Triton FP8 kernel vs PyTorch, without a fallback route."""
    try:
        from hivemind.swarm import swarm_fabric as sf
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "status": "ERROR", "error": f"import failed: {type(exc).__name__}: {exc}"}
    out = {"ran": False, "triton_importable": bool(sf._HAS_TRITON),
           "note": "_triton_fused_int4_fp8_moe_router_kernel is defined but never launched anywhere in src/."}
    if not sf._HAS_TRITON:
        out.update(status="SKIPPED_UNAVAILABLE", reason="triton not importable; kernel parity was not tested")
        return out
    try:
        x = torch.randn(64, 256, device="cuda", dtype=torch.float32)
        fp8_data, actual_scale = _triton_fp8_no_fallback(sf, x)
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = absmax / 448.0
        expected = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        mismatch = (fp8_data.view(torch.uint8) != expected.view(torch.uint8)).float().mean().item()
        scale_diff = float((actual_scale.float() - scale.half().float()).abs().max())
        out.update(ran=True, backend="triton_direct_no_fallback", fp8_byte_mismatch_fraction=mismatch,
                   scale_max_abs_diff=scale_diff, passed=mismatch == 0.0 and scale_diff == 0.0,
                   tolerances={"fp8_byte_mismatch_fraction": 0.0, "scale_max_abs_diff": 0.0})
        out["status"] = "PASSED" if out["passed"] else "FAILED"
    except Exception as exc:  # noqa: BLE001
        out.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version_cuda": getattr(torch.version, "cuda", None),
        "torch_version_hip": getattr(torch.version, "hip", None),
        "numpy": np.__version__,
    }
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu_name"] = props.name
        env["gpu_capability"] = f"{props.major}.{props.minor}"
        env["gpu_backend"] = "hip" if env["torch_version_hip"] else "cuda"
        if env["gpu_backend"] == "hip":
            env["warning"] = "ROCm/HIP build: results are not CUDA acceptance evidence."
    return env


SHAPES = [(1,), (63,), (64,), (65,), (127,), (1, 1), (3, 5), (7, 13), (33, 129), (64, 64), (128, 127), (257, 1), (4, 3, 5), (97, 191)]
KINDS_ALL = ["gaussian", "zeros", "constant", "alternating_sign", "single_outlier", "mixed_block_magnitude", "tiny", "huge"]


def _compact(case: dict) -> dict:
    if not case.get("passed") or "metrics" not in case:
        return case
    keep = ("roundtrip_max_abs_err", "roundtrip_max_err_over_block_scale", "roundtrip_rel_l2",
            "repo_vs_reference_max_abs", "scale_byte_mismatches", "nibble_mismatches",
            "nibble_mismatches_that_are_near_ties", "info_blocks_collapsed_to_zero", "info_blocks_total",
            "info_blocks_scale_floor_hit", "finding_decoded_nonfinite", "scale_fp16_overflow_rows")
    slim = {k: v for k, v in case.items() if k not in ("checks", "metrics")}
    slim["checks_passed"] = len(case["checks"])
    slim["metrics"] = {k: case["metrics"][k] for k in keep if k in case["metrics"]}
    return slim


def run_correctness(seed: int, quick: bool, verbose: bool = False) -> dict:
    from hivemind.core.quantization import NF4_CODEBOOK

    cases = []
    repo_codebook = [float(v) for v in NF4_CODEBOOK.tolist()]
    codebook_ok = repo_codebook == REFERENCE_NF4_CODEBOOK and sorted(REFERENCE_NF4_CODEBOOK) == REFERENCE_NF4_CODEBOOK
    cases.append({"case": "repo_codebook_equals_published_nf4_table", "passed": bool(codebook_ok)})

    dtypes = [d for d in (torch.float32, torch.float16, torch.bfloat16) if dtype_supported(d)]
    shapes = SHAPES[::3] if quick else SHAPES
    block_sizes = (64,) if quick else (64, 128, 33)
    for dtype in dtypes:
        for shape in shapes:
            for kind in (["gaussian", "zeros"] if quick else KINDS_ALL):
                if kind in ("single_outlier", "mixed_block_magnitude") and math.prod(shape) < 8:
                    continue
                for block in block_sizes if kind == "gaussian" else (64,):
                    w = make_tensor(kind, shape, dtype, seed)
                    result = check_nf4_case(w, block, kind)
                    result.update({"case": f"nf4/{kind}", "shape": list(shape), "dtype": str(dtype).replace("torch.", ""),
                                   "block_size": block, "passed": all(result["checks"].values())})
                    cases.append(result)
    empty = torch.empty(0, 4)
    r = check_nf4_case(empty, 64, "empty")
    r.update({"case": "nf4/empty", "shape": [0, 4], "dtype": "float32", "block_size": 64, "passed": all(r["checks"].values())})
    cases.append(r)
    cases.extend(check_nonfinite_and_invalid())

    int4 = {"available": True, "cases": []}
    try:
        for dtype in dtypes:
            for shape in ((3, 5), (7, 13), (33, 129), (64, 64)):
                for kind in ("gaussian", "zeros", "single_outlier", "huge"):
                    res = check_int4_row_case(make_tensor(kind, shape, dtype, seed), kind)
                    res.update({"case": f"int4_row/{kind}", "shape": list(shape),
                                "dtype": str(dtype).replace("torch.", ""), "passed": all(res["checks"].values())})
                    int4["cases"].append(res)
    except Exception as exc:  # noqa: BLE001
        # Preserve earlier failed cases; an incomplete path is never a passing suite.
        int4.update(available=False, reason=f"{type(exc).__name__}: {exc}")

    failed = [c for c in cases if not c["passed"]]
    int4_failed = [c for c in int4["cases"] if not c["passed"]]
    info = [c for c in cases if c.get("metrics", {}).get("info_blocks_collapsed_to_zero")]
    worst = max((c for c in cases if "metrics" in c and "roundtrip_max_err_over_block_scale" in c["metrics"]),
                key=lambda c: c["metrics"]["roundtrip_max_err_over_block_scale"], default=None)
    if not verbose:
        cases = [_compact(c) for c in cases]
        int4["cases"] = [_compact(c) if c["passed"] else c for c in int4["cases"]]
    return {
        "nf4_cases": cases,
        "int4_row_path": int4,
        "summary": {
            "nf4_cases": len(cases), "nf4_failed": len(failed),
            "int4_row_cases": len(int4["cases"]), "int4_row_failed": len(int4_failed),
            "failed_case_ids": [f"{c['case']} {c.get('shape')} {c.get('dtype')}" for c in failed + int4_failed],
            "cases_with_blocks_collapsed_to_zero_informational": len(info),
            "worst_roundtrip_err_over_block_scale": None if worst is None else worst["metrics"]["roundtrip_max_err_over_block_scale"],
            "worst_case_id": None if worst is None else f"{worst['case']} {worst['shape']} {worst['dtype']}",
            "half_max_code_gap": round(0.5 * MAX_CODE_GAP, 6),
            "half_max_code_gap_note": "pure code-rounding error is <= this fraction of the block scale; larger ratios come only from absmax clamping or dtype rounding and are covered by the gated analytic bound",
        },
        "passed": not failed and not int4_failed and int4["available"],
    }


def parse_shapes(text: str) -> list[tuple[int, ...]]:
    try:
        shapes = [tuple(int(d) for d in item.lower().split("x")) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shapes must be comma-separated positive dimensions, e.g. 64x64,128") from exc
    if not shapes or any(not shape or any(d <= 0 for d in shape) for shape in shapes):
        raise argparse.ArgumentTypeError("every CUDA shape must have positive dimensions")
    return shapes


def _valid_measurement_payload(row: dict) -> bool:
    """A success label alone is not evidence that the measurement completed."""
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    scalar_keys = ("quantize_peak_allocated_over_baseline_bytes", "dequantize_peak_allocated_over_pre_call_bytes",
                   "dense_output_bytes")
    memory = row.get("memory_bytes_torch_allocator")
    latency = row.get("latency")
    if not all(number(row.get(key)) for key in scalar_keys) or not isinstance(memory, dict) or not isinstance(latency, dict):
        return False
    for key in ("baseline", "original_dense_weight_resident", "plus_packed_buffers_on_gpu", "plus_dequantized_dense_output"):
        snapshot = memory.get(key)
        if not isinstance(snapshot, dict) or not all(number(snapshot.get(k)) for k in (
            "allocated_bytes", "reserved_bytes", "device_free_bytes", "device_total_bytes"
        )):
            return False
    if not all(number(memory.get(key)) for key in ("dense_weight_alone_bytes", "packed_buffers_alone_bytes")):
        return False
    required = ["dequantize_gpu_resident_packed", "dequantize_cpu_resident_packed_includes_h2d_copy"]
    if len(row["shape"]) == 2:
        required += ["dense_linear_reference", "dequantize_then_linear_gpu_resident"]
    for name in required:
        timing = latency.get(name)
        if not isinstance(timing, dict) or timing.get("unit") != "ms (torch.cuda.Event)":
            return False
        if not all(number(timing.get(k)) for k in ("warmup_iters", "iters", "median", "p95", "min")):
            return False
        if timing["iters"] <= 0:
            return False
    return True


def cuda_measurement_summary(measurements: list[dict], shapes, dtypes) -> dict:
    """Fail closed unless each requested shape/dtype has one complete measurement."""
    expected = [(tuple(shape), str(dtype).replace("torch.", "")) for shape in shapes for dtype in dtypes]
    actual, invalid_rows = [], []
    errors = measured = skipped = 0
    if not isinstance(measurements, list):
        measurements = [measurements]
    for index, row in enumerate(measurements):
        if not isinstance(row, dict):
            invalid_rows.append(index)
            errors += 1
            continue
        shape, dtype = row.get("shape"), row.get("dtype")
        identity_valid = (isinstance(shape, (list, tuple)) and bool(shape)
                          and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape)
                          and isinstance(dtype, str))
        if identity_valid:
            actual.append((tuple(shape), dtype))
        if row.get("status") == "ERROR" or row.get("error"):
            errors += 1
        elif not identity_valid or (row.get("status") == "MEASURED" and not _valid_measurement_payload(row)):
            invalid_rows.append(index)
            errors += 1
        elif row.get("status") == "MEASURED":
            measured += 1
        elif row.get("status") == "SKIPPED_UNSUPPORTED":
            skipped += 1
    complete = bool(expected) and sorted(actual) == sorted(expected) and measured == len(expected) and errors == 0
    return {"requested": len(expected), "measured": measured, "errors": errors,
            "skipped": skipped, "invalid_rows": invalid_rows, "complete": complete}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the JSON report to this path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fewer shapes/kinds (used by the unit tests)")
    ap.add_argument("--require-cuda", action="store_true", help="acceptance mode: exit 2 if CUDA is unavailable or measurements are incomplete")
    ap.add_argument("--cuda-shapes", type=parse_shapes, default="4096x4096,11008x4096", help="comma list for GPU measurements")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--skip-cpu-timing", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="keep full checks/metrics for passing cases")
    args = ap.parse_args(argv)
    if args.warmup < 0 or args.iters <= 0:
        ap.error("--warmup must be nonnegative and --iters must be positive")

    cuda = torch.cuda.is_available()
    report: dict = {"tool": "nf4_dense_reference", "schema_version": SCHEMA_VERSION, "environment": environment(),
                    "tolerances": TOLERANCES,
                    "scope_notes": [
                        "Repo execution path is dense reconstruction (quantization.py sets _nf4_storage_mode='dense_reconstruction'); this is not a fused NF4 matmul.",
                        "Correctness of decode says nothing about memory residency; see cuda.memory_bytes_torch_allocator when CUDA ran.",
                        "swarm_fabric.pack_int4_nf4_weights is a uniform signed-int4 per-row scheme, not the NF4 codebook.",
                    ]}
    if cuda and getattr(torch.version, "hip", None):
        report["status"] = "BLOCKED_NON_CUDA_BACKEND"
        report["cpu_correctness"] = None
        report["cuda"] = {"available": False, "measurements": None,
                          "message": "ROCm/HIP uses torch.cuda APIs but is not CUDA acceptance evidence."}
        _emit(report, args.out)
        return 2
    if not cuda:
        print(NO_CUDA_MESSAGE, file=sys.stderr)
        report["cuda"] = {"available": False, "message": NO_CUDA_MESSAGE, "measurements": None}
        if args.require_cuda:
            report["status"] = "BLOCKED_NO_CUDA"
            report["cpu_correctness"] = None
            _emit(report, args.out)
            return 2
    report["cpu_correctness"] = run_correctness(args.seed, args.quick, args.verbose)
    if not args.skip_cpu_timing:
        report["cpu_wall_clock_timings"] = cpu_wall_clock(3 if args.quick else 5)
    if cuda:
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        try:
            measurements = measure_cuda(args.cuda_shapes, dtypes, args.warmup, args.iters)
        except Exception as exc:  # noqa: BLE001
            measurements = [{"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}]
        try:
            parity = cuda_triton_parity()
        except Exception as exc:  # noqa: BLE001
            parity = {"ran": False, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        report["cuda"] = {
            "available": True,
            "measurements": measurements,
            "measurement_summary": cuda_measurement_summary(measurements, args.cuda_shapes, dtypes),
            "triton_fp8_parity": parity,
            "memory_note": "torch allocator bytes, not process-level nvidia-smi; record driver/CUDA context overhead separately.",
        }
    if not report["cpu_correctness"]["passed"]:
        status, code = "FAIL_CORRECTNESS", 1
    elif cuda:
        summary = report["cuda"]["measurement_summary"]
        parity = report["cuda"]["triton_fp8_parity"]
        if summary["errors"]:
            status, code = "FAIL_CUDA_MEASUREMENTS", 1
        elif parity.get("error") or (parity.get("ran") and parity.get("passed") is not True):
            status, code = "FAIL_CUDA_PARITY", 1
        elif not summary["complete"]:
            status, code = "BLOCKED_INCOMPLETE_CUDA_MEASUREMENTS", 2
        elif not parity.get("ran"):
            status, code = "BLOCKED_CUDA_PARITY_UNAVAILABLE", 2
        else:
            status, code = "PASS_CORRECTNESS_AND_CUDA_MEASURED", 0
    else:
        status, code = "PASS_CORRECTNESS_ONLY_NO_CUDA", 0
    report["status"] = status
    _emit(report, args.out)
    return code


def _emit(report: dict, out: str | None) -> None:
    text = json.dumps(report, indent=2, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
