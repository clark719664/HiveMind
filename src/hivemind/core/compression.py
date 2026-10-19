"""Pseudo-Gradient Delta Compression via Top-k Sparsification & FP8 (E4M3) Quantization.

Implements the Project HiveMind communication compression layer:
  Delta_k^(t) = theta^(t-1) - theta_k^(t, H)
Compressed via Top-k magnitude selection and FP8 (float8_e4m3fn) quantization,
achieving 4x-10x wire reduction on top of DiLoCo's 500x step-frequency reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


FP8_E4M3_MAX = 448.0


@dataclass
class CompressedTensorPayload:
    """Wire representation of a single Top-k + FP8 compressed LoRA parameter delta."""

    original_shape: tuple[int, ...]
    original_numel: int
    indices: torch.Tensor       # int32 indices of top-k entries
    fp8_values: torch.Tensor    # float8_e4m3fn (or uint8-packed) quantized values
    scale: float                # absmax / FP8_E4M3_MAX dequantization factor

    @property
    def compressed_bytes(self) -> int:
        # The payload currently carries int32 indices, not an encoded uint16 stream.
        # Count the actual tensor buffers plus the float64 scale (excluding framing).
        return int(
            self.fp8_values.numel() * self.fp8_values.element_size()
            + self.indices.numel() * self.indices.element_size()
            + 8
        )


@dataclass
class CompressedDeltaPackage:
    """Complete compressed pseudo-gradient package uploaded by a BOINC worker node."""

    worker_id: str
    branch_name: str
    outer_step: int
    inner_steps_completed: int
    topk_ratio: float
    tensors: dict[str, CompressedTensorPayload]
    uncompressed_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 1.0
        return round(self.uncompressed_bytes / self.compressed_bytes, 2)


def compute_pseudo_gradient_delta(
    theta_init: dict[str, torch.Tensor],
    theta_final: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute the DiLoCo pseudo-gradient: Delta_k^(t) = theta^(t-1) - theta_k^(t, H)."""
    delta: dict[str, torch.Tensor] = {}
    for name, init_t in theta_init.items():
        if name not in theta_final:
            raise KeyError(f"Missing LoRA parameter '{name}' in final worker state.")
        fin_t = theta_final[name]
        delta[name] = (init_t.detach().to(torch.float32).cpu() - fin_t.detach().to(torch.float32).cpu())
    return delta


def compress_tensor_topk_fp8(
    tensor: torch.Tensor,
    topk_ratio: float = 0.20,
) -> CompressedTensorPayload:
    """Compress a 1D/2D LoRA delta tensor using Top-k sparsification and FP8 E4M3 quantization."""
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    flat = tensor.detach().to(torch.float32).cpu().reshape(-1)
    if not torch.isfinite(flat).all():
        raise ValueError("Cannot compress non-finite delta values.")
    numel = flat.numel()
    k = max(1, int(round(numel * topk_ratio)))

    if k >= numel:
        indices = torch.arange(numel, dtype=torch.int32)
        selected = flat
    else:
        abs_vals = flat.abs()
        # Deterministic top-k selection
        _, top_idx = torch.topk(abs_vals, k=k, largest=True, sorted=True)
        indices = top_idx.to(torch.int32)
        selected = flat[top_idx]

    absmax = float(selected.abs().max().item()) if selected.numel() > 0 else 0.0
    if absmax == 0.0:
        scale = 1.0
        scaled = torch.zeros_like(selected)
    else:
        scale = absmax / FP8_E4M3_MAX
        scaled = (selected / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    if hasattr(torch, "float8_e4m3fn"):
        fp8_vals = scaled.to(torch.float8_e4m3fn)
    else:
        # Fallback uniform int8 quantization if float8 dtype unavailable
        fp8_vals = torch.round(scaled / (FP8_E4M3_MAX / 127.0)).to(torch.int8)

    return CompressedTensorPayload(
        original_shape=tuple(tensor.shape),
        original_numel=numel,
        indices=indices,
        fp8_values=fp8_vals,
        scale=scale,
    )


def decompress_tensor_topk_fp8(
    payload: CompressedTensorPayload,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decompress a Top-k + FP8 payload back into a dense tensor of original_shape."""
    if (
        not isinstance(payload.original_numel, int)
        or payload.original_numel < 0
        or any(not isinstance(dim, int) or dim < 0 for dim in payload.original_shape)
        or math.prod(payload.original_shape) != payload.original_numel
    ):
        raise ValueError("Compressed payload shape and element count disagree.")
    if not math.isfinite(payload.scale) or payload.scale <= 0:
        raise ValueError("Compressed payload scale must be finite and positive.")
    if payload.indices.ndim != 1 or payload.fp8_values.ndim != 1:
        raise ValueError("Compressed payload indices and values must be one-dimensional.")
    if payload.indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Compressed payload indices must be integers.")
    if payload.fp8_values.dtype not in (torch.int8, getattr(torch, "float8_e4m3fn", torch.int8)):
        raise ValueError("Unsupported compressed value dtype.")
    indices = payload.indices.detach().to(device="cpu", dtype=torch.long)
    values = payload.fp8_values.detach().to(device="cpu", dtype=torch.float32)
    if indices.numel() != values.numel() or indices.numel() > payload.original_numel:
        raise ValueError("Compressed payload indices and values have inconsistent lengths.")
    if indices.numel() and (
        int(indices.min()) < 0
        or int(indices.max()) >= payload.original_numel
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("Compressed payload indices must be unique and within bounds.")
    if not torch.isfinite(values).all():
        raise ValueError("Compressed payload contains non-finite values.")
    out = torch.zeros(payload.original_numel, dtype=torch.float32, device="cpu")
    if hasattr(torch, "float8_e4m3fn") and payload.fp8_values.dtype == torch.float8_e4m3fn:
        dequant = values * payload.scale
    else:
        dequant = values * (FP8_E4M3_MAX / 127.0) * payload.scale

    if not torch.isfinite(dequant).all():
        raise ValueError("Compressed payload dequantization overflowed.")
    out.index_copy_(0, indices, dequant)
    return out.reshape(payload.original_shape).to(device=device, dtype=dtype)


def compress_lora_delta(
    delta_dict: dict[str, torch.Tensor],
    worker_id: str,
    branch_name: str,
    outer_step: int,
    inner_steps_completed: int = 500,
    topk_ratio: float = 0.20,
) -> CompressedDeltaPackage:
    """Compress an entire dictionary of LoRA pseudo-gradients for BOINC upload."""
    tensors: dict[str, CompressedTensorPayload] = {}
    uncompressed_bytes = 0
    compressed_bytes = 0

    for name, tensor in delta_dict.items():
        # Compare against the actual submitted tensor representation.
        uncompressed_bytes += tensor.numel() * tensor.element_size()
        payload = compress_tensor_topk_fp8(tensor, topk_ratio=topk_ratio)
        tensors[name] = payload
        compressed_bytes += payload.compressed_bytes

    return CompressedDeltaPackage(
        worker_id=worker_id,
        branch_name=branch_name,
        outer_step=outer_step,
        inner_steps_completed=inner_steps_completed,
        topk_ratio=topk_ratio,
        tensors=tensors,
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
    )


def decompress_lora_delta(
    package: CompressedDeltaPackage,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Reconstruct dense LoRA pseudo-gradient dictionary from a CompressedDeltaPackage."""
    return {
        name: decompress_tensor_topk_fp8(payload, dtype=dtype, device=device)
        for name, payload in package.tensors.items()
    }
