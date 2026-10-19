"""Phase A & Phase C: Branch-Train-Mix (BTM) Domain Partitioning & Sparse MoE Upcycling.

Implements:
- Phase A: Initializing E=8 domain-specific expert branches from a shared base checkpoint W_0.
- Phase C: Upcycling E converged QLoRA adapters into a unified Sparse Mixture-of-Experts (MoE):
    1. Self-Attention projections (q_proj, k_proj, v_proj, o_proj) are uniformly averaged
       across all E adapters and merged into the shared attention backbone:
         W_attn = W_0_attn + (1/E) * sum_{e=1}^E (alpha/r) * B_{e,attn} @ A_{e,attn}
    2. FFN/MLP projections (gate_proj, up_proj, down_proj) are merged per expert e in {1..E}
       to create E distinct SwiGLU FFN experts:
         W_{ffn, e} = W_0_ffn + (alpha/r) * B_{e,ffn} @ A_{e,ffn}
    3. A Top-2 Softmax Router G(x) = Softmax(TopK(x @ W_g, k=2)) routes each token to its
       top-2 domain experts with Switch-style load-balancing auxiliary loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EXPERT_DOMAINS = [
    "Code",
    "Mathematics",
    "Creative_Writing",
    "STEM",
    "Legal_Medical",
    "Multilingual_FR",
    "Reasoning",
    "General",
]


@dataclass
class ExpertBranchState:
    """Tracks the outer-loop LoRA state for one BTM domain branch."""

    branch_id: int
    domain_name: str
    outer_step: int
    lora_state: dict[str, torch.Tensor] = field(repr=False)


@dataclass
class MoEUpcycleReport:
    """Telemetry report produced when upcycling E QLoRA branches into a Sparse MoE."""

    num_experts: int
    top_k: int
    expert_domains: list[str]
    attention_layers_merged: int
    ffn_expert_modules_created: int
    total_moe_parameters: int
    active_parameters_per_token: int
    router_initial_loss: float = 0.0
    router_final_loss: float = 0.0
    expert_utilization_distribution: dict[str, float] = field(default_factory=dict)


def initialize_domain_branches(
    initial_lora_state: dict[str, torch.Tensor],
    expert_domains: list[str] | None = None,
) -> dict[str, ExpertBranchState]:
    """Phase A: Initialize E=8 independent expert branches from identical base checkpoint W_0."""
    domains = expert_domains or DEFAULT_EXPERT_DOMAINS
    branches: dict[str, ExpertBranchState] = {}
    for idx, domain in enumerate(domains):
        branches[domain] = ExpertBranchState(
            branch_id=idx,
            domain_name=domain,
            outer_step=0,
            lora_state={k: v.detach().cpu().clone() for k, v in initial_lora_state.items()},
        )
    return branches


def _dequantize_linear_weight(module: nn.Module) -> torch.Tensor:
    """Extract dense FP32 weight matrix from either nn.Linear or bitsandbytes Linear4bit."""
    base_layer = getattr(module, "base_layer", module)
    weight = base_layer.weight
    if hasattr(weight, "quant_state") and weight.quant_state is not None:
        import bitsandbytes.functional as bnb_F

        return bnb_F.dequantize_4bit(weight.data, weight.quant_state).to(torch.float32).cpu()
    return weight.detach().to(torch.float32).cpu()


def _extract_ab_pair(
    lora_state: dict[str, torch.Tensor],
    layer_idx: int,
    subpath: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Locate (lora_A, lora_B) tensors for a specific decoder layer and projection."""
    prefix = f"layers.{layer_idx}.{subpath}."
    a_tensor: torch.Tensor | None = None
    b_tensor: torch.Tensor | None = None
    for k, v in lora_state.items():
        if prefix in k and "lora_A" in k:
            a_tensor = v.detach().to(torch.float32).cpu()
        elif prefix in k and "lora_B" in k:
            b_tensor = v.detach().to(torch.float32).cpu()
    if a_tensor is not None and b_tensor is not None:
        return a_tensor, b_tensor
    return None


@dataclass
class SVDLoARankProjectionReport:
    """Telemetry describing a truncated-SVD LoRA rank projection (`HM-S4-05`).

    Attributes:
        source_rank_r: Rank `r_src` detected on the incoming adapter.
        target_rank_r: Canonical rank `r_dst` the adapter was projected onto.
        num_projected_pairs: Number of (A, B) LoRA factor pairs reprojected.
        num_padded_pairs: Number of pairs whose source rank was below `target_rank_r`.
        num_identity_pairs: Number of pairs already at `target_rank_r`.
        mean_spectral_energy_retained: Mean fraction of the Frobenius spectral
            energy of `D = B @ A` preserved by the truncated SVD.
        min_spectral_energy_retained: Minimum per-pair spectral energy retention.
        max_relative_delta_error: Maximum relative Frobenius error between the
            source effective delta and the projected effective delta.
        source_params: Total number of LoRA parameters before projection.
        target_params: Total number of LoRA parameters after projection.
        source_lora_alpha: `alpha` assumed for the source adapter.
        target_lora_alpha: `alpha` the projected adapter is meant to be used with.
    """

    source_rank_r: int
    target_rank_r: int
    num_projected_pairs: int
    num_padded_pairs: int
    num_identity_pairs: int
    mean_spectral_energy_retained: float
    min_spectral_energy_retained: float
    max_relative_delta_error: float
    source_params: int
    target_params: int
    source_lora_alpha: float
    target_lora_alpha: float

    @property
    def compression_ratio(self) -> float:
        """Parameter compression factor achieved by the rank projection."""
        if self.target_params <= 0:
            return 1.0
        return round(self.source_params / self.target_params, 4)


def _orient_lora_pair(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return a LoRA (A, B) pair in the canonical PEFT layout `A=[r, in]`, `B=[out, r]`."""
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        raise ValueError(
            "LoRA factors must be 2-D tensors, got "
            f"{tuple(lora_a.shape)} and {tuple(lora_b.shape)}."
        )
    if lora_a.shape[0] == lora_b.shape[1]:
        return lora_a, lora_b, False
    if lora_a.shape[1] == lora_b.shape[0]:
        return lora_a.t().contiguous(), lora_b.t().contiguous(), True
    raise ValueError(
        f"Incompatible LoRA factor shapes {tuple(lora_a.shape)} / {tuple(lora_b.shape)}: "
        "expected A=[r, in_features] and B=[out_features, r] (or the transposed layout)."
    )


def pair_lora_ab_keys(lora_state_dict: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Pair every `lora_A` key of a LoRA state dict with its matching `lora_B` key."""
    pairs: list[tuple[str, str]] = []
    for key in lora_state_dict:
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key in lora_state_dict:
            pairs.append((key, b_key))
    return pairs


def detect_lora_rank(lora_state_dict: dict[str, torch.Tensor]) -> int | None:
    """Detect the rank `r` of a LoRA state dict (`None` when it holds no LoRA pair).

    Adapters mixing several ranks across layers report the **maximum** rank; every
    factor pair is nonetheless projected independently by
    `project_lora_state_dict_to_rank`.
    """
    ranks: set[int] = set()
    for a_key, b_key in pair_lora_ab_keys(lora_state_dict):
        a_t = lora_state_dict[a_key]
        b_t = lora_state_dict[b_key]
        if a_t.shape[0] == b_t.shape[1]:
            ranks.add(int(a_t.shape[0]))
        elif a_t.shape[1] == b_t.shape[0]:
            ranks.add(int(a_t.shape[1]))
    if not ranks:
        return None
    return max(ranks)


def svd_project_lora_pair(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    target_rank_r: int,
    scale_b: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Truncated-SVD projection of a LoRA (A, B) factor pair onto `target_rank_r`.

    Computes the Eckart-Young optimal rank-`target_rank_r` approximation of the
    effective weight delta `D = B @ A` and refactors it back into a LoRA pair
    `(A', B')` such that `B' @ A' ~= D_k` (the truncated spectral expansion of `D`).

    The computation never materializes the dense `D` matrix: a QR decomposition of
    `B` is combined with the SVD of the (small) `R @ A` core, which is
    mathematically equivalent to the SVD of `D` but costs `O(r * n)` instead of
    `O(min(m, n)^3)` — critical to stay inside the `< 2.0 GB` VRAM budget while
    hot-plugging 32 experts.

    Args:
        lora_a: LoRA down-projection factor `A` with shape `[r, in_features]`.
        lora_b: LoRA up-projection factor `B` with shape `[out_features, r]`.
        target_rank_r: Canonical rank the pair is projected onto.
        scale_b: Multiplicative factor applied to `B'` (used to keep the
            `alpha / r` scaling convention consistent across ranks).

    Returns:
        `(lora_a_new, lora_b_new, spectral_energy_retained)` where
        `spectral_energy_retained = sum(s[:k]^2) / sum(s^2)` is the fraction of the
        Frobenius spectral energy of `D` captured by the truncated expansion.
    """
    if target_rank_r < 1:
        raise ValueError(f"target_rank_r must be >= 1, got {target_rank_r}.")

    a_t = lora_a.detach().to(torch.float32)
    b_t = lora_b.detach().to(torch.float32)

    q_b, r_core = torch.linalg.qr(b_t)  # q_b: [out, k], r_core: [k, r]
    core = r_core @ a_t  # [k, in] - the only matrix we ever factorize
    u, s, vh = torch.linalg.svd(core, full_matrices=False)

    total_energy = float(torch.sum(s * s).item())
    k_eff = int(min(target_rank_r, s.numel()))
    u_k = u[:, :k_eff]
    s_k = s[:k_eff]
    vh_k = vh[:k_eff, :]

    sqrt_s = torch.sqrt(s_k.clamp_min(0.0))
    left = q_b @ u_k  # [out, k_eff] left singular vectors of D
    b_new = (left * sqrt_s.unsqueeze(0)) * float(scale_b)
    a_new = sqrt_s.unsqueeze(1) * vh_k

    if k_eff < target_rank_r:
        # Pad lower-rank adapters up to the canonical rank with zero rows/columns:
        # the effective delta is preserved exactly while the nominal rank matches.
        pad = target_rank_r - k_eff
        a_new = torch.cat([a_new, a_new.new_zeros((pad, a_new.shape[1]))], dim=0)
        b_new = torch.cat([b_new, b_new.new_zeros((b_new.shape[0], pad))], dim=1)

    retained = 1.0 if total_energy <= 0.0 else float(torch.sum(s_k * s_k).item()) / total_energy
    return a_new, b_new, retained


def project_lora_state_dict_to_rank(
    lora_state_dict: dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
    source_rank_r: int | None = None,
) -> tuple[dict[str, torch.Tensor], SVDLoARankProjectionReport]:
    """Project a whole LoRA state dict onto a canonical rank via truncated SVD.

    Higher-rank adapters (`r=32`, `r=64`) are compressed and lower-rank adapters
    (`r=8`) are zero-padded, always landing on the same canonical rank so that
    heterogeneous community adapters can be hot-plugged into one unified
    `UpcycledSparseMoEModel` without VRAM bloat.

    The `alpha / r` scaling convention is preserved: the returned adapter obeys

        (target_lora_alpha / target_rank_r) * B' @ A'
            ~= (source_lora_alpha / source_rank_r) * B @ A

    so the effective weight delta of the caller is unchanged (up to the truncation
    error of the discarded spectral tail).

    Args:
        lora_state_dict: LoRA state dict with `lora_A` / `lora_B` tensor pairs.
        target_rank_r: Canonical rank to project onto.
        source_lora_alpha: `alpha` of the source adapter (defaults to the
            community-standard `2 * r` convention).
        target_lora_alpha: `alpha` the projected adapter will be used with
            (defaults to the community-standard `2 * target_rank_r` convention).
        source_rank_r: Explicit source rank override (auto-detected by default).

    Returns:
        `(projected_state_dict, report)`.
    """
    if target_rank_r < 1:
        raise ValueError(f"target_rank_r must be >= 1, got {target_rank_r}.")

    pairs = pair_lora_ab_keys(lora_state_dict)
    detected_rank = detect_lora_rank(lora_state_dict)
    src_r = int(source_rank_r) if source_rank_r is not None else int(detected_rank or 0)
    if src_r <= 0:
        empty = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in lora_state_dict.items()}
        report = SVDLoARankProjectionReport(
            source_rank_r=0,
            target_rank_r=int(target_rank_r),
            num_projected_pairs=0,
            num_padded_pairs=0,
            num_identity_pairs=0,
            mean_spectral_energy_retained=1.0,
            min_spectral_energy_retained=1.0,
            max_relative_delta_error=0.0,
            source_params=0,
            target_params=0,
            source_lora_alpha=0.0,
            target_lora_alpha=0.0,
        )
        return empty, report

    src_alpha = float(source_lora_alpha) if source_lora_alpha is not None else 2.0 * src_r
    dst_alpha = float(target_lora_alpha) if target_lora_alpha is not None else 2.0 * float(target_rank_r)
    target_scaling = dst_alpha / float(target_rank_r)

    projected: dict[str, torch.Tensor] = {}
    retentions: list[float] = []
    rel_errors: list[float] = []
    source_params = 0
    target_params = 0
    num_padded = 0
    num_identity = 0
    consumed: set[str] = set()

    for a_key, b_key in pairs:
        consumed.add(a_key)
        consumed.add(b_key)
        raw_a = lora_state_dict[a_key]
        raw_b = lora_state_dict[b_key]
        a_t, b_t, transposed = _orient_lora_pair(raw_a, raw_b)
        pair_rank = int(a_t.shape[0])

        # The `alpha / r` convention is applied per factor pair so that adapters
        # mixing several ranks across layers keep their exact effective delta.
        pair_src_alpha = float(source_lora_alpha) if source_lora_alpha is not None else 2.0 * pair_rank
        scale_b = (pair_src_alpha / float(pair_rank)) / target_scaling if target_scaling != 0.0 else 1.0

        a_new, b_new, retained = svd_project_lora_pair(
            a_t, b_t, target_rank_r=target_rank_r, scale_b=scale_b
        )
        retentions.append(retained)
        if pair_rank < target_rank_r:
            num_padded += 1
        elif pair_rank == target_rank_r:
            num_identity += 1

        # Relative Frobenius error between the source and the projected delta.
        delta_src = b_t @ a_t
        delta_dst = b_new @ a_new
        denom = float(torch.linalg.norm(delta_src).item())
        err = float(torch.linalg.norm(delta_dst - delta_src).item())
        rel_errors.append(err / denom if denom > 0.0 else 0.0)

        source_params += int(raw_a.numel()) + int(raw_b.numel())
        target_params += int(a_new.numel()) + int(b_new.numel())

        out_a = a_new.t().contiguous() if transposed else a_new.contiguous()
        out_b = b_new.t().contiguous() if transposed else b_new.contiguous()
        projected[a_key] = out_a.to(dtype=raw_a.dtype)
        projected[b_key] = out_b.to(dtype=raw_b.dtype)

    for key, value in lora_state_dict.items():
        if key in consumed:
            continue
        projected[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value

    report = SVDLoARankProjectionReport(
        source_rank_r=src_r,
        target_rank_r=int(target_rank_r),
        num_projected_pairs=len(pairs),
        num_padded_pairs=num_padded,
        num_identity_pairs=num_identity,
        mean_spectral_energy_retained=round(sum(retentions) / max(1, len(retentions)), 6),
        min_spectral_energy_retained=round(min(retentions) if retentions else 1.0, 6),
        max_relative_delta_error=round(max(rel_errors) if rel_errors else 0.0, 8),
        source_params=source_params,
        target_params=target_params,
        source_lora_alpha=round(src_alpha, 6),
        target_lora_alpha=round(dst_alpha, 6),
    )
    return projected, report


class UpcycledSwiGLUExpert(nn.Module):
    """Single domain-specialized SwiGLU FFN expert constructed via W_e = W_0 + (alpha/r) B_e A_e."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dev = gate_weight.device
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=dev)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=dev)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=dev)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight.to(device=dev, dtype=dtype))
            self.up_proj.weight.copy_(up_weight.to(device=dev, dtype=dtype))
            self.down_proj.weight.copy_(down_weight.to(device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class UpcycledFactoredSwiGLUExpert(nn.Module):
    """VRAM-efficient SwiGLU FFN expert evaluating (W_0 + (alpha/r) B_e A_e) x on shared NF4/FP16 W_0."""

    def __init__(
        self,
        shared_base_ffn: tuple[nn.Linear, nn.Linear, nn.Linear],
        ab_gate: tuple[torch.Tensor, torch.Tensor] | None,
        ab_up: tuple[torch.Tensor, torch.Tensor] | None,
        ab_down: tuple[torch.Tensor, torch.Tensor] | None,
        scaling: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Keep reference in a plain Python tuple so PyTorch does not duplicate shared base parameters
        self._shared_base_ffn = shared_base_ffn
        for prefix, ab in (("gate", ab_gate), ("up", ab_up), ("down", ab_down)):
            if ab is not None:
                a_buf = ab[0].to(device=device, dtype=dtype).contiguous()
                b_buf = (ab[1].to(device=device, dtype=torch.float32) * scaling).to(dtype=dtype).contiguous()
                self.register_buffer(f"{prefix}_A", a_buf)
                self.register_buffer(f"{prefix}_B", b_buf)
            else:
                self.register_buffer(f"{prefix}_A", None)
                self.register_buffer(f"{prefix}_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_gate, base_up, base_down = self._shared_base_ffn
        g = base_gate(x)
        if self.gate_A is not None and self.gate_B is not None:
            g = g + (x @ self.gate_A.t()) @ self.gate_B.t()
        u = base_up(x)
        if self.up_A is not None and self.up_B is not None:
            u = u + (x @ self.up_A.t()) @ self.up_B.t()
        h = F.silu(g) * u
        d = base_down(h)
        if self.down_A is not None and self.down_B is not None:
            d = d + (h @ self.down_A.t()) @ self.down_B.t()
        return d


class TopKMoERouter(nn.Module):
    """Lightweight Top-k Softmax Gating Network: G(x) = Softmax(TopK(x @ W_g, k=2))."""

    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        _ = dtype
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        # Keep tiny router matrix W_g in FP32 so AdamW (eps=1e-8) never underflows in FP16
        self.wg = nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)
        nn.init.normal_(self.wg.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Top-k gating weights, selected expert indices, and Switch load-balancing loss.

        Args:
            hidden_states: [batch * seq_len, hidden_size]
        Returns:
            routing_weights: [N, top_k] normalized via Softmax over top-k logits
            selected_experts: [N, top_k] integer indices in [0, E-1]
            aux_loss: scalar load-balancing loss
        """
        logits = self.wg(hidden_states.to(torch.float32))  # [N, E]
        probs_all = F.softmax(logits, dim=-1, dtype=torch.float32)

        top_logits, selected_experts = torch.topk(logits, k=self.top_k, dim=-1)
        routing_weights = F.softmax(top_logits, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # Load-balancing auxiliary loss: E * sum_i (f_i * P_i)
        num_tokens = max(1, hidden_states.size(0))
        one_hot = F.one_hot(selected_experts[:, 0], num_classes=self.num_experts).to(torch.float32)
        fraction_routed = one_hot.mean(dim=0)
        mean_prob = probs_all.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(fraction_routed * mean_prob) * (num_tokens / num_tokens)

        return routing_weights, selected_experts, aux_loss

    def expand_for_new_expert(self) -> None:
        """Incrementally expand the Top-k gating matrix W_g from R^{d x E} to R^{d x (E+1)}."""
        old_wg = self.wg
        device = old_wg.weight.device
        hidden_size = old_wg.in_features
        new_num_experts = self.num_experts + 1

        new_wg = nn.Linear(hidden_size, new_num_experts, bias=False, dtype=torch.float32, device=device)
        with torch.no_grad():
            new_wg.weight[: self.num_experts].copy_(old_wg.weight)
            # Initialize the new expert's gating column near the mean of existing columns
            mean_col = old_wg.weight.mean(dim=0, keepdim=True)
            new_wg.weight[self.num_experts :].copy_(mean_col + torch.randn_like(mean_col) * 0.01)

        self.wg = new_wg
        self.num_experts = new_num_experts


class SparseMoEMLPBlock(nn.Module):
    """Phase C Upcycled Sparse MoE Layer replacing a dense Qwen/Gemma MLP block.

    Evaluates:
      y = sum_{i in Top-2(G(x))} G(x)_i * FFN_i(x)
    """

    def __init__(
        self,
        hidden_size: int,
        experts: list[nn.Module],
        expert_domains: list[str],
        top_k: int = 2,
        dtype: torch.dtype = torch.float32,
        shared_base_ffn: tuple[nn.Linear, nn.Linear, nn.Linear] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = len(experts)
        self.top_k = min(top_k, self.num_experts)
        self.expert_domains = list(expert_domains)
        if shared_base_ffn is not None:
            self.shared_gate_proj = shared_base_ffn[0]
            self.shared_up_proj = shared_base_ffn[1]
            self.shared_down_proj = shared_base_ffn[2]
        self.router = TopKMoERouter(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=dtype,
        )
        self.experts = nn.ModuleList(experts)
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.register_buffer("expert_token_counts", torch.zeros(self.num_experts, dtype=torch.long))

    def hotplug_expert(self, domain_name: str, expert_module: nn.Module) -> None:
        """Hot-plug a newly converged domain expert E_{N+1} into this MoE layer."""
        self.experts.append(expert_module)
        self.expert_domains.append(domain_name)
        self.num_experts = len(self.experts)
        self.router.expand_for_new_expert()
        old_counts = self.expert_token_counts
        new_counts = torch.zeros(self.num_experts, dtype=torch.long, device=old_counts.device)
        new_counts[: old_counts.numel()] = old_counts
        self.expert_token_counts = new_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.hidden_size)
        routing_weights, selected_experts, aux_loss = self.router(flat_x)
        self.last_aux_loss = aux_loss

        final_hidden = torch.zeros_like(flat_x)
        with torch.no_grad():
            for k_idx in range(self.top_k):
                binc = torch.bincount(selected_experts[:, k_idx], minlength=self.num_experts)
                self.expert_token_counts.add_(binc.to(self.expert_token_counts.device))

        for expert_idx, expert_layer in enumerate(self.experts):
            # Find tokens routed to expert_idx in any of the top_k slots
            mask = selected_experts == expert_idx  # [N, top_k]
            if not mask.any():
                continue
            token_indices, slot_indices = torch.where(mask)
            expert_in = flat_x[token_indices]
            expert_out = expert_layer(expert_in)
            weights = routing_weights[token_indices, slot_indices].unsqueeze(-1).to(expert_out.dtype)
            final_hidden.index_add_(0, token_indices, expert_out * weights)

        return final_hidden.reshape(orig_shape)


class UpcycledSparseMoEModel(nn.Module):
    """Unified Sparse MoE Causal LM upcycled from E domain-specific QLoRA branches."""

    def __init__(
        self,
        upcycled_backbone: nn.Module,
        expert_domains: list[str],
        top_k: int = 2,
        router_aux_loss_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.backbone = upcycled_backbone
        self.expert_domains = list(expert_domains)
        self.num_experts = len(expert_domains)
        self.top_k = top_k
        self.router_aux_loss_coef = router_aux_loss_coef
        # Season 4 (`HM-S4-05`): telemetry of the last SVD rank-adaptive projection.
        self.last_rank_projection_report: SVDLoARankProjectionReport | None = None

    def get_moe_blocks(self) -> list[SparseMoEMLPBlock]:
        blocks: list[SparseMoEMLPBlock] = []
        for mod in self.backbone.modules():
            if isinstance(mod, SparseMoEMLPBlock):
                blocks.append(mod)
        return blocks

    def reset_routing_telemetry(self) -> None:
        for blk in self.get_moe_blocks():
            blk.expert_token_counts.zero_()

    def get_expert_utilization(self) -> dict[str, float]:
        counts = torch.zeros(self.num_experts, dtype=torch.float64)
        for blk in self.get_moe_blocks():
            counts += blk.expert_token_counts.detach().cpu().to(torch.float64)
        total = float(counts.sum().item())
        if total <= 0:
            return {dom: round(1.0 / self.num_experts, 4) for dom in self.expert_domains}
        return {
            dom: round(float(counts[i].item()) / total, 4)
            for i, dom in enumerate(self.expert_domains)
        }

    def hotplug_domain_expert(
        self,
        domain_name: str,
        lora_state_dict: dict[str, torch.Tensor],
        lora_rank_r: int = 16,
        lora_alpha: int = 32,
        target_rank_r: int | None = None,
        source_lora_alpha: float | None = None,
    ) -> int:
        """Phase 3.5 (BTM MoE v2): Hot-plug a new domain LoRA expert (E -> E+1) without retraining existing experts.

        Args:
            domain_name: Human-readable domain identifier of the new expert.
            lora_state_dict: LoRA state dict of the incoming community adapter.
            lora_rank_r: Rank declared by the caller (drives the `alpha / r` scaling).
            lora_alpha: LoRA alpha declared by the caller.
            target_rank_r: When set, the adapter is first projected onto this
                canonical rank with a truncated SVD (`HM-S4-05`) so that
                heterogeneous `r=8..64` adapters can share one canonical layout.
            source_lora_alpha: `alpha` of the incoming adapter (defaults to `lora_alpha`).

        Returns:
            The new total number of experts `E+1`.
        """
        scaling = float(lora_alpha) / float(lora_rank_r)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        cfg = self.backbone.config
        blocks = self.get_moe_blocks()

        if target_rank_r is not None:
            src_alpha = float(source_lora_alpha) if source_lora_alpha is not None else float(lora_alpha)
            # The model keeps applying `lora_alpha / lora_rank_r` to the projected
            # adapter, so the projection must compensate with the same scaling.
            dst_alpha = float(lora_alpha) * float(target_rank_r) / float(lora_rank_r)
            lora_state_dict, projection_report = project_lora_state_dict_to_rank(
                lora_state_dict,
                target_rank_r=target_rank_r,
                source_lora_alpha=src_alpha,
                target_lora_alpha=dst_alpha,
            )
            self.last_rank_projection_report = projection_report

        for layer_idx, blk in enumerate(blocks):
            ab_gate = _extract_ab_pair(lora_state_dict, layer_idx, "mlp.gate_proj")
            ab_up = _extract_ab_pair(lora_state_dict, layer_idx, "mlp.up_proj")
            ab_down = _extract_ab_pair(lora_state_dict, layer_idx, "mlp.down_proj")

            if hasattr(blk, "shared_gate_proj"):
                shared_ffn = (blk.shared_gate_proj, blk.shared_up_proj, blk.shared_down_proj)
                expert_mod: nn.Module = UpcycledFactoredSwiGLUExpert(
                    shared_base_ffn=shared_ffn,
                    ab_gate=ab_gate,
                    ab_up=ab_up,
                    ab_down=ab_down,
                    scaling=scaling,
                    dtype=dtype,
                    device=device,
                )
            else:
                ref_exp: UpcycledSwiGLUExpert = blk.experts[0]  # type: ignore[assignment]
                w0_gate = ref_exp.gate_proj.weight.detach().to(torch.float32)
                w0_up = ref_exp.up_proj.weight.detach().to(torch.float32)
                w0_down = ref_exp.down_proj.weight.detach().to(torch.float32)
                w_gate = w0_gate + (
                    scaling * (ab_gate[1].to(device) @ ab_gate[0].to(device)) if ab_gate else 0.0
                )
                w_up = w0_up + (
                    scaling * (ab_up[1].to(device) @ ab_up[0].to(device)) if ab_up else 0.0
                )
                w_down = w0_down + (
                    scaling * (ab_down[1].to(device) @ ab_down[0].to(device)) if ab_down else 0.0
                )
                expert_mod = UpcycledSwiGLUExpert(
                    hidden_size=cfg.hidden_size,
                    intermediate_size=cfg.intermediate_size,
                    gate_weight=w_gate,
                    up_weight=w_up,
                    down_weight=w_down,
                    dtype=dtype,
                ).to(device)

            blk.hotplug_expert(domain_name=domain_name, expert_module=expert_mod)

        self.expert_domains.append(domain_name)
        self.num_experts = len(self.expert_domains)
        return self.num_experts

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any) -> Any:
        out = self.backbone(input_ids=input_ids, labels=labels, **kwargs)
        if labels is not None and hasattr(out, "loss") and out.loss is not None:
            aux_losses = [blk.last_aux_loss.to(out.loss.device) for blk in self.get_moe_blocks()]
            if aux_losses:
                total_aux = torch.stack(aux_losses).mean()
                out.loss = out.loss + self.router_aux_loss_coef * total_aux
        return out


def measure_moe_vram_footprint_gb(
    moe_model: UpcycledSparseMoEModel,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Measure the resident VRAM footprint of an upcycled Sparse MoE in GiB.

    The factored (VRAM-efficient) expert layout stores only the LoRA factors
    `(A, B)` of every expert on top of the *shared* base FFN, so the footprint
    grows linearly with the number of hot-plugged experts while the base weights
    are paid for only once.

    Args:
        moe_model: The upcycled Sparse MoE to profile.
        device: Device to query for live CUDA allocations (auto-detected by default).

    Returns:
        Dictionary with the weight/buffer breakdown (MiB & GiB) and, when running
        on CUDA, the live and peak allocated VRAM.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in moe_model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in moe_model.buffers())
    total_bytes = param_bytes + buffer_bytes

    if device is None:
        try:
            device = next(moe_model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            device = torch.device("cpu")

    footprint: dict[str, float] = {
        "num_experts": float(moe_model.num_experts),
        "param_mb": round(param_bytes / (1024**2), 4),
        "buffer_mb": round(buffer_bytes / (1024**2), 4),
        "total_mb": round(total_bytes / (1024**2), 4),
        "total_gb": round(total_bytes / (1024**3), 6),
        "cuda_allocated_gb": 0.0,
        "cuda_peak_gb": 0.0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        footprint["cuda_allocated_gb"] = round(torch.cuda.memory_allocated(device) / (1024**3), 6)
        footprint["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 6)
    return footprint


def hotplug_rank_adaptive_experts(
    moe_model: UpcycledSparseMoEModel,
    expert_lora_states: dict[str, dict[str, torch.Tensor]],
    target_rank_r: int = 16,
    lora_alpha: int | None = None,
    expert_lora_alphas: dict[str, float] | None = None,
    vram_budget_gb: float = 2.0,
) -> dict[str, Any]:
    """Season 4 (`HM-S4-05`): hot-plug heterogeneous community adapters at scale.

    Every incoming adapter is projected onto the canonical `target_rank_r` with a
    truncated SVD (`project_lora_state_dict_to_rank`) before being hot-plugged,
    so `r=8`, `r=16`, `r=32` and `r=64` community adapters can all live inside a
    single `UpcycledSparseMoEModel` without VRAM bloat.

    Args:
        moe_model: Target Sparse MoE receiving the new experts.
        expert_lora_states: Mapping `domain_name -> LoRA state dict`.
        target_rank_r: Canonical rank every adapter is projected onto.
        lora_alpha: Alpha the projected adapters are used with (defaults to the
            community-standard `2 * target_rank_r`).
        expert_lora_alphas: Optional per-domain source alpha of the incoming adapter.
        vram_budget_gb: VRAM budget used to flag budget compliance.

    Returns:
        Telemetry dictionary with the expert counts, per-expert projection reports
        and the VRAM footprint measured before and after the hot-plug.
    """
    if target_rank_r < 1:
        raise ValueError(f"target_rank_r must be >= 1, got {target_rank_r}.")

    dst_alpha = float(lora_alpha) if lora_alpha is not None else 2.0 * float(target_rank_r)
    footprint_before = measure_moe_vram_footprint_gb(moe_model)

    projection_reports: dict[str, dict[str, Any]] = {}
    plugged_domains: list[str] = []
    for domain_name, lora_state in expert_lora_states.items():
        src_alpha = (
            float(expert_lora_alphas[domain_name])
            if expert_lora_alphas is not None and domain_name in expert_lora_alphas
            else None  # fall back to the per-pair community-standard `alpha = 2 * r`
        )
        projected, report = project_lora_state_dict_to_rank(
            lora_state,
            target_rank_r=target_rank_r,
            source_lora_alpha=src_alpha,
            target_lora_alpha=dst_alpha,
        )
        projection_reports[domain_name] = {
            "source_rank_r": report.source_rank_r,
            "target_rank_r": report.target_rank_r,
            "num_projected_pairs": report.num_projected_pairs,
            "num_padded_pairs": report.num_padded_pairs,
            "num_identity_pairs": report.num_identity_pairs,
            "mean_spectral_energy_retained": report.mean_spectral_energy_retained,
            "min_spectral_energy_retained": report.min_spectral_energy_retained,
            "max_relative_delta_error": report.max_relative_delta_error,
            "compression_ratio": report.compression_ratio,
            "source_params": report.source_params,
            "target_params": report.target_params,
        }
        moe_model.hotplug_domain_expert(
            domain_name=domain_name,
            lora_state_dict=projected,
            lora_rank_r=target_rank_r,
            lora_alpha=int(dst_alpha),
        )
        plugged_domains.append(domain_name)

    footprint_after = measure_moe_vram_footprint_gb(moe_model)
    return {
        "initial_num_experts": int(footprint_before["num_experts"]),
        "final_num_experts": int(moe_model.num_experts),
        "num_plugged_experts": len(plugged_domains),
        "plugged_domains": plugged_domains,
        "target_rank_r": int(target_rank_r),
        "lora_alpha": dst_alpha,
        "projection_reports": projection_reports,
        "vram_footprint_before_gb": footprint_before["total_gb"],
        "vram_footprint_after_gb": footprint_after["total_gb"],
        "vram_footprint_after_mb": footprint_after["total_mb"],
        "vram_budget_gb": float(vram_budget_gb),
        "within_vram_budget": bool(footprint_after["total_gb"] < float(vram_budget_gb)),
    }


def upcycle_branches_to_sparse_moe(
    peft_or_base_model: nn.Module,
    branch_lora_states: dict[str, dict[str, torch.Tensor]],
    lora_rank_r: int = 64,
    lora_alpha: int = 128,
    top_k: int = 2,
    router_aux_loss_coef: float = 0.01,
    target_device: torch.device | None = None,
    target_dtype: torch.dtype = torch.float32,
    in_place: bool = False,
) -> tuple[UpcycledSparseMoEModel, MoEUpcycleReport]:
    """Phase C: Convert E converged QLoRA domain branches into a unified Sparse MoE model."""
    expert_domains = list(branch_lora_states.keys())
    E = len(expert_domains)
    if E < 2:
        raise ValueError(f"Sparse MoE upcycling requires at least 2 expert branches, got {E}.")

    scaling = float(lora_alpha) / float(lora_rank_r)

    # Unwrap PeftModel to obtain the underlying CausalLM decoder structure
    raw_causal_lm = getattr(peft_or_base_model, "base_model", peft_or_base_model)
    if hasattr(raw_causal_lm, "model") and hasattr(raw_causal_lm.model, "model"):
        raw_causal_lm = raw_causal_lm.model

    if target_device is None:
        target_device = next(raw_causal_lm.parameters()).device

    cfg = copy.deepcopy(raw_causal_lm.config)

    if in_place:
        merged_causal_lm = raw_causal_lm
    else:
        from transformers import Qwen2ForCausalLM

        merged_causal_lm = Qwen2ForCausalLM(cfg).to(dtype=target_dtype)
        with torch.no_grad():
            merged_causal_lm.model.embed_tokens.weight.copy_(
                raw_causal_lm.model.embed_tokens.weight.detach().to(dtype=target_dtype, device="cpu")
            )
            merged_causal_lm.model.norm.weight.copy_(
                raw_causal_lm.model.norm.weight.detach().to(dtype=target_dtype, device="cpu")
            )
            if not cfg.tie_word_embeddings:
                merged_causal_lm.lm_head.weight.copy_(
                    raw_causal_lm.lm_head.weight.detach().to(dtype=target_dtype, device="cpu")
                )

    attn_merged_count = 0
    ffn_experts_created = 0
    work_dev = target_device if in_place else torch.device("cpu")

    with torch.no_grad():
        for layer_idx, src_layer in enumerate(raw_causal_lm.model.layers):
            dst_layer = merged_causal_lm.model.layers[layer_idx]
            if not in_place:
                dst_layer.input_layernorm.weight.copy_(
                    src_layer.input_layernorm.weight.detach().to(dtype=target_dtype, device="cpu")
                )
                dst_layer.post_attention_layernorm.weight.copy_(
                    src_layer.post_attention_layernorm.weight.detach().to(dtype=target_dtype, device="cpu")
                )

            # 1. Average self-attention LoRA adapters across all E branches and merge into W_0
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                src_proj = getattr(src_layer.self_attn, proj_name)
                base_lin = getattr(src_proj, "base_layer", src_proj)
                avg_delta = torch.zeros_like(base_lin.weight.data, dtype=torch.float32, device=work_dev)
                for domain in expert_domains:
                    ab = _extract_ab_pair(
                        branch_lora_states[domain],
                        layer_idx=layer_idx,
                        subpath=f"self_attn.{proj_name}",
                    )
                    if ab is not None:
                        a_mat = ab[0].to(device=work_dev, dtype=torch.float32)
                        b_mat = ab[1].to(device=work_dev, dtype=torch.float32)
                        avg_delta.add_(scaling * (b_mat @ a_mat))
                        del a_mat, b_mat
                avg_delta.div_(float(E))

                if in_place:
                    base_lin.weight.data.copy_(
                        (base_lin.weight.data.to(torch.float32) + avg_delta).to(target_dtype)
                    )
                    setattr(dst_layer.self_attn, proj_name, base_lin)
                else:
                    w0 = base_lin.weight.detach().to(device=work_dev, dtype=torch.float32)
                    new_attn_lin = nn.Linear(
                        base_lin.in_features,
                        base_lin.out_features,
                        bias=base_lin.bias is not None,
                        device=work_dev,
                        dtype=target_dtype,
                    )
                    new_attn_lin.weight.copy_((w0 + avg_delta).to(target_dtype))
                    if base_lin.bias is not None and new_attn_lin.bias is not None:
                        new_attn_lin.bias.copy_(base_lin.bias.detach().to(device=work_dev, dtype=target_dtype))
                    setattr(dst_layer.self_attn, proj_name, new_attn_lin)
                    del w0
                del avg_delta
                attn_merged_count += 1

            # 2. Construct E distinct FFN experts: W_{ffn, e} = W_0_ffn + (alpha/r) * B_e @ A_e
            base_gate = getattr(src_layer.mlp.gate_proj, "base_layer", src_layer.mlp.gate_proj)
            base_up = getattr(src_layer.mlp.up_proj, "base_layer", src_layer.mlp.up_proj)
            base_down = getattr(src_layer.mlp.down_proj, "base_layer", src_layer.mlp.down_proj)

            layer_experts: list[nn.Module] = []
            if in_place:
                shared_ffn = (base_gate, base_up, base_down)
                for domain in expert_domains:
                    b_state = branch_lora_states[domain]
                    ab_gate = _extract_ab_pair(b_state, layer_idx, "mlp.gate_proj")
                    ab_up = _extract_ab_pair(b_state, layer_idx, "mlp.up_proj")
                    ab_down = _extract_ab_pair(b_state, layer_idx, "mlp.down_proj")
                    expert_mod = UpcycledFactoredSwiGLUExpert(
                        shared_base_ffn=shared_ffn,
                        ab_gate=ab_gate,
                        ab_up=ab_up,
                        ab_down=ab_down,
                        scaling=scaling,
                        dtype=target_dtype,
                        device=work_dev,
                    )
                    layer_experts.append(expert_mod)
                    ffn_experts_created += 1

                dst_layer.mlp = SparseMoEMLPBlock(
                    hidden_size=cfg.hidden_size,
                    experts=layer_experts,
                    expert_domains=expert_domains,
                    top_k=top_k,
                    dtype=target_dtype,
                    shared_base_ffn=shared_ffn,
                ).to(work_dev)
            else:
                w0_gate = _dequantize_linear_weight(src_layer.mlp.gate_proj).to(work_dev)
                w0_up = _dequantize_linear_weight(src_layer.mlp.up_proj).to(work_dev)
                w0_down = _dequantize_linear_weight(src_layer.mlp.down_proj).to(work_dev)

                for domain in expert_domains:
                    b_state = branch_lora_states[domain]
                    ab_gate = _extract_ab_pair(b_state, layer_idx, "mlp.gate_proj")
                    ab_up = _extract_ab_pair(b_state, layer_idx, "mlp.up_proj")
                    ab_down = _extract_ab_pair(b_state, layer_idx, "mlp.down_proj")

                    w_gate = w0_gate + (
                        scaling * (ab_gate[1].to(work_dev) @ ab_gate[0].to(work_dev)) if ab_gate else 0.0
                    )
                    w_up = w0_up + (
                        scaling * (ab_up[1].to(work_dev) @ ab_up[0].to(work_dev)) if ab_up else 0.0
                    )
                    w_down = w0_down + (
                        scaling * (ab_down[1].to(work_dev) @ ab_down[0].to(work_dev)) if ab_down else 0.0
                    )

                    expert_mod = UpcycledSwiGLUExpert(
                        hidden_size=cfg.hidden_size,
                        intermediate_size=cfg.intermediate_size,
                        gate_weight=w_gate,
                        up_weight=w_up,
                        down_weight=w_down,
                        dtype=target_dtype,
                    ).to(work_dev)
                    layer_experts.append(expert_mod)
                    del w_gate, w_up, w_down
                    ffn_experts_created += 1

                del w0_gate, w0_up, w0_down
                dst_layer.mlp = SparseMoEMLPBlock(
                    hidden_size=cfg.hidden_size,
                    experts=layer_experts,
                    expert_domains=expert_domains,
                    top_k=top_k,
                    dtype=target_dtype,
                ).to(work_dev)

    moe_model = UpcycledSparseMoEModel(
        upcycled_backbone=merged_causal_lm.to(target_device),
        expert_domains=expert_domains,
        top_k=top_k,
        router_aux_loss_coef=router_aux_loss_coef,
    )

    num_moe_layers = len(moe_model.get_moe_blocks())
    single_ffn_params = num_moe_layers * 3 * int(cfg.hidden_size) * int(cfg.intermediate_size)
    raw_param_sum = sum(p.numel() for p in moe_model.parameters())
    if in_place:
        non_ffn_params = raw_param_sum - single_ffn_params
        total_moe_params = non_ffn_params + single_ffn_params * E
    else:
        total_moe_params = raw_param_sum
        non_ffn_params = total_moe_params - single_ffn_params * E
    active_params = non_ffn_params + single_ffn_params * min(top_k, E)

    report = MoEUpcycleReport(
        num_experts=E,
        top_k=min(top_k, E),
        expert_domains=expert_domains,
        attention_layers_merged=attn_merged_count,
        ffn_expert_modules_created=ffn_experts_created,
        total_moe_parameters=total_moe_params,
        active_parameters_per_token=active_params,
    )
    return moe_model, report


def train_moe_router_gating(
    moe_model: UpcycledSparseMoEModel,
    domain_token_batches: dict[str, torch.Tensor],
    steps: int = 40,
    lr: float = 3e-3,
) -> tuple[float, float, dict[str, float]]:
    """Train only the lightweight Top-2 gating network W_g while freezing upcycled experts."""
    device = next(moe_model.parameters()).device
    moe_model.train()
    moe_model.reset_routing_telemetry()

    router_params: list[nn.Parameter] = []
    for name, param in moe_model.named_parameters():
        if "router.wg" in name:
            param.requires_grad = True
            router_params.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(router_params, lr=lr, weight_decay=0.0)
    domains = list(domain_token_batches.keys())
    num_doms = max(1, len(domains))
    first_epoch_losses: list[float] = []
    last_epoch_losses: list[float] = []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        dom_idx = (step - 1) % num_doms
        domain = domains[dom_idx]
        tokens = domain_token_batches[domain]
        batch = tokens[0:1].to(device, dtype=torch.long)
        out = moe_model(input_ids=batch, labels=batch)
        loss: torch.Tensor = out.loss
        loss.backward()
        optimizer.step()

        lval = float(loss.detach().cpu().item())
        if step <= num_doms:
            first_epoch_losses.append(lval)
        if step > max(0, steps - num_doms):
            last_epoch_losses.append(lval)

    for blk in moe_model.get_moe_blocks():
        blk.last_aux_loss = blk.last_aux_loss.detach()

    initial_loss = sum(first_epoch_losses) / max(1, len(first_epoch_losses))
    final_loss = sum(last_epoch_losses) / max(1, len(last_epoch_losses))
    return round(initial_loss, 6), round(final_loss, 6), moe_model.get_expert_utilization()

