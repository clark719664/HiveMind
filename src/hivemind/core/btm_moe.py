"""Season 3 (`HM-S3-04`), Season 4 (`HM-S4-05`) & Season 5 (`HM-S5-02`) BTM Sparse MoE.

Re-exports the core Branch-Train-Merge (`btm_upcycle`) primitives and provides
`Season3DomainExpertRegistry` for managing and routing specialized domain QLoRA
adapters (`Rust_Systems_Security`, `Solidity_Smart_Contracts`,
`Biomedical_Clinical_QA`, and `Lean4_Theorem_Prover`).

Season 4 (`HM-S4-05`) adds `project_lora_rank_svd`: a truncated-SVD rank-adaptive
LoRA projection that compresses higher-rank community adapters (`r=32`, `r=64`) and
zero-pads lower-rank ones (`r=8`) onto a single canonical rank (`r=16`), so that
16-32 heterogeneous community experts can be hot-plugged into one unified
`UpcycledSparseMoEModel` inside a `< 2.0 GB` VRAM budget.

Season 5 (`HM-S5-02`) adds the **64-Expert 2-Level Hierarchical Sparse MoE**:

- `HierarchicalMoERouter`: a `G_macro -> G_micro` router over `8` macro-domains x
  `8` micro-experts (`E = 64`) that selects the **Top-2 micro-experts inside the
  winning macro-cluster**. It implements the exact `TopKMoERouter` contract
  (`forward(hidden_states) -> (routing_weights, selected_experts, aux_loss)`) and
  returns **global** expert indices, so it is a drop-in replacement inside
  `SparseMoEMLPBlock`.
- `hotplug_hierarchical_experts` / `build_hierarchical_sparse_moe`: hot-plug 64
  heterogeneous community adapters (all SVD-projected onto the canonical `r=16`)
  and install the hierarchical router, keeping the whole model inside the
  `< 2.2 GB` VRAM ceiling enforced by the FP8 paged KV-cache of
  `hivemind.swarm.swarm_fabric`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.core.btm_upcycle import (
    ExpertBranchState,
    MoEUpcycleReport,
    SparseMoEMLPBlock,
    SVDLoARankProjectionReport,
    TopKMoERouter,
    UpcycledFactoredSwiGLUExpert,
    UpcycledSparseMoEModel,
    UpcycledSwiGLUExpert,
    detect_lora_rank,
    hotplug_rank_adaptive_experts,
    initialize_domain_branches,
    measure_moe_vram_footprint_gb,
    pair_lora_ab_keys,
    project_lora_state_dict_to_rank,
    svd_project_lora_pair,
    train_moe_router_gating,
    upcycle_branches_to_sparse_moe,
)

SEASON_3_DOMAIN_EXPERT_SPECS: Dict[str, Dict[str, Any]] = {
    "Rust_Systems_Security": {
        "dataset_id": "anon/rust-code-instructions",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["fn", "pub", "mut", "arc", "mutex", "unsafe", "borrow", "lifetime", "rust"],
        "description": "Zero-copy systems programming, borrow-checker invariants, and memory-safe concurrency in Rust.",
    },
    "Solidity_Smart_Contracts": {
        "dataset_id": "hivemind/solidity-audit-invariants",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["solidity", "pragma", "contract", "reentrancy", "erc20", "uint256", "msg.sender", "require"],
        "description": "EVM smart contract security auditing, CEI reentrancy defense, and fixed-point tokenomics invariants.",
    },
    "Biomedical_Clinical_QA": {
        "dataset_id": "qiaojin/PubMedQA",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["clinical", "patient", "atp", "mitochondria", "apoptosis", "therapy", "trial", "enzyme"],
        "description": "Evidence-based clinical synthesis, cellular bioenergetics, and PubMedQA diagnostic reasoning.",
    },
    "Lean4_Theorem_Prover": {
        "dataset_id": "hivemind/lean4-mathlib-proofs",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["theorem", "lemma", "omega", "simp", "nat", "lean", "by", "induction", "proof"],
        "description": "Machine-checked Lean 4 formal verification with zero sorrys and zero custom axioms.",
    },
}


@dataclass
class RegisteredDomainExpert:
    """Metadata and CAS location for a trained Season 3 domain LoRA adapter."""

    domain_name: str
    dataset_id: str
    expert_cid: str
    shard_cid: str
    lora_rank_r: int
    lora_alpha: int
    initial_loss: float
    final_loss: float
    compression_ratio: float
    compressed_bytes: int
    lora_state_dict: Dict[str, torch.Tensor] = field(repr=False)


class ProjectedLoRAStateDict(dict):
    """Dict of projected LoRA tensors that also exposes metadata keys (`state_dict`, `factors`, `target_rank`, `parameter_bytes`)."""

    def __init__(self, tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
        super().__init__(tensors)
        self._meta = meta

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key == "state_dict":
            return dict(self)
        if key in self._meta:
            return self._meta[key]
        raise KeyError(key)


def project_lora_rank_svd(
    lora_state_dict: Dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
) -> Dict[str, Any]:
    """Season 4 (`HM-S4-05`): truncated-SVD rank-adaptive LoRA projection.

    Supports both paired `(lora_A, lora_B)` PEFT state dicts (PR #34) and standalone
    2D LoRA factor dictionaries (PR #33), returning a `ProjectedLoRAStateDict` that behaves
    as a tensor `state_dict` while also exposing `["state_dict"]`, `["factors"]`,
    `["target_rank"]`, and `["parameter_bytes"]`.
    """
    if target_rank_r < 1:
        raise ValueError("target_rank_r must be positive")

    pairs = pair_lora_ab_keys(lora_state_dict)
    if pairs:
        projected, report = project_lora_state_dict_to_rank(
            lora_state_dict,
            target_rank_r=target_rank_r,
            source_lora_alpha=source_lora_alpha,
            target_lora_alpha=target_lora_alpha,
        )
        factors: List[Dict[str, Any]] = [
            {
                "name": name,
                "source_rank": report.source_rank_r,
                "kept_rank": min(report.source_rank_r, target_rank_r),
                "target_rank": target_rank_r,
                "output_shape": list(t.shape),
                "spectral_energy_kept": report.mean_spectral_energy_retained,
            }
            for name, t in projected.items()
            if isinstance(t, torch.Tensor)
        ]
        param_bytes = sum(
            t.numel() * t.element_size() for t in projected.values() if isinstance(t, torch.Tensor)
        )
        return ProjectedLoRAStateDict(
            projected,
            {
                "target_rank": target_rank_r,
                "factors": factors,
                "parameter_bytes": param_bytes,
                "report": report,
            },
        )

    projected_single: Dict[str, torch.Tensor] = {}
    reports: List[Dict[str, Any]] = []
    for name, weight in lora_state_dict.items():
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            projected_single[name] = weight.detach().clone() if isinstance(weight, torch.Tensor) else weight
            continue
        matrix = weight.detach().to(torch.float32)
        rank_on_rows = matrix.shape[0] <= matrix.shape[1]
        current_rank = int(matrix.shape[0] if rank_on_rows else matrix.shape[1])
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        keep = min(target_rank_r, int(singular.shape[0]), current_rank)
        core = (u[:, :keep] * singular[:keep].unsqueeze(0)) @ vh[:keep, :]
        if rank_on_rows:
            if keep < target_rank_r:
                core = torch.nn.functional.pad(core, (0, 0, 0, target_rank_r - keep))
            else:
                core = core[:target_rank_r, :]
        else:
            if keep < target_rank_r:
                core = torch.nn.functional.pad(core, (0, target_rank_r - keep, 0, 0))
            else:
                core = core[:, :target_rank_r]
        projected_single[name] = core.to(dtype=weight.dtype)
        kept_energy = float(torch.linalg.vector_norm(singular[:keep]).item())
        full_energy = float(torch.linalg.vector_norm(singular).item())
        reports.append(
            {
                "name": name,
                "source_rank": current_rank,
                "kept_rank": keep,
                "target_rank": target_rank_r,
                "output_shape": list(core.shape),
                "spectral_energy_kept": round(kept_energy / max(full_energy, 1e-8), 6),
            }
        )
    parameter_bytes = sum(
        t.numel() * t.element_size() for t in projected_single.values() if isinstance(t, torch.Tensor)
    )
    return ProjectedLoRAStateDict(
        projected_single,
        {
            "target_rank": target_rank_r,
            "factors": reports,
            "parameter_bytes": parameter_bytes,
        },
    )


def project_lora_rank_svd_with_report(
    lora_state_dict: Dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
) -> Tuple[Dict[str, torch.Tensor], SVDLoARankProjectionReport]:
    """Same as `project_lora_rank_svd` but also returns the spectral-energy report."""
    return project_lora_state_dict_to_rank(
        lora_state_dict,
        target_rank_r=target_rank_r,
        source_lora_alpha=source_lora_alpha,
        target_lora_alpha=target_lora_alpha,
    )


class Season3DomainExpertRegistry:
    """Registry managing Season 3 domain LoRA adapters and dynamic Top-2 MoE hot-plugging (`HM-S3-04`)."""

    def __init__(self) -> None:
        self.experts: Dict[str, RegisteredDomainExpert] = {}

    def register_expert(self, expert: RegisteredDomainExpert) -> None:
        self.experts[expert.domain_name] = expert

    def list_domains(self) -> List[str]:
        return list(self.experts.keys())

    def hot_plug_all_into_moe(
        self,
        moe_model: UpcycledSparseMoEModel,
        target_rank_r: int | None = None,
    ) -> Dict[str, Any]:
        """Hot-plugs all registered Season 3 domain adapters into a live `UpcycledSparseMoEModel`.

        Args:
            moe_model: Live Sparse MoE receiving the domain experts.
            target_rank_r: When set, every adapter is first projected onto this
                canonical rank with `project_lora_rank_svd` (`HM-S4-05`), so
                heterogeneous community ranks can coexist without VRAM bloat.
        """
        initial_experts = moe_model.num_experts
        plugged_domains: List[str] = []
        for domain_name, exp in self.experts.items():
            if domain_name not in moe_model.expert_domains:
                lora_state = exp.lora_state_dict
                if target_rank_r is not None:
                    lora_state = project_lora_rank_svd(
                        lora_state,
                        target_rank_r=target_rank_r,
                        source_lora_alpha=float(exp.lora_alpha),
                    )
                moe_model.hotplug_domain_expert(
                    domain_name=domain_name,
                    lora_state_dict=lora_state,
                    lora_rank_r=exp.lora_rank_r,
                    lora_alpha=exp.lora_alpha,
                )
                plugged_domains.append(domain_name)
        return {
            "initial_num_experts": initial_experts,
            "final_num_experts": moe_model.num_experts,
            "plugged_domains": plugged_domains,
            "active_branch_names": list(moe_model.expert_domains),
        }

    def route_domain_query(self, prompt: str, top_k: int = 2) -> Tuple[List[str], Dict[str, float]]:
        """Scores a prompt against Season 3 domain keyword signatures and returns Top-k expert weights."""
        p_low = prompt.lower()
        raw_scores: Dict[str, float] = {}
        for domain, spec in SEASON_3_DOMAIN_EXPERT_SPECS.items():
            hits = sum(1 for kw in spec["keywords"] if kw in p_low)
            raw_scores[domain] = 1.0 + hits * 2.5

        ranked = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = ranked[: max(1, int(top_k))]
        total = sum(score for _, score in selected) or 1.0
        weights = {dom: round(score / total, 4) for dom, score in selected}
        return [dom for dom, _ in selected], weights




class HierarchicalMoERouter(nn.Module):
    """Season 5 (`HM-S5-02`): 2-Level `G_macro -> G_micro` Router for 64-Expert Sparse MoE.

    Partitions `E = num_macro_domains * experts_per_domain` (default `8 * 8 = 64`) experts into
    `num_macro_domains` macro-clusters. For each token:
      1. `macro_gate(x)` selects the winning macro-domain `m* in {0..M-1}` (`O(M)`).
      2. `micro_gates[m*](x)` selects the Top-`k` (`k=2`) micro-experts inside cluster `m*` (`O(E/M)`).
    Total router gating cost is `M + E/M = 16` inner products instead of `64` (`4.0x` router speedup).
    """

    def __init__(
        self,
        hidden_size: int,
        num_macro_domains: int = 8,
        experts_per_domain: int = 8,
        top_k: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _ = dtype
        if num_macro_domains < 1 or experts_per_domain < 1:
            raise ValueError("num_macro_domains and experts_per_domain must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_macro_domains = int(num_macro_domains)
        self.experts_per_domain = int(experts_per_domain)
        self.num_experts = self.num_macro_domains * self.experts_per_domain
        self.top_k = min(int(top_k), self.experts_per_domain)

        self.macro_gate = nn.Linear(
            self.hidden_size, self.num_macro_domains, bias=False, dtype=torch.float32
        )
        self.micro_gates = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.experts_per_domain, bias=False, dtype=torch.float32)
                for _ in range(self.num_macro_domains)
            ]
        )
        nn.init.normal_(self.macro_gate.weight, mean=0.0, std=0.02)
        for mg in self.micro_gates:
            nn.init.normal_(mg.weight, mean=0.0, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical Top-k routing weights, global expert indices `[0..E-1]`, and aux loss."""
        hs_fp32 = hidden_states.to(torch.float32)
        num_tokens = hs_fp32.shape[0]
        device = hs_fp32.device

        macro_logits = self.macro_gate(hs_fp32)  # [N, M]
        macro_probs = F.softmax(macro_logits, dim=-1, dtype=torch.float32)
        winning_macro_prob, winning_macro_idx = torch.max(macro_probs, dim=-1)  # [N]

        # Stack micro gates `[M, E_m, d]` for vectorized per-token dispatch
        micro_weight_stack = torch.stack(
            [mg.weight for mg in self.micro_gates], dim=0
        )  # [M, E_m, d]
        token_micro_weights = micro_weight_stack[winning_macro_idx]  # [N, E_m, d]
        micro_logits = torch.bmm(
            token_micro_weights, hs_fp32.unsqueeze(-1)
        ).squeeze(-1)  # [N, E_m]

        top_micro_logits, top_micro_indices = torch.topk(
            micro_logits, k=self.top_k, dim=-1
        )  # [N, top_k]
        micro_probs = F.softmax(top_micro_logits, dim=-1, dtype=torch.float32)
        routing_weights = (
            micro_probs * winning_macro_prob.unsqueeze(-1)
        )
        routing_weights = (
            routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ).to(hidden_states.dtype)

        global_selected_experts = (
            winning_macro_idx.unsqueeze(-1) * self.experts_per_domain + top_micro_indices
        ).clamp(0, self.num_experts - 1)

        # Switch-style hierarchical load-balancing auxiliary loss
        one_hot_macro = F.one_hot(
            winning_macro_idx, num_classes=self.num_macro_domains
        ).to(torch.float32)
        frac_macro = one_hot_macro.mean(dim=0)
        mean_macro_prob = macro_probs.mean(dim=0)
        aux_loss = self.num_macro_domains * torch.sum(frac_macro * mean_macro_prob)
        _ = num_tokens, device
        return routing_weights, global_selected_experts, aux_loss

    def expand_for_new_expert(self) -> None:
        """Expand the last macro cluster's micro-gate to accommodate a newly hot-plugged expert."""
        self.num_experts += 1
        last_mg = self.micro_gates[-1]
        old_out = last_mg.out_features
        new_mg = nn.Linear(
            self.hidden_size,
            old_out + 1,
            bias=False,
            dtype=torch.float32,
            device=last_mg.weight.device,
        )
        with torch.no_grad():
            new_mg.weight[:old_out].copy_(last_mg.weight)
            mean_col = last_mg.weight.mean(dim=0, keepdim=True)
            new_mg.weight[old_out:].copy_(mean_col)
        self.micro_gates[-1] = new_mg
        self.experts_per_domain = max(self.experts_per_domain, old_out + 1)


class FP8PagedKVCache:
    """Season 5 (`HM-S5-02`): Block-Scaled FP8 (`float8_e4m3fn`) Paged KV-Cache Manager.

    Stores attention Key/Value states in fixed-size pages (`page_size_tokens=16`) quantized to
    8-bit (`float8_e4m3fn` with per-page FP16 absmax scales), achieving `2.0x` KV-cache VRAM
    compression so a 64-Expert Sparse MoE + active KV pages stay strictly `< 2.2 GB` VRAM.
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        page_size_tokens: int = 16,
        max_pages: int = 128,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size_tokens = int(page_size_tokens)
        self.max_pages = int(max_pages)
        self.device = torch.device(device)
        self._pages_k: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pages_v: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _quantize_page_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_fp32 = tensor.detach().to(torch.float32)
        absmax = t_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (absmax / 448.0).to(torch.float16)
        scaled = (t_fp32 / scale.to(torch.float32)).clamp(-448.0, 448.0)
        if hasattr(torch, "float8_e4m3fn"):
            q = scaled.to(torch.float8_e4m3fn)
        else:
            q = torch.round(scaled / 448.0 * 127.0).to(torch.int8)
        return q, scale

    @staticmethod
    def _dequantize_page_fp8(
        q: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        if hasattr(torch, "float8_e4m3fn") and q.dtype == torch.float8_e4m3fn:
            return (q.to(torch.float32) * scale.to(torch.float32)).to(target_dtype)
        return ((q.to(torch.float32) / 127.0) * 448.0 * scale.to(torch.float32)).to(target_dtype)

    def write_kv_page(
        self, layer_idx: int, page_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Quantize and store a `(K, V)` page in FP8 with per-token FP16 scales."""
        if len(self._pages_k) >= self.max_pages and (layer_idx, page_idx) not in self._pages_k:
            oldest_key = next(iter(self._pages_k))
            self._pages_k.pop(oldest_key, None)
            self._pages_v.pop(oldest_key, None)
        qk, sk = self._quantize_page_fp8(k_tensor.to(self.device))
        qv, sv = self._quantize_page_fp8(v_tensor.to(self.device))
        self._pages_k[(layer_idx, page_idx)] = (qk, sk)
        self._pages_v[(layer_idx, page_idx)] = (qv, sv)
        fp16_bytes = (k_tensor.numel() + v_tensor.numel()) * 2
        fp8_bytes = (
            qk.numel() * qk.element_size()
            + qv.numel() * qv.element_size()
            + (sk.numel() + sv.numel()) * 2
        )
        return {
            "layer_idx": layer_idx,
            "page_idx": page_idx,
            "fp16_equivalent_bytes": fp16_bytes,
            "fp8_paged_bytes": fp8_bytes,
            "compression_ratio": round(fp16_bytes / max(1, fp8_bytes), 4),
        }

    def read_kv_page(
        self, layer_idx: int, page_idx: int, target_dtype: torch.dtype = torch.float16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return a stored `(K, V)` page."""
        qk, sk = self._pages_k[(layer_idx, page_idx)]
        qv, sv = self._pages_v[(layer_idx, page_idx)]
        return (
            self._dequantize_page_fp8(qk, sk, target_dtype=target_dtype),
            self._dequantize_page_fp8(qv, sv, target_dtype=target_dtype),
        )

    def total_kv_cache_bytes(self) -> int:
        total = 0
        for q, s in list(self._pages_k.values()) + list(self._pages_v.values()):
            total += q.numel() * q.element_size() + s.numel() * s.element_size()
        return total


def hotplug_hierarchical_experts(
    moe_model: UpcycledSparseMoEModel,
    community_adapters: Dict[str, Dict[str, torch.Tensor]],
    canonical_rank_r: int = 16,
    num_macro_domains: int = 8,
    experts_per_domain: int = 8,
    top_k: int = 2,
    vram_ceiling_gb: float = 2.2,
) -> Dict[str, Any]:
    """Season 5 (`HM-S5-02`): Hot-plug up to 64 heterogeneous community adapters into `moe_model`
    using SVD rank normalization (`r -> canonical_rank_r`) and install `HierarchicalMoERouter`.
    """
    hotplug_report = hotplug_rank_adaptive_experts(
        moe_model=moe_model,
        community_adapters=community_adapters,
        target_rank_r=canonical_rank_r,
        vram_ceiling_gb=vram_ceiling_gb,
    )
    total_experts = moe_model.num_experts
    eff_macro = min(num_macro_domains, max(1, total_experts))
    eff_micro = max(1, (total_experts + eff_macro - 1) // eff_macro)

    for block in moe_model.moe_blocks:
        dev = block.router.wg.weight.device
        hier_router = HierarchicalMoERouter(
            hidden_size=moe_model.hidden_size,
            num_macro_domains=eff_macro,
            experts_per_domain=eff_micro,
            top_k=top_k,
        ).to(dev)
        block.router = hier_router

    vram_stats = measure_moe_vram_footprint_gb(moe_model)
    return {
        **hotplug_report,
        "router_type": "HierarchicalMoERouter",
        "num_macro_domains": eff_macro,
        "experts_per_domain": eff_micro,
        "total_experts": total_experts,
        "gating_flops_reduction_factor": round(
            float(total_experts) / float(eff_macro + eff_micro), 2
        ),
        "vram_footprint_gb": vram_stats["total_footprint_gb"],
        "fits_under_2_2gb_vram": vram_stats["total_footprint_gb"] < vram_ceiling_gb,
    }


__all__ = [
    "ExpertBranchState",
    "FP8PagedKVCache",
    "HierarchicalMoERouter",
    "MoEUpcycleReport",
    "ProjectedLoRAStateDict",
    "RegisteredDomainExpert",
    "SEASON_3_DOMAIN_EXPERT_SPECS",
    "SVDLoARankProjectionReport",
    "Season3DomainExpertRegistry",
    "SparseMoEMLPBlock",
    "TopKMoERouter",
    "UpcycledFactoredSwiGLUExpert",
    "UpcycledSparseMoEModel",
    "UpcycledSwiGLUExpert",
    "detect_lora_rank",
    "hotplug_hierarchical_experts",
    "hotplug_rank_adaptive_experts",
    "initialize_domain_branches",
    "measure_moe_vram_footprint_gb",
    "pair_lora_ab_keys",
    "project_lora_rank_svd",
    "project_lora_rank_svd_with_report",
    "project_lora_state_dict_to_rank",
    "svd_project_lora_pair",
    "train_moe_router_gating",
    "upcycle_branches_to_sparse_moe",
]
