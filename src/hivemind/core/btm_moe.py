"""Season 3 (`HM-S3-04`) BTM Sparse MoE Domain Expert Registry & Hot-Plug Router.

Re-exports the core Branch-Train-Merge (`btm_upcycle`) primitives and provides
`Season3DomainExpertRegistry` for managing and routing specialized domain QLoRA
adapters (`Rust_Systems_Security`, `Solidity_Smart_Contracts`,
`Biomedical_Clinical_QA`, and `Lean4_Theorem_Prover`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from hivemind.core.btm_upcycle import (
    ExpertBranchState,
    MoEUpcycleReport,
    SparseMoEMLPBlock,
    TopKMoERouter,
    UpcycledFactoredSwiGLUExpert,
    UpcycledSparseMoEModel,
    UpcycledSwiGLUExpert,
    initialize_domain_branches,
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


class Season3DomainExpertRegistry:
    """Registry managing Season 3 domain LoRA adapters and dynamic Top-2 MoE hot-plugging (`HM-S3-04`)."""

    def __init__(self) -> None:
        self.experts: Dict[str, RegisteredDomainExpert] = {}

    def register_expert(self, expert: RegisteredDomainExpert) -> None:
        self.experts[expert.domain_name] = expert

    def list_domains(self) -> List[str]:
        return list(self.experts.keys())

    def hot_plug_all_into_moe(self, moe_model: UpcycledSparseMoEModel) -> Dict[str, Any]:
        """Hot-plugs all registered Season 3 domain adapters into a live `UpcycledSparseMoEModel`."""
        initial_experts = moe_model.num_experts
        plugged_domains: List[str] = []
        for domain_name, exp in self.experts.items():
            if domain_name not in moe_model.expert_domains:
                moe_model.hotplug_domain_expert(
                    domain_name=domain_name,
                    lora_state_dict=exp.lora_state_dict,
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


def project_lora_rank_svd(
    lora_state_dict: Dict[str, torch.Tensor],
    target_rank_r: int = 16,
) -> Dict[str, Any]:
    """HM-S4-05: truncated-SVD projection of heterogeneous LoRA factors onto one rank.

    The smaller matrix dimension is treated as the adapter rank. Higher ranks
    are truncated to `target_rank_r`. Lower ranks are zero-padded up to it.
    The reconstructed factor keeps the non-rank dimension unchanged.
    """
    if target_rank_r < 1:
        raise ValueError("target_rank_r must be positive")
    projected: Dict[str, torch.Tensor] = {}
    reports: List[Dict[str, Any]] = []
    for name, weight in lora_state_dict.items():
        if weight.ndim != 2:
            projected[name] = weight.detach().clone()
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
        projected[name] = core.to(dtype=weight.dtype)
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
    parameter_bytes = sum(t.numel() * t.element_size() for t in projected.values())
    return {
        "state_dict": projected,
        "target_rank": target_rank_r,
        "factors": reports,
        "parameter_bytes": parameter_bytes,
    }


__all__ = [
    "ExpertBranchState",
    "MoEUpcycleReport",
    "RegisteredDomainExpert",
    "SEASON_3_DOMAIN_EXPERT_SPECS",
    "Season3DomainExpertRegistry",
    "SparseMoEMLPBlock",
    "TopKMoERouter",
    "UpcycledFactoredSwiGLUExpert",
    "UpcycledSparseMoEModel",
    "UpcycledSwiGLUExpert",
    "initialize_domain_branches",
    "project_lora_rank_svd",
    "train_moe_router_gating",
    "upcycle_branches_to_sparse_moe",
]
