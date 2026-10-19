"""Season 3 (`HM-S3-04`) & Season 4 (`HM-S4-05`) BTM Sparse MoE Domain Expert Registry.

Re-exports the core Branch-Train-Merge (`btm_upcycle`) primitives and provides
`Season3DomainExpertRegistry` for managing and routing specialized domain QLoRA
adapters (`Rust_Systems_Security`, `Solidity_Smart_Contracts`,
`Biomedical_Clinical_QA`, and `Lean4_Theorem_Prover`).

Season 4 (`HM-S4-05`) adds `project_lora_rank_svd`: a truncated-SVD rank-adaptive
LoRA projection that compresses higher-rank community adapters (`r=32`, `r=64`) and
zero-pads lower-rank ones (`r=8`) onto a single canonical rank (`r=16`), so that
16-32 heterogeneous community experts can be hot-plugged into one unified
`UpcycledSparseMoEModel` inside a `< 2.0 GB` VRAM budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

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


def project_lora_rank_svd(
    lora_state_dict: Dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Season 4 (`HM-S4-05`): truncated-SVD rank-adaptive LoRA projection.

    Community miners train LoRA adapters with heterogeneous ranks (`r=8`, `r=16`,
    `r=32`, `r=64`). This projects any of them onto a single canonical rank
    (`target_rank_r`, default `16`) so that they can be hot-plugged into one
    unified `UpcycledSparseMoEModel` without VRAM bloat:

    - **Compression** (`r=32`, `r=64`): the effective weight delta `D = B @ A` is
      expanded in its singular basis and truncated to the `target_rank_r` largest
      singular values (Eckart-Young optimal), then refactored back into a LoRA pair.
    - **Padding** (`r=8`): the adapter is zero-padded up to `target_rank_r`, so the
      effective delta is preserved *exactly* while the nominal rank becomes canonical.

    The `alpha / r` scaling convention is preserved across ranks, i.e. the returned
    adapter satisfies

        (target_lora_alpha / target_rank_r) * B' @ A'
            ~= (source_lora_alpha / source_rank_r) * B @ A

    so the merged weights of the caller stay identical (up to the truncation error
    of the discarded spectral tail).

    Args:
        lora_state_dict: LoRA state dict with `lora_A` / `lora_B` tensor pairs
            (PEFT naming: `...lora_A.default.weight` / `...lora_B.default.weight`).
        target_rank_r: Canonical rank to project onto.
        source_lora_alpha: `alpha` of the source adapter (defaults to the
            community-standard `2 * r` convention).
        target_lora_alpha: `alpha` the projected adapter will be used with
            (defaults to the community-standard `2 * target_rank_r` convention).

    Returns:
        A new LoRA state dict with the same keys and canonical rank.
    """
    projected, _ = project_lora_state_dict_to_rank(
        lora_state_dict,
        target_rank_r=target_rank_r,
        source_lora_alpha=source_lora_alpha,
        target_lora_alpha=target_lora_alpha,
    )
    return projected


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


__all__ = [
    "ExpertBranchState",
    "MoEUpcycleReport",
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
