"""Unit tests for Phase A Branching & Phase C Sparse MoE Upcycling (Top-2 Router)."""

from __future__ import annotations

import torch

from hivemind.core.btm_upcycle import (
    DEFAULT_EXPERT_DOMAINS,
    initialize_domain_branches,
    train_moe_router_gating,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)


def test_eight_branch_sparse_moe_upcycling_and_top2_routing() -> None:
    device = torch.device("cpu")
    base_model, _, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        device=device,
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=8,
        lora_alpha=16,
        gradient_checkpointing=False,
        seed=42,
    )
    init_lora = extract_lora_state_dict(bundle.model)

    # Phase A: Initialize E = 8 branches
    branches = initialize_domain_branches(init_lora, expert_domains=DEFAULT_EXPERT_DOMAINS)
    assert len(branches) == 8

    # Perturb each branch's LoRA B matrices distinctly to simulate domain specialization
    branch_states: dict[str, dict[str, torch.Tensor]] = {}
    for idx, (dom, b_state) in enumerate(branches.items()):
        specialized: dict[str, torch.Tensor] = {}
        for k, v in b_state.lora_state.items():
            if "lora_B" in k:
                specialized[k] = v + (idx + 1) * 0.01 * torch.ones_like(v)
            else:
                specialized[k] = v.clone()
        branch_states[dom] = specialized

    # Phase C: Upcycle 8 branches into a single 8-Expert Sparse MoE with Top-2 routing
    moe_model, report = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=branch_states,
        lora_rank_r=8,
        lora_alpha=16,
        top_k=2,
        target_device=device,
        target_dtype=torch.float32,
    )

    assert report.num_experts == 8
    assert report.top_k == 2
    assert report.attention_layers_merged == 2 * 4  # 2 layers * 4 attn projections
    assert report.ffn_expert_modules_created == 2 * 8  # 2 layers * 8 FFN experts
    assert report.total_moe_parameters > report.active_parameters_per_token

    domain_batches = {
        dom: torch.randint(0, 512, (4, 16), dtype=torch.long) for dom in DEFAULT_EXPERT_DOMAINS
    }
    init_loss, final_loss, util = train_moe_router_gating(
        moe_model=moe_model,
        domain_token_batches=domain_batches,
        steps=16,
        lr=1e-2,
    )
    assert init_loss > 0.0
    assert final_loss > 0.0
    assert abs(sum(util.values()) - 1.0) < 1e-2
