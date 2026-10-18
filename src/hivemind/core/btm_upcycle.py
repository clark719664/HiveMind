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
    ) -> int:
        """Phase 3.5 (BTM MoE v2): Hot-plug a new domain LoRA expert (E -> E+1) without retraining existing experts."""
        scaling = float(lora_alpha) / float(lora_rank_r)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        cfg = self.backbone.config
        blocks = self.get_moe_blocks()

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

