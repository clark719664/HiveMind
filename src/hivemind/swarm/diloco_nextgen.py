"""Next-Generation Outer-Loop DiLoCo Optimizers & Aggregators (`SI-06`..`SI-09`).

Implements verified algorithmic improvements synthesized from 2024-2026 arXiv literature
and open-source reference repositories cloned in `sub_projects/hivemind/reference_repos/`:

1. `SI-06` — `CosineGatedAdamDecayOuterOptimizer` (`CGAD` & `PA-CGAD`, Shah & Sun,
   `arXiv:2605.09126`):
   Drop-in staleness- and per-fragment-age-aware outer optimizer that gates incoming
   pseudo-gradients by:
     gamma(tau_eff) = I[tau_eff <= tau_cut] * 0.5 * (1 + cos(pi * tau_eff / tau_cut))
     sigma(tau_eff) = gamma(tau_eff) * exp(-alpha * tau_eff)
   BEFORE Adam's first and second moment buffers (`m`, `v`), preventing stale rounding
   noise from inflating `sqrt(v)` and bounding the staleness bias by `1 / (e * alpha)`
   independently of `tau_max`.

2. `SI-07` — `ReputationWeightedIsoCTSMerger` (`IsoLoCo` / `ISO-C` / `ISO-CTS`, Horoi et al.,
   `arXiv:2607.03011`, `github.com/shoroi/isoloco`):
   Combines Reputation-Weighted TIES sign election (`SI-04`) with Isotropic Covariance
   (`ISO-C`) and Common + Task-Specific (`ISO-CTS`) subspace orthogonalization, replacing
   skewed singular value spectra with their isotropic mean so dominant domain updates do
   not drown out specialized experts as worker count `M` scales.

3. `SI-08` — `HeLoCoDirectionCorrector` & `AutoLoCoIntervalCorrector` (`HeLoCo`, Asif et al.,
   `arXiv:2606.00271`; `AutoLoCo`, He et al., `arXiv:2609.36662`):
   Projects out directionally conflicting components of stale non-IID pseudo-gradients
   relative to the outer momentum trajectory `m_t`, and rescales the outer learning rate
   and momentum when the inner step interval `H_t` changes dynamically (`SI-03`).

4. `SI-09` — `FoMoEPartialReplicaPlanner` & `ComputeVisibleSparseLoCoCompressor` (`FoMoE`,
   Sani et al., `arXiv:2606.19025`; `SparseLoCo`, Sarfi et al., `arXiv:2508.15706`;
   `PULSELoCo`, Miahi & Belilovsky, `arXiv:2602.03839`; `DeMo`, Peng et al.,
   `arXiv:2411.19870`):
   Models partial expert replication overlap `O_e in {1..M}` with Ghost-Expert Skip-Token
   routing and combines compute-visible BF16/FP16 rounding-threshold gating with
   chunked 2-bit statistical centroid quantization and deterministic index reduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class CGADStepTelemetry:
    """Per-fragment outer step telemetry for Cosine-Gated Adam-Decay (`SI-06`)."""

    fragment_name: str
    worker_delay_tau: float
    fragment_age_rounds: float
    effective_tau: float
    cosine_gate_gamma: float
    decay_gate_sigma: float
    dropped_by_cutoff: bool
    staleness_bias_bound: float
    step_linf_norm: float


class CosineGatedAdamDecayOuterOptimizer:
    """SI-06: Cosine-Gated Adam-Decay (`CGAD` / `PA-CGAD`, `arXiv:2605.09126`).

    Replaces delay-unaware outer Nesterov SGD with an age-aware outer Adam update that
    gates each incoming pseudo-gradient `g_f` by `sigma(tau_eff) = gamma(tau_eff) * exp(-alpha * tau_eff)`
    before updating `m_f` and `v_f`, where `tau_eff = max(worker_delay_tau, fragment_age_f)`.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        alpha: float = 0.2,
        tau_cut: float = 32.0,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if tau_cut <= 0.0:
            raise ValueError("tau_cut must be > 0")
        b1, b2 = float(betas[0]), float(betas[1])
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.tau_cut = float(tau_cut)
        self.beta1 = b1
        self.beta2 = b2
        self.eps = float(eps)
        self.step_count: Dict[str, int] = {}
        self.m_buffers: Dict[str, torch.Tensor] = {}
        self.v_buffers: Dict[str, torch.Tensor] = {}
        self.fragment_ages: Dict[str, int] = {}

    def compute_gate(self, effective_tau: float) -> Tuple[float, float]:
        """Return `(gamma(tau), sigma(tau))` for effective staleness `effective_tau >= 0`."""
        tau = max(0.0, float(effective_tau))
        if tau >= self.tau_cut:
            return 0.0, 0.0
        gamma = 0.5 * (1.0 + math.cos(math.pi * tau / self.tau_cut))
        sigma = gamma * math.exp(-self.alpha * tau)
        return float(gamma), float(sigma)

    def theoretical_staleness_bias_bound(self) -> float:
        """Return `max_{tau >= 0} tau * sigma(tau) <= 1 / (e * alpha)` (Theorem 1, `arXiv:2605.09126`)."""
        return 1.0 / (math.e * self.alpha)

    @torch.no_grad()
    def step_fragments(
        self,
        params: Dict[str, torch.Tensor],
        pseudo_gradients: Dict[str, torch.Tensor],
        worker_delays: Optional[Dict[str, float]] = None,
        synced_fragments: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[CGADStepTelemetry]]:
        """Apply one PA-CGAD outer step across synced parameter fragments."""
        active_set = set(synced_fragments) if synced_fragments is not None else set(pseudo_gradients.keys())
        updated_params: Dict[str, torch.Tensor] = {}
        telemetries: List[CGADStepTelemetry] = []

        for name, param in params.items():
            prev_age = self.fragment_ages.get(name, 0)
            if name not in active_set or name not in pseudo_gradients:
                self.fragment_ages[name] = prev_age + 1
                updated_params[name] = param.clone()
                continue

            g_f = pseudo_gradients[name].detach().to(torch.float32).cpu()
            p_fp32 = param.detach().to(torch.float32).cpu()
            tau_w = float((worker_delays or {}).get(name, 0.0))
            eff_tau = max(0.0, tau_w, float(prev_age))
            gamma, sigma = self.compute_gate(eff_tau)

            if sigma <= 0.0:
                # Hard cosine cutoff: do not touch m_f or v_f (prevents stale variance pollution)
                self.fragment_ages[name] = prev_age + 1
                updated_params[name] = param.clone()
                telemetries.append(
                    CGADStepTelemetry(
                        fragment_name=name,
                        worker_delay_tau=round(tau_w, 4),
                        fragment_age_rounds=float(prev_age),
                        effective_tau=round(eff_tau, 4),
                        cosine_gate_gamma=0.0,
                        decay_gate_sigma=0.0,
                        dropped_by_cutoff=True,
                        staleness_bias_bound=round(self.theoretical_staleness_bias_bound(), 6),
                        step_linf_norm=0.0,
                    )
                )
                continue

            g_gated = sigma * g_f
            if name not in self.m_buffers:
                self.m_buffers[name] = torch.zeros_like(g_gated)
                self.v_buffers[name] = torch.zeros_like(g_gated)
                self.step_count[name] = 0

            self.step_count[name] += 1
            t = self.step_count[name]
            m = self.m_buffers[name]
            v = self.v_buffers[name]

            m.mul_(self.beta1).add_(g_gated, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_gated, g_gated, value=1.0 - self.beta2)

            m_hat = m / (1.0 - (self.beta1 ** t))
            v_hat = v / (1.0 - (self.beta2 ** t))

            step_vec = self.lr * sigma * (m_hat / (torch.sqrt(v_hat) + self.eps))
            new_p = (p_fp32 - step_vec).to(device=param.device, dtype=param.dtype)
            updated_params[name] = new_p
            self.fragment_ages[name] = 0

            telemetries.append(
                CGADStepTelemetry(
                    fragment_name=name,
                    worker_delay_tau=round(tau_w, 4),
                    fragment_age_rounds=float(prev_age),
                    effective_tau=round(eff_tau, 4),
                    cosine_gate_gamma=round(gamma, 6),
                    decay_gate_sigma=round(sigma, 6),
                    dropped_by_cutoff=False,
                    staleness_bias_bound=round(self.theoretical_staleness_bias_bound(), 6),
                    step_linf_norm=round(float(step_vec.abs().max().item()), 8),
                )
            )

        return updated_params, telemetries


class ReputationWeightedIsoCTSMerger:
    """SI-07: Reputation-Weighted Isotropic Common + Task-Specific Merger (`IsoLoCo`, `arXiv:2607.03011`).

    Unifies HiveMind's `SI-04` reputation-weighted TIES sign election with `ISO-C` / `ISO-CTS`
    spectral isotropy so that multi-worker LoRA pseudo-gradients preserve both shared
    representation directions and orthogonal specialist directions without singular-value
    dominance collapse.
    """

    def __init__(
        self,
        common_space_fraction: float = 0.75,
        trim_density: float = 0.35,
        scaling_lambda: float = 1.10,
    ) -> None:
        if not (0.1 <= common_space_fraction <= 1.0):
            raise ValueError("common_space_fraction must be in [0.1, 1.0]")
        if not (0.0 < trim_density <= 1.0):
            raise ValueError("trim_density must be in (0, 1.0]")
        self.common_space_fraction = float(common_space_fraction)
        self.trim_density = float(trim_density)
        self.scaling_lambda = float(scaling_lambda)

    @staticmethod
    @torch.no_grad()
    def apply_isoc_2d(matrix: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Flatten the singular value spectrum of a 2D delta matrix to its isotropic mean."""
        if matrix.ndim != 2:
            return matrix.clone(), 1.0, 1.0
        mat_fp32 = matrix.detach().to(torch.float32).cpu()
        u, s, vh = torch.linalg.svd(mat_fp32, full_matrices=False)
        if s.numel() == 0 or float(s.max().item()) <= 1e-12:
            return matrix.clone(), 1.0, 1.0
        cond_before = float(s.max().item() / s.clamp_min(1e-8).min().item())
        s_mean = torch.ones_like(s) * s.mean()
        reconstructed = (u * s_mean.unsqueeze(0)) @ vh
        return reconstructed.to(device=matrix.device, dtype=matrix.dtype), cond_before, 1.0

    @torch.no_grad()
    def merge_worker_deltas(
        self,
        base_lora_state: Dict[str, torch.Tensor],
        worker_lora_states: Sequence[Dict[str, torch.Tensor]],
        worker_reputation_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Merge worker LoRA states using Reputation-Weighted TIES + ISO-CTS orthogonalization."""
        if not worker_lora_states:
            return {k: v.clone() for k, v in base_lora_state.items()}, {"num_workers": 0}

        num_workers = len(worker_lora_states)
        if worker_reputation_weights is None or len(worker_reputation_weights) != num_workers:
            weights = [1.0 for _ in range(num_workers)]
        else:
            weights = [max(1e-4, float(w)) for w in worker_reputation_weights]
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        w_tensor = w_tensor / w_tensor.sum().clamp_min(1e-8)

        merged_state: Dict[str, torch.Tensor] = {}
        cond_before_list: List[float] = []
        cond_after_list: List[float] = []

        for key, base_t in base_lora_state.items():
            base_fp32 = base_t.detach().to(torch.float32).cpu()
            trimmed_vectors: List[torch.Tensor] = []
            valid_w: List[float] = []

            for idx, w_state in enumerate(worker_lora_states):
                if key not in w_state:
                    continue
                tau = w_state[key].detach().to(torch.float32).cpu() - base_fp32
                flat = tau.reshape(-1)
                k_keep = max(1, int(round(flat.numel() * self.trim_density)))
                if k_keep < flat.numel():
                    thresh = torch.topk(flat.abs(), k=k_keep, largest=True).values.min()
                    tau = torch.where(tau.abs() >= thresh, tau, torch.zeros_like(tau))
                trimmed_vectors.append(tau)
                valid_w.append(float(w_tensor[idx].item()))

            if not trimmed_vectors:
                merged_state[key] = base_t.clone()
                continue

            stacked = torch.stack(trimmed_vectors, dim=0)
            w_view = torch.tensor(valid_w, dtype=torch.float32).view(
                -1, *([1] * (stacked.ndim - 1))
            )
            weighted_mass = torch.sum(stacked * w_view, dim=0)
            elected_sign = torch.sign(weighted_mass)
            agree_mask = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (stacked != 0)
            agree_weights = w_view * agree_mask.to(torch.float32)
            denom = torch.clamp(torch.sum(agree_weights, dim=0), min=1e-6)
            ties_delta = torch.sum(stacked * agree_weights, dim=0) / denom

            if ties_delta.ndim == 2 and min(ties_delta.shape) >= 2:
                # Extract common subspace + task-specific residual subspace (ISO-CTS)
                min_dim = min(ties_delta.shape)
                c_dim = max(1, int(round(min_dim * self.common_space_fraction)))
                u_c, s_c, vh_c = torch.linalg.svd(ties_delta, full_matrices=False)
                cond_before = float(s_c.max().item() / s_c.clamp_min(1e-8).min().item())
                cond_before_list.append(cond_before)

                u_common = u_c[:, :c_dim]
                rem_dim = min_dim - c_dim
                if rem_dim > 0 and len(trimmed_vectors) >= 1:
                    # Aggregate task-specific residuals orthogonal to u_common
                    residuals = [
                        (vec - u_common @ (u_common.t() @ vec)) * wt
                        for vec, wt in zip(trimmed_vectors, valid_w)
                    ]
                    res_sum = sum(residuals)
                    if float(res_sum.norm().item()) > 1e-8:
                        u_ts, _, vh_ts = torch.linalg.svd(res_sum, full_matrices=False)
                        u_comb = torch.cat([u_common, u_ts[:, :rem_dim]], dim=1)
                        vh_comb = torch.cat([vh_c[:c_dim, :], vh_ts[:rem_dim, :]], dim=0)
                        pu, _, quh = torch.linalg.svd(u_comb, full_matrices=False)
                        pv, _, qvh = torch.linalg.svd(vh_comb, full_matrices=False)
                        u_orth = pu @ quh
                        vh_orth = pv @ qvh
                        s_iso = torch.ones_like(s_c) * s_c.mean()
                        iso_delta = (u_orth * s_iso.unsqueeze(0)) @ vh_orth
                    else:
                        iso_delta, _, _ = self.apply_isoc_2d(ties_delta)
                else:
                    iso_delta, _, _ = self.apply_isoc_2d(ties_delta)
                cond_after_list.append(1.0)
                final_delta = iso_delta
            else:
                final_delta = ties_delta

            merged_state[key] = (base_fp32 + self.scaling_lambda * final_delta).to(
                device=base_t.device, dtype=base_t.dtype
            )

        return merged_state, {
            "strategy_id": "SI-07-REPUTATION-WEIGHTED-ISO-CTS",
            "num_workers": num_workers,
            "common_space_fraction": self.common_space_fraction,
            "mean_condition_number_before": round(
                sum(cond_before_list) / max(1, len(cond_before_list)), 4
            ),
            "mean_condition_number_after": round(
                sum(cond_after_list) / max(1, len(cond_after_list)), 4
            ),
        }


class HeLoCoDirectionCorrector:
    """SI-08A: Direction-Aware Staleness Correction (`HeLoCo`, Asif et al., `arXiv:2606.00271`).

    Uses the outer momentum trajectory `m_ref` as a directional reference and dampens or
    projects out negative-cosine (conflicting) components of stale non-IID pseudo-gradients.
    """

    def __init__(self, conflict_damping: float = 0.25) -> None:
        if not (0.0 <= conflict_damping <= 1.0):
            raise ValueError("conflict_damping must be in [0, 1]")
        self.conflict_damping = float(conflict_damping)

    @torch.no_grad()
    def correct_pseudo_gradient(
        self,
        pseudo_grad: torch.Tensor,
        outer_momentum_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Align a stale worker pseudo-gradient against `outer_momentum_ref`."""
        g = pseudo_grad.detach().to(torch.float32)
        m = outer_momentum_ref.detach().to(torch.float32)
        g_flat = g.reshape(-1)
        m_flat = m.reshape(-1)
        m_norm_sq = float(torch.dot(m_flat, m_flat).item())
        g_norm = float(g_flat.norm().item())
        if m_norm_sq <= 1e-12 or g_norm <= 1e-12:
            return pseudo_grad.clone(), {"cosine_before": 0.0, "cosine_after": 0.0, "conflict_detected": 0.0}

        dot_gm = float(torch.dot(g_flat, m_flat).item())
        cos_before = dot_gm / (g_norm * math.sqrt(m_norm_sq))
        if dot_gm >= 0.0:
            return pseudo_grad.clone(), {
                "cosine_before": round(cos_before, 6),
                "cosine_after": round(cos_before, 6),
                "conflict_detected": 0.0,
            }

        # Project out the conflicting component along m_ref and retain a damped fraction
        proj_coeff = dot_gm / m_norm_sq
        conflicting_component = proj_coeff * m
        corrected = g - (1.0 - self.conflict_damping) * conflicting_component
        c_flat = corrected.reshape(-1)
        cos_after = float(torch.dot(c_flat, m_flat).item()) / (
            max(1e-12, float(c_flat.norm().item())) * math.sqrt(m_norm_sq)
        )
        return corrected.to(device=pseudo_grad.device, dtype=pseudo_grad.dtype), {
            "cosine_before": round(cos_before, 6),
            "cosine_after": round(cos_after, 6),
            "conflict_detected": 1.0,
        }


class AutoLoCoIntervalCorrector:
    """SI-08B: Adaptive Inner-Interval Outer Optimizer Correction (`AutoLoCo`, `arXiv:2609.36662`).

    When `AdaptiveDiLoCoCadenceController` (`SI-03`) dynamically adapts the local inner step
    count `H_t` (or inner learning rate schedule `eta_inner`), corrects the outer learning
    rate `eta_outer` and outer momentum `mu_outer` by the accumulated inner learning rate
    ratio `S_t / S_ref`, eliminating scale mismatch across variable-length inner rounds.
    """

    def __init__(
        self,
        reference_inner_steps_h: int = 32,
        reference_inner_lr: float = 1e-3,
        base_outer_lr: float = 0.7,
        base_outer_momentum: float = 0.9,
    ) -> None:
        if reference_inner_steps_h < 1 or reference_inner_lr <= 0.0:
            raise ValueError("reference_inner_steps_h >= 1 and reference_inner_lr > 0 required")
        self.reference_accumulated_lr = float(reference_inner_steps_h) * float(reference_inner_lr)
        self.base_outer_lr = float(base_outer_lr)
        self.base_outer_momentum = float(base_outer_momentum)

    def correct_outer_hyperparams(
        self,
        actual_inner_steps_h: int,
        actual_inner_lr: float,
    ) -> Dict[str, float]:
        """Compute corrected `(outer_lr, outer_momentum)` for the realized inner interval."""
        if actual_inner_steps_h < 1 or actual_inner_lr <= 0.0:
            raise ValueError("actual_inner_steps_h >= 1 and actual_inner_lr > 0 required")
        accum_lr = float(actual_inner_steps_h) * float(actual_inner_lr)
        ratio = accum_lr / self.reference_accumulated_lr
        # Normalize outer step inversely with sqrt(ratio) and adapt momentum memory exp(-ratio)
        corrected_outer_lr = self.base_outer_lr / math.sqrt(max(0.25, min(4.0, ratio)))
        corrected_momentum = self.base_outer_momentum ** max(0.25, min(4.0, ratio))
        return {
            "accumulated_inner_lr": round(accum_lr, 6),
            "interval_scale_ratio": round(ratio, 6),
            "corrected_outer_lr": round(corrected_outer_lr, 6),
            "corrected_outer_momentum": round(corrected_momentum, 6),
        }


class FoMoEPartialReplicaPlanner:
    """SI-09A: Federated Partial Expert Replication & Ghost-Expert Skip-Token Planner (`FoMoE`, `arXiv:2606.19025`).

    Computes the communication ring factor `R(O_e) = 2 * (O_e - 1) / O_e`, steady-state
    expert memory fraction `O_e / M`, and Ghost-Expert Skip-Token compute scaling `sigma_st`
    for a 128-expert Sparse MoE distributed across `M` WAN workers.
    """

    def __init__(self, total_experts: int = 128, active_top_k: int = 2) -> None:
        if total_experts < 2 or active_top_k < 1:
            raise ValueError("total_experts >= 2 and active_top_k >= 1 required")
        self.total_experts = int(total_experts)
        self.active_top_k = int(active_top_k)

    def evaluate_overlap_configuration(
        self,
        num_workers_m: int,
        overlap_factor_oe: int,
        reshuffle_frequency_gamma: float = 0.0,
        dense_bytes: int = 32 * 1024 * 1024,
        single_expert_bytes: int = 2 * 1024 * 1024,
    ) -> Dict[str, Any]:
        """Return memory, communication, and Skip-Token FLOP metrics for `(M, O_e, gamma)`."""
        if num_workers_m < 1:
            raise ValueError("num_workers_m must be >= 1")
        if not (1 <= overlap_factor_oe <= num_workers_m):
            raise ValueError("overlap_factor_oe must be in [1, num_workers_m]")
        if not (0.0 <= reshuffle_frequency_gamma <= 1.0):
            raise ValueError("reshuffle_frequency_gamma must be in [0, 1]")

        m = float(num_workers_m)
        oe = float(overlap_factor_oe)
        local_experts = max(1, int(round((oe / m) * self.total_experts)))

        ring_factor_r = 0.0 if overlap_factor_oe == 1 else (2.0 * (oe - 1.0) / oe)
        migration_factor_p = float(reshuffle_frequency_gamma) * (1.0 - (oe / m))

        dense_comm_bytes = (2.0 * (m - 1.0) / max(1.0, m)) * float(dense_bytes)
        sparse_comm_bytes = (ring_factor_r + migration_factor_p) * float(
            local_experts * single_expert_bytes
        )
        full_replica_comm_bytes = (2.0 * (m - 1.0) / max(1.0, m)) * float(
            dense_bytes + self.total_experts * single_expert_bytes
        )

        comm_reduction_x = full_replica_comm_bytes / max(1.0, dense_comm_bytes + sparse_comm_bytes)
        skip_token_factor = local_experts / float(self.total_experts)

        return {
            "strategy_id": "SI-09A-FOMOE-PARTIAL-REPLICA-SKIP-TOKEN",
            "num_workers_m": num_workers_m,
            "overlap_factor_oe": overlap_factor_oe,
            "local_experts_per_worker": local_experts,
            "ghost_experts_per_worker": self.total_experts - local_experts,
            "ring_aggregation_factor_R": round(ring_factor_r, 6),
            "migration_broadcast_factor_P": round(migration_factor_p, 6),
            "expert_vram_fraction": round(oe / m, 6),
            "skip_token_active_fraction": round(skip_token_factor, 6),
            "per_site_comm_bytes": int(round(dense_comm_bytes + sparse_comm_bytes)),
            "full_replica_comm_bytes": int(round(full_replica_comm_bytes)),
            "wan_payload_reduction_x": round(comm_reduction_x, 4),
        }


class ComputeVisibleSparseLoCoCompressor:
    """SI-09B: Compute-Visible + 2-Bit Statistical Centroid Compressor (`PULSELoCo` + `SparseLoCo`).

    Combines:
    1. `PULSELoCo` (`arXiv:2602.03839`): Compute-visible gating that drops per-coordinate
       updates smaller than half the BF16/FP16 unit-in-last-place (`ulp / 2`) relative to
       the parameter magnitude.
    2. `SparseLoCo` (`arXiv:2508.15706`): Top-k sparsification (`1-3%` density) with 2-bit
       (`4`-bin) empirical centroid lookup-table quantization and deterministic index
       reconstruction (avoiding `DeMo`'s non-deterministic CUDA `scatter_reduce_`).
    """

    def __init__(
        self,
        topk_ratio: float = 0.03,
        n_bins: int = 4,
        range_in_sigmas: float = 4.0,
        bf16_ulp_gate: float = 2.0 ** -8,
    ) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be in (0, 1]")
        if n_bins < 2 or n_bins > 256:
            raise ValueError("n_bins must be in [2, 256]")
        self.topk_ratio = float(topk_ratio)
        self.n_bins = int(n_bins)
        self.range_in_sigmas = float(range_in_sigmas)
        self.bf16_ulp_gate = float(bf16_ulp_gate)

    @torch.no_grad()
    def compress_and_reconstruct(
        self,
        delta: torch.Tensor,
        reference_param: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress `delta` via PULSE compute-visible gate + 2-bit centroid Top-k and reconstruct."""
        d_fp32 = delta.detach().to(torch.float32).cpu()
        flat = d_fp32.reshape(-1)
        numel = flat.numel()

        if reference_param is not None:
            ref_abs = reference_param.detach().to(torch.float32).cpu().reshape(-1).abs().clamp_min(1e-4)
            visible_mask = flat.abs() >= (ref_abs * self.bf16_ulp_gate)
            gated_flat = torch.where(visible_mask, flat, torch.zeros_like(flat))
            invisible_fraction = 1.0 - float(visible_mask.to(torch.float32).mean().item())
        else:
            gated_flat = flat
            invisible_fraction = 0.0

        k = max(1, int(round(numel * self.topk_ratio)))
        # Deterministic sorted top-k index selection (Golden Invariant #2)
        _, idx = torch.topk(gated_flat.abs(), k=k, largest=True, sorted=True)
        selected = gated_flat[idx]

        shift = float(selected.mean().item())
        centered = selected - shift
        std = float(centered.norm().item() / math.sqrt(max(1, selected.numel() - 1)))
        scale = (self.range_in_sigmas * std / float(self.n_bins)) if std > 1e-12 else 1.0
        offset = self.n_bins // 2

        q_bins = ((centered / scale + offset).round().clamp(0, self.n_bins - 1)).to(torch.long)
        lookup = torch.zeros(self.n_bins, dtype=torch.float32)
        for b in range(self.n_bins):
            mask_b = q_bins == b
            if bool(mask_b.any().item()):
                lookup[b] = centered[mask_b].mean()

        dequant_selected = lookup[q_bins] + shift
        reconstructed = torch.zeros_like(flat)
        reconstructed.index_copy_(0, idx, dequant_selected)
        reconstructed = reconstructed.reshape(delta.shape).to(device=delta.device, dtype=delta.dtype)

        bits_per_val = math.ceil(math.log2(float(self.n_bins)))
        wire_bytes = int(math.ceil(k * (bits_per_val / 8.0)) + k * 2 + self.n_bins * 4 + 8)
        raw_bf16_bytes = numel * 2

        return reconstructed, {
            "strategy_id": "SI-09B-PULSE-SPARSELOCO-2BIT",
            "numel": numel,
            "k_transmitted": k,
            "quantization_bins": self.n_bins,
            "compute_invisible_fraction": round(invisible_fraction, 6),
            "raw_bf16_bytes": raw_bf16_bytes,
            "compressed_wire_bytes": wire_bytes,
            "wire_compression_ratio_x": round(raw_bf16_bytes / max(1, wire_bytes), 2),
        }
