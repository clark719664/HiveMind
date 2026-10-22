"""Phase B (BOINC Aggregator): Byzantine-Resilient Multi-Krum, Adaptive Norm Clipping & Nesterov Outer Loop.

Implements:
0. Phase S-3 Pre-Krum Finiteness & Free-Rider Zero-Gradient Sanitization (`FL-01`, `FL-02`).
1. Multi-Krum scoring over flattened pseudo-gradients Delta_k^(t) (sum of distances to
   K - f - 2 nearest neighbors).
2. Median pairwise cosine similarity 2-sigma outlier rejection.
3. Phase S-3 Adaptive Median-L2 & Neurotoxin Coordinate-L_infty Clipping + Coordinate-Wise
   Trimmed Mean + optional DP-DiLoCo Gaussian Privacy Sanitization (`FL-03`, `FL-04`, `FL-07`).
4. DiLoCo Nesterov Momentum outer optimization step (mu = 0.9, eta_outer = 0.7):
     Delta^(t) = TrimmedMean_{k in K_valid}( Clip(Delta_k^(t)) )
     v^(t)     = mu * v^(t-1) + Delta^(t)
     theta^(t) = theta^(t-1) - eta_outer * (mu * v^(t) + Delta^(t))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch

from hivemind.security.federated_learning_hardening import (
    DPGradientSanitizerConfig,
    FederatedDeltaAuditReport,
    FederatedLearningSecurityGuard,
)


@dataclass
class ByzantineFilterReport:
    """Detailed audit report of Multi-Krum + Cosine 2-sigma + Pre-Krum Byzantine filtering."""

    total_submissions: int
    max_byzantine_f: int
    accepted_workers: list[str]
    rejected_workers: list[str]
    krum_scores: dict[str, float]
    mean_cosine_similarities: dict[str, float]
    median_cosine: float
    cosine_threshold_lower: float
    quarantined_pre_krum: dict[str, str] = field(default_factory=dict)


@dataclass
class OuterStepResult:
    """Result of one DiLoCo outer aggregation step on the BOINC server."""

    branch_name: str
    outer_step: int
    theta_updated: dict[str, torch.Tensor] = field(repr=False)
    velocity_updated: dict[str, torch.Tensor] = field(repr=False)
    aggregated_delta_norm: float
    filter_report: ByzantineFilterReport
    federated_security_report: FederatedDeltaAuditReport | None = None
    outer_optimizer_mode: str = "nesterov"
    mean_cosine_gate: float | None = None
    extra_gpu_vram_bytes: int = 0


def flatten_delta_dict(delta_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of LoRA tensors in deterministic key order into a 1D FP32 vector."""
    parts = [
        delta_dict[k].detach().to(torch.float32).cpu().reshape(-1)
        for k in sorted(delta_dict.keys())
    ]
    return torch.cat(parts, dim=0)


def filter_byzantine_deltas_multikrum(
    worker_deltas: dict[str, dict[str, torch.Tensor]],
    max_byzantine_f: int = 1,
    cosine_sigma: float = 2.0,
) -> ByzantineFilterReport:
    """Filter worker pseudo-gradients using Pre-Krum Sanitization, Multi-Krum, and a 2-sigma pairwise cosine filter."""
    all_worker_ids = list(worker_deltas.keys())
    total_K = len(all_worker_ids)
    if total_K == 0:
        raise ValueError("Cannot run Multi-Krum on empty worker_deltas.")

    fl_guard = FederatedLearningSecurityGuard()
    clean_deltas, quarantined = fl_guard.sanitize_worker_deltas_pre_krum(worker_deltas)
    worker_ids = list(clean_deltas.keys())
    K = len(worker_ids)

    if K == 0:
        raise ValueError(
            f"All {total_K} worker deltas were quarantined during pre-Krum security sanitization: {quarantined}"
        )

    if K <= 2:
        return ByzantineFilterReport(
            total_submissions=total_K,
            max_byzantine_f=0,
            accepted_workers=worker_ids,
            rejected_workers=list(quarantined.keys()),
            krum_scores={wid: 0.0 for wid in worker_ids},
            mean_cosine_similarities={wid: 1.0 for wid in worker_ids},
            median_cosine=1.0,
            cosine_threshold_lower=0.0,
            quarantined_pre_krum=quarantined,
        )

    # Clamp f so K - f - 2 >= 1
    f = min(max_byzantine_f, max(0, K - 3))
    num_neighbors = max(1, K - f - 2)

    # Accumulate exact [K, K] inner-product Gram matrix parameter-by-parameter (<1 MB peak RAM)
    gram = torch.zeros((K, K), dtype=torch.float64)
    param_keys = sorted(clean_deltas[worker_ids[0]].keys())
    for k in param_keys:
        p_mat = torch.stack(
            [
                clean_deltas[wid][k].detach().to(dtype=torch.float32, device="cpu").reshape(-1)
                for wid in worker_ids
            ],
            dim=0,
        )
        gram += torch.mm(p_mat, p_mat.t()).to(torch.float64)
        del p_mat

    diag = torch.diag(gram).clamp_min(0.0)
    # Pairwise squared Euclidean distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 <u, v>
    dist_matrix = (diag.unsqueeze(1) + diag.unsqueeze(0) - 2.0 * gram).clamp_min(0.0).to(torch.float32)

    krum_scores: dict[str, float] = {}
    for i, wid in enumerate(worker_ids):
        # Exclude self (distance 0 at index i)
        row = torch.cat([dist_matrix[i, :i], dist_matrix[i, i + 1 :]])
        nearest_vals, _ = torch.topk(row, k=min(num_neighbors, row.numel()), largest=False)
        krum_scores[wid] = round(float(nearest_vals.sum().item()), 6)

    # Pairwise Cosine Similarity: [K, K]
    norms = torch.sqrt(diag).clamp_min(1e-12)
    cos_matrix = (gram / (norms.unsqueeze(1) * norms.unsqueeze(0))).clamp(-1.0, 1.0).to(torch.float32)

    mean_cos_per_worker: dict[str, float] = {}
    off_diag_cos_values: list[float] = []
    for i, wid in enumerate(worker_ids):
        others = torch.cat([cos_matrix[i, :i], cos_matrix[i, i + 1 :]])
        # Median cosine similarity to peers is robust against a single huge outlier
        peer_median_cos = float(others.median().item())
        mean_cos_per_worker[wid] = round(peer_median_cos, 6)
        off_diag_cos_values.extend(others.tolist())

    off_diag_tensor = torch.tensor(off_diag_cos_values, dtype=torch.float32)
    median_cos = float(off_diag_tensor.median().item())
    std_cos = float(off_diag_tensor.std(unbiased=False).item())
    # Reject anything falling outside cosine_sigma below the median (or negative alignment)
    cos_lower_bound = max(-0.05, median_cos - cosine_sigma * max(std_cos, 0.05))

    # Multi-Krum selects the top (K - f) workers with smallest Krum distance score
    sorted_by_krum = sorted(worker_ids, key=lambda w: krum_scores[w])
    krum_survivors = set(sorted_by_krum[: max(1, K - f)])

    accepted: list[str] = []
    rejected: list[str] = list(quarantined.keys())
    for wid in worker_ids:
        if wid in krum_survivors and mean_cos_per_worker[wid] >= cos_lower_bound:
            accepted.append(wid)
        else:
            rejected.append(wid)

    # Ensure at least the single best Multi-Krum worker survives
    if not accepted:
        best_worker = sorted_by_krum[0]
        accepted = [best_worker]
        rejected = list(quarantined.keys()) + [w for w in worker_ids if w != best_worker]

    return ByzantineFilterReport(
        total_submissions=total_K,
        max_byzantine_f=f,
        accepted_workers=accepted,
        rejected_workers=rejected,
        krum_scores=krum_scores,
        mean_cosine_similarities=mean_cos_per_worker,
        median_cosine=round(median_cos, 6),
        cosine_threshold_lower=round(cos_lower_bound, 6),
        quarantined_pre_krum=quarantined,
    )


class DiLoCoOuterOptimizer:
    """Hardened CPU outer aggregation with Nesterov or branch-isolated CGAD state.

    CGAD consumes fresh, synchronous aggregates, so its cosine gate is 1.0.
    ``velocity_updated`` contains Nesterov velocity or CGAD's first moment.
    """

    def __init__(
        self,
        outer_lr_eta: float = 0.7,
        outer_momentum_mu: float = 0.9,
        max_byzantine_f: int = 1,
        cosine_sigma: float = 2.0,
        dp_config: DPGradientSanitizerConfig | None = None,
        outer_optimizer_mode: str = "nesterov",
        use_cpu_flat_bucket: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.01,
    ) -> None:
        if outer_optimizer_mode not in ("nesterov", "cosine_gated_adam_decay"):
            raise ValueError("outer_optimizer_mode must be nesterov or cosine_gated_adam_decay")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if (outer_optimizer_mode != "nesterov" or use_cpu_flat_bucket) and (
            not math.isfinite(outer_lr_eta) or outer_lr_eta <= 0.0
        ):
            raise ValueError("outer_lr_eta must be finite and positive")
        self.outer_lr_eta = outer_lr_eta
        self.outer_momentum_mu = outer_momentum_mu
        self.max_byzantine_f = max_byzantine_f
        self.cosine_sigma = cosine_sigma
        self.dp_config = dp_config
        self.fl_guard = FederatedLearningSecurityGuard()
        # Velocity buffers per expert branch: branch_name -> {param_name: tensor}
        self.velocities: dict[str, dict[str, torch.Tensor]] = {}
        self.outer_optimizer_mode = outer_optimizer_mode
        self.use_cpu_flat_bucket = use_cpu_flat_bucket
        self.beta1, self.beta2 = beta1, beta2
        self.weight_decay = weight_decay
        self.cgad_optimizers = {}
        self.cpu_flat_buckets = {}

    def step(
        self,
        branch_name: str,
        outer_step: int,
        theta_prev: dict[str, torch.Tensor],
        worker_deltas: dict[str, dict[str, torch.Tensor]],
    ) -> OuterStepResult:
        """Filter and clip worker deltas before applying the selected CPU outer update."""
        filter_report = filter_byzantine_deltas_multikrum(
            worker_deltas=worker_deltas,
            max_byzantine_f=self.max_byzantine_f,
            cosine_sigma=self.cosine_sigma,
        )
        valid_ids = filter_report.accepted_workers
        surviving_deltas = {wid: worker_deltas[wid] for wid in valid_ids}

        aggregated_deltas, fl_report = self.fl_guard.clip_and_aggregate_surviving_deltas(
            surviving_deltas=surviving_deltas,
            quarantined_pre_filter=filter_report.quarantined_pre_krum,
            total_submitted_count=filter_report.total_submissions,
            dp_config=self.dp_config,
        )

        mean_cosine_gate = None
        if self.outer_optimizer_mode != "nesterov" or self.use_cpu_flat_bucket:
            if set(aggregated_deltas) != set(theta_prev) or any(
                aggregated_deltas[k].shape != param.shape for k, param in theta_prev.items()
            ):
                raise ValueError("aggregated deltas must match all branch parameter keys and shapes")
            if branch_name not in self.velocities:
                # Preserve the branch-reset convention used by existing coordinator clients.
                self.cgad_optimizers.pop(branch_name, None)
                self.cpu_flat_buckets.pop(branch_name, None)
            # Convert directly to CPU FP32 without a temporary FP32 allocation on the GPU.
            params_cpu = {
                k: v.detach().to(device="cpu", dtype=torch.float32)
                for k, v in theta_prev.items()
            }
            if self.use_cpu_flat_bucket:
                from hivemind.swarm.diloco_nextgen import CPUOffloadedFlatBucketOuterState

                bucket = self.cpu_flat_buckets.get(branch_name)
                if bucket is None:
                    bucket = CPUOffloadedFlatBucketOuterState(params_cpu)
                    self.cpu_flat_buckets[branch_name] = bucket
                else:
                    if set(params_cpu) != set(bucket.keys) or any(
                        params_cpu[k].shape != bucket.shapes[k] for k in bucket.keys
                    ):
                        raise ValueError("branch parameters must match the existing flat bucket")
                    bucket.apply_outer_update_and_sync_inner(params_cpu)

            if self.outer_optimizer_mode == "cosine_gated_adam_decay":
                from hivemind.swarm.diloco_nextgen import CosineGatedAdamDecayOuterOptimizer

                cgad = self.cgad_optimizers.get(branch_name)
                if cgad is None:
                    cgad = CosineGatedAdamDecayOuterOptimizer(
                        lr=self.outer_lr_eta, betas=(self.beta1, self.beta2),
                        weight_decay=self.weight_decay,
                    )
                    self.cgad_optimizers[branch_name] = cgad
                theta_updated, telemetry = cgad.step_fragments(params_cpu, aggregated_deltas)
                vel_branch = {k: v.clone() for k, v in cgad.m_buffers.items()}
                mean_cosine_gate = sum(t.cosine_gate_gamma for t in telemetry) / len(telemetry)
                if self.use_cpu_flat_bucket:
                    bucket.apply_outer_update_and_sync_inner(theta_updated)
            else:
                theta_updated, vel_branch = bucket.outer_nesterov_step(
                    aggregated_deltas, lr=self.outer_lr_eta, momentum=self.outer_momentum_mu,
                )
            self.velocities[branch_name] = vel_branch
            return OuterStepResult(
                branch_name=branch_name, outer_step=outer_step,
                theta_updated=theta_updated, velocity_updated=vel_branch,
                aggregated_delta_norm=round(sum(
                    float((delta**2).sum().item()) for delta in aggregated_deltas.values()
                ) ** 0.5, 6),
                filter_report=filter_report, federated_security_report=fl_report,
                outer_optimizer_mode=self.outer_optimizer_mode,
                mean_cosine_gate=mean_cosine_gate,
            )

        if branch_name not in self.velocities:
            self.velocities[branch_name] = {
                k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                for k, v in theta_prev.items()
            }

        vel_branch = self.velocities[branch_name]
        theta_updated: dict[str, torch.Tensor] = {}
        sq_norm_sum = 0.0

        for name, param_prev in theta_prev.items():
            avg_delta = aggregated_deltas[name]
            sq_norm_sum += float((avg_delta**2).sum().item())

            # v^(t) = mu * v^(t-1) + Delta^(t)
            v_t = self.outer_momentum_mu * vel_branch[name] + avg_delta
            vel_branch[name] = v_t

            # theta^(t) = theta^(t-1) - eta_outer * (mu * v^(t) + Delta^(t))
            nesterov_step = self.outer_momentum_mu * v_t + avg_delta
            theta_updated[name] = (
                param_prev.detach().to(torch.float32).cpu() - self.outer_lr_eta * nesterov_step
            )

        return OuterStepResult(
            branch_name=branch_name,
            outer_step=outer_step,
            theta_updated=theta_updated,
            velocity_updated=vel_branch,
            aggregated_delta_norm=round(sq_norm_sum**0.5, 6),
            filter_report=filter_report,
            federated_security_report=fl_report,
        )
