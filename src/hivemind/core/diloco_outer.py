"""Phase B (BOINC Aggregator): Byzantine-Resilient Multi-Krum & Nesterov Outer Loop.

Implements:
1. Multi-Krum scoring over flattened pseudo-gradients Delta_k^(t) (sum of distances to
   K - f - 2 nearest neighbors).
2. Median pairwise cosine similarity 2-sigma outlier rejection.
3. DiLoCo Nesterov Momentum outer optimization step (mu = 0.9, eta_outer = 0.7):
     Delta^(t) = (1 / |K_valid|) * sum_{k in K_valid} Delta_k^(t)
     v^(t)     = mu * v^(t-1) + Delta^(t)
     theta^(t) = theta^(t-1) - eta_outer * (mu * v^(t) + Delta^(t))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class ByzantineFilterReport:
    """Detailed audit report of Multi-Krum + Cosine 2-sigma Byzantine filtering."""

    total_submissions: int
    max_byzantine_f: int
    accepted_workers: list[str]
    rejected_workers: list[str]
    krum_scores: dict[str, float]
    mean_cosine_similarities: dict[str, float]
    median_cosine: float
    cosine_threshold_lower: float


@dataclass
class OuterStepResult:
    """Result of one DiLoCo outer aggregation step on the BOINC server."""

    branch_name: str
    outer_step: int
    theta_updated: dict[str, torch.Tensor] = field(repr=False)
    velocity_updated: dict[str, torch.Tensor] = field(repr=False)
    aggregated_delta_norm: float
    filter_report: ByzantineFilterReport


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
    """Filter worker pseudo-gradients using Multi-Krum and a 2-sigma pairwise cosine filter."""
    worker_ids = list(worker_deltas.keys())
    K = len(worker_ids)
    if K == 0:
        raise ValueError("Cannot run Multi-Krum on empty worker_deltas.")

    if K <= 2:
        return ByzantineFilterReport(
            total_submissions=K,
            max_byzantine_f=0,
            accepted_workers=worker_ids,
            rejected_workers=[],
            krum_scores={wid: 0.0 for wid in worker_ids},
            mean_cosine_similarities={wid: 1.0 for wid in worker_ids},
            median_cosine=1.0,
            cosine_threshold_lower=0.0,
        )

    # Clamp f so K - f - 2 >= 1
    f = min(max_byzantine_f, max(0, K - 3))
    num_neighbors = max(1, K - f - 2)

    # Accumulate exact [K, K] inner-product Gram matrix parameter-by-parameter (<1 MB peak RAM)
    gram = torch.zeros((K, K), dtype=torch.float64)
    param_keys = sorted(worker_deltas[worker_ids[0]].keys())
    for k in param_keys:
        p_mat = torch.stack(
            [
                worker_deltas[wid][k].detach().to(dtype=torch.float32, device="cpu").reshape(-1)
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
    rejected: list[str] = []
    for wid in worker_ids:
        if wid in krum_survivors and mean_cos_per_worker[wid] >= cos_lower_bound:
            accepted.append(wid)
        else:
            rejected.append(wid)

    # Ensure at least the single best Multi-Krum worker survives
    if not accepted:
        best_worker = sorted_by_krum[0]
        accepted = [best_worker]
        rejected = [w for w in worker_ids if w != best_worker]

    return ByzantineFilterReport(
        total_submissions=K,
        max_byzantine_f=f,
        accepted_workers=accepted,
        rejected_workers=rejected,
        krum_scores=krum_scores,
        mean_cosine_similarities=mean_cos_per_worker,
        median_cosine=round(median_cos, 6),
        cosine_threshold_lower=round(cos_lower_bound, 6),
    )


class DiLoCoOuterOptimizer:
    """Server-side DiLoCo Nesterov Momentum Outer Optimizer with Multi-Krum Defense."""

    def __init__(
        self,
        outer_lr_eta: float = 0.7,
        outer_momentum_mu: float = 0.9,
        max_byzantine_f: int = 1,
        cosine_sigma: float = 2.0,
    ) -> None:
        self.outer_lr_eta = outer_lr_eta
        self.outer_momentum_mu = outer_momentum_mu
        self.max_byzantine_f = max_byzantine_f
        self.cosine_sigma = cosine_sigma
        # Velocity buffers per expert branch: branch_name -> {param_name: tensor}
        self.velocities: dict[str, dict[str, torch.Tensor]] = {}

    def step(
        self,
        branch_name: str,
        outer_step: int,
        theta_prev: dict[str, torch.Tensor],
        worker_deltas: dict[str, dict[str, torch.Tensor]],
    ) -> OuterStepResult:
        """Aggregate valid worker deltas via Multi-Krum and apply Nesterov outer update."""
        filter_report = filter_byzantine_deltas_multikrum(
            worker_deltas=worker_deltas,
            max_byzantine_f=self.max_byzantine_f,
            cosine_sigma=self.cosine_sigma,
        )
        valid_ids = filter_report.accepted_workers
        num_valid = float(len(valid_ids))

        if branch_name not in self.velocities:
            self.velocities[branch_name] = {
                k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                for k, v in theta_prev.items()
            }

        vel_branch = self.velocities[branch_name]
        theta_updated: dict[str, torch.Tensor] = {}
        sq_norm_sum = 0.0

        for name, param_prev in theta_prev.items():
            # Delta^(t) = (1 / |K_valid|) * sum_{k in K_valid} Delta_k^(t)
            avg_delta = torch.zeros_like(param_prev, dtype=torch.float32, device="cpu")
            for wid in valid_ids:
                avg_delta.add_(worker_deltas[wid][name].detach().to(torch.float32).cpu())
            avg_delta.div_(num_valid)

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
        )
