"""SI-12: Decoupled DiLoCo syncer primitives (Google DeepMind, `arXiv:2604.21428`).

Decoupled DiLoCo (Douillard, Rush, Donchev, Charles et al., Google DeepMind / Google
Research, April 2026) replaces the lock-step outer all-reduce of DiLoCo by a central,
CPU-resident *syncer* that aggregates learner fragments as soon as a **minimum quorum**
of `K <= M` learners has reported, optionally waits an **adaptive grace window** bounded
by a fraction `gamma < 1` of the measured network slack, weights each learner by a
**token-weighted quality weight**, and merges per-learner outer gradients with
**Radial-Directional Averaging (RDA)** so that the merged norm no longer shrinks like
`R / sqrt(M)` when per-learner outer gradients are nearly orthogonal.

This module implements those four mechanisms as small, individually testable pieces:

* :func:`token_weighted_quality_weight` — `Weight(c_tokens, c_steps) = c_tokens * (c_tokens / c_steps)`
  (Algorithm 2, line 9, "quantity x quality").
* :func:`select_quorum` — first-`K`-by-arrival quorum plus every learner that arrives
  inside the grace window measured from the moment the quorum was reached.
* :class:`AdaptiveGraceWindowController` — `xi_slack = tau * xi_step - (xi_quorum + xi_sync)`
  and `xi_grace = gamma * max(0, xi_slack)` with exponential-moving-average tracking of
  `xi_step` and `xi_quorum` (Eq. 3 of the paper).
* :func:`radial_directional_average` — weighted RDA (Appendix D.2):
  `RDA({(v_i, w_i)}) = (sum_i w_i ||v_i|| / sum_i w_i) * phi(sum_i w_i phi(v_i) / sum_i w_i)`
  with `phi(x) = x / ||x||`.
* :class:`DecoupledDiLoCoQuorumSyncer` — one syncer round for one parameter fragment,
  producing the outer gradient `Delta_p` that the existing outer optimizers
  (`DiLoCoOuterOptimizer`, `CPUOffloadedFlatBucketOuterState`, `CosineGatedAdamDecayOuterOptimizer`)
  consume unchanged.

What this module does **not** do: it does not implement the asynchronous RPC transport,
the Chandy-Lamport checkpointing or the learner recovery protocol of the paper, and it has
not been exercised across real machines. Every number produced here is a controlled
single-process check; see ``scripts/verify_decoupled_diloco_quorum.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


DECOUPLED_DILOCO_REFERENCE: Dict[str, str] = {
    "name": "Decoupled DiLoCo for Resilient Distributed Pre-training",
    "authors": "Decoupled DiLoCo Team (Google DeepMind & Google Research), incl. A. Douillard, K. Rush, Y. Donchev, Z. Charles",
    "arxiv_url": "https://arxiv.org/abs/2604.21428",
    "integrated_components": (
        "Minimum-quorum aggregation, adaptive grace window bounded by gamma * slack, "
        "token-weighted learner weights, and Radial-Directional Averaging (RDA)"
    ),
}


@dataclass(frozen=True)
class LearnerArrival:
    """Metadata a learner sends to the syncer for one fragment (Algorithm 1, line 13)."""

    learner_id: str
    arrival_time_s: float
    steps_since_sync: int
    tokens_since_sync: int

    def __post_init__(self) -> None:
        if not self.learner_id:
            raise ValueError("learner_id must be non-empty")
        if not math.isfinite(self.arrival_time_s):
            raise ValueError("arrival_time_s must be finite")
        if self.steps_since_sync < 0 or self.tokens_since_sync < 0:
            raise ValueError("steps_since_sync and tokens_since_sync must be >= 0")


@dataclass(frozen=True)
class QuorumSelection:
    """Outcome of one quorum-gathering attempt for a fragment."""

    reached: bool
    min_quorum_k: int
    grace_window_s: float
    admitted_learner_ids: Tuple[str, ...]
    deferred_learner_ids: Tuple[str, ...]
    quorum_reached_at_s: Optional[float]
    window_close_s: Optional[float]


@dataclass(frozen=True)
class SyncerRoundTelemetry:
    """Telemetry for one syncer merge of one fragment."""

    fragment_name: str
    merge_mode: str
    num_arrivals: int
    num_admitted: int
    num_deferred: int
    min_quorum_k: int
    grace_window_s: float
    admitted_learner_ids: Tuple[str, ...]
    learner_weights: Dict[str, float]
    admitted_tokens: int
    deferred_tokens: int
    merged_delta_norm: float
    direct_average_norm: float
    mean_learner_delta_norm: float


def token_weighted_quality_weight(tokens_since_sync: int, steps_since_sync: int) -> float:
    """Decoupled DiLoCo learner weight `Weight(c_tokens, c_steps) = c_tokens * (c_tokens / c_steps)`.

    The paper reads this as `(quantity) x (quality)`: more tokens are worth more, and the
    same tokens are worth more when amortised over fewer inner steps. A learner that took
    no inner step since the last sync has no new information and receives weight `0`.
    """
    if tokens_since_sync < 0 or steps_since_sync < 0:
        raise ValueError("tokens_since_sync and steps_since_sync must be >= 0")
    if steps_since_sync == 0:
        return 0.0
    tokens = float(tokens_since_sync)
    return tokens * (tokens / float(steps_since_sync))


def select_quorum(
    arrivals: Sequence[LearnerArrival],
    min_quorum_k: int,
    grace_window_s: float,
) -> QuorumSelection:
    """Admit the first `K` learners by arrival time plus every learner arriving within the grace window.

    Ties on arrival time are broken by `learner_id` so the selection is deterministic.
    If fewer than `K` learners have arrived the quorum is not reached and nothing is admitted.
    """
    if min_quorum_k < 1:
        raise ValueError("min_quorum_k must be >= 1")
    if grace_window_s < 0.0 or not math.isfinite(grace_window_s):
        raise ValueError("grace_window_s must be finite and >= 0")
    ids = [a.learner_id for a in arrivals]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate learner_id in arrivals")

    ordered = sorted(arrivals, key=lambda a: (a.arrival_time_s, a.learner_id))
    if len(ordered) < min_quorum_k:
        return QuorumSelection(
            reached=False,
            min_quorum_k=min_quorum_k,
            grace_window_s=float(grace_window_s),
            admitted_learner_ids=(),
            deferred_learner_ids=tuple(a.learner_id for a in ordered),
            quorum_reached_at_s=None,
            window_close_s=None,
        )

    quorum_reached_at = ordered[min_quorum_k - 1].arrival_time_s
    window_close = quorum_reached_at + float(grace_window_s)
    admitted: List[str] = []
    deferred: List[str] = []
    for idx, a in enumerate(ordered):
        if idx < min_quorum_k or a.arrival_time_s <= window_close:
            admitted.append(a.learner_id)
        else:
            deferred.append(a.learner_id)
    return QuorumSelection(
        reached=True,
        min_quorum_k=min_quorum_k,
        grace_window_s=float(grace_window_s),
        admitted_learner_ids=tuple(admitted),
        deferred_learner_ids=tuple(deferred),
        quorum_reached_at_s=float(quorum_reached_at),
        window_close_s=float(window_close),
    )


class AdaptiveGraceWindowController:
    """Adaptive grace window `xi_grace = gamma * max(0, xi_slack)` (Decoupled DiLoCo, Eq. 3).

    `xi_slack = tau * xi_step - (xi_quorum + xi_sync)` where `tau` is the number of compute
    steps the syncer overlaps with (the paper double-buffers with `tau = 2`), `xi_step` and
    `xi_quorum` are tracked as exponential moving averages, and `xi_sync` is the measured
    fragment synchronisation time. The safety margin `gamma < 1` guarantees that the grace
    window never consumes the whole slack, so waiting for extra learners cannot stall the
    learners' compute.
    """

    def __init__(
        self,
        overlap_steps_tau: int = 2,
        safety_margin_gamma: float = 0.5,
        ema_beta: float = 0.9,
        sync_duration_s: float = 0.0,
    ) -> None:
        if overlap_steps_tau < 1:
            raise ValueError("overlap_steps_tau must be >= 1")
        if not (0.0 <= safety_margin_gamma < 1.0):
            raise ValueError("safety_margin_gamma must be in [0, 1)")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError("ema_beta must be in [0, 1)")
        if sync_duration_s < 0.0 or not math.isfinite(sync_duration_s):
            raise ValueError("sync_duration_s must be finite and >= 0")
        self.overlap_steps_tau = int(overlap_steps_tau)
        self.safety_margin_gamma = float(safety_margin_gamma)
        self.ema_beta = float(ema_beta)
        self.sync_duration_s = float(sync_duration_s)
        self.step_duration_ema_s: Optional[float] = None
        self.quorum_duration_ema_s: Optional[float] = None

    @staticmethod
    def _ema(previous: Optional[float], observation: float, beta: float) -> float:
        if observation < 0.0 or not math.isfinite(observation):
            raise ValueError("durations must be finite and >= 0")
        if previous is None:
            return float(observation)
        return beta * previous + (1.0 - beta) * float(observation)

    def observe_step_duration(self, seconds: float) -> float:
        """Record one learner compute-step duration; returns the updated EMA."""
        self.step_duration_ema_s = self._ema(self.step_duration_ema_s, seconds, self.ema_beta)
        return self.step_duration_ema_s

    def observe_quorum_duration(self, seconds: float) -> float:
        """Record how long the syncer needed to reach the minimum quorum; returns the updated EMA."""
        self.quorum_duration_ema_s = self._ema(self.quorum_duration_ema_s, seconds, self.ema_beta)
        return self.quorum_duration_ema_s

    def slack_seconds(self) -> float:
        """`xi_slack = tau * xi_step - (xi_quorum + xi_sync)`; `0` until a step duration is observed."""
        if self.step_duration_ema_s is None:
            return 0.0
        quorum = self.quorum_duration_ema_s if self.quorum_duration_ema_s is not None else 0.0
        return self.overlap_steps_tau * self.step_duration_ema_s - (quorum + self.sync_duration_s)

    def grace_window_seconds(self) -> float:
        """`xi_grace = gamma * max(0, xi_slack)` — strictly below the slack whenever slack is positive."""
        return self.safety_margin_gamma * max(0.0, self.slack_seconds())

    def telemetry(self) -> Dict[str, float]:
        return {
            "overlap_steps_tau": float(self.overlap_steps_tau),
            "safety_margin_gamma": self.safety_margin_gamma,
            "step_duration_ema_s": float(self.step_duration_ema_s or 0.0),
            "quorum_duration_ema_s": float(self.quorum_duration_ema_s or 0.0),
            "sync_duration_s": self.sync_duration_s,
            "slack_s": float(self.slack_seconds()),
            "grace_window_s": float(self.grace_window_seconds()),
        }


def _validate_vectors_and_weights(
    vectors: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(vectors) == 0:
        raise ValueError("at least one vector is required")
    shape = vectors[0].shape
    for v in vectors:
        if v.shape != shape:
            raise ValueError("all vectors must share the same shape")
    stacked = torch.stack([v.detach().to(torch.float32).reshape(-1) for v in vectors], dim=0)
    if weights is None:
        w = torch.ones(stacked.shape[0], dtype=torch.float32)
    else:
        if len(weights) != stacked.shape[0]:
            raise ValueError("weights must have one entry per vector")
        w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
        if bool((w < 0).any()) or not bool(torch.isfinite(w).all()):
            raise ValueError("weights must be finite and >= 0")
    if float(w.sum()) <= 0.0:
        raise ValueError("weights must not all be zero")
    return stacked, w / w.sum()


def direct_weighted_average(
    vectors: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Plain (weighted) mean of outer gradients, the DiLoCo baseline merge (Appendix D.2)."""
    stacked, w = _validate_vectors_and_weights(vectors, weights)
    return (w.unsqueeze(1) * stacked).sum(dim=0).reshape(vectors[0].shape)


def radial_directional_average(
    vectors: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Weighted Radial-Directional Averaging (RDA, Decoupled DiLoCo Appendix D.2).

    Averages norms and unit directions separately and multiplies them back, so that if every
    `||v_i|| = R` the result has norm exactly `R` regardless of how many nearly-orthogonal
    learners contributed. If the weighted mean direction vanishes (perfectly cancelling
    directions) the zero vector is returned; the all-zero input also yields zero.
    """
    stacked, w = _validate_vectors_and_weights(vectors, weights)
    norms = stacked.norm(dim=1)
    units = stacked / norms.clamp_min(eps).unsqueeze(1)
    units = torch.where(norms.unsqueeze(1) > eps, units, torch.zeros_like(units))
    radius = float((w * norms).sum())
    mean_dir = (w.unsqueeze(1) * units).sum(dim=0)
    dir_norm = float(mean_dir.norm())
    if dir_norm <= eps or radius <= eps:
        return torch.zeros_like(mean_dir).reshape(vectors[0].shape)
    return (radius * mean_dir / dir_norm).reshape(vectors[0].shape)


class DecoupledDiLoCoQuorumSyncer:
    """One Decoupled DiLoCo syncer shard: quorum, grace window, token weights and RDA merge."""

    MERGE_MODES = ("rda", "direct")

    def __init__(
        self,
        num_learners_m: int,
        min_quorum_k: int,
        merge_mode: str = "rda",
        grace_controller: Optional[AdaptiveGraceWindowController] = None,
    ) -> None:
        if num_learners_m < 1:
            raise ValueError("num_learners_m must be >= 1")
        if not (1 <= min_quorum_k <= num_learners_m):
            raise ValueError("min_quorum_k must satisfy 1 <= K <= M")
        if merge_mode not in self.MERGE_MODES:
            raise ValueError(f"merge_mode must be one of {self.MERGE_MODES}")
        self.num_learners_m = int(num_learners_m)
        self.min_quorum_k = int(min_quorum_k)
        self.merge_mode = merge_mode
        self.grace_controller = grace_controller

    @torch.no_grad()
    def merge_fragment(
        self,
        fragment_name: str,
        global_prev: torch.Tensor,
        learner_fragments: Dict[str, torch.Tensor],
        arrivals: Sequence[LearnerArrival],
        grace_window_s: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], QuorumSelection, SyncerRoundTelemetry]:
        """Merge one fragment. Returns `(Delta_p, selection, telemetry)`; `Delta_p` is `None`
        when the quorum is not reached (the syncer must keep waiting).

        `Delta_p = Merge({theta_m, w_m}, Theta_prev)` aggregates per-learner outer gradients
        `Delta_m = Theta_prev - theta_m` with weights `w_m = Weight(c_tokens, c_steps)`.
        """
        if len(arrivals) > self.num_learners_m:
            raise ValueError("more arrivals than learners in the swarm")
        if grace_window_s is None:
            grace_window_s = (
                self.grace_controller.grace_window_seconds() if self.grace_controller is not None else 0.0
            )
        selection = select_quorum(arrivals, self.min_quorum_k, grace_window_s)
        by_id = {a.learner_id: a for a in arrivals}
        deferred_tokens = sum(by_id[i].tokens_since_sync for i in selection.deferred_learner_ids)

        if not selection.reached:
            telemetry = SyncerRoundTelemetry(
                fragment_name=fragment_name,
                merge_mode=self.merge_mode,
                num_arrivals=len(arrivals),
                num_admitted=0,
                num_deferred=len(selection.deferred_learner_ids),
                min_quorum_k=self.min_quorum_k,
                grace_window_s=float(grace_window_s),
                admitted_learner_ids=(),
                learner_weights={},
                admitted_tokens=0,
                deferred_tokens=int(deferred_tokens),
                merged_delta_norm=0.0,
                direct_average_norm=0.0,
                mean_learner_delta_norm=0.0,
            )
            return None, selection, telemetry

        prev32 = global_prev.detach().to(torch.float32)
        deltas: List[torch.Tensor] = []
        weights: List[float] = []
        learner_weights: Dict[str, float] = {}
        admitted_tokens = 0
        for learner_id in selection.admitted_learner_ids:
            if learner_id not in learner_fragments:
                raise KeyError(f"admitted learner {learner_id!r} has no fragment payload")
            theta_m = learner_fragments[learner_id].detach().to(torch.float32)
            if theta_m.shape != prev32.shape:
                raise ValueError(f"fragment shape mismatch for learner {learner_id!r}")
            a = by_id[learner_id]
            w_m = token_weighted_quality_weight(a.tokens_since_sync, a.steps_since_sync)
            deltas.append(prev32 - theta_m)
            weights.append(w_m)
            learner_weights[learner_id] = w_m
            admitted_tokens += a.tokens_since_sync
        if sum(weights) <= 0.0:
            raise ValueError("no admitted learner performed any inner step since the last sync")

        direct = direct_weighted_average(deltas, weights)
        merged = radial_directional_average(deltas, weights) if self.merge_mode == "rda" else direct
        mean_delta_norm = float(torch.stack([d.norm() for d in deltas]).mean())

        telemetry = SyncerRoundTelemetry(
            fragment_name=fragment_name,
            merge_mode=self.merge_mode,
            num_arrivals=len(arrivals),
            num_admitted=len(selection.admitted_learner_ids),
            num_deferred=len(selection.deferred_learner_ids),
            min_quorum_k=self.min_quorum_k,
            grace_window_s=float(grace_window_s),
            admitted_learner_ids=selection.admitted_learner_ids,
            learner_weights=learner_weights,
            admitted_tokens=int(admitted_tokens),
            deferred_tokens=int(deferred_tokens),
            merged_delta_norm=float(merged.norm()),
            direct_average_norm=float(direct.norm()),
            mean_learner_delta_norm=mean_delta_norm,
        )
        return merged.to(dtype=global_prev.dtype, device=global_prev.device), selection, telemetry
