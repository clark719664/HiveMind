"""Phase 2: Regional DHT Swarm Router & 5-Token Speculative Decoding Engine.

Once an epoch of the 8x Sparse MoE is finalized, idle Steam nodes switch to Inference Mode:
- Residential fiber nodes (ping < 25 ms via STUN/TURN) form regional rings (e.g., EU-West-Paris).
- Local 0.5B draft model proposes gamma=5 speculative tokens locally.
- The P2P Sparse MoE swarm verifies all 5 tokens in a single parallel forward pass,
  dividing network latency overhead by 5 to achieve 15-25 tokens/sec over consumer broadband.

Season 4 Milestone `HM-S4-02` (Issue #27) upgrades the flat regional ring into a 2-Tier
Kademlia Super-Ring: the regional Geo-Ring stays the Tier-1 `< 20 ms` data plane while a
`ContinentalSuperRingRouter` elects *Continental Bridge* nodes per continent (`EU`, `US`,
`APAC`) that aggregate the Tier-1 rings into a single WAN-transit Tier-2 plane.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

from hivemind.core.btm_upcycle import UpcycledSparseMoEModel

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular runtime import.
    from hivemind.swarm.quic_transport import ContinentalSuperRingRouter


@contextmanager
def _temporary_model_top_k(model: nn.Module, top_k: int):
    """Keep draft/target routing separate even when their modules are shared."""
    blocks = list(model.get_moe_blocks()) if isinstance(model, UpcycledSparseMoEModel) else []
    previous = [(block, block.top_k, block.router.top_k) for block in blocks]
    try:
        for block in blocks:
            block.top_k = top_k
            block.router.top_k = top_k
        yield
    finally:
        for block, block_top_k, router_top_k in previous:
            block.top_k = block_top_k
            block.router.top_k = router_top_k


@contextmanager
def _evaluation_mode(*models: nn.Module):
    # Save individual flags so shared modules and mixed train/eval states survive.
    previous = {module: module.training for model in models for module in model.modules()}
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for module, training in previous.items():
            module.training = training


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else fallback


def _sampling_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Normalize on CPU in float64 to reduce residual subtraction roundoff."""
    probabilities = torch.softmax(logits.detach().to(device="cpu", dtype=torch.float64), dim=-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("model logits must define finite categorical probabilities")
    return probabilities


def _sample_token(probabilities: torch.Tensor, generator: torch.Generator | None) -> int:
    device = generator.device if generator is not None else probabilities.device
    return int(torch.multinomial(probabilities.to(device), 1, generator=generator).item())


@dataclass
class SwarmPeerNode:
    """Represents an idle gamer GPU hosting a subset of Sparse MoE expert layers in 4-bit."""

    peer_id: str
    steam_id64: str
    region: str
    ping_ms: float
    vram_gb: float
    hosted_expert_domain: str
    layer_range: tuple[int, int]
    stun_verified: bool = False
    jitter_ms: float = 0.0
    packet_loss_ratio: float = 0.0

    @property
    def eligible_for_low_latency_ring(self) -> bool:
        return self.ping_ms < 25.0

    @property
    def continent(self) -> str:
        """Continental Super-Ring bucket (`EU`/`US`/`APAC`) of this regional peer (`HM-S4-02`)."""
        from hivemind.swarm.quic_transport import classify_continent

        return classify_continent(self.region)


@dataclass
class SpeculativeSwarmResult:
    """Telemetry and output from a Regional P2P Swarm Speculative Decoding session."""

    region_ring: str
    active_ring_peers: int
    mean_ring_ping_ms: float
    prompt_length: int
    generated_tokens: list[int]
    draft_gamma: int
    swarm_verification_passes: int
    mean_acceptance_rate: float
    effective_tokens_per_sec: float
    latency_reduction_factor: float
    expert_routing_share: dict[str, float] = field(default_factory=dict)
    routing_tier: str = "regional_geo_ring"
    cross_continent_hops: int = 0
    estimated_route_rtt_ms: float = 0.0


class RegionalDHTSwarmRing:
    """Petals-style DHT Swarm coordinator grouping low-latency (<25ms) peers into regional rings."""

    def __init__(self, region: str = "EU-West-Paris", max_ping_ms: float = 25.0) -> None:
        self.region = region
        self.max_ping_ms = max_ping_ms
        self.peers: list[SwarmPeerNode] = []
        self.super_ring_router: "ContinentalSuperRingRouter | None" = None
        self.hierarchical_routes: list[Dict[str, Any]] = []

    def register_peer(self, peer: SwarmPeerNode) -> bool:
        if peer.region == self.region and peer.ping_ms < self.max_ping_ms:
            self.peers.append(peer)
            self._sync_peer_to_super_ring(peer)
            return True
        return False

    # ------------------------------------------------------------------
    # HM-S4-02: 2-Tier Kademlia Continental Super-Ring
    # ------------------------------------------------------------------
    def attach_continental_super_ring(self, router: "ContinentalSuperRingRouter") -> None:
        """Bind a 2-Tier Continental Super-Ring router to this regional ring (`HM-S4-02`).

        Every peer already registered in the Tier-1 regional ring is mirrored into the
        Super-Ring so that Continental Bridge elections see the full regional membership.
        """
        self.super_ring_router = router
        for peer in self.peers:
            self._sync_peer_to_super_ring(peer)

    def _ensure_super_ring(self) -> "ContinentalSuperRingRouter":
        """Return the attached Super-Ring router, lazily creating a default one."""
        if self.super_ring_router is None:
            from hivemind.swarm.quic_transport import ContinentalSuperRingRouter

            self.super_ring_router = ContinentalSuperRingRouter()
        return self.super_ring_router

    def _sync_peer_to_super_ring(self, peer: SwarmPeerNode) -> bool:
        """Mirror a Tier-1 peer into the Tier-2 Super-Ring (respecting the `< 20 ms` SLA)."""
        if self.super_ring_router is None:
            return False
        from hivemind.swarm.quic_transport import SuperRingPeerNode

        node = SuperRingPeerNode(
            peer_id=peer.peer_id,
            region=peer.region,
            rtt_ms=float(peer.ping_ms),
            jitter_ms=float(peer.jitter_ms),
            packet_loss_ratio=float(peer.packet_loss_ratio),
            stun_verified=bool(peer.stun_verified),
        )
        return self.super_ring_router.register_peer(node)

    def register_super_ring_peer(self, peer: SwarmPeerNode) -> bool:
        """Admit a peer of *any* continent into the Tier-2 Continental Super-Ring.

        Unlike `register_peer`, the peer does not need to belong to this regional ring: the
        Super-Ring aggregates `EU`/`US`/`APAC` rings into one WAN-transit plane (`HM-S4-02`).
        """
        router = self._ensure_super_ring()
        from hivemind.swarm.quic_transport import SuperRingPeerNode

        node = SuperRingPeerNode(
            peer_id=peer.peer_id,
            region=peer.region,
            rtt_ms=float(peer.ping_ms),
            jitter_ms=float(peer.jitter_ms),
            packet_loss_ratio=float(peer.packet_loss_ratio),
            stun_verified=bool(peer.stun_verified),
        )
        return router.register_peer(node)

    def resolve_hierarchical_route(
        self, source_peer_id: str, target_peer_id: str
    ) -> Dict[str, Any]:
        """Resolve the 2-Tier path (`Continental Bridge -> Regional Geo-Ring`) between two peers."""
        router = self._ensure_super_ring()
        route = router.route(source_peer_id, target_peer_id)
        self.hierarchical_routes.append(route)
        return route

    def route_to_region(self, source_peer_id: str, target_region: str) -> Dict[str, Any]:
        """Route towards the lowest-RTT peer of `target_region` via the Super-Ring."""
        router = self._ensure_super_ring()
        target = router.find_peer_in_region(target_region)
        if target is None:
            raise KeyError(f"No Super-Ring peer registered for region '{target_region}'.")
        return self.resolve_hierarchical_route(source_peer_id, target.peer_id)

    def run_speculative_inference(
        self,
        moe_target_model: UpcycledSparseMoEModel,
        draft_model: nn.Module,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        draft_gamma: int = 5,
        simulated_network_rtt_ms: float = 18.5,
        *,
        generator: torch.Generator | None = None,
    ) -> SpeculativeSwarmResult:
        """Sample from the target distribution using single-pass speculative verification.

        Each proposal is sampled from its draft distribution q and accepted with
        probability min(1, p(token)/q(token)). A rejection emits a sample from
        normalized max(p-q, 0) and discards the remaining proposal suffix. If all
        proposals are accepted, the final target position supplies a bonus sample.
        Both models must be causal and share token IDs/vocabulary. One prompt is
        supported. Pass a seeded generator for reproducible sampling; the reported
        network RTT remains a simulation, not a measured WAN execution.
        """
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if isinstance(draft_gamma, bool) or not isinstance(draft_gamma, Integral) or draft_gamma < 1:
            raise ValueError("draft_gamma must be a positive integer")
        if not math.isfinite(simulated_network_rtt_ms) or simulated_network_rtt_ms < 0:
            raise ValueError("simulated_network_rtt_ms must be finite and nonnegative")
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1 or prompt_ids.size(1) == 0:
            raise ValueError("prompt_ids must contain one nonempty prompt")
        device = _model_device(moe_target_model, prompt_ids.device)
        draft_device = _model_device(draft_model, device)
        moe_target_model.reset_routing_telemetry()
        seq = prompt_ids.to(device, dtype=torch.long)
        prompt_len = seq.size(1)

        generated: list[int] = []
        verification_passes = 0
        accepted_total = 0
        proposed_total = 0

        t0 = time.perf_counter()
        with torch.no_grad(), _evaluation_mode(moe_target_model, draft_model):
            while len(generated) < max_new_tokens:
                proposal_count = min(draft_gamma, max_new_tokens - len(generated))
                draft_seq = seq.to(draft_device)
                draft_tokens: list[int] = []
                draft_probabilities: list[torch.Tensor] = []
                with _temporary_model_top_k(draft_model, 1):
                    for _ in range(proposal_count):
                        d_logits = draft_model(input_ids=draft_seq).logits
                        if d_logits.ndim != 3 or d_logits.shape[:2] != draft_seq.shape:
                            raise ValueError("draft model must return logits for every input position")
                        q = _sampling_probabilities(d_logits[0, -1, :])
                        next_tok = _sample_token(q, generator)
                        draft_tokens.append(next_tok)
                        draft_probabilities.append(q)
                        draft_seq = torch.cat(
                            [draft_seq, draft_seq.new_tensor([[next_tok]])], dim=1
                        )
                proposed_total += len(draft_tokens)

                # A causal target forward supplies p at every proposed prefix and
                # at the extra position after the last proposal.
                with _temporary_model_top_k(moe_target_model, getattr(moe_target_model, "top_k", 2)):
                    m_logits = moe_target_model(input_ids=draft_seq.to(device)).logits
                verification_passes += 1
                if m_logits.ndim != 3 or m_logits.shape[:2] != draft_seq.shape:
                    raise ValueError("target model must return logits for every input position")
                start_pos = seq.size(1) - 1
                target_probs = _sampling_probabilities(
                    m_logits[0, start_pos : start_pos + proposal_count + 1, :]
                )
                if any(q.shape != target_probs[0].shape for q in draft_probabilities):
                    raise ValueError("draft and target models must share the same vocabulary")

                emitted: list[int] = []
                all_accepted = True
                for i, d_tok in enumerate(draft_tokens):
                    p, q = target_probs[i], draft_probabilities[i]
                    acceptance = min(1.0, float((p[d_tok] / q[d_tok]).item()))
                    random_device = generator.device if generator is not None else p.device
                    uniform = float(torch.rand((), dtype=torch.float64, device=random_device, generator=generator).item())
                    if uniform < acceptance:
                        emitted.append(d_tok)
                        accepted_total += 1
                    else:
                        residual = torch.clamp(p - q, min=0)
                        residual_mass = residual.sum()
                        if float(residual_mass.item()) <= 0:
                            raise RuntimeError("a rejected proposal must have positive residual mass")
                        emitted.append(_sample_token(residual / residual_mass, generator))
                        all_accepted = False
                        break

                if all_accepted and len(generated) + len(emitted) < max_new_tokens:
                    emitted.append(_sample_token(target_probs[proposal_count], generator))
                generated.extend(emitted)
                seq = torch.cat([seq, seq.new_tensor([emitted])], dim=1)

        compute_elapsed_sec = time.perf_counter() - t0
        # Account for single-pass regional ring RTT (1 RTT per verification pass instead of 1 RTT per token)
        network_elapsed_sec = verification_passes * (simulated_network_rtt_ms / 1000.0)
        total_time_sec = max(1e-4, compute_elapsed_sec + network_elapsed_sec)
        tps = round(len(generated) / total_time_sec, 2)

        mean_ping = (
            round(sum(p.ping_ms for p in self.peers) / len(self.peers), 2)
            if self.peers
            else simulated_network_rtt_ms
        )

        return SpeculativeSwarmResult(
            region_ring=self.region,
            active_ring_peers=len(self.peers),
            mean_ring_ping_ms=mean_ping,
            prompt_length=prompt_len,
            generated_tokens=generated[:max_new_tokens],
            draft_gamma=draft_gamma,
            swarm_verification_passes=verification_passes,
            mean_acceptance_rate=round(accepted_total / max(1, proposed_total), 3),
            effective_tokens_per_sec=tps,
            latency_reduction_factor=round(len(generated[:max_new_tokens]) / max(1, verification_passes), 2),
            expert_routing_share=moe_target_model.get_expert_utilization(),
        )
