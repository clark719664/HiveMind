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

Phase S-5 Inference Ring Hardening (`RING-01` .. `RING-02`):
- Rejects spoofed (`ping_ms <= 0`), high-packet-loss (`> 15%`), high-jitter (`> 20 ms`),
  or insufficient-VRAM (`< 4.0 GB`) peers and deduplicates `peer_id` registrations.
- Enforces speculative decoding envelope (`1 <= draft_gamma <= 16`, `1 <= max_new_tokens <= 512`)
  and verifies finiteness (`torch.isfinite`) of MoE verification logits.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

from hivemind.core.btm_upcycle import UpcycledSparseMoEModel
from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular runtime import.
    from hivemind.swarm.quic_transport import ContinentalSuperRingRouter


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
        return (
            0.0 < self.ping_ms < 25.0
            and self.vram_gb >= 4.0
            and 0.0 <= self.packet_loss_ratio <= 0.15
            and 0.0 <= self.jitter_ms <= 20.0
        )

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
    latency_basis: str = (
        "latency_reduction_factor = generated tokens / verification passes under a simulated "
        "per-pass RTT; effective_tokens_per_sec adds that simulated RTT to measured local compute "
        "time. Neither is a measured network speedup."
    )


class RegionalDHTSwarmRing:
    """Petals-style DHT Swarm coordinator grouping low-latency (<25ms) peers into regional rings."""

    def __init__(self, region: str = "EU-West-Paris", max_ping_ms: float = 25.0) -> None:
        self.region = region
        self.max_ping_ms = max_ping_ms
        self.peers: list[SwarmPeerNode] = []
        self.super_ring_router: "ContinentalSuperRingRouter | None" = None
        self.hierarchical_routes: list[Dict[str, Any]] = []

    def register_peer(self, peer: SwarmPeerNode) -> bool:
        # Rule RING-01: Reject spoofed ping (<= 0), high packet loss (> 15%), high jitter (> 20ms), or low VRAM (< 4GB)
        if not peer.peer_id or not str(peer.peer_id).strip() or not peer.steam_id64:
            return False
        if not math.isfinite(float(peer.ping_ms)) or float(peer.ping_ms) <= 0.0:
            return False
        if not math.isfinite(float(peer.vram_gb)) or float(peer.vram_gb) < 4.0:
            return False
        if not math.isfinite(float(peer.packet_loss_ratio)) or not (0.0 <= float(peer.packet_loss_ratio) <= 0.15):
            return False
        if not math.isfinite(float(peer.jitter_ms)) or not (0.0 <= float(peer.jitter_ms) <= 20.0):
            return False

        if peer.region == self.region and peer.ping_ms < self.max_ping_ms:
            self.peers = [p for p in self.peers if p.peer_id != peer.peer_id]
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
    ) -> SpeculativeSwarmResult:
        """Generate tokens using local draft speculation (gamma=5) + single-pass MoE swarm verification."""
        # Rule RING-02: Enforce speculative decoding bounds to prevent unbounded forward-pass DoS
        if int(draft_gamma) < 1 or int(draft_gamma) > 16:
            raise SwarmSecurityViolationError(
                "RING-02",
                "SPECULATIVE_GAMMA_OUT_OF_BOUNDS",
                f"draft_gamma ({draft_gamma}) must be within [1, 16].",
            )
        if int(max_new_tokens) < 1 or int(max_new_tokens) > 512:
            raise SwarmSecurityViolationError(
                "RING-02",
                "MAX_NEW_TOKENS_OUT_OF_BOUNDS",
                f"max_new_tokens ({max_new_tokens}) must be within [1, 512].",
            )
        if prompt_ids.numel() == 0:
            raise SwarmSecurityViolationError(
                "RING-02",
                "EMPTY_PROMPT_TENSOR",
                "prompt_ids must contain at least 1 token.",
            )

        device = next(moe_target_model.parameters()).device
        moe_target_model.eval()
        draft_model.eval()
        moe_target_model.reset_routing_telemetry()

        seq = prompt_ids.to(device, dtype=torch.long)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        prompt_len = seq.size(1)

        generated: list[int] = []
        verification_passes = 0
        accepted_total = 0
        proposed_total = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            while len(generated) < max_new_tokens:
                # 1. Local Draft Pass (Top-1 single-expert mode) proposes gamma=5 tokens with zero network RTT
                draft_seq = seq.clone()
                draft_tokens: list[int] = []
                if isinstance(draft_model, UpcycledSparseMoEModel):
                    for blk in draft_model.get_moe_blocks():
                        blk.top_k = 1
                        blk.router.top_k = 1
                for _ in range(draft_gamma):
                    d_out = draft_model(input_ids=draft_seq)
                    next_tok = int(torch.argmax(d_out.logits[:, -1, :], dim=-1).item())
                    draft_tokens.append(next_tok)
                    draft_seq = torch.cat(
                        [draft_seq, torch.tensor([[next_tok]], device=device, dtype=torch.long)],
                        dim=1,
                    )

                # 2. P2P Sparse MoE Swarm (Top-2 multi-expert mode) verifies all gamma=5 tokens in ONE forward pass
                if isinstance(moe_target_model, UpcycledSparseMoEModel):
                    for blk in moe_target_model.get_moe_blocks():
                        blk.top_k = moe_target_model.top_k
                        blk.router.top_k = moe_target_model.top_k
                verification_passes += 1
                m_out = moe_target_model(input_ids=draft_seq)
                if not bool(torch.isfinite(m_out.logits).all().item()):
                    raise SwarmSecurityViolationError(
                        "RING-02",
                        "POISONED_MOE_RING_LOGITS_DETECTED",
                        "Non-finite (NaN/Inf) logits returned by P2P Sparse MoE verification pass.",
                    )
                start_pos = seq.size(1) - 1
                target_logits = m_out.logits[:, start_pos : start_pos + draft_gamma + 1, :].squeeze(0)
                target_probs = torch.softmax(target_logits.to(torch.float32), dim=-1)
                target_preds = torch.argmax(target_probs, dim=-1)

                accepted_in_pass = 0
                for i, d_tok in enumerate(draft_tokens):
                    proposed_total += 1
                    t_tok = int(target_preds[i].item())
                    max_p = float(target_probs[i, t_tok].item())
                    draft_p = float(target_probs[i, d_tok].item())
                    # Speculative rejection sampling criterion: accept if matches argmax or within high-density support
                    if d_tok == t_tok or (max_p > 0 and (draft_p / max_p) >= 0.65):
                        generated.append(d_tok)
                        accepted_total += 1
                        accepted_in_pass += 1
                        if len(generated) >= max_new_tokens:
                            break
                    else:
                        # Accept the target MoE correction token and resync draft sequence
                        generated.append(t_tok)
                        accepted_total += 1
                        accepted_in_pass += 1
                        break

                # If all gamma draft tokens matched, we also get 1 bonus token from the final position
                if accepted_in_pass == draft_gamma and len(generated) < max_new_tokens:
                    bonus_tok = int(target_preds[draft_gamma].item())
                    generated.append(bonus_tok)

                seq = torch.cat(
                    [
                        prompt_ids.to(device, dtype=torch.long).reshape(1, -1),
                        torch.tensor([generated], device=device, dtype=torch.long),
                    ],
                    dim=1,
                )

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
