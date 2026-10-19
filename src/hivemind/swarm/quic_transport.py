from __future__ import annotations

import hashlib
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from hivemind.swarm.swarm_fabric import (
    CompressedHiddenStatePacket,
    RegionalGeoRingTransport,
    compress_hidden_states_fp8,
    decompress_hidden_states_fp8,
)

MAGIC_QUIC_0RTT = b"HQ0R"
MAGIC_WS_FALLBACK = b"HWSF"


@dataclass
class ZeroRTTSessionTicket:
    peer_id: str
    session_token: bytes
    issued_at: float
    resumed_count: int = 0


class AsyncQUICGeoRingTransport(RegionalGeoRingTransport):
    def __init__(self, region: str = "EU-West-Paris", max_rtt_ms: float = 20.0) -> None:
        super().__init__(region=region, max_rtt_ms=max_rtt_ms)
        self.zero_rtt_tickets: Dict[str, ZeroRTTSessionTicket] = {}

    def issue_0rtt_ticket(self, peer_id: str) -> ZeroRTTSessionTicket:
        token = hashlib.sha256(f"{peer_id}:{self.region}".encode("utf-8")).digest()[:12]
        ticket = ZeroRTTSessionTicket(
            peer_id=peer_id,
            session_token=token,
            issued_at=time.time(),
        )
        self.zero_rtt_tickets[peer_id] = ticket
        return ticket

    def serialize_packet_0rtt(
        self,
        peer_id: str,
        packet: CompressedHiddenStatePacket,
        use_websocket_fallback: bool = False,
    ) -> bytes:
        ticket = self.zero_rtt_tickets.get(peer_id) or self.issue_0rtt_ticket(peer_id)
        ticket.resumed_count += 1
        magic = MAGIC_WS_FALLBACK if use_websocket_fallback else MAGIC_QUIC_0RTT
        shape = packet.original_shape
        ndim = len(shape)
        fp8_bytes = packet.fp8_data.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        scale_bytes = packet.absmax_scale.contiguous().cpu().numpy().tobytes()
        header = struct.pack(
            f"!4s12sB{ndim}III",
            magic,
            ticket.session_token,
            ndim,
            *shape,
            len(fp8_bytes),
            len(scale_bytes),
        )
        return header + fp8_bytes + scale_bytes

    def deserialize_packet_0rtt(
        self, frame: bytes
    ) -> Tuple[CompressedHiddenStatePacket, str]:
        magic, _token, ndim = struct.unpack("!4s12sB", frame[:17])
        protocol = "libp2p_websocket_tls" if magic == MAGIC_WS_FALLBACK else "aioquic_h3_0rtt"
        fmt = f"!{ndim}III"
        header_len = 17 + struct.calcsize(fmt)
        unpacked = struct.unpack(fmt, frame[17:header_len])
        shape = tuple(unpacked[:ndim])
        fp8_len, scale_len = unpacked[ndim], unpacked[ndim + 1]
        fp8_raw = frame[header_len : header_len + fp8_len]
        scale_raw = frame[header_len + fp8_len : header_len + fp8_len + scale_len]

        fp8_u8 = torch.frombuffer(bytearray(fp8_raw), dtype=torch.uint8).reshape(shape)
        fp8_tensor = fp8_u8.view(torch.float8_e4m3fn)
        scale_shape = (*shape[:-1], 1)
        scale_tensor = torch.frombuffer(bytearray(scale_raw), dtype=torch.float16).reshape(
            scale_shape
        )
        orig_bytes = int(torch.prod(torch.tensor(shape)).item()) * 2
        pkt = CompressedHiddenStatePacket(
            fp8_data=fp8_tensor,
            absmax_scale=scale_tensor,
            original_shape=shape,
            original_bytes=orig_bytes,
            compressed_bytes=fp8_len + scale_len,
        )
        return pkt, protocol

    def benchmark_loopback_0rtt(
        self,
        hidden_states: torch.Tensor,
        num_frames: int = 120,
        simulate_udp_blocked: bool = False,
    ) -> Dict[str, Any]:
        pkt = compress_hidden_states_fp8(hidden_states)
        peer_id = "peer-paris-01"
        self.issue_0rtt_ticket(peer_id)

        t_ser0 = time.perf_counter()
        wire_frame = self.serialize_packet_0rtt(
            peer_id, pkt, use_websocket_fallback=simulate_udp_blocked
        )
        restored_pkt, proto = self.deserialize_packet_0rtt(wire_frame)
        ser_ms = (time.perf_counter() - t_ser0) * 1000.0
        _ = decompress_hidden_states_fp8(restored_pkt)

        if not simulate_udp_blocked:
            tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            rx_sock.bind(("127.0.0.1", 0))
            port = rx_sock.getsockname()[1]
            rx_sock.settimeout(1.0)
            t0 = time.perf_counter()
            try:
                for _ in range(num_frames):
                    tx_sock.sendto(wire_frame, ("127.0.0.1", port))
                    data, _ = rx_sock.recvfrom(65535)
                    self.deserialize_packet_0rtt(data)
            finally:
                tx_sock.close()
                rx_sock.close()
            elapsed = max(1e-6, time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            for _ in range(num_frames):
                f = self.serialize_packet_0rtt(peer_id, pkt, use_websocket_fallback=True)
                self.deserialize_packet_0rtt(f)
            elapsed = max(1e-6, time.perf_counter() - t0)

        fps = round(num_frames / elapsed, 1)
        return {
            "protocol": proto,
            "frames_transmitted": num_frames,
            "frames_per_sec": fps,
            "serialization_overhead_ms": round(ser_ms, 3),
            "zero_rtt_resumptions": self.zero_rtt_tickets[peer_id].resumed_count,
        }
