"""QUIC hidden-state link with TLS 1.3 0-RTT resumption and a libp2p WebSocket fallback (`HM-M1-01`).

- Frames travel on a QUIC (RFC 9000) stream, length-prefixed, via `aioquic`.
- The server issues TLS 1.3 session tickets. A client that presents one resumes the session and
  sends its frames as 0-RTT early data (RFC 9001 §4.6), before the handshake finishes. Tickets
  are single-use on the server, the RFC 8446 §8.1 defence against replayed 0-RTT data.
- When QUIC cannot complete (for example UDP is blocked), frames go over a WebSocket (RFC 6455)
  after libp2p multistream-select 1.0 protocol negotiation.

Peers authenticate with an ephemeral CA: `LinkIdentity.generate()` creates a CA and a server
certificate signed by it, and clients trust only that CA.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import ipaddress
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import ConnectionTerminated, HandshakeCompleted, QuicEvent, StreamDataReceived
from aioquic.tls import SessionTicket
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve

ALPN = "hivemind-hidden-state/1"
SERVER_NAME = "hivemind.local"
MULTISTREAM = "/multistream/1.0.0"
LIBP2P_PROTOCOL = "/hivemind/hidden-state/1.0.0"
_LENGTH = struct.Struct("!I")

FrameHandler = Callable[[bytes], None]


@dataclass
class LinkIdentity:
    """An ephemeral CA plus a server certificate (SAN: `SERVER_NAME`, 127.0.0.1) it signed."""

    ca_certificate: x509.Certificate
    certificate: x509.Certificate
    private_key: ec.EllipticCurvePrivateKey

    @property
    def ca_pem(self) -> str:
        return self.ca_certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")

    @classmethod
    def generate(cls, server_name: str = SERVER_NAME, days: int = 7) -> "LinkIdentity":
        now = datetime.datetime.now(datetime.timezone.utc)
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "HiveMind link CA")])
        ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
        ca_certificate = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False, content_commitment=False, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=True,
                    crl_sign=True, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(ca_ski, critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        key = ec.generate_private_key(ec.SECP256R1())
        certificate = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, server_name)]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.DNSName(server_name), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                ),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski), critical=False
            )
            .sign(ca_key, hashes.SHA256())
        )
        return cls(ca_certificate, certificate, key)


def _drain_frames(buffer: bytearray) -> List[bytes]:
    frames = []
    while len(buffer) >= _LENGTH.size:
        (length,) = _LENGTH.unpack_from(buffer)
        if len(buffer) < _LENGTH.size + length:
            break
        frames.append(bytes(buffer[_LENGTH.size : _LENGTH.size + length]))
        del buffer[: _LENGTH.size + length]
    return frames


class _ServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, link: "QuicHiddenStateServer", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._link = link
        self._handshake_done = False
        self._buffers: Dict[int, bytearray] = {}
        self._counts: Dict[int, int] = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, HandshakeCompleted):
            self._handshake_done = True
            self._link.handshakes.append(
                {"session_resumed": event.session_resumed, "early_data_accepted": event.early_data_accepted}
            )
        elif isinstance(event, StreamDataReceived):
            buffer = self._buffers.setdefault(event.stream_id, bytearray())
            buffer += event.data
            for frame in _drain_frames(buffer):
                self._counts[event.stream_id] = self._counts.get(event.stream_id, 0) + 1
                if not self._handshake_done:
                    self._link.early_frames += 1
                self._link.deliver(frame)
            if event.end_stream:
                count = self._counts.pop(event.stream_id, 0)
                self._buffers.pop(event.stream_id, None)
                self._quic.send_stream_data(event.stream_id, _LENGTH.pack(count), end_stream=True)
                self.transmit()


class QuicHiddenStateServer:
    """QUIC endpoint that receives length-prefixed hidden-state frames and acknowledges counts."""

    def __init__(
        self,
        identity: LinkIdentity,
        host: str = "127.0.0.1",
        port: int = 0,
        on_frame: Optional[FrameHandler] = None,
    ) -> None:
        self.identity = identity
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.tickets: Dict[bytes, SessionTicket] = {}
        self.handshakes: List[Dict[str, bool]] = []
        self.frames_received = 0
        self.early_frames = 0
        """Frames that arrived before this server's handshake completed, i.e. as 0-RTT data."""
        self._server = None

    def deliver(self, frame: bytes) -> None:
        self.frames_received += 1
        if self.on_frame is not None:
            self.on_frame(frame)

    async def start(self) -> "QuicHiddenStateServer":
        configuration = QuicConfiguration(is_client=False, alpn_protocols=[ALPN])
        configuration.certificate = self.identity.certificate
        configuration.private_key = self.identity.private_key
        self._server = await serve(
            self.host,
            self.port,
            configuration=configuration,
            create_protocol=functools.partial(_ServerProtocol, link=self),
            session_ticket_fetcher=lambda ticket: self.tickets.pop(ticket, None),
            session_ticket_handler=lambda ticket: self.tickets.__setitem__(ticket.ticket, ticket),
        )
        self.port = self._server._transport.get_extra_info("sockname")[1]
        return self

    def close(self) -> None:
        if self._server is not None:
            self._server.close()


class _ClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.handshake: Optional[HandshakeCompleted] = None
        self.reply: asyncio.Future = asyncio.get_running_loop().create_future()
        self._buffer = bytearray()

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, HandshakeCompleted):
            self.handshake = event
        elif isinstance(event, StreamDataReceived):
            self._buffer += event.data
            if event.end_stream and not self.reply.done():
                self.reply.set_result(bytes(self._buffer))
        elif isinstance(event, ConnectionTerminated) and not self.reply.done():
            self.reply.set_exception(ConnectionError(f"QUIC connection closed: {event.reason_phrase}"))


@dataclass
class QuicSendReport:
    delivered: int
    elapsed_s: float
    session_resumed: bool
    early_data_accepted: bool
    new_ticket: Optional[SessionTicket]


async def quic_send_frames(
    host: str,
    port: int,
    frames: Sequence[bytes],
    ca_pem: str,
    session_ticket: Optional[SessionTicket] = None,
    server_name: str = SERVER_NAME,
    timeout: float = 3.0,
) -> QuicSendReport:
    """Send `frames` on one QUIC stream and wait for the server's delivered-frame count.

    With a `session_ticket` the frames are written before the handshake completes, so the
    client sends them as 0-RTT early data. Raises `TimeoutError`/`ConnectionError` when no
    QUIC path exists, which callers use to fall back to WebSocket.
    """
    configuration = QuicConfiguration(is_client=True, alpn_protocols=[ALPN], server_name=server_name)
    configuration.load_verify_locations(cadata=ca_pem.encode("ascii"))
    configuration.session_ticket = session_ticket
    tickets: List[SessionTicket] = []
    connection = QuicConnection(configuration=configuration, session_ticket_handler=tickets.append)
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: _ClientProtocol(connection), local_addr=("0.0.0.0", 0), family=socket.AF_INET
    )
    succeeded = False
    try:
        t0 = time.perf_counter()
        protocol.connect((socket.gethostbyname(host), port))
        stream_id = connection.get_next_available_stream_id()
        payload = b"".join(_LENGTH.pack(len(frame)) + frame for frame in frames)
        connection.send_stream_data(stream_id, payload, end_stream=True)
        protocol.transmit()
        reply = await asyncio.wait_for(protocol.reply, timeout)
        elapsed = time.perf_counter() - t0
        # aioquic's wait_connected() only resolves if it was awaited before the handshake ended,
        # so watch for the HandshakeCompleted event (and the server's ticket) directly.
        deadline = time.monotonic() + timeout
        while protocol.handshake is None and time.monotonic() < deadline:
            await asyncio.sleep(0.002)
        if protocol.handshake is None:
            raise TimeoutError("QUIC handshake did not complete")
        ticket_deadline = time.monotonic() + 0.5
        while session_ticket is None and not tickets and time.monotonic() < ticket_deadline:
            await asyncio.sleep(0.002)
        succeeded = True
        handshake = protocol.handshake
        return QuicSendReport(
            delivered=_LENGTH.unpack(reply[: _LENGTH.size])[0],
            elapsed_s=elapsed,
            session_resumed=bool(handshake and handshake.session_resumed),
            early_data_accepted=bool(handshake and handshake.early_data_accepted),
            new_ticket=tickets[-1] if tickets else None,
        )
    finally:
        protocol.close()
        if succeeded:
            try:
                await asyncio.wait_for(protocol.wait_closed(), 1.0)
            except asyncio.TimeoutError:
                pass
        transport.close()


def _uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _multistream_message(text: str) -> bytes:
    payload = (text + "\n").encode("utf-8")
    return _uvarint(len(payload)) + payload


def _parse_multistream(data: bytes) -> str:
    length, shift, index = 0, 0, 0
    while True:
        byte = data[index]
        length |= (byte & 0x7F) << shift
        index += 1
        if not byte & 0x80:
            break
        shift += 7
    return data[index : index + length].decode("utf-8").rstrip("\n")


class WebSocketHiddenStateServer:
    """WebSocket endpoint speaking libp2p multistream-select before streaming frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, on_frame: Optional[FrameHandler] = None) -> None:
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.frames_received = 0
        self._server = None

    async def start(self) -> "WebSocketHiddenStateServer":
        self._server = await ws_serve(self._handle, self.host, self.port, max_size=None)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def _handle(self, websocket) -> None:
        if _parse_multistream(await websocket.recv()) != MULTISTREAM:
            return
        await websocket.send(_multistream_message(MULTISTREAM))
        if _parse_multistream(await websocket.recv()) != LIBP2P_PROTOCOL:
            await websocket.send(_multistream_message("na"))
            return
        await websocket.send(_multistream_message(LIBP2P_PROTOCOL))
        count = 0
        async for message in websocket:
            if message == b"":
                break
            count += 1
            self.frames_received += 1
            if self.on_frame is not None:
                self.on_frame(bytes(message))
        await websocket.send(_LENGTH.pack(count))

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


async def websocket_send_frames(url: str, frames: Sequence[bytes], timeout: float = 5.0) -> int:
    """Negotiate `LIBP2P_PROTOCOL` via multistream-select, stream frames, return delivered count."""
    async with ws_connect(url, max_size=None) as websocket:
        await websocket.send(_multistream_message(MULTISTREAM))
        await websocket.send(_multistream_message(LIBP2P_PROTOCOL))
        header = _parse_multistream(await asyncio.wait_for(websocket.recv(), timeout))
        selected = _parse_multistream(await asyncio.wait_for(websocket.recv(), timeout))
        if header != MULTISTREAM or selected != LIBP2P_PROTOCOL:
            raise ConnectionError(f"libp2p negotiation failed: {header!r} / {selected!r}")
        for frame in frames:
            await websocket.send(frame)
        await websocket.send(b"")
        reply = await asyncio.wait_for(websocket.recv(), timeout)
        return _LENGTH.unpack(reply)[0]


@dataclass
class LinkRun:
    """Outcome of `run_local_link`."""

    protocol: str
    """`quic_0rtt` (resumed, early data accepted), `quic` (no 0-RTT) or `libp2p_websocket`."""
    frames_sent: int
    frames_delivered: int
    elapsed_s: float
    session_resumed: bool = False
    early_data_accepted: bool = False
    server_early_frames: int = 0
    fallback_reason: Optional[str] = None
    handshakes: List[Dict[str, bool]] = field(default_factory=list)

    @property
    def frames_per_sec(self) -> float:
        return self.frames_delivered / max(self.elapsed_s, 1e-9)


def _unused_udp_port(host: str) -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind((host, 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


async def run_local_link(
    frames: Sequence[bytes],
    host: str = "127.0.0.1",
    on_frame: Optional[FrameHandler] = None,
    udp_blocked: bool = False,
    quic_timeout: float = 2.0,
) -> LinkRun:
    """Start QUIC + WebSocket servers on `host` and push `frames` the way a peer would.

    A first connection obtains a session ticket; the second resumes with it and sends `frames`
    as 0-RTT early data. With `udp_blocked` the client is pointed at a UDP port nothing listens
    on, so the QUIC handshake genuinely times out and the WebSocket fallback carries the frames.
    """
    identity = LinkIdentity.generate()
    quic_server = await QuicHiddenStateServer(identity, host, on_frame=on_frame).start()
    ws_server = await WebSocketHiddenStateServer(host, on_frame=on_frame).start()
    quic_port = _unused_udp_port(host) if udp_blocked else quic_server.port
    try:
        try:
            warmup = await quic_send_frames(host, quic_port, [b"warmup"], identity.ca_pem, timeout=quic_timeout)
            report = await quic_send_frames(
                host, quic_port, frames, identity.ca_pem, session_ticket=warmup.new_ticket, timeout=quic_timeout
            )
        except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
            t0 = time.perf_counter()
            delivered = await websocket_send_frames(ws_server.url, frames)
            return LinkRun(
                protocol="libp2p_websocket",
                frames_sent=len(frames),
                frames_delivered=delivered,
                elapsed_s=time.perf_counter() - t0,
                fallback_reason=type(exc).__name__,
            )
        return LinkRun(
            protocol="quic_0rtt" if report.early_data_accepted else "quic",
            frames_sent=len(frames),
            frames_delivered=report.delivered,
            elapsed_s=report.elapsed_s,
            session_resumed=report.session_resumed,
            early_data_accepted=report.early_data_accepted,
            server_early_frames=quic_server.early_frames,
            handshakes=list(quic_server.handshakes),
        )
    finally:
        quic_server.close()
        await ws_server.close()
