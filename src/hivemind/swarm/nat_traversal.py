"""RFC 5389 STUN, RFC 5766 TURN and UDP hole punching for HiveMind swarm peers (`HM-S3-01`).

Everything in this module speaks the real wire protocols over real UDP sockets:

- `StunMessage` encodes and decodes RFC 5389 messages, including MESSAGE-INTEGRITY
  (HMAC-SHA1 with short- or long-term credentials) and FINGERPRINT. It reproduces the
  RFC 5769 test vectors byte for byte.
- `stun_binding` sends Binding requests with RFC 5389 §7.2.1 retransmission and returns the
  XOR-MAPPED-ADDRESS (server-reflexive transport address).
- `classify_nat_mapping` applies the RFC 5780 §4.3 mapping-behaviour test against two STUN
  servers to tell endpoint-independent (cone) mappings from address-dependent (symmetric) ones.
- `TurnClient` performs RFC 5766 Allocate (401 long-term-credential challenge, 438 stale-nonce
  retry), CreatePermission, Refresh and Send/Data indications.
- `udp_hole_punch` runs ICE-style (RFC 8445) simultaneous-open connectivity checks.
- `StunServer` and `TurnRelayServer` let any HiveMind node act as a STUN reflector or a TURN
  relay for peers stuck behind symmetric NATs.

Passwords must already be SASLprep-normalised (RFC 4013); plain ASCII passwords need no change.
Only IPv4 transport is used for sockets; IPv6 addresses are still encoded and decoded.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import select
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

Address = Tuple[str, int]
ServerSpec = Union[str, Address]

MAGIC_COOKIE = 0x2112A442
FINGERPRINT_XOR = 0x5354554E
HEADER = struct.Struct("!HHI12s")

BINDING = 0x001
ALLOCATE = 0x003
REFRESH = 0x004
SEND = 0x006
DATA = 0x007
CREATE_PERMISSION = 0x008

REQUEST = 0b00
INDICATION = 0b01
SUCCESS = 0b10
ERROR = 0b11

MAPPED_ADDRESS = 0x0001
USERNAME = 0x0006
MESSAGE_INTEGRITY = 0x0008
ERROR_CODE = 0x0009
LIFETIME = 0x000D
XOR_PEER_ADDRESS = 0x0012
DATA_ATTR = 0x0013
REALM = 0x0014
NONCE = 0x0015
XOR_RELAYED_ADDRESS = 0x0016
REQUESTED_TRANSPORT = 0x0019
XOR_MAPPED_ADDRESS = 0x0020
SOFTWARE = 0x8022
FINGERPRINT = 0x8028

UDP_PROTOCOL_NUMBER = 17
SOFTWARE_NAME = "HiveMind STUN/TURN"
PERMISSION_LIFETIME_S = 300.0
DEFAULT_STUN_SERVERS: Tuple[str, ...] = ("stun.l.google.com:19302", "stun.cloudflare.com:3478")
"""Public STUN servers used when a caller does not name one."""


class StunError(Exception):
    """A STUN/TURN transaction failed: timeout, error response or failed integrity check."""

    def __init__(self, message: str, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


def message_type(method: int, cls: int) -> int:
    """Interleave the 12-bit method and 2-bit class into the 14-bit STUN message type."""
    return (
        ((method & 0x0F80) << 2)
        | ((method & 0x0070) << 1)
        | (method & 0x000F)
        | ((cls & 0b10) << 7)
        | ((cls & 0b01) << 4)
    )


def split_message_type(msg_type: int) -> Tuple[int, int]:
    method = (msg_type & 0x000F) | ((msg_type & 0x00E0) >> 1) | ((msg_type & 0x3E00) >> 2)
    cls = ((msg_type & 0x0100) >> 7) | ((msg_type & 0x0010) >> 4)
    return method, cls


def long_term_key(username: str, realm: str, password: str) -> bytes:
    """RFC 5389 §15.4 long-term credential key: MD5(username ":" realm ":" password)."""
    return hashlib.md5(f"{username}:{realm}:{password}".encode("utf-8")).digest()


def short_term_key(password: str) -> bytes:
    """RFC 5389 §15.4 short-term credential key: the password itself."""
    return password.encode("utf-8")


def encode_xor_address(addr: Address, transaction_id: bytes) -> bytes:
    ip = ipaddress.ip_address(addr[0])
    port = addr[1] ^ (MAGIC_COOKIE >> 16)
    if ip.version == 4:
        return struct.pack("!BBHI", 0, 0x01, port, int(ip) ^ MAGIC_COOKIE)
    key = int.from_bytes(struct.pack("!I", MAGIC_COOKIE) + transaction_id, "big")
    return struct.pack("!BBH", 0, 0x02, port) + (int(ip) ^ key).to_bytes(16, "big")


def decode_xor_address(value: bytes, transaction_id: bytes) -> Address:
    family = value[1]
    port = struct.unpack("!H", value[2:4])[0] ^ (MAGIC_COOKIE >> 16)
    if family == 0x01:
        ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address] = ipaddress.IPv4Address(
            struct.unpack("!I", value[4:8])[0] ^ MAGIC_COOKIE
        )
    elif family == 0x02:
        key = int.from_bytes(struct.pack("!I", MAGIC_COOKIE) + transaction_id, "big")
        ip = ipaddress.IPv6Address(int.from_bytes(value[4:20], "big") ^ key)
    else:
        raise StunError(f"unknown address family {family:#x}")
    return str(ip), port


def decode_plain_address(value: bytes) -> Address:
    family = value[1]
    port = struct.unpack("!H", value[2:4])[0]
    if family == 0x01:
        return str(ipaddress.IPv4Address(value[4:8])), port
    if family == 0x02:
        return str(ipaddress.IPv6Address(value[4:20])), port
    raise StunError(f"unknown address family {family:#x}")


def encode_error_code(code: int, reason: str) -> bytes:
    return struct.pack("!HBB", 0, code // 100, code % 100) + reason.encode("utf-8")


def decode_error_code(value: Optional[bytes]) -> Tuple[int, str]:
    if not value or len(value) < 4:
        return 0, ""
    return (value[2] & 0x07) * 100 + value[3], value[4:].decode("utf-8", "replace")


def _pack_attribute(attr_type: int, value: bytes, pad_byte: bytes) -> bytes:
    return struct.pack("!HH", attr_type, len(value)) + value + pad_byte * (-len(value) % 4)


@dataclass
class StunMessage:
    """A STUN message: method, class, transaction id and ordered attributes."""

    method: int
    cls: int
    transaction_id: bytes = field(default_factory=lambda: secrets.token_bytes(12))
    attributes: List[Tuple[int, bytes]] = field(default_factory=list)

    def add(self, attr_type: int, value: bytes) -> "StunMessage":
        self.attributes.append((attr_type, value))
        return self

    def get(self, attr_type: int) -> Optional[bytes]:
        for current, value in self.attributes:
            if current == attr_type:
                return value
        return None

    def encode(
        self,
        integrity_key: Optional[bytes] = None,
        fingerprint: bool = False,
        pad_byte: bytes = b"\x00",
    ) -> bytes:
        """Serialise the message, appending MESSAGE-INTEGRITY and/or FINGERPRINT last.

        RFC 5389 lets padding bytes take any value; `pad_byte` exists so the RFC 5769 vectors
        (which pad with spaces) can be reproduced exactly.
        """
        mtype = message_type(self.method, self.cls)
        body = b"".join(
            _pack_attribute(t, v, pad_byte)
            for t, v in self.attributes
            if t not in (MESSAGE_INTEGRITY, FINGERPRINT)
        )
        if integrity_key is not None:
            header = HEADER.pack(mtype, len(body) + 24, MAGIC_COOKIE, self.transaction_id)
            mac = hmac.new(integrity_key, header + body, hashlib.sha1).digest()
            body += _pack_attribute(MESSAGE_INTEGRITY, mac, pad_byte)
        if fingerprint:
            header = HEADER.pack(mtype, len(body) + 8, MAGIC_COOKIE, self.transaction_id)
            crc = (zlib.crc32(header + body) & 0xFFFFFFFF) ^ FINGERPRINT_XOR
            body += _pack_attribute(FINGERPRINT, struct.pack("!I", crc), pad_byte)
        return HEADER.pack(mtype, len(body), MAGIC_COOKIE, self.transaction_id) + body

    @classmethod
    def decode(cls, data: bytes) -> "StunMessage":
        if len(data) < HEADER.size:
            raise StunError("datagram shorter than a STUN header")
        mtype, length, cookie, transaction_id = HEADER.unpack_from(data)
        if mtype & 0xC000 or cookie != MAGIC_COOKIE or length % 4 or len(data) < HEADER.size + length:
            raise StunError("not a STUN message")
        method, klass = split_message_type(mtype)
        message = cls(method, klass, transaction_id)
        offset, end = HEADER.size, HEADER.size + length
        while offset + 4 <= end:
            attr_type, attr_len = struct.unpack_from("!HH", data, offset)
            message.attributes.append((attr_type, data[offset + 4 : offset + 4 + attr_len]))
            offset += 4 + attr_len + (-attr_len % 4)
        return message


def verify_message_integrity(raw: bytes, key: bytes) -> bool:
    """Check MESSAGE-INTEGRITY per RFC 5389 §15.4 (length field covers up to the MAC)."""
    mtype, length, _, _ = HEADER.unpack_from(raw)
    offset, end = HEADER.size, HEADER.size + length
    while offset + 4 <= end:
        attr_type, attr_len = struct.unpack_from("!HH", raw, offset)
        if attr_type == MESSAGE_INTEGRITY:
            header = struct.pack("!HH", mtype, offset + 24 - HEADER.size) + raw[4:HEADER.size]
            expected = hmac.new(key, header + raw[HEADER.size : offset], hashlib.sha1).digest()
            return hmac.compare_digest(expected, raw[offset + 4 : offset + 24])
        offset += 4 + attr_len + (-attr_len % 4)
    return False


def verify_fingerprint(raw: bytes) -> bool:
    """Check the trailing FINGERPRINT attribute (CRC-32 XOR 0x5354554E, RFC 5389 §15.5)."""
    if len(raw) < HEADER.size + 8:
        return False
    attr_type, attr_len = struct.unpack_from("!HH", raw, len(raw) - 8)
    if attr_type != FINGERPRINT or attr_len != 4:
        return False
    expected = (zlib.crc32(raw[:-8]) & 0xFFFFFFFF) ^ FINGERPRINT_XOR
    return struct.unpack_from("!I", raw, len(raw) - 4)[0] == expected


def udp_socket(bind: Address = ("0.0.0.0", 0)) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        # Windows otherwise turns an ICMP port-unreachable into an error on the next recvfrom.
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)
    sock.bind(bind)
    return sock


def resolve(server: ServerSpec) -> Address:
    """Resolve `"host:port"` (or a `(host, port)` tuple) to an IPv4 transport address."""
    if isinstance(server, tuple):
        host, port = server[0], int(server[1])
    else:
        host, _, port_text = server.rpartition(":")
        host, port = host.strip("[]"), int(port_text)
    info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    return info[0][4][0], info[0][4][1]


def outbound_interface_ip(target: Address) -> str:
    """Local IP the OS would use to reach `target` (UDP connect sends no packets)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(target)
        return probe.getsockname()[0]
    finally:
        probe.close()


def stun_transaction(
    sock: socket.socket,
    server: Address,
    request: bytes,
    transaction_id: bytes,
    timeout: float = 3.0,
    initial_rto: float = 0.1,
    on_other: Optional[Callable[[bytes, Address], None]] = None,
) -> Tuple[StunMessage, bytes, float]:
    """Send `request` with RFC 5389 §7.2.1 retransmission (RTO doubling) and await its response.

    Datagrams that are not the matching response are handed to `on_other`, so a TURN client
    sharing the socket does not lose Data indications. Returns `(message, raw, rtt_ms)`.
    """
    deadline = time.monotonic() + timeout
    rto = initial_rto
    next_send = time.monotonic()
    t0 = time.perf_counter()
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise StunError(f"no STUN response from {server[0]}:{server[1]} within {timeout:.1f}s")
        if now >= next_send:
            sock.sendto(request, server)
            next_send = now + rto
            rto *= 2
        wait = max(0.0, min(next_send, deadline) - time.monotonic())
        ready, _, _ = select.select([sock], [], [], wait)
        if not ready:
            continue
        try:
            data, addr = sock.recvfrom(65535)
        except OSError:
            continue
        try:
            message = StunMessage.decode(data)
        except StunError:
            message = None
        if message is None or message.transaction_id != transaction_id:
            if on_other is not None:
                on_other(data, addr)
            continue
        return message, data, (time.perf_counter() - t0) * 1000.0


@dataclass
class StunBindingResult:
    server: Address
    local: Address
    mapped: Address
    rtt_ms: float


def stun_binding(sock: socket.socket, server: ServerSpec, timeout: float = 3.0) -> StunBindingResult:
    """Send an RFC 5389 Binding request and return the server-reflexive address it reports."""
    server_addr = resolve(server)
    request = StunMessage(BINDING, REQUEST).add(SOFTWARE, SOFTWARE_NAME.encode("utf-8"))
    message, _, rtt_ms = stun_transaction(
        sock, server_addr, request.encode(fingerprint=True), request.transaction_id, timeout
    )
    if message.cls == ERROR:
        code, reason = decode_error_code(message.get(ERROR_CODE))
        raise StunError(f"Binding error {code} {reason}", code)
    xor_value = message.get(XOR_MAPPED_ADDRESS)
    plain_value = message.get(MAPPED_ADDRESS)
    if xor_value is not None:
        mapped = decode_xor_address(xor_value, message.transaction_id)
    elif plain_value is not None:
        mapped = decode_plain_address(plain_value)
    else:
        raise StunError("Binding response carried no mapped address")
    return StunBindingResult(server_addr, sock.getsockname(), mapped, rtt_ms)


@dataclass
class NatMapping:
    """RFC 5780 §4.3 mapping behaviour observed for one local socket."""

    behaviour: str
    """`no_nat`, `endpoint_independent`, `address_dependent` or `unknown`."""
    local: Address
    mapped: List[Address]
    servers: List[Address]
    rtt_ms: List[float] = field(default_factory=list)

    @property
    def nat_type(self) -> str:
        return {
            "no_nat": "open",
            "endpoint_independent": "cone",
            "address_dependent": "symmetric",
        }.get(self.behaviour, "unknown")

    @property
    def hole_punchable(self) -> bool:
        return self.behaviour in ("no_nat", "endpoint_independent")


def classify_nat_mapping(
    sock: socket.socket, servers: Sequence[ServerSpec], timeout: float = 2.0
) -> NatMapping:
    """Compare the mappings two STUN servers report for the same local socket."""
    results: List[StunBindingResult] = []
    for server in servers:
        try:
            results.append(stun_binding(sock, server, timeout))
        except (StunError, OSError):
            continue
        if len(results) == 2:
            break
    if not results:
        raise StunError("no STUN server answered")
    local_ip, local_port = sock.getsockname()
    if local_ip in ("0.0.0.0", ""):
        local_ip = outbound_interface_ip(results[0].server)
    mapped = [r.mapped for r in results]
    if mapped[0] == (local_ip, local_port):
        behaviour = "no_nat"
    elif len(mapped) < 2:
        behaviour = "unknown"
    elif mapped[0] == mapped[1]:
        behaviour = "endpoint_independent"
    else:
        behaviour = "address_dependent"
    return NatMapping(
        behaviour, (local_ip, local_port), mapped, [r.server for r in results], [r.rtt_ms for r in results]
    )


PUNCH_MAGIC = b"HMPUNCH1"


@dataclass
class PunchResult:
    succeeded: bool
    peer: Optional[Address]
    rtt_ms: Optional[float]
    probes_sent: int


def udp_hole_punch(
    sock: socket.socket,
    candidates: Sequence[Address],
    session: bytes,
    timeout: float = 2.0,
    interval: float = 0.01,
    linger: float = 0.05,
) -> PunchResult:
    """Simultaneous-open UDP hole punching with ICE-style connectivity checks (RFC 8445 §7.2).

    Both peers call this at the same time with each other's candidates and a shared `session`
    id from signalling. Each side keeps probing every candidate; a probe that gets through is
    answered from the address it arrived on, which is the hole the peer's NAT just opened. A
    side succeeds when it receives that answer, then lingers briefly to answer late probes.
    """
    syn = PUNCH_MAGIC + b"S" + session
    ack = PUNCH_MAGIC + b"A" + session
    deadline = time.monotonic() + timeout
    t0 = time.perf_counter()
    probes = 0
    next_probe = 0.0
    winner: Optional[Address] = None
    rtt_ms: Optional[float] = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if winner is None and now >= next_probe:
            for candidate in candidates:
                sock.sendto(syn, candidate)
                probes += 1
            next_probe = now + interval
        wake = min(next_probe, deadline) if winner is None else deadline
        ready, _, _ = select.select([sock], [], [], max(0.0, wake - time.monotonic()))
        if not ready:
            continue
        try:
            data, addr = sock.recvfrom(2048)
        except OSError:
            continue
        if data == syn:
            sock.sendto(ack, addr)
        elif data == ack and winner is None:
            winner = addr
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            deadline = time.monotonic() + linger
    return PunchResult(winner is not None, winner, rtt_ms, probes)


class TurnClient:
    """RFC 5766 TURN client over UDP using long-term credentials."""

    def __init__(
        self,
        server: ServerSpec,
        username: str,
        password: str,
        sock: Optional[socket.socket] = None,
        timeout: float = 3.0,
    ) -> None:
        self.server = resolve(server)
        self.username = username
        self.password = password
        self.sock = sock if sock is not None else udp_socket()
        self._owns_socket = sock is None
        self.timeout = timeout
        self.realm: Optional[str] = None
        self.nonce: Optional[bytes] = None
        self.key: Optional[bytes] = None
        self.relayed: Optional[Address] = None
        self.mapped: Optional[Address] = None
        self.lifetime = 0
        self._inbox: List[Tuple[Address, bytes]] = []

    def _on_other(self, data: bytes, addr: Address) -> None:
        if addr != self.server:
            return
        try:
            message = StunMessage.decode(data)
        except StunError:
            return
        if message.method == DATA and message.cls == INDICATION:
            peer, payload = message.get(XOR_PEER_ADDRESS), message.get(DATA_ATTR)
            if peer is not None and payload is not None:
                self._inbox.append((decode_xor_address(peer, message.transaction_id), payload))

    def _request(
        self, method: int, attributes: Sequence[Tuple[int, Callable[[bytes], bytes]]]
    ) -> StunMessage:
        retried_stale_nonce = False
        while True:
            request = StunMessage(method, REQUEST)
            for attr_type, build in attributes:
                request.add(attr_type, build(request.transaction_id))
            if self.key is not None and self.realm is not None and self.nonce is not None:
                request.add(USERNAME, self.username.encode("utf-8"))
                request.add(REALM, self.realm.encode("utf-8"))
                request.add(NONCE, self.nonce)
                raw = request.encode(integrity_key=self.key, fingerprint=True)
            else:
                raw = request.encode(fingerprint=True)
            message, data, _ = stun_transaction(
                self.sock, self.server, raw, request.transaction_id, self.timeout, on_other=self._on_other
            )
            if message.cls == SUCCESS:
                if (
                    self.key is not None
                    and message.get(MESSAGE_INTEGRITY) is not None
                    and not verify_message_integrity(data, self.key)
                ):
                    raise StunError("TURN response failed its MESSAGE-INTEGRITY check")
                return message
            code, reason = decode_error_code(message.get(ERROR_CODE))
            fresh_nonce = message.get(NONCE)
            had_credentials = self.key is not None
            if code in (401, 438) and fresh_nonce is not None:
                stale = code == 438 or fresh_nonce != self.nonce
                if had_credentials and code == 401 and not stale:
                    raise StunError("TURN server rejected the credentials (401)", code)
                if code == 438 and retried_stale_nonce:
                    raise StunError("TURN server keeps reporting a stale nonce", code)
                retried_stale_nonce = retried_stale_nonce or code == 438
                realm = message.get(REALM)
                if realm is not None:
                    self.realm = realm.decode("utf-8")
                if self.realm is None:
                    raise StunError("TURN challenge carried no REALM", code)
                self.nonce = fresh_nonce
                self.key = long_term_key(self.username, self.realm, self.password)
                continue
            raise StunError(f"TURN request {method:#05x} failed: {code} {reason}", code)

    def allocate(self, lifetime: int = 600) -> Address:
        message = self._request(
            ALLOCATE,
            [
                (REQUESTED_TRANSPORT, lambda _t: struct.pack("!B3x", UDP_PROTOCOL_NUMBER)),
                (LIFETIME, lambda _t: struct.pack("!I", lifetime)),
            ],
        )
        relayed = message.get(XOR_RELAYED_ADDRESS)
        if relayed is None:
            raise StunError("Allocate success carried no XOR-RELAYED-ADDRESS")
        self.relayed = decode_xor_address(relayed, message.transaction_id)
        mapped = message.get(XOR_MAPPED_ADDRESS)
        if mapped is not None:
            self.mapped = decode_xor_address(mapped, message.transaction_id)
        granted = message.get(LIFETIME)
        self.lifetime = struct.unpack("!I", granted)[0] if granted else lifetime
        return self.relayed

    def create_permission(self, peer_ip: str) -> None:
        self._request(
            CREATE_PERMISSION,
            [(XOR_PEER_ADDRESS, lambda txid: encode_xor_address((peer_ip, 0), txid))],
        )

    def refresh(self, lifetime: int = 600) -> int:
        message = self._request(REFRESH, [(LIFETIME, lambda _t: struct.pack("!I", lifetime))])
        granted = message.get(LIFETIME)
        self.lifetime = struct.unpack("!I", granted)[0] if granted else lifetime
        return self.lifetime

    def send_to(self, peer: Address, payload: bytes) -> None:
        """Relay `payload` to `peer` through the allocation (Send indication, RFC 5766 §10.1)."""
        indication = StunMessage(SEND, INDICATION)
        indication.add(XOR_PEER_ADDRESS, encode_xor_address(peer, indication.transaction_id))
        indication.add(DATA_ATTR, payload)
        self.sock.sendto(indication.encode(fingerprint=True), self.server)

    def recv_from(self, timeout: float = 2.0) -> Tuple[Address, bytes]:
        """Wait for the next Data indication and return `(peer, payload)`."""
        deadline = time.monotonic() + timeout
        while not self._inbox:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StunError("no relayed data before the timeout")
            ready, _, _ = select.select([self.sock], [], [], remaining)
            if not ready:
                continue
            try:
                data, addr = self.sock.recvfrom(65535)
            except OSError:
                continue
            self._on_other(data, addr)
        return self._inbox.pop(0)

    def close(self) -> None:
        """Release the allocation (Refresh with LIFETIME 0) and close an owned socket."""
        try:
            if self.relayed is not None:
                self.refresh(0)
        except (StunError, OSError):
            pass
        finally:
            self.relayed = None
            if self._owns_socket:
                self.sock.close()


class StunServer:
    """RFC 5389 Binding responder serving on a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.sock = udp_socket((host, port))
        self.address: Address = self.sock.getsockname()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StunServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"{self.address[0]}:{self.address[1]}"

    def start(self) -> "StunServer":
        self._thread = threading.Thread(target=self._serve, name=type(self).__name__, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_sockets()

    def _close_sockets(self) -> None:
        self.sock.close()

    def _sockets(self) -> List[socket.socket]:
        return [self.sock]

    def _serve(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select(self._sockets(), [], [], 0.05)
            for sock in ready:
                self._on_readable(sock)
            self._housekeeping()

    def _housekeeping(self) -> None:
        pass

    def _on_readable(self, sock: socket.socket) -> None:
        try:
            data, addr = sock.recvfrom(65535)
        except OSError:
            return
        self.handle_datagram(data, addr)

    def mapped_address_for(self, addr: Address) -> Address:
        """The reflexive address reported to `addr` (the address the datagram came from)."""
        return addr

    def handle_datagram(self, data: bytes, addr: Address) -> None:
        try:
            message = StunMessage.decode(data)
        except StunError:
            return
        if message.method == BINDING and message.cls == REQUEST:
            response = StunMessage(BINDING, SUCCESS, message.transaction_id)
            response.add(
                XOR_MAPPED_ADDRESS,
                encode_xor_address(self.mapped_address_for(addr), message.transaction_id),
            )
            response.add(SOFTWARE, SOFTWARE_NAME.encode("utf-8"))
            self.sock.sendto(response.encode(fingerprint=True), addr)


@dataclass
class _Allocation:
    client: Address
    username: str
    relay: socket.socket
    expires: float
    permissions: Dict[str, float] = field(default_factory=dict)

    def permits(self, ip: str) -> bool:
        return self.permissions.get(ip, 0.0) > time.monotonic()


class TurnRelayServer(StunServer):
    """Minimal RFC 5766 TURN relay (UDP allocations, permissions, Send/Data indications).

    Also answers Binding requests, like production TURN servers do. Intended for HiveMind nodes
    that volunteer as relays; channels (ChannelBind) and TCP allocations are not implemented.
    """

    def __init__(
        self,
        users: Dict[str, str],
        host: str = "127.0.0.1",
        port: int = 0,
        realm: str = "hivemind.local",
        max_lifetime: int = 3600,
    ) -> None:
        super().__init__(host, port)
        self.users = dict(users)
        self.realm = realm
        self.max_lifetime = max_lifetime
        self.nonce = secrets.token_hex(16).encode("ascii")
        self.allocations: Dict[Address, _Allocation] = {}
        self._by_relay: Dict[socket.socket, _Allocation] = {}

    def _sockets(self) -> List[socket.socket]:
        return [self.sock, *self._by_relay]

    def _close_sockets(self) -> None:
        for allocation in list(self.allocations.values()):
            allocation.relay.close()
        self.allocations.clear()
        self._by_relay.clear()
        super()._close_sockets()

    def _housekeeping(self) -> None:
        now = time.monotonic()
        for client, allocation in list(self.allocations.items()):
            if allocation.expires <= now:
                self._release(client)

    def _release(self, client: Address) -> None:
        allocation = self.allocations.pop(client, None)
        if allocation is not None:
            self._by_relay.pop(allocation.relay, None)
            allocation.relay.close()

    def _on_readable(self, sock: socket.socket) -> None:
        if sock is self.sock:
            super()._on_readable(sock)
            return
        allocation = self._by_relay.get(sock)
        try:
            payload, peer = sock.recvfrom(65535)
        except OSError:
            return
        if allocation is None or not allocation.permits(peer[0]):
            return  # RFC 5766 §10.3: drop data from peers without a permission
        indication = StunMessage(DATA, INDICATION)
        indication.add(XOR_PEER_ADDRESS, encode_xor_address(peer, indication.transaction_id))
        indication.add(DATA_ATTR, payload)
        self.sock.sendto(indication.encode(fingerprint=True), allocation.client)

    def handle_datagram(self, data: bytes, addr: Address) -> None:
        try:
            message = StunMessage.decode(data)
        except StunError:
            return
        if message.cls == REQUEST and message.method == BINDING:
            super().handle_datagram(data, addr)
        elif message.cls == INDICATION and message.method == SEND:
            self._on_send(message, addr)
        elif message.cls == REQUEST and message.method in (ALLOCATE, REFRESH, CREATE_PERMISSION):
            self._on_authenticated_request(message, data, addr)

    def _send_error(
        self, request: StunMessage, addr: Address, code: int, reason: str, challenge: bool = False
    ) -> None:
        response = StunMessage(request.method, ERROR, request.transaction_id)
        response.add(ERROR_CODE, encode_error_code(code, reason))
        if challenge:
            response.add(REALM, self.realm.encode("utf-8"))
            response.add(NONCE, self.nonce)
        self.sock.sendto(response.encode(fingerprint=True), addr)

    def _on_authenticated_request(self, request: StunMessage, raw: bytes, addr: Address) -> None:
        username = request.get(USERNAME)
        if request.get(MESSAGE_INTEGRITY) is None or username is None:
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        user = username.decode("utf-8", "replace")
        if user not in self.users:
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        if request.get(NONCE) != self.nonce:
            self._send_error(request, addr, 438, "Stale Nonce", challenge=True)
            return
        key = long_term_key(user, self.realm, self.users[user])
        if not verify_message_integrity(raw, key):
            self._send_error(request, addr, 401, "Unauthorized", challenge=True)
            return
        handler = {
            ALLOCATE: self._allocate,
            REFRESH: self._refresh,
            CREATE_PERMISSION: self._create_permission,
        }[request.method]
        response = handler(request, addr, user)
        if response is not None:
            response.add(SOFTWARE, SOFTWARE_NAME.encode("utf-8"))
            self.sock.sendto(response.encode(integrity_key=key, fingerprint=True), addr)

    def _requested_lifetime(self, request: StunMessage) -> int:
        value = request.get(LIFETIME)
        requested = struct.unpack("!I", value)[0] if value else 600
        return min(requested, self.max_lifetime)

    def _allocate(self, request: StunMessage, addr: Address, user: str) -> Optional[StunMessage]:
        if addr in self.allocations:
            self._send_error(request, addr, 437, "Allocation Mismatch")
            return None
        transport = request.get(REQUESTED_TRANSPORT)
        if transport is None or transport[0] != UDP_PROTOCOL_NUMBER:
            self._send_error(request, addr, 442, "Unsupported Transport Protocol")
            return None
        lifetime = self._requested_lifetime(request)
        relay = udp_socket((self.address[0], 0))
        allocation = _Allocation(addr, user, relay, time.monotonic() + lifetime)
        self.allocations[addr] = allocation
        self._by_relay[relay] = allocation
        response = StunMessage(ALLOCATE, SUCCESS, request.transaction_id)
        response.add(XOR_RELAYED_ADDRESS, encode_xor_address(relay.getsockname(), request.transaction_id))
        response.add(LIFETIME, struct.pack("!I", lifetime))
        response.add(XOR_MAPPED_ADDRESS, encode_xor_address(addr, request.transaction_id))
        return response

    def _refresh(self, request: StunMessage, addr: Address, user: str) -> Optional[StunMessage]:
        allocation = self.allocations.get(addr)
        if allocation is None:
            self._send_error(request, addr, 437, "Allocation Mismatch")
            return None
        lifetime = self._requested_lifetime(request)
        if lifetime == 0:
            self._release(addr)
        else:
            allocation.expires = time.monotonic() + lifetime
        response = StunMessage(REFRESH, SUCCESS, request.transaction_id)
        response.add(LIFETIME, struct.pack("!I", lifetime))
        return response

    def _create_permission(
        self, request: StunMessage, addr: Address, user: str
    ) -> Optional[StunMessage]:
        allocation = self.allocations.get(addr)
        peer = request.get(XOR_PEER_ADDRESS)
        if allocation is None:
            self._send_error(request, addr, 437, "Allocation Mismatch")
            return None
        if peer is None:
            self._send_error(request, addr, 400, "Bad Request")
            return None
        peer_ip = decode_xor_address(peer, request.transaction_id)[0]
        allocation.permissions[peer_ip] = time.monotonic() + PERMISSION_LIFETIME_S
        return StunMessage(CREATE_PERMISSION, SUCCESS, request.transaction_id)

    def _on_send(self, indication: StunMessage, addr: Address) -> None:
        allocation = self.allocations.get(addr)
        peer = indication.get(XOR_PEER_ADDRESS)
        payload = indication.get(DATA_ATTR)
        if allocation is None or peer is None or payload is None:
            return
        peer_addr = decode_xor_address(peer, indication.transaction_id)
        if allocation.permits(peer_addr[0]):
            allocation.relay.sendto(payload, peer_addr)
