"""Offline STUN/TURN security regressions over real loopback UDP sockets."""

from __future__ import annotations

import random
import secrets
import struct
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

from hivemind.swarm import nat_traversal as nt


USER, PASSWORD, REALM = "alice", "test-secret", "test.local"
KEY = nt.long_term_key(USER, REALM, PASSWORD)


def _append_attribute(raw: bytes, attr: int, value: bytes) -> bytes:
    body = raw[nt.HEADER.size :] + struct.pack("!HH", attr, len(value)) + value
    body += b"\x00" * (-len(value) % 4)
    return raw[:2] + struct.pack("!H", len(body)) + raw[4:nt.HEADER.size] + body


def _authenticated_request(method: int, nonce: bytes, txid: bytes | None = None) -> nt.StunMessage:
    message = nt.StunMessage(method, nt.REQUEST, txid or secrets.token_bytes(12))
    if method == nt.ALLOCATE:
        message.add(nt.REQUESTED_TRANSPORT, struct.pack("!B3x", nt.UDP_PROTOCOL_NUMBER))
    return message.add(nt.USERNAME, USER.encode()).add(nt.REALM, REALM.encode()).add(nt.NONCE, nonce)


@contextmanager
def _scripted_server(
    handler: Callable[[nt.StunServer, bytes, nt.Address], None],
) -> Iterator[nt.StunServer]:
    failures: list[Exception] = []

    class ScriptedServer(nt.StunServer):
        def handle_datagram(self, data: bytes, addr: nt.Address) -> None:
            try:
                handler(self, data, addr)
            except Exception as exc:
                failures.append(exc)

    with ScriptedServer() as server:
        yield server
    assert not failures, failures


def _primed_client(server: nt.Address, timeout: float = 0.6) -> nt.TurnClient:
    client = nt.TurnClient(server, USER, PASSWORD, timeout=timeout)
    client.realm, client.nonce, client.key = REALM, b"nonce", KEY
    return client


@pytest.mark.parametrize(
    "attr,value",
    [
        (nt.LIFETIME, b""), (nt.LIFETIME, b"\x00" * 3), (nt.LIFETIME, b"\x00" * 8),
        (nt.REQUESTED_TRANSPORT, b""), (nt.REQUESTED_TRANSPORT, b"\x11"),
        (nt.XOR_PEER_ADDRESS, b""), (nt.XOR_PEER_ADDRESS, b"\x00\x01\x00\x00"),
        (nt.XOR_MAPPED_ADDRESS, b"\x00\x02" + b"\x00" * 6),
        (nt.MAPPED_ADDRESS, b"\x00\x03" + b"\x00" * 6),
        (nt.ERROR_CODE, b"\x00"), (nt.ERROR_CODE, b"\x00\x00\x04\xff"),
    ],
)
def test_decoder_rejects_invalid_known_attribute_lengths(attr: int, value: bytes) -> None:
    raw = nt.StunMessage(nt.ALLOCATE, nt.REQUEST).add(attr, value).encode()
    with pytest.raises(nt.StunError):
        nt.StunMessage.decode(raw)
    assert not nt.verify_message_integrity(raw, KEY)


@pytest.mark.parametrize("size", [0, 1, 3, 19, 21, 24])
def test_integrity_attribute_requires_exact_sha1_length(size: int) -> None:
    raw = _append_attribute(
        nt.StunMessage(nt.REFRESH, nt.SUCCESS).encode(), nt.MESSAGE_INTEGRITY, b"x" * size
    )
    with pytest.raises(nt.StunError):
        nt.StunMessage.decode(raw)
    assert not nt.verify_message_integrity(raw, KEY)


def test_decoder_checks_declared_bounds_and_fingerprint() -> None:
    raw = nt.StunMessage(nt.BINDING, nt.SUCCESS).encode(fingerprint=True)
    bad_length = raw[:nt.HEADER.size + 2] + struct.pack("!H", 100) + raw[nt.HEADER.size + 4:]
    corrupted = raw[:-1] + bytes([raw[-1] ^ 1])
    for malformed in (b"", raw[:19], raw[:-1], raw + b"trailing", bad_length, corrupted):
        with pytest.raises(nt.StunError):
            nt.StunMessage.decode(malformed)
        assert not nt.verify_message_integrity(malformed, KEY)
        assert not nt.verify_fingerprint(malformed)


def test_post_integrity_attributes_cannot_change_semantics() -> None:
    message = nt.StunMessage(nt.REFRESH, nt.SUCCESS).add(nt.LIFETIME, struct.pack("!I", 42))
    signed = message.encode(integrity_key=KEY)
    tampered = _append_attribute(signed, nt.LIFETIME, struct.pack("!I", 999))
    tampered = _append_attribute(tampered, nt.REALM, b"attacker.example")
    tampered = _append_attribute(tampered, nt.XOR_RELAYED_ADDRESS, b"malformed and ignored")
    assert nt.verify_message_integrity(tampered, KEY)
    decoded = nt.StunMessage.decode(tampered)
    assert decoded.get(nt.LIFETIME) == struct.pack("!I", 42)
    assert decoded.get(nt.REALM) is None
    assert decoded.get(nt.XOR_RELAYED_ADDRESS) is None
    assert len([value for attr, value in decoded.attributes if attr == nt.LIFETIME]) == 1


@pytest.mark.parametrize(
    "attack",
    ["wrong_source", "wrong_method", "wrong_class", "wrong_transaction", "unsigned_success",
     "bad_integrity", "unsigned_error", "bad_fingerprint", "malformed_attribute", "changed_realm"],
)
def test_authenticated_transaction_discards_invalid_replies_and_waits_for_real_reply(attack: str) -> None:
    observed: list[bytes] = []

    def respond(server: nt.StunServer, raw: bytes, addr: nt.Address) -> None:
        request = nt.StunMessage.decode(raw)
        assert nt.verify_message_integrity(raw, KEY)
        observed.append(request.transaction_id)
        forged = nt.StunMessage(nt.REFRESH, nt.SUCCESS, request.transaction_id)
        forged.add(nt.LIFETIME, struct.pack("!I", 999))
        if attack == "wrong_method":
            forged.method = nt.ALLOCATE
        elif attack == "wrong_class":
            forged.cls = nt.REQUEST
        elif attack == "wrong_transaction":
            forged.transaction_id = secrets.token_bytes(12)
        elif attack in ("unsigned_error", "changed_realm"):
            forged.cls = nt.ERROR
            code = 438 if attack == "changed_realm" else 437
            forged.add(nt.ERROR_CODE, nt.encode_error_code(code, "forged"))
            if attack == "changed_realm":
                forged.add(nt.REALM, b"other.realm").add(nt.NONCE, b"new-nonce")
        elif attack == "malformed_attribute":
            forged.attributes = [(nt.LIFETIME, b"\x00")]
        key = None if attack in ("unsigned_success", "unsigned_error", "changed_realm") else KEY
        if attack == "bad_integrity":
            key = b"wrong-key"
        forged_raw = forged.encode(integrity_key=key, fingerprint=True)
        if attack == "bad_fingerprint":
            forged_raw = forged_raw[:-1] + bytes([forged_raw[-1] ^ 1])
        if attack == "wrong_source":
            with nt.udp_socket(("127.0.0.1", 0)) as attacker:
                attacker.sendto(forged_raw, addr)
        else:
            server.sock.sendto(forged_raw, addr)
        valid = nt.StunMessage(nt.REFRESH, nt.SUCCESS, request.transaction_id)
        valid.add(nt.LIFETIME, struct.pack("!I", 42))
        server.sock.sendto(valid.encode(integrity_key=KEY, fingerprint=True), addr)

    with _scripted_server(respond) as server:
        client = _primed_client(server.address)
        try:
            assert client.refresh(42) == 42
            assert len(observed) == 1
        finally:
            client.close()


@pytest.mark.parametrize("cls", [nt.SUCCESS, nt.ERROR])
def test_unsigned_authenticated_response_alone_does_not_end_transaction(cls: int) -> None:
    requests: list[bytes] = []

    def respond(server: nt.StunServer, raw: bytes, addr: nt.Address) -> None:
        request = nt.StunMessage.decode(raw)
        requests.append(request.transaction_id)
        reply = nt.StunMessage(request.method, cls, request.transaction_id)
        reply.add(nt.LIFETIME, struct.pack("!I", 999))
        if cls == nt.ERROR:
            reply.add(nt.ERROR_CODE, nt.encode_error_code(437, "forged"))
        server.sock.sendto(reply.encode(fingerprint=True), addr)

    with _scripted_server(respond) as server:
        client = _primed_client(server.address, timeout=0.16)
        try:
            with pytest.raises(nt.StunError, match="no STUN response"):
                client.refresh()
            assert len(requests) >= 2
            assert len(set(requests)) == 1
        finally:
            client.close()


def test_changing_401_nonce_cannot_create_unbounded_authentication_retries() -> None:
    requests: list[bytes] = []

    def respond(server: nt.StunServer, raw: bytes, addr: nt.Address) -> None:
        request = nt.StunMessage.decode(raw)
        requests.append(request.transaction_id)
        reply = nt.StunMessage(request.method, nt.ERROR, request.transaction_id)
        reply.add(nt.ERROR_CODE, nt.encode_error_code(401, "Unauthorized"))
        reply.add(nt.REALM, REALM.encode()).add(nt.NONCE, secrets.token_bytes(16))
        server.sock.sendto(reply.encode(fingerprint=True), addr)

    with _scripted_server(respond) as server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.3)
        try:
            with pytest.raises(nt.StunError) as caught:
                client.allocate()
            assert caught.value.code == 401
            assert len(requests) == 2
        finally:
            client.close()


def test_stale_nonce_challenges_are_bounded() -> None:
    requests: list[bytes] = []

    def respond(server: nt.StunServer, raw: bytes, addr: nt.Address) -> None:
        request = nt.StunMessage.decode(raw)
        requests.append(request.transaction_id)
        reply = nt.StunMessage(request.method, nt.ERROR, request.transaction_id)
        reply.add(nt.ERROR_CODE, nt.encode_error_code(438, "Stale Nonce"))
        reply.add(nt.REALM, REALM.encode()).add(nt.NONCE, secrets.token_bytes(16))
        server.sock.sendto(reply.encode(fingerprint=True), addr)

    with _scripted_server(respond) as server:
        client = _primed_client(server.address)
        try:
            with pytest.raises(nt.StunError, match="keeps reporting") as caught:
                client.refresh()
            assert caught.value.code == 438
            assert len(requests) == 2
        finally:
            client.close()


class _DropFirstSuccess:
    """Drop a real server UDP send once, leaving its retransmission path intact."""

    def __init__(self, sock: object, method: int) -> None:
        self.sock, self.method = sock, method
        self.successes: list[bytes] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.sock, name)

    def sendto(self, raw: bytes, addr: nt.Address) -> int:
        message = nt.StunMessage.decode(raw)
        if message.method == self.method and message.cls == nt.SUCCESS:
            self.successes.append(raw)
            if len(self.successes) == 1:
                return len(raw)
        return self.sock.sendto(raw, addr)


@pytest.mark.parametrize("method", [nt.ALLOCATE, nt.REFRESH])
def test_lost_success_replays_exact_response_without_reexecuting_request(method: int) -> None:
    server = nt.TurnRelayServer({USER: PASSWORD}, realm=REALM)
    lossy = _DropFirstSuccess(server.sock, method)
    server.sock = lossy
    with server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        try:
            relay = client.allocate()
            assert len(server.allocations) == 1
            if method == nt.REFRESH:
                assert client.refresh(0) == 0
                assert not server.allocations
                client.relayed = None
            else:
                assert next(iter(server.allocations.values())).relay.getsockname() == relay
            assert len(lossy.successes) == 2
            assert lossy.successes[0] == lossy.successes[1]
        finally:
            client.close()


def test_response_cache_is_bounded_expires_and_rejects_changed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nt, "TRANSACTION_CACHE_LIMIT", 3)
    server = nt.TurnRelayServer({USER: PASSWORD}, realm=REALM)
    client = nt.udp_socket(("127.0.0.1", 0))
    client.settimeout(0.15)
    try:
        for _ in range(5):
            request = _authenticated_request(nt.REFRESH, server.nonce)
            raw = request.encode(integrity_key=KEY)
            server.handle_datagram(raw, client.getsockname())
            response, _ = client.recvfrom(65535)
            assert nt.verify_message_integrity(response, KEY)
        assert len(server._responses) == 3
        request.add(nt.LIFETIME, struct.pack("!I", 123))
        server.handle_datagram(request.encode(integrity_key=KEY), client.getsockname())
        with pytest.raises(TimeoutError):
            client.recvfrom(65535)
        for cache_key, (_, digest, response) in list(server._responses.items()):
            server._responses[cache_key] = (time.monotonic() - 1, digest, response)
        server._housekeeping()
        assert not server._responses
    finally:
        client.close()
        server.stop()


def test_authenticated_operation_errors_are_signed_and_allocation_owner_is_enforced() -> None:
    with nt.TurnRelayServer({USER: PASSWORD, "bob": "bob-secret"}, realm=REALM) as server:
        alice = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        try:
            alice.allocate()
            with pytest.raises(nt.StunError) as duplicate:
                alice.allocate()
            assert duplicate.value.code == 437
            bob = nt.TurnClient(server.address, "bob", "bob-secret", sock=alice.sock, timeout=0.6)
            bob.realm, bob.nonce = REALM, server.nonce
            bob.key = nt.long_term_key("bob", REALM, "bob-secret")
            for action in (lambda: bob.refresh(0), lambda: bob.create_permission("127.0.0.1")):
                with pytest.raises(nt.StunError) as caught:
                    action()
                assert caught.value.code == 441
            assert len(server.allocations) == 1
            assert alice.refresh(120) == 120
        finally:
            alice.close()


def test_malformed_datagrams_do_not_kill_server_or_create_allocation() -> None:
    with nt.TurnRelayServer({USER: PASSWORD}, realm=REALM) as server:
        with nt.udp_socket(("127.0.0.1", 0)) as client:
            for attr, value in ((nt.LIFETIME, b"\x00"), (nt.REQUESTED_TRANSPORT, b""),
                                (nt.XOR_PEER_ADDRESS, b"\x00\x02")):
                request = _authenticated_request(nt.ALLOCATE, server.nonce).add(attr, value)
                client.sendto(request.encode(integrity_key=KEY), server.address)
            assert nt.stun_binding(client, server.address, timeout=0.6).mapped == client.getsockname()
            assert not server.allocations
            assert server._thread.is_alive()


@pytest.mark.parametrize("value", [b"", b"\x00", b"\x00\x01\x00\x00", b"\x00\x02" + b"\x00" * 6])
def test_address_decoders_report_stun_errors(value: bytes) -> None:
    for decode in (lambda: nt.decode_plain_address(value), lambda: nt.decode_xor_address(value, b"x" * 12)):
        with pytest.raises(nt.StunError):
            decode()


def test_wildcard_listener_advertises_usable_relay_address() -> None:
    with nt.TurnRelayServer({USER: PASSWORD}, host="0.0.0.0", realm=REALM) as server:
        client = nt.TurnClient(("127.0.0.1", server.address[1]), USER, PASSWORD, timeout=0.6)
        with nt.udp_socket(("127.0.0.1", 0)) as peer:
            peer.settimeout(0.6)
            try:
                relay = client.allocate()
                assert relay[0] == "127.0.0.1"
                client.create_permission("127.0.0.1")
                client.send_to(peer.getsockname(), b"usable-relay")
                assert peer.recvfrom(1024) == (b"usable-relay", relay)
            finally:
                client.close()


@pytest.mark.parametrize("advertised_ip", ["0.0.0.0", "::", "::1", "224.0.0.1", "255.255.255.255", "invalid"])
def test_invalid_relay_advertisement_is_rejected(advertised_ip: str) -> None:
    with pytest.raises(ValueError):
        nt.TurnRelayServer({USER: PASSWORD}, advertised_ip=advertised_ip)


def test_explicit_relay_advertisement_is_used() -> None:
    with nt.TurnRelayServer({USER: PASSWORD}, realm=REALM, advertised_ip="192.0.2.55") as server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        try:
            assert client.allocate()[0] == "192.0.2.55"
        finally:
            client.close()


@pytest.mark.parametrize("forged_cls,code", [(nt.SUCCESS, None), (nt.ERROR, 438)])
def test_initial_unsigned_success_or_stale_challenge_cannot_skip_authentication(
    forged_cls: int, code: int | None,
) -> None:
    class InjectingRelay(nt.TurnRelayServer):
        def handle_datagram(self, data: bytes, addr: nt.Address) -> None:
            request = nt.StunMessage.decode(data)
            if request.get(nt.MESSAGE_INTEGRITY) is None:
                forged = nt.StunMessage(request.method, forged_cls, request.transaction_id)
                if code is not None:
                    forged.add(nt.ERROR_CODE, nt.encode_error_code(code, "forged"))
                    forged.add(nt.REALM, REALM.encode()).add(nt.NONCE, b"forged-nonce")
                self.sock.sendto(forged.encode(fingerprint=True), addr)
            super().handle_datagram(data, addr)

    with InjectingRelay({USER: PASSWORD}, realm=REALM) as server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        try:
            assert client.allocate()[0] == "127.0.0.1"
            assert client.nonce == server.nonce
        finally:
            client.close()


@pytest.mark.parametrize("method", [nt.REFRESH, nt.CREATE_PERMISSION])
def test_server_cannot_act_on_attributes_appended_after_integrity(method: int) -> None:
    with nt.TurnRelayServer({USER: PASSWORD}, realm=REALM) as server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        try:
            client.allocate()
            request = _authenticated_request(method, server.nonce)
            signed = request.encode(integrity_key=KEY)
            if method == nt.REFRESH:
                raw = _append_attribute(signed, nt.LIFETIME, struct.pack("!I", 0))
            else:
                raw = _append_attribute(
                    signed, nt.XOR_PEER_ADDRESS,
                    nt.encode_xor_address(("127.0.0.1", 0), request.transaction_id),
                )
            reply, response, _ = nt.stun_transaction(
                client.sock, server.address, raw, request.transaction_id, timeout=0.6,
                response_validator=client._valid_response,
            )
            assert nt.verify_message_integrity(response, KEY)
            assert len(server.allocations) == 1
            allocation = next(iter(server.allocations.values()))
            assert not allocation.permissions
            if method == nt.REFRESH:
                assert reply.cls == nt.SUCCESS
                assert reply.get(nt.LIFETIME) == struct.pack("!I", 600)
            else:
                assert reply.cls == nt.ERROR
                assert nt.decode_error_code(reply.get(nt.ERROR_CODE))[0] == 400
        finally:
            client.close()


def test_random_wire_attributes_never_raise_non_protocol_exceptions() -> None:
    rng = random.Random(731)
    for _ in range(1000):
        body = rng.randbytes(rng.randrange(65) * 4)
        raw = nt.HEADER.pack(nt.message_type(nt.ALLOCATE, nt.REQUEST), len(body), nt.MAGIC_COOKIE, b"t" * 12)
        raw += body
        try:
            nt.StunMessage.decode(raw)
        except nt.StunError:
            pass
        assert isinstance(nt.verify_message_integrity(raw, KEY), bool)
        assert isinstance(nt.verify_fingerprint(raw), bool)


def test_data_indication_queue_is_bounded_and_malformed_addresses_are_ignored() -> None:
    with nt.StunServer() as server:
        client = _primed_client(server.address)
        try:
            malformed = nt.StunMessage(nt.DATA, nt.INDICATION)
            malformed.add(nt.XOR_PEER_ADDRESS, b"\x00").add(nt.DATA_ATTR, b"bad")
            client._on_other(malformed.encode(), server.address)
            assert not client._inbox
            valid = nt.StunMessage(nt.DATA, nt.INDICATION)
            valid.add(nt.XOR_PEER_ADDRESS, nt.encode_xor_address(("127.0.0.1", 9000), valid.transaction_id))
            valid.add(nt.DATA_ATTR, b"payload")
            raw = valid.encode(fingerprint=True)
            for _ in range(nt.MAX_PENDING_INDICATIONS + 10):
                client._on_other(raw, server.address)
            assert len(client._inbox) == nt.MAX_PENDING_INDICATIONS
        finally:
            client.close()


def test_ipv6_permission_on_ipv4_relay_returns_signed_error() -> None:
    with nt.TurnRelayServer({USER: PASSWORD}, realm=REALM) as server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        try:
            client.allocate()
            with pytest.raises(nt.StunError) as caught:
                client.create_permission("::1")
            assert caught.value.code == 443
            assert client.refresh(120) == 120
        finally:
            client.close()


def test_oversized_peer_datagram_does_not_stop_relay() -> None:
    with nt.TurnRelayServer({USER: PASSWORD}, realm=REALM) as server:
        client = nt.TurnClient(server.address, USER, PASSWORD, timeout=0.6)
        with nt.udp_socket(("127.0.0.1", 0)) as peer:
            try:
                relay = client.allocate()
                client.create_permission("127.0.0.1")
                # The inbound UDP payload fits, but adding TURN framing would exceed
                # IPv4's UDP payload limit. A failed send must not kill the server.
                peer.sendto(b"x" * 65507, relay)
                peer.sendto(b"still-running", relay)
                assert client.recv_from(timeout=0.6) == (peer.getsockname(), b"still-running")
                assert client.refresh(120) == 120
            finally:
                client.close()
