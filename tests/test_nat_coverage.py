"""Deterministic NAT protocol edges using codecs and loopback UDP boundaries."""

from __future__ import annotations

from contextlib import ExitStack
import socket
import struct

import pytest

from hivemind.swarm import nat_traversal as nt


class _ReplyServer(nt.StunServer):
    """A controlled remote endpoint; the client protocol remains real."""

    def __init__(self, reply):
        super().__init__()
        self.reply = reply
        self.requests = 0

    def handle_datagram(self, data, addr):
        request = nt.StunMessage.decode(data)
        self.requests += 1
        response, key = self.reply(request)
        self.sock.sendto(response.encode(integrity_key=key, fingerprint=True), addr)


class _ReceiveFailureOnce:
    """An OS socket boundary that raises once without consuming its datagram."""

    def __init__(self, sock):
        self.sock = sock
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def recvfrom(self, size):
        if not self.failed:
            self.failed = True
            raise OSError("transient receive failure")
        return self.sock.recvfrom(size)


@pytest.fixture
def udp():
    with nt.udp_socket(("127.0.0.1", 0)) as sock:
        sock.settimeout(0.5)
        yield sock


@pytest.fixture
def relay():
    server = nt.TurnRelayServer({"alice": "secret", "bob": "other"})
    try:
        yield server
    finally:
        server.stop()


def _exchange(server, sock, method, attributes=(), *, user=b"alice", omit=(), key=None):
    request = nt.StunMessage(method, nt.REQUEST)
    credentials = ((nt.USERNAME, user), (nt.REALM, server.realm.encode()), (nt.NONCE, server.nonce))
    for kind, value in (*credentials, *attributes):
        if kind not in omit:
            request.add(kind, value)
    if key is None:
        key = nt.long_term_key("alice", server.realm, "secret")
    server.handle_datagram(request.encode(integrity_key=key, fingerprint=True), sock.getsockname())
    raw, source = sock.recvfrom(65535)
    response = nt.StunMessage.decode(raw)
    assert source == server.address
    assert (response.method, response.transaction_id) == (method, request.transaction_id)
    return response, raw


def _allocate(server, sock):
    response, raw = _exchange(server, sock, nt.ALLOCATE, [(nt.REQUESTED_TRANSPORT, b"\x11\0\0\0")])
    assert response.cls == nt.SUCCESS
    assert nt.verify_message_integrity(raw, nt.long_term_key("alice", server.realm, "secret"))
    return server.allocations[sock.getsockname()]


def _bind(server, sock, peer, number=0x4000):
    # IPv4's XOR mask is independent of the transaction ID supplied by _exchange.
    response, raw = _exchange(
        server, sock, nt.CHANNEL_BIND,
        [(nt.CHANNEL_NUMBER, struct.pack("!HH", number, 0)),
         (nt.XOR_PEER_ADDRESS, nt.encode_xor_address(peer, b"\0" * 12))],
    )
    assert response.cls == nt.SUCCESS
    assert nt.verify_message_integrity(raw, nt.long_term_key("alice", server.realm, "secret"))
    return server.channels[(sock.getsockname(), number)]


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_encode_channel_data_rejects_non_integer_channel(value):
    with pytest.raises(TypeError, match="must be an integer"):
        nt.encode_channel_data(value, b"payload")


@pytest.mark.parametrize("value", [-1, 65536])
def test_encode_channel_data_rejects_channel_overflow(value):
    with pytest.raises(ValueError, match="16 bits"):
        nt.encode_channel_data(value, b"payload")


def test_encode_channel_data_rejects_payload_overflow():
    with pytest.raises(ValueError, match="payload exceeds"):
        nt.encode_channel_data(0x4000, b"x" * 65536)


@pytest.mark.parametrize("value", [True, None, 1.5])
def test_encode_channel_number_rejects_non_integer(value):
    with pytest.raises(ValueError, match="must be an integer"):
        nt.encode_channel_number(value)


@pytest.mark.parametrize("wire", [b"", bytes.fromhex("00000000"), bytes.fromhex("4000000261")])
def test_decode_channel_data_rejects_invalid_or_truncated_frames(wire):
    with pytest.raises(nt.StunError, match="ChannelData message"):
        nt.decode_channel_data(wire)


@pytest.mark.parametrize(
    "wire,expected",
    [("00010000c0000201", ("192.0.2.1", 0)),
     ("0002ffff20010db8000000000000000000000001", ("2001:db8::1", 65535))],
)
def test_plain_address_decoder_preserves_ipv4_ipv6_and_port_boundaries(wire, expected):
    assert nt.decode_plain_address(bytes.fromhex(wire)) == expected


def test_fingerprint_must_be_last_wire_attribute():
    body = struct.pack("!HHIHH", nt.FINGERPRINT, 4, 0, nt.SOFTWARE, 0)
    wire = nt.HEADER.pack(0x0101, len(body), nt.MAGIC_COOKIE, b"x" * 12) + body
    with pytest.raises(nt.StunError, match="final four-byte attribute"):
        nt.StunMessage.decode(wire)


@pytest.mark.parametrize("size", [0, 11, 13])
def test_xor_address_requires_twelve_byte_transaction_id(size):
    with pytest.raises(nt.StunError, match="transaction ID length"):
        nt.decode_xor_address(bytes.fromhex("00010000c0000201"), b"x" * size)


def test_absent_error_attribute_has_no_error():
    assert nt.decode_error_code(None) == (0, "")


@pytest.mark.parametrize("cls,matching_id", [(nt.SUCCESS, True), (nt.REQUEST, False)])
def test_transaction_rejects_request_identity_mismatch_before_sending(udp, cls, matching_id):
    request = nt.StunMessage(nt.BINDING, cls, b"a" * 12)
    with pytest.raises(nt.StunError, match="does not match"):
        nt.stun_transaction(udp, udp.getsockname(), request.encode(), b"a" * 12 if matching_id else b"b" * 12)
    udp.settimeout(0.02)
    with pytest.raises(socket.timeout):
        udp.recvfrom(1024)


@pytest.mark.parametrize("timeout,rto", [(0, 0.1), (-1, 0.1), (1, 0), (1, -1)])
def test_transaction_rejects_nonpositive_timing(udp, timeout, rto):
    request = nt.StunMessage(nt.BINDING, nt.REQUEST)
    with pytest.raises(ValueError, match="must be positive"):
        nt.stun_transaction(udp, udp.getsockname(), request.encode(), request.transaction_id,
                            timeout=timeout, initial_rto=rto)


@pytest.mark.parametrize("kind", ["plain", "missing", "error"])
def test_binding_handles_plain_address_missing_mapping_and_error(udp, kind):
    def reply(request):
        response = nt.StunMessage(nt.BINDING, nt.ERROR if kind == "error" else nt.SUCCESS,
                                  request.transaction_id)
        if kind == "plain":
            response.add(nt.MAPPED_ADDRESS, bytes.fromhex("00010d96c0000201"))
        elif kind == "error":
            response.add(nt.ERROR_CODE, nt.encode_error_code(500, "fixture failure"))
        return response, None

    with _ReplyServer(reply) as server:
        if kind == "plain":
            result = nt.stun_binding(udp, server.address, timeout=0.5)
            assert result.mapped == ("192.0.2.1", 3478)
            assert result.local == udp.getsockname()
            assert result.server == server.address
            assert result.rtt_ms >= 0
        else:
            with pytest.raises(nt.StunError) as error:
                nt.stun_binding(udp, server.address, timeout=0.5)
            assert error.value.code == (500 if kind == "error" else None)
            assert ("Binding error" if kind == "error" else "no mapped address") in str(error.value)


@pytest.mark.parametrize(
    "mappings,behaviour,nat_type,punchable",
    [([("192.0.2.1", 4000)], "unknown", "unknown", False),
     ([("192.0.2.1", 4000)] * 2, "endpoint_independent", "cone", True),
     ([("192.0.2.1", 4000), ("192.0.2.1", 4001)], "address_dependent", "symmetric", False)],
)
def test_mapping_classification_uses_actual_remote_observations(udp, mappings, behaviour, nat_type, punchable):
    with ExitStack() as stack:
        servers = []
        for mapping in mappings:
            def reply(request, mapping=mapping):
                response = nt.StunMessage(nt.BINDING, nt.SUCCESS, request.transaction_id)
                response.add(nt.XOR_MAPPED_ADDRESS, nt.encode_xor_address(mapping, request.transaction_id))
                return response, None
            servers.append(stack.enter_context(_ReplyServer(reply)))
        # An IPv6 literal cannot resolve to the IPv4-only transport; no packet leaves the host.
        result = nt.classify_nat_mapping(udp, [("::1", 3478), *(s.address for s in servers)], timeout=0.3)
        assert result.behaviour == behaviour
        assert result.nat_type == nat_type
        assert result.hole_punchable is punchable
        assert result.local == udp.getsockname()
        assert result.mapped == mappings
        assert result.servers == [s.address for s in servers]
        assert len(result.rtt_ms) == len(mappings)


def test_mapping_with_no_answer_raises_protocol_error(udp):
    with pytest.raises(nt.StunError, match="no STUN server answered"):
        nt.classify_nat_mapping(udp, [("::1", 3478)], timeout=0.1)


@pytest.mark.parametrize("realm,signed,accepted", [(b"new.realm", True, True), (b"\xff", True, False),
                                                    (b"\xff", False, False), (None, True, False)])
def test_initial_challenge_authentication_validates_realm_without_mutating_client(udp, realm, signed, accepted):
    client = nt.TurnClient(("127.0.0.1", 9), "alice", "secret", sock=udp)
    response = nt.StunMessage(nt.ALLOCATE, nt.ERROR)
    response.add(nt.ERROR_CODE, nt.encode_error_code(401, "Unauthorized")).add(nt.NONCE, b"nonce")
    if realm is not None:
        response.add(nt.REALM, realm)
    raw = response.encode(integrity_key=nt.long_term_key("alice", "new.realm", "secret") if signed else None)
    assert client._valid_response(nt.StunMessage.decode(raw), raw) is accepted
    assert (client.realm, client.key, client.nonce) == (None, None, None)


@pytest.mark.parametrize("missing", [nt.USERNAME, nt.REALM, nt.NONCE])
def test_signed_request_missing_credential_is_bad_request_without_allocation(relay, udp, missing):
    response, _ = _exchange(relay, udp, nt.ALLOCATE, [(nt.REQUESTED_TRANSPORT, b"\x11\0\0\0")], omit=(missing,))
    assert response.cls == nt.ERROR
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 400
    assert relay.allocations == {} and relay._responses == {}


@pytest.mark.parametrize("user", [b"\xff", b"unknown"])
def test_invalid_username_receives_challenge_without_allocation(relay, udp, user):
    response, _ = _exchange(relay, udp, nt.ALLOCATE, user=user)
    assert response.cls == nt.ERROR
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 401
    assert response.get(nt.REALM) == relay.realm.encode()
    assert response.get(nt.NONCE) == relay.nonce
    assert relay.allocations == {} and relay._responses == {}


@pytest.mark.parametrize("transport", [None, b"\x06\0\0\0"])
def test_allocate_rejects_absent_or_non_udp_transport_with_signed_error(relay, udp, transport):
    attributes = [] if transport is None else [(nt.REQUESTED_TRANSPORT, transport)]
    response, raw = _exchange(relay, udp, nt.ALLOCATE, attributes)
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 442
    assert nt.verify_message_integrity(raw, nt.long_term_key("alice", relay.realm, "secret"))
    assert relay.allocations == {} and relay._by_relay == {}


@pytest.mark.parametrize("method", [nt.CREATE_PERMISSION, nt.CHANNEL_BIND])
def test_permission_and_channel_require_an_existing_allocation(relay, udp, method):
    response, raw = _exchange(relay, udp, method)
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 437
    assert nt.verify_message_integrity(raw, nt.long_term_key("alice", relay.realm, "secret"))
    assert relay.allocations == {} and relay.channels == {}


def test_channel_binding_rejects_another_allocation_owner(relay, udp):
    _allocate(relay, udp)
    response, raw = _exchange(relay, udp, nt.CHANNEL_BIND, user=b"bob",
                              key=nt.long_term_key("bob", relay.realm, "other"))
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 441
    assert nt.verify_message_integrity(raw, nt.long_term_key("bob", relay.realm, "other"))
    assert relay.channels == {} and relay._channel_by_peer == {}


def test_ipv6_channel_binding_on_ipv4_allocation_returns_signed_family_error(relay, udp):
    _allocate(relay, udp)
    response, raw = _exchange(
        relay, udp, nt.CHANNEL_BIND,
        [(nt.CHANNEL_NUMBER, bytes.fromhex("40000000")),
         (nt.XOR_PEER_ADDRESS, nt.encode_xor_address(("::1", 1234), b"\0" * 12))],
    )
    assert response.cls == nt.ERROR
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 443
    assert nt.verify_message_integrity(raw, nt.long_term_key("alice", relay.realm, "secret"))
    assert relay.channels == {} and relay._channel_by_peer == {}


@pytest.mark.parametrize("method", ["channel", "indication"])
def test_relay_send_error_does_not_break_control_requests(relay, udp, method):
    allocation = _allocate(relay, udp)
    with nt.udp_socket(("127.0.0.1", 0)) as peer:
        _bind(relay, udp, peer.getsockname())
        allocation.relay.close()  # A real failed OS transport, rather than a mocked handler.
        if method == "channel":
            wire = nt.encode_channel_data(0x4000, b"undeliverable")
        else:
            indication = nt.StunMessage(nt.SEND, nt.INDICATION)
            indication.add(nt.XOR_PEER_ADDRESS, nt.encode_xor_address(peer.getsockname(), indication.transaction_id))
            wire = indication.add(nt.DATA_ATTR, b"undeliverable").encode()
        relay.handle_datagram(wire, udp.getsockname())
        peer.settimeout(0.02)
        with pytest.raises(socket.timeout):
            peer.recvfrom(1024)
        response, raw = _exchange(relay, udp, nt.REFRESH, [(nt.LIFETIME, struct.pack("!I", 120))])
        assert response.cls == nt.SUCCESS
        assert response.get(nt.LIFETIME) == struct.pack("!I", 120)
        assert nt.verify_message_integrity(raw, nt.long_term_key("alice", relay.realm, "secret"))


def test_os_allocation_failure_returns_signed_capacity_error_without_partial_state(relay, udp, monkeypatch):
    def unavailable_socket(*args, **kwargs):
        raise OSError("socket capacity exhausted")

    with monkeypatch.context() as patch:
        patch.setattr(nt.socket, "socket", unavailable_socket)
        response, raw = _exchange(relay, udp, nt.ALLOCATE, [(nt.REQUESTED_TRANSPORT, b"\x11\0\0\0")])
    assert response.cls == nt.ERROR
    assert nt.decode_error_code(response.get(nt.ERROR_CODE))[0] == 508
    assert nt.verify_message_integrity(raw, nt.long_term_key("alice", relay.realm, "secret"))
    assert relay.allocations == {} and relay._by_relay == {}


@pytest.mark.parametrize("invalid", ["truncated", "unbound", "expired", "permission"])
def test_channel_data_drops_unusable_channel_and_then_relays_valid_payload(relay, udp, invalid):
    allocation = _allocate(relay, udp)
    with nt.udp_socket(("127.0.0.1", 0)) as peer:
        peer.settimeout(0.02)
        channel = _bind(relay, udp, peer.getsockname())
        packet = nt.encode_channel_data(0x4000, b"discard")
        if invalid == "truncated":
            packet = bytes.fromhex("4000000261")
        elif invalid == "unbound":
            packet = nt.encode_channel_data(0x4001, b"discard")
        elif invalid == "expired":
            channel.expires = 0
        else:
            allocation.permissions[peer.getsockname()[0]] = 0
        relay.handle_datagram(packet, udp.getsockname())
        with pytest.raises(socket.timeout):
            peer.recvfrom(1024)
        if invalid == "expired":
            assert relay.channels == {} and relay._channel_by_peer == {}
        _bind(relay, udp, peer.getsockname())
        relay.handle_datagram(nt.encode_channel_data(0x4000, b"delivered"), udp.getsockname())
        peer.settimeout(0.5)
        assert peer.recvfrom(1024) == (b"delivered", allocation.relay.getsockname())


@pytest.mark.parametrize("expire", ["allocation", "channel"])
def test_housekeeping_expires_resources_and_cleans_reverse_indexes(relay, udp, expire):
    allocation = _allocate(relay, udp)
    channel = _bind(relay, udp, ("127.0.0.1", 23456))
    if expire == "allocation":
        allocation.expires = 0
    else:
        channel.expires = 0
    relay._housekeeping()
    assert relay.channels == {} and relay._channel_by_peer == {}
    if expire == "allocation":
        assert relay.allocations == {} and relay._by_relay == {}
        assert allocation.relay.fileno() == -1
    else:
        assert relay.allocations[udp.getsockname()] is allocation
        assert allocation.relay.fileno() >= 0


def test_expired_cached_request_is_reexecuted_without_replacing_allocation(relay, udp):
    request = nt.StunMessage(nt.ALLOCATE, nt.REQUEST)
    request.add(nt.USERNAME, b"alice").add(nt.REALM, relay.realm.encode()).add(nt.NONCE, relay.nonce)
    request.add(nt.REQUESTED_TRANSPORT, b"\x11\0\0\0")
    key = nt.long_term_key("alice", relay.realm, "secret")
    wire = request.encode(integrity_key=key, fingerprint=True)
    relay.handle_datagram(wire, udp.getsockname())
    first, _ = udp.recvfrom(65535)
    assert nt.StunMessage.decode(first).cls == nt.SUCCESS
    allocation = relay.allocations[udp.getsockname()]
    cache_key = (udp.getsockname(), nt.ALLOCATE, request.transaction_id)
    _, digest, response = relay._responses[cache_key]
    relay._responses[cache_key] = (0, digest, response)
    relay.handle_datagram(wire, udp.getsockname())
    repeated, _ = udp.recvfrom(65535)
    error = nt.StunMessage.decode(repeated)
    assert error.cls == nt.ERROR
    assert nt.decode_error_code(error.get(nt.ERROR_CODE))[0] == 437
    assert nt.verify_message_integrity(repeated, key)
    assert relay.allocations[udp.getsockname()] is allocation
    assert relay._responses[cache_key][2] == repeated


def test_client_discards_truncated_channel_data_before_accepting_valid_payload(udp):
    client = nt.TurnClient(("127.0.0.1", 9), "alice", "secret", sock=udp)
    peer = ("192.0.2.1", 1234)
    client._channel_peers[0x4000] = peer
    client._handle_datagram(bytes.fromhex("4000000261"), client.server)
    assert client._inbox == []
    client._handle_datagram(nt.encode_channel_data(0x4000, b"ok"), client.server)
    assert client.recv_from(timeout=0) == (peer, b"ok")
    assert client._inbox == []


def test_client_cannot_send_on_unbound_or_exhausted_channels(udp):
    client = nt.TurnClient(("127.0.0.1", 9), "alice", "secret", sock=udp)
    with pytest.raises(nt.StunError, match="no TURN channel"):
        client.send_channel(("127.0.0.1", 1234), b"data")
    client._channel_peers.update({number: ("127.0.0.1", number) for number in range(0x4000, 0x8000)})
    with pytest.raises(nt.StunError, match="every TURN channel"):
        client.bind_channel(("127.0.0.1", 1234))
    assert client.channels == {}


@pytest.mark.parametrize(
    "realm,nonce,error_text",
    [(b"\xff", b"fresh", "invalid REALM"),
     (b"changed.realm", b"fresh", "changed REALM"),
     (b"fixture.realm", b"", "invalid REALM or NONCE")],
)
def test_client_rejects_invalid_authenticated_stale_challenge(udp, realm, nonce, error_text):
    key = nt.long_term_key("alice", "fixture.realm", "secret")

    def reply(request):
        initial = request.get(nt.USERNAME) is None
        response = nt.StunMessage(request.method, nt.ERROR, request.transaction_id)
        response.add(nt.ERROR_CODE, nt.encode_error_code(401 if initial else 438, "challenge"))
        response.add(nt.REALM, b"fixture.realm" if initial else realm)
        response.add(nt.NONCE, b"original" if initial else nonce)
        return response, None if initial else key

    with _ReplyServer(reply) as server:
        client = nt.TurnClient(server.address, "alice", "secret", sock=udp, timeout=0.5)
        with pytest.raises(nt.StunError, match=error_text) as error:
            client.allocate()
        assert error.value.code == 438
        assert server.requests == 2
        assert client.relayed is None and client.channels == {}


def test_allocate_success_requires_relay_address_after_real_authentication(udp):
    def reply(request):
        initial = request.get(nt.USERNAME) is None
        response = nt.StunMessage(request.method, nt.ERROR if initial else nt.SUCCESS, request.transaction_id)
        if initial:
            response.add(nt.ERROR_CODE, nt.encode_error_code(401, "challenge"))
            response.add(nt.REALM, b"fixture.realm").add(nt.NONCE, b"original")
        return response, None if initial else nt.long_term_key("alice", "fixture.realm", "secret")

    with _ReplyServer(reply) as server:
        client = nt.TurnClient(server.address, "alice", "secret", sock=udp, timeout=0.5)
        with pytest.raises(nt.StunError, match="no XOR-RELAYED-ADDRESS"):
            client.allocate()
        assert client.relayed is None
        assert server.requests == 2


@pytest.mark.parametrize("owns_socket", [False, True])
def test_close_cleans_client_state_even_when_release_times_out(relay, udp, owns_socket):
    # This server is deliberately not started: its bound socket cannot answer Refresh.
    client = nt.TurnClient(relay.address, "alice", "secret", sock=None if owns_socket else udp, timeout=0.02)
    client.relayed = ("127.0.0.1", 1234)
    client.channels[("127.0.0.1", 2345)] = 0x4000
    client._channel_peers[0x4000] = ("127.0.0.1", 2345)
    client.close()
    assert client.relayed is None and client.channels == {} and client._channel_peers == {}
    assert (client.sock.fileno() == -1) is owns_socket


def test_client_receive_retries_transient_socket_failure_without_losing_data(udp):
    with nt.udp_socket(("127.0.0.1", 0)) as remote:
        boundary = _ReceiveFailureOnce(udp)
        client = nt.TurnClient(remote.getsockname(), "alice", "secret", sock=boundary)
        client._channel_peers[0x4000] = ("192.0.2.1", 1234)
        remote.sendto(nt.encode_channel_data(0x4000, b"retained"), udp.getsockname())
        assert client.recv_from(timeout=0.5) == (("192.0.2.1", 1234), b"retained")
        assert boundary.failed
        assert client._inbox == []


@pytest.mark.parametrize("turn", [False, True])
def test_server_survives_stale_socket_and_malformed_packet_before_valid_binding(udp, turn):
    server = nt.TurnRelayServer({"alice": "secret"}) if turn else nt.StunServer()
    try:
        stale = nt.udp_socket(("127.0.0.1", 0))
        stale.close()
        server._on_readable(stale)
        udp.sendto(b"invalid STUN", server.address)
        server._on_readable(server.sock)
        request = nt.StunMessage(nt.BINDING, nt.REQUEST)
        udp.sendto(request.encode(), server.address)
        server._on_readable(server.sock)
        raw, source = udp.recvfrom(1024)
        response = nt.StunMessage.decode(raw)
        assert source == server.address
        assert response.cls == nt.SUCCESS and response.transaction_id == request.transaction_id
        assert nt.decode_xor_address(response.get(nt.XOR_MAPPED_ADDRESS), response.transaction_id) == udp.getsockname()
    finally:
        server.stop()


@pytest.mark.parametrize("field", ["loss_rate", "reorder_rate", "jitter_ms"])
@pytest.mark.parametrize("value", [True, "0.1", None])
def test_fault_profile_rejects_non_numeric_probabilities_and_delay(field, value):
    with pytest.raises(TypeError, match=field):
        nt.FaultProfile(**{field: value})


def test_fault_proxy_lifecycle_guards_and_context_cleanup(udp):
    proxy = nt.DeterministicFaultProxy(udp.getsockname())
    for action in (lambda: proxy.client_address, lambda: proxy.upstream_address, proxy.rebind):
        with pytest.raises(RuntimeError, match="not started"):
            action()
    proxy.close()
    with proxy:
        sockets = (proxy._client_sock, proxy._upstream_sock)
        assert proxy.client_address[0] == proxy.upstream_address[0] == "127.0.0.1"
        with pytest.raises(RuntimeError, match="already started"):
            proxy.start()
    assert all(sock.fileno() == -1 for sock in sockets)
    assert proxy._thread is None
    with pytest.raises(RuntimeError, match="not started"):
        _ = proxy.client_address


@pytest.mark.parametrize("failure", ["select_oserror", "select_valueerror", "receive"])
def test_fault_proxy_recovers_transport_errors_and_still_completes_binding(udp, monkeypatch, failure):
    original_select = nt.select.select
    original_socket = nt.socket.socket
    injected = []
    made = []
    with nt.StunServer() as server:
        proxy = nt.DeterministicFaultProxy(server.address)

        def failing_select(readers, writers, errors, timeout):
            if readers and readers[0] is proxy._client_sock and not injected:
                injected.append(failure)
                if failure == "select_oserror":
                    raise OSError("interrupted selector")
                raise ValueError("temporarily invalid selector")
            return original_select(readers, writers, errors, timeout)

        def receive_failure_socket(*args, **kwargs):
            sock = original_socket(*args, **kwargs)
            if not made:
                sock = _ReceiveFailureOnce(sock)
            made.append(sock)
            return sock

        with monkeypatch.context() as patch:
            if failure == "receive":
                patch.setattr(nt.socket, "socket", receive_failure_socket)
            else:
                patch.setattr(nt.select, "select", failing_select)
            with proxy:
                result = nt.stun_binding(udp, proxy.client_address, timeout=1)
                assert result.mapped == proxy.upstream_address
                assert proxy._thread.is_alive()
                if failure == "receive":
                    assert made[0].failed
                else:
                    assert injected == [failure]


def test_fault_proxy_releases_single_reordered_response_and_drops_unregistered_destination(udp):
    with nt.udp_socket(("127.0.0.1", 0)) as upstream:
        upstream.settimeout(1)
        with nt.DeterministicFaultProxy(upstream.getsockname(), nt.FaultProfile(reorder_rate=1, seed=7)) as proxy:
            # No destination exists until a client speaks: unsolicited downlink
            # data must not survive in the reorder queue for a future client.
            proxy._forward("down", b"unsolicited", upstream.getsockname())
            assert proxy._held["down"] is None and proxy._pending == []
            udp.sendto(b"request", proxy.client_address)
            request, source = upstream.recvfrom(1024)
            assert request == b"request" and source == proxy.upstream_address
            upstream.sendto(b"response", source)
            assert udp.recvfrom(1024) == (b"response", proxy.client_address)
        assert proxy.stats()["held"] == 2
        assert proxy.stats()["forwarded"] == 2
