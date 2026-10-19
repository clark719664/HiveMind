"""Protocol-level tests for `hivemind.swarm.nat_traversal` and `hivemind.swarm.quic_link`.

Set `HIVEMIND_LIVE_NET=1` to also probe public STUN servers, and `TURN_SERVER`,
`TURN_USERNAME`, `TURN_PASSWORD` to run the TURN client against a real server such as coturn.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import secrets
import threading

import pytest

from hivemind.swarm import nat_traversal as nt
from hivemind.swarm.quic_link import (
    LIBP2P_PROTOCOL,
    MULTISTREAM,
    _multistream_message,
    _parse_multistream,
    run_local_link,
)

# RFC 5769 test vectors (§2.1-§2.4), copied from the RFC text.
RFC5769_REQUEST = bytes.fromhex(
    "000100582112a442b7e7a701bc34d686fa87dfae802200105354554e207465737420636c69656e74"
    "002400046e0001ff80290008932ff9b151263b36000600096576746a3a68367659202020000800149a"
    "eaa70cbfd8cb56781ef2b5b2d3f249c1b571a280280004e57a3bcf"
)
RFC5769_IPV4_RESPONSE = bytes.fromhex(
    "0101003c2112a442b7e7a701bc34d686fa87dfae8022000b7465737420766563746f722000200008"
    "0001a147e112a643000800142b91f599fd9e90c38c7489f92af9ba53f06be7d780280004c07d4c96"
)
RFC5769_IPV6_RESPONSE = bytes.fromhex(
    "010100482112a442b7e7a701bc34d686fa87dfae8022000b7465737420766563746f722000200014"
    "0002a1470113a9faa5d3f179bc25f4b5bed2b9d900080014a382954e4be67bf11784c97c8292c275"
    "bfe3ed4180280004c8fb0b4c"
)
RFC5769_LONG_TERM_REQUEST = bytes.fromhex(
    "000100602112a44278ad3433c6ad72c029da412e00060012e3839ee38388e383aae38383e382afe3"
    "82b900000015001c662f2f3439396b39353464364f4c33346f4c394653547679363473410014000b"
    "6578616d706c652e6f72670000080014f67024656dd64a3e02b8e0712e85c9a28ca89666"
)
RFC5769_PASSWORD = "VOkJxbRl1RmTxUk/WvJxBt"


def test_rfc5769_short_term_vectors_verify_and_reencode() -> None:
    key = nt.short_term_key(RFC5769_PASSWORD)
    for vector in (RFC5769_REQUEST, RFC5769_IPV4_RESPONSE, RFC5769_IPV6_RESPONSE):
        assert nt.verify_message_integrity(vector, key)
        assert nt.verify_fingerprint(vector)
        message = nt.StunMessage.decode(vector)
        assert message.encode(integrity_key=key, fingerprint=True, pad_byte=b" ") == vector
    assert not nt.verify_message_integrity(RFC5769_REQUEST, nt.short_term_key("wrong"))


def test_rfc5769_xor_mapped_addresses() -> None:
    v4 = nt.StunMessage.decode(RFC5769_IPV4_RESPONSE)
    assert (v4.method, v4.cls) == (nt.BINDING, nt.SUCCESS)
    assert nt.decode_xor_address(v4.get(nt.XOR_MAPPED_ADDRESS), v4.transaction_id) == ("192.0.2.1", 32853)
    v6 = nt.StunMessage.decode(RFC5769_IPV6_RESPONSE)
    assert nt.decode_xor_address(v6.get(nt.XOR_MAPPED_ADDRESS), v6.transaction_id) == (
        "2001:db8:1234:5678:11:2233:4455:6677",
        32853,
    )
    assert nt.encode_xor_address(("192.0.2.1", 32853), v4.transaction_id) == v4.get(nt.XOR_MAPPED_ADDRESS)


def test_rfc5769_long_term_credential_vector() -> None:
    username = "マトリックス"
    key = nt.long_term_key(username, "example.org", "TheMatrIX")
    assert nt.verify_message_integrity(RFC5769_LONG_TERM_REQUEST, key)
    message = nt.StunMessage.decode(RFC5769_LONG_TERM_REQUEST)
    assert message.get(nt.USERNAME).decode("utf-8") == username
    assert message.get(nt.NONCE) == b"f//499k954d6OL34oL9FSTvy64sA"
    assert message.encode(integrity_key=key) == RFC5769_LONG_TERM_REQUEST


def test_message_type_round_trip() -> None:
    assert nt.message_type(nt.BINDING, nt.REQUEST) == 0x0001
    assert nt.message_type(nt.ALLOCATE, nt.SUCCESS) == 0x0103
    assert nt.message_type(nt.SEND, nt.INDICATION) == 0x0016
    assert nt.message_type(nt.DATA, nt.INDICATION) == 0x0017
    for method in (nt.BINDING, nt.ALLOCATE, nt.REFRESH, nt.CREATE_PERMISSION):
        for cls in (nt.REQUEST, nt.INDICATION, nt.SUCCESS, nt.ERROR):
            assert nt.split_message_type(nt.message_type(method, cls)) == (method, cls)


def test_stun_binding_and_mapping_against_local_server() -> None:
    with nt.StunServer() as first, nt.StunServer() as second:
        sock = nt.udp_socket(("127.0.0.1", 0))
        try:
            result = nt.stun_binding(sock, first.url)
            assert result.mapped == sock.getsockname()
            mapping = nt.classify_nat_mapping(sock, [first.url, second.url])
        finally:
            sock.close()
    assert mapping.behaviour == "no_nat"
    assert mapping.nat_type == "open" and mapping.hole_punchable


def test_stun_binding_times_out_without_server() -> None:
    sock = nt.udp_socket(("127.0.0.1", 0))
    dead = nt.udp_socket(("127.0.0.1", 0))
    dead_addr = dead.getsockname()
    dead.close()
    try:
        with pytest.raises(nt.StunError):
            nt.stun_binding(sock, dead_addr, timeout=0.3)
    finally:
        sock.close()


def test_turn_allocation_permissions_and_relay() -> None:
    with nt.TurnRelayServer({"alice": "s3cret", "bob": "hunter2"}) as turn:
        alice = nt.TurnClient(turn.url, "alice", "s3cret")
        bob = nt.TurnClient(turn.url, "bob", "hunter2")
        stranger = nt.udp_socket(("127.0.0.1", 0))
        try:
            relay_a, relay_b = alice.allocate(), bob.allocate()
            assert alice.lifetime == 600 and relay_a != relay_b
            assert alice.mapped == ("127.0.0.1", alice.sock.getsockname()[1])

            # No permission yet: RFC 5766 §10.3 says the relay must drop the datagram.
            stranger.sendto(b"unsolicited", relay_a)
            with pytest.raises(nt.StunError):
                alice.recv_from(timeout=0.3)

            alice.create_permission(relay_b[0])
            bob.create_permission(relay_a[0])
            alice.send_to(relay_b, b"ping")
            assert bob.recv_from() == (relay_a, b"ping")
            bob.send_to(relay_a, b"pong")
            assert alice.recv_from() == (relay_b, b"pong")
            assert alice.refresh(120) == 120
        finally:
            alice.close()
            bob.close()
            stranger.close()
        assert turn.allocations == {}


def test_turn_rejects_bad_credentials_and_recovers_from_stale_nonce() -> None:
    with nt.TurnRelayServer({"alice": "s3cret"}) as turn:
        intruder = nt.TurnClient(turn.url, "alice", "guess")
        with pytest.raises(nt.StunError) as rejected:
            intruder.allocate()
        assert rejected.value.code == 401
        intruder.close()

        client = nt.TurnClient(turn.url, "alice", "s3cret")
        try:
            client.allocate()
            turn.nonce = secrets.token_hex(16).encode("ascii")  # server rotates its nonce
            assert client.refresh(300) == 300  # 438 Stale Nonce, then retried with the new nonce
            assert client.nonce == turn.nonce
        finally:
            client.close()


def test_udp_hole_punch_simultaneous_open() -> None:
    a, b = nt.udp_socket(("127.0.0.1", 0)), nt.udp_socket(("127.0.0.1", 0))
    addr_a, addr_b = a.getsockname(), b.getsockname()
    session = secrets.token_bytes(8)
    other: list = []
    try:
        worker = threading.Thread(
            target=lambda: other.append(nt.udp_hole_punch(b, [a.getsockname()], session, timeout=1.0))
        )
        worker.start()
        mine = nt.udp_hole_punch(a, [b.getsockname()], session, timeout=1.0)
        worker.join()
    finally:
        a.close()
        b.close()
    assert mine.succeeded and other[0].succeeded
    assert mine.peer == addr_b and other[0].peer == addr_a


def test_udp_hole_punch_ignores_other_sessions() -> None:
    a, b = nt.udp_socket(("127.0.0.1", 0)), nt.udp_socket(("127.0.0.1", 0))
    other: list = []
    try:
        worker = threading.Thread(
            target=lambda: other.append(nt.udp_hole_punch(b, [a.getsockname()], b"session-B", timeout=0.4))
        )
        worker.start()
        mine = nt.udp_hole_punch(a, [b.getsockname()], b"session-A", timeout=0.4)
        worker.join()
    finally:
        a.close()
        b.close()
    assert not mine.succeeded and not other[0].succeeded


def test_multistream_select_framing() -> None:
    message = _multistream_message(MULTISTREAM)
    assert message[0] == len(MULTISTREAM) + 1 and message.endswith(b"\n")
    assert _parse_multistream(message) == MULTISTREAM
    long_id = "/x" * 100
    assert _parse_multistream(_multistream_message(long_id)) == long_id
    assert _parse_multistream(_multistream_message(LIBP2P_PROTOCOL)) == LIBP2P_PROTOCOL


def test_quic_link_resumes_with_0rtt_and_delivers_every_frame() -> None:
    received: list = []
    frames = [secrets.token_bytes(700) for _ in range(64)]
    run = asyncio.run(run_local_link(frames, on_frame=received.append))
    assert run.protocol == "quic_0rtt"
    assert run.handshakes[0] == {"session_resumed": False, "early_data_accepted": False}
    assert run.handshakes[1] == {"session_resumed": True, "early_data_accepted": True}
    assert run.server_early_frames > 0
    assert received[0] == b"warmup" and received[1:] == frames
    assert run.frames_delivered == len(frames)


def test_quic_link_falls_back_to_libp2p_websocket_when_udp_is_blocked() -> None:
    received: list = []
    frames = [secrets.token_bytes(300) for _ in range(10)]
    run = asyncio.run(run_local_link(frames, on_frame=received.append, udp_blocked=True, quic_timeout=0.5))
    assert run.protocol == "libp2p_websocket"
    assert run.fallback_reason == "TimeoutError"
    assert received == frames and run.frames_delivered == len(frames)


live_net = pytest.mark.skipif(os.environ.get("HIVEMIND_LIVE_NET") != "1", reason="set HIVEMIND_LIVE_NET=1")


@live_net
def test_live_public_stun_servers() -> None:
    sock = nt.udp_socket()
    try:
        mapping = nt.classify_nat_mapping(sock, nt.DEFAULT_STUN_SERVERS, timeout=3.0)
    finally:
        sock.close()
    assert ipaddress.ip_address(mapping.mapped[0][0]).is_global
    assert mapping.behaviour in ("no_nat", "endpoint_independent", "address_dependent")


@pytest.mark.skipif(not os.environ.get("TURN_SERVER"), reason="set TURN_SERVER/TURN_USERNAME/TURN_PASSWORD")
def test_live_turn_server_relays_between_two_allocations() -> None:
    server, user, password = (os.environ[k] for k in ("TURN_SERVER", "TURN_USERNAME", "TURN_PASSWORD"))
    first, second = nt.TurnClient(server, user, password), nt.TurnClient(server, user, password)
    try:
        relay_1, relay_2 = first.allocate(), second.allocate()
        first.create_permission(relay_2[0])
        second.create_permission(relay_1[0])
        first.send_to(relay_2, b"through-a-real-turn-server")
        assert second.recv_from(timeout=3.0) == (relay_1, b"through-a-real-turn-server")
        second.send_to(relay_1, b"and-back")
        assert first.recv_from(timeout=3.0) == (relay_2, b"and-back")
        assert first.refresh(300) > 0
    finally:
        first.close()
        second.close()
