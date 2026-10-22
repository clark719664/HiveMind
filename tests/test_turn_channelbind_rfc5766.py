"""RFC 5766 §11 ChannelBind conformance and honest 0-RTT probe verdicts.

These tests pin the behaviour a real TURN server (coturn) expects from the client and the
behaviour the benchmark report claims about QUIC 0-RTT resumption.
"""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Optional

import pytest

from hivemind.swarm import nat_traversal as nt
from hivemind.swarm.nat_traversal import TurnClient, TurnRelayServer
from hivemind.swarm.quic_link import (
    FrameEnvelope,
    LinkIdentity,
    Quic0RttProbe,
    QuicHiddenStateServer,
    quic_0rtt_resumption_probe,
    quic_send_frames,
)


def _channel(turn: TurnRelayServer, number: int):
    """The one channel bound with `number` (channels are keyed by (client, number))."""
    matches = [channel for (_client, n), channel in turn.channels.items() if n == number]
    assert len(matches) == 1, f"expected exactly one channel {number:#06x}, found {len(matches)}"
    return matches[0]


def _numbers(turn: TurnRelayServer) -> set:
    return {number for (_client, number) in turn.channels}


def _xor_peer(peer):
    return (nt.XOR_PEER_ADDRESS, lambda txid: nt.encode_xor_address(peer, txid))


def _channel_number_attr(value: bytes):
    return (nt.CHANNEL_NUMBER, lambda _txid: value)


def _raw_channel_bind(client: TurnClient, peer, channel_number_value: Optional[bytes]) -> int:
    """Send a ChannelBind request with hand-built attributes and return the error code."""
    attributes = [_xor_peer(peer)]
    if channel_number_value is not None:
        attributes.append(_channel_number_attr(channel_number_value))
    with pytest.raises(nt.StunError) as excinfo:
        client._request(nt.CHANNEL_BIND, attributes)
    return excinfo.value.code


# ---------------------------------------------------------------------------
# RFC 5766 §11.1: the CLIENT chooses the channel number
# ---------------------------------------------------------------------------
def test_channel_number_attribute_is_four_bytes_with_a_zero_rffu_field() -> None:
    # RFC 5766 §14.1: 16-bit channel number followed by a two-octet RFFU field set to 0.
    assert nt.encode_channel_number(0x4000) == b"\x40\x00\x00\x00"
    assert nt.decode_channel_number(b"\x7f\xff\x00\x00") == 0x7FFF
    # RFFU "MUST be ignored on reception".
    assert nt.decode_channel_number(b"\x40\x01\xab\xcd") == 0x4001
    with pytest.raises(nt.StunError):
        nt.decode_channel_number(b"\x40\x00")


def test_client_chosen_channel_number_is_bound_by_the_server() -> None:
    with TurnRelayServer({"alice": "s3cret", "bob": "hunter2"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        bob = TurnClient(turn.url, "bob", "hunter2")
        try:
            alice.allocate()
            relay_b = bob.allocate()
            number = alice.bind_channel(relay_b, channel_number=0x4123)
            assert number == 0x4123
            assert _channel(turn, 0x4123).peer == relay_b
            assert alice.channels[relay_b] == 0x4123
        finally:
            alice.close()
            bob.close()


def test_client_picks_distinct_in_range_numbers_for_distinct_peers() -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            peer_one, peer_two = ("127.0.0.1", 5001), ("127.0.0.1", 5002)
            first = alice.bind_channel(peer_one)
            second = alice.bind_channel(peer_two)
            assert first != second
            for number, peer in ((first, peer_one), (second, peer_two)):
                assert nt.CHANNEL_NUMBER_MIN <= number <= nt.CHANNEL_NUMBER_MAX
                assert _channel(turn, number).peer == peer
        finally:
            alice.close()


def test_client_rejects_an_out_of_range_channel_number_before_sending() -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            for bad in (0x3FFF, 0x8000, -1):
                with pytest.raises(ValueError):
                    alice.bind_channel(("127.0.0.1", 5000), channel_number=bad)
            assert not turn.channels
        finally:
            alice.close()


# ---------------------------------------------------------------------------
# RFC 5766 §11.2: what the server must reject with 400 Bad Request
# ---------------------------------------------------------------------------
def test_server_rejects_channel_bind_without_a_channel_number() -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            assert _raw_channel_bind(alice, ("127.0.0.1", 5000), None) == 400
            assert not turn.channels
        finally:
            alice.close()


@pytest.mark.parametrize(
    "value",
    [
        struct.pack("!HH", 0x3FFF, 0),  # below the channel range
        struct.pack("!HH", 0x8000, 0),  # above the channel range
        struct.pack("!H", 0x4000),  # 2 bytes: RFFU field missing (§14.1 says 4 bytes)
    ],
)
def test_server_rejects_malformed_or_out_of_range_channel_numbers(value: bytes) -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            assert _raw_channel_bind(alice, ("127.0.0.1", 5000), value) == 400
            assert not turn.channels
        finally:
            alice.close()


def test_server_rejects_a_channel_number_already_bound_to_another_peer() -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            alice.bind_channel(("127.0.0.1", 5001), channel_number=0x4000)
            with pytest.raises(nt.StunError) as excinfo:
                alice.bind_channel(("127.0.0.1", 5002), channel_number=0x4000)
            assert excinfo.value.code == 400
            assert _channel(turn, 0x4000).peer == ("127.0.0.1", 5001)
        finally:
            alice.close()


def test_server_rejects_a_second_channel_number_for_the_same_peer() -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            peer = ("127.0.0.1", 5000)
            alice.bind_channel(peer, channel_number=0x4000)
            with pytest.raises(nt.StunError) as excinfo:
                alice.bind_channel(peer, channel_number=0x4001)
            assert excinfo.value.code == 400
            assert _numbers(turn) == {0x4000}
        finally:
            alice.close()


def test_rebinding_the_same_number_to_the_same_peer_is_a_refresh() -> None:
    with TurnRelayServer({"alice": "s3cret"}, channel_lifetime=600) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            peer = ("127.0.0.1", 5000)
            assert alice.bind_channel(peer, channel_number=0x4000) == 0x4000
            first_expiry = _channel(turn, 0x4000).expires
            time.sleep(0.05)  # Windows' monotonic clock ticks every ~16 ms.
            # Without an explicit number the client refreshes the channel it already bound.
            assert alice.bind_channel(peer) == 0x4000
            assert _numbers(turn) == {0x4000}
            assert _channel(turn, 0x4000).expires > first_expiry
        finally:
            alice.close()


# ---------------------------------------------------------------------------
# RFC 5766 §11.2: ChannelBind installs (or refreshes) the peer permission
# ---------------------------------------------------------------------------
def test_channel_bind_installs_the_peer_permission_in_both_directions() -> None:
    with TurnRelayServer({"alice": "s3cret", "bob": "hunter2"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        bob = TurnClient(turn.url, "bob", "hunter2")
        try:
            relay_a, relay_b = alice.allocate(), bob.allocate()
            # Alice never sends CreatePermission: ChannelBind alone must permit relay_b.
            alice.bind_channel(relay_b)
            bob.create_permission(relay_a[0])

            alice.send_channel(relay_b, b"outbound-over-channel")
            _peer, payload = bob.recv_from()
            assert payload == b"outbound-over-channel"

            bob.send_to(relay_a, b"inbound-from-peer")
            _peer, payload = alice.recv_from()
            assert payload == b"inbound-from-peer"
        finally:
            alice.close()
            bob.close()


def test_channel_bind_success_response_carries_no_attributes() -> None:
    # RFC 5766 §11.2: "There are no required attributes in a successful ChannelBind response."
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            response = alice._request(
                nt.CHANNEL_BIND,
                [
                    _xor_peer(("127.0.0.1", 5000)),
                    _channel_number_attr(nt.encode_channel_number(0x4000)),
                ],
            )
            assert response.get(nt.CHANNEL_NUMBER) is None
            assert response.get(nt.CHANNEL_LIFETIME) is None
            assert _channel(turn, 0x4000).peer == ("127.0.0.1", 5000)
        finally:
            alice.close()


# ---------------------------------------------------------------------------
# 0-RTT probe honesty
# ---------------------------------------------------------------------------
def _probe(**overrides) -> Quic0RttProbe:
    fields = dict(
        frames_sent=4,
        frames_delivered=4,
        session_resumed=True,
        early_data_accepted=True,
        server_early_frames=4,
        handshake_latency_ms=1.0,
        resumption_latency_ms=1.0,
    )
    fields.update(overrides)
    return Quic0RttProbe(**fields)


def test_probe_ok_requires_resumption_and_accepted_early_data() -> None:
    assert _probe().ok
    assert not _probe(session_resumed=False).ok
    assert not _probe(early_data_accepted=False).ok
    assert not _probe(frames_delivered=3).ok
    assert not _probe(error="TimeoutError").ok


def test_probe_delivers_non_envelope_frames_to_the_server_handler() -> None:
    async def run() -> None:
        identity = LinkIdentity.generate()
        received: list = []
        server = await QuicHiddenStateServer(
            identity, "127.0.0.1", 0, on_frame=received.append
        ).start()
        try:
            frames = [b"plain-frame-one", b"plain-frame-two"]
            probe = await quic_0rtt_resumption_probe(server, frames)
            assert probe.frames_delivered == len(frames)
            assert b"plain-frame-one" in received
            assert b"plain-frame-two" in received
            # Not timing envelopes, so they contribute no latency sample.
            assert probe.frame_latencies_ms == []
        finally:
            server.close()

    asyncio.run(run())


def test_probe_still_times_envelope_frames_and_forwards_them() -> None:
    async def run() -> None:
        identity = LinkIdentity.generate()
        received: list = []
        server = await QuicHiddenStateServer(
            identity, "127.0.0.1", 0, on_frame=received.append
        ).start()
        try:
            frames = [FrameEnvelope(i, time.time(), b"y" * 32).encode() for i in range(3)]
            probe = await quic_0rtt_resumption_probe(server, frames)
            assert probe.ok
            assert len(probe.frame_latencies_ms) == 3
            assert all(frame in received for frame in frames)
        finally:
            server.close()

    asyncio.run(run())


def test_probe_is_not_ok_when_the_server_rejects_a_consumed_ticket() -> None:
    async def run() -> None:
        identity = LinkIdentity.generate()
        server = await QuicHiddenStateServer(identity, "127.0.0.1", 0).start()
        try:
            frames = [FrameEnvelope(i, time.time(), b"z" * 32).encode() for i in range(2)]
            warmup = await quic_send_frames(
                server.host, server.port, [b"warmup"], server.identity.ca_pem
            )
            ticket = warmup.new_ticket
            assert ticket is not None
            second = await quic_0rtt_resumption_probe(server, frames, session_ticket=ticket)
            assert second.ok
            # Replaying the now-consumed ticket: the server rejects 0-RTT and falls back to 1-RTT.
            third = await quic_0rtt_resumption_probe(server, frames, session_ticket=ticket)
            assert third.error is None
            assert third.frames_delivered == len(frames)
            assert not third.session_resumed
            assert not third.ok
        finally:
            server.close()

    asyncio.run(run())
