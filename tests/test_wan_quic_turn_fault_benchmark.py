"""Tests for the HM-S5-05 WAN QUIC 0-RTT & RFC 5766 TURN fault-injection benchmark.

Covers the deterministic fault proxy, the RFC 5766 channel path, the `aioquic` 0-RTT
resumption probe and the structured telemetry report.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path

import pytest

from hivemind.swarm import nat_traversal as nt
from hivemind.swarm.nat_traversal import (
    DeterministicFaultProxy,
    FaultProfile,
    TurnClient,
    TurnRelayServer,
)
from hivemind.swarm.quic_link import (
    FrameEnvelope,
    LinkIdentity,
    LinkProtocolError,
    QuicHiddenStateServer,
    quic_0rtt_resumption_probe,
)
from hivemind.swarm.wan_fault_benchmark import (
    DEFAULT_LOSS_RATES,
    REPORT_SCHEMA,
    ExternalEndpoints,
    FaultScenario,
    StrikeRegister,
    default_scenarios,
    latency_summary,
    percentile,
    run_fault_scenario,
    write_fault_report,
)


# ---------------------------------------------------------------------------
# Fault profile and proxy
# ---------------------------------------------------------------------------
def test_fault_profile_rejects_invalid_impairments() -> None:
    with pytest.raises(ValueError):
        FaultProfile(loss_rate=1.5)
    with pytest.raises(ValueError):
        FaultProfile(reorder_rate=-0.1)
    with pytest.raises(ValueError):
        FaultProfile(jitter_ms=-1.0)
    with pytest.raises(TypeError):
        FaultProfile(seed="abc")
    assert FaultProfile(loss_rate=0.05, seed=7).to_dict()["loss_rate"] == 0.05


def test_fault_proxy_drops_every_datagram_at_total_loss() -> None:
    server = nt.udp_socket(("127.0.0.1", 0))
    client = nt.udp_socket(("127.0.0.1", 0))
    client.settimeout(0.3)
    proxy = DeterministicFaultProxy(
        server.getsockname(), FaultProfile(loss_rate=1.0, seed=11)
    ).start()
    try:
        client.sendto(b"lost", proxy.client_address)
        with pytest.raises(TimeoutError):
            client.recvfrom(64)
        assert proxy.stats()["dropped"] >= 1
        assert proxy.stats()["forwarded"] == 0
    finally:
        proxy.close()
        client.close()
        server.close()


def test_fault_proxy_is_deterministic_for_the_same_seed() -> None:
    """The same seed must drop the same datagrams in the same order."""
    decisions = []
    for _ in range(2):
        server = nt.udp_socket(("127.0.0.1", 0))
        client = nt.udp_socket(("127.0.0.1", 0))
        client.settimeout(0.4)
        proxy = DeterministicFaultProxy(
            server.getsockname(), FaultProfile(loss_rate=0.5, reorder_rate=0.25, seed=99)
        ).start()
        try:
            for index in range(12):
                client.sendto(f"packet-{index}".encode(), proxy.client_address)
            time.sleep(0.25)
            decisions.append((proxy.stats()["dropped"], proxy.stats()["reordered"]))
        finally:
            proxy.close()
            client.close()
            server.close()
    assert decisions[0] == decisions[1]


def test_fault_proxy_reorders_datagrams() -> None:
    server = nt.udp_socket(("127.0.0.1", 0))
    server.settimeout(1.0)
    proxy = DeterministicFaultProxy(
        server.getsockname(), FaultProfile(reorder_rate=1.0, seed=5)
    ).start()
    try:
        for index in range(3):
            proxy_client = nt.udp_socket(("127.0.0.1", 0))
            proxy_client.sendto(f"packet-{index}".encode(), proxy.client_address)
            proxy_client.close()
        time.sleep(0.2)
        received = [server.recvfrom(64)[0] for _ in range(2)]
        # The first datagram is held back, so the second one overtakes it.
        assert received[0] == b"packet-1"
        assert proxy.stats()["reordered"] >= 1
    finally:
        proxy.close()
        server.close()


def test_fault_proxy_rebind_changes_the_server_visible_address() -> None:
    server = nt.udp_socket(("127.0.0.1", 0))
    server.settimeout(1.0)
    proxy = DeterministicFaultProxy(server.getsockname(), FaultProfile(seed=3)).start()
    try:
        client = nt.udp_socket(("127.0.0.1", 0))
        client.sendto(b"first", proxy.client_address)
        first_addr = server.recvfrom(64)[1]
        proxy.rebind()
        time.sleep(0.05)
        client.sendto(b"second", proxy.client_address)
        second_addr = server.recvfrom(64)[1]
        assert second_addr != first_addr  # the NAT mapping moved
        assert proxy.stats()["rebinds"] == 1
        client.close()
    finally:
        proxy.close()
        server.close()


# ---------------------------------------------------------------------------
# RFC 5766 channels
# ---------------------------------------------------------------------------
def test_turn_channel_relays_payloads_in_both_directions() -> None:
    with TurnRelayServer({"alice": "s3cret", "bob": "hunter2"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        bob = TurnClient(turn.url, "bob", "hunter2")
        try:
            relay_a, relay_b = alice.allocate(), bob.allocate()
            alice.create_permission(relay_b[0])
            bob.create_permission(relay_a[0])
            number = alice.bind_channel(relay_b)
            assert nt.CHANNEL_NUMBER_MIN <= number <= nt.CHANNEL_NUMBER_MAX
            bob.bind_channel(relay_a)

            alice.send_channel(relay_b, b"through-a-channel")
            peer, payload = bob.recv_from()
            assert payload == b"through-a-channel" and peer[0] == "127.0.0.1"
            bob.send_channel(relay_a, b"and-back")
            _peer, back = alice.recv_from()
            assert back == b"and-back"
            assert len(turn.channels) == 2
        finally:
            alice.close()
            bob.close()


def test_turn_channel_rejects_a_second_bind_for_the_same_peer() -> None:
    with TurnRelayServer({"alice": "s3cret"}) as turn:
        alice = TurnClient(turn.url, "alice", "s3cret")
        try:
            alice.allocate()
            peer = ("127.0.0.1", 5000)
            alice.bind_channel(peer)
            with pytest.raises(nt.StunError):
                alice.bind_channel(peer)
        finally:
            alice.close()


def test_channel_data_framing_round_trip() -> None:
    encoded = nt.encode_channel_data(0x4001, b"payload")
    assert nt.is_channel_data(encoded)
    assert not nt.is_channel_data(b"\x00\x01\x00\x05")
    assert nt.decode_channel_data(encoded) == (0x4001, b"payload")
    with pytest.raises(nt.StunError):
        nt.decode_channel_data(encoded[:-2])


# ---------------------------------------------------------------------------
# Frame envelopes and the 0-RTT probe
# ---------------------------------------------------------------------------
def test_frame_envelope_round_trip_and_validation() -> None:
    envelope = FrameEnvelope(7, time.time(), b"hidden-state")
    decoded = FrameEnvelope.decode(envelope.encode())
    assert decoded.sequence == 7 and decoded.payload == b"hidden-state"
    assert abs(decoded.sent_at - envelope.sent_at) < 1e-6
    with pytest.raises(LinkProtocolError):
        FrameEnvelope.decode(b"short")
    with pytest.raises(LinkProtocolError):
        FrameEnvelope.decode(struct.pack("!dI", -1.0, 1))


def test_quic_0rtt_probe_resumes_behind_the_fault_proxy() -> None:
    async def run() -> None:
        identity = LinkIdentity.generate()
        received: list = []
        server = await QuicHiddenStateServer(
            identity, "127.0.0.1", 0, on_frame=received.append
        ).start()
        proxy = DeterministicFaultProxy(
            (server.host, server.port), FaultProfile(seed=17)
        ).start()
        try:
            frames = [
                FrameEnvelope(index, time.time(), b"x" * 64).encode() for index in range(8)
            ]
            probe = await quic_0rtt_resumption_probe(
                server, frames, client_target=proxy.client_address
            )
            assert probe.ok
            assert probe.session_resumed and probe.early_data_accepted
            assert probe.server_early_frames > 0
            assert len(probe.frame_latencies_ms) == len(frames)
            assert all(latency >= 0 for latency in probe.frame_latencies_ms)
            assert probe.handshake_latency_ms is not None
        finally:
            proxy.close()
            server.close()

    asyncio.run(run())


def test_quic_0rtt_probe_reports_an_unreachable_path() -> None:
    async def run() -> None:
        identity = LinkIdentity.generate()
        server = await QuicHiddenStateServer(identity, "127.0.0.1", 0).start()
        try:
            probe = await quic_0rtt_resumption_probe(
                server, [b"x"], client_target=("127.0.0.1", 1), timeout=0.3
            )
            assert not probe.ok
            assert probe.error == "TimeoutError"
        finally:
            server.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Strike register and telemetry helpers
# ---------------------------------------------------------------------------
def test_strike_register_rejects_a_replayed_nonce() -> None:
    strike = StrikeRegister("unit-test-peer")
    nonce = strike.open_session()
    assert strike.register(nonce)
    assert not strike.replay(nonce)
    assert strike.rejection_rate == 0.5
    payload = strike.to_dict()
    assert payload["accepted"] == 1 and payload["rejected"] == 1
    assert any("replay_nonce_detected" in reason for reason in payload["rejection_reasons"])


def test_latency_summary_reports_p50_p95_p99() -> None:
    values = [float(index) for index in range(1, 101)]
    summary = latency_summary(values)
    assert summary["count"] == 100
    assert summary["p50_ms"] == 50.0
    assert summary["p95_ms"] == 95.0
    assert summary["p99_ms"] == 99.0
    assert percentile([], 0.5) == 0.0
    with pytest.raises(ValueError):
        percentile([1.0], 0.0)


def test_default_scenarios_cover_the_requested_loss_rates() -> None:
    scenarios = default_scenarios()
    assert [scenario.profile.loss_rate for scenario in scenarios] == list(DEFAULT_LOSS_RATES)
    assert [scenario.scenario_id for scenario in scenarios] == [
        "loss_0pct",
        "loss_1pct",
        "loss_3pct",
        "loss_5pct",
    ]
    assert all(scenario.nat_rebinds >= 1 for scenario in scenarios)


# ---------------------------------------------------------------------------
# The benchmark itself
# ---------------------------------------------------------------------------
def test_fault_scenario_reports_the_required_telemetry() -> None:
    scenario = FaultScenario(
        scenario_id="unit_loss_1pct",
        profile=FaultProfile(loss_rate=0.01, reorder_rate=0.02, jitter_ms=1.0, seed=101),
        nat_rebinds=1,
        frames=6,
        frame_bytes=128,
    )
    report = asyncio.run(run_fault_scenario(scenario))

    assert report["scenario_id"] == "unit_loss_1pct"
    assert report["packet_loss_rate"] == 0.01
    latency = report["frame_delivery_latency_ms"]
    for key in ("p50_ms", "p95_ms", "p99_ms", "count"):
        assert key in latency
    assert latency["count"] > 0

    quic = report["quic_0rtt"]
    assert quic["session_resumed"] and quic["early_data_accepted"]
    assert quic["frames_delivered"] > 0
    assert quic["resumption_latency_ms"] is not None

    turn = report["turn_relay"]
    assert turn["allocate_ok"]
    assert turn["channel_number"] >= nt.CHANNEL_NUMBER_MIN
    assert turn["data_relay"]["frames_delivered"] > 0
    assert turn["channel_relay"]["frames_delivered"] > 0
    assert turn["nat_rebind_events"] == 1
    assert turn["reallocations_after_rebind"] == 1

    websocket = report["websocket_fallback"]
    assert websocket["protocol"] == "libp2p_websocket"
    assert websocket["frames_delivered"] == scenario.frames

    register = report["zero_rtt_strike_register"]
    assert register["rejected"] >= 1
    assert register["rejection_rate"] > 0.0

    pacing = report["bbr_v1_pacing"]
    assert pacing["model"] == "bbr_v1_delivery_rate_max_filter"
    assert pacing["cwnd_bytes"] > 0
    assert pacing["gain"] > 0
    assert [sample["leg"] for sample in pacing["pacing_samples"]]


def test_write_fault_report_produces_valid_json(tmp_path: Path) -> None:
    scenario = FaultScenario(
        scenario_id="unit_loss_0pct",
        profile=FaultProfile(seed=7),
        nat_rebinds=0,
        frames=4,
        frame_bytes=64,
    )
    report = asyncio.run(run_fault_scenario(scenario))
    target = write_fault_report(
        {"schema": REPORT_SCHEMA, "scenarios": [report]}, tmp_path / "r.json"
    )
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == REPORT_SCHEMA
    assert payload["scenarios"][0]["scenario_id"] == "unit_loss_0pct"


# ---------------------------------------------------------------------------
# Multi-host mode
# ---------------------------------------------------------------------------
def test_external_endpoints_are_read_from_the_environment(tmp_path: Path) -> None:
    pem = tmp_path / "ca.pem"
    pem.write_text("-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n", encoding="utf-8")
    endpoints = ExternalEndpoints.from_environment(
        {
            "HIVEMIND_WAN_QUIC_HOST": "198.51.100.7",
            "HIVEMIND_WAN_QUIC_PORT": "4433",
            "HIVEMIND_WAN_QUIC_CA_PEM": str(pem),
            "HIVEMIND_WAN_TURN_HOST": "198.51.100.9",
            "HIVEMIND_WAN_TURN_PORT": "3478",
            "HIVEMIND_WAN_TURN_USERNAME": "alice",
            "HIVEMIND_WAN_TURN_PASSWORD": "s3cret",
        }
    )
    assert endpoints.has_quic and endpoints.has_turn
    assert endpoints.quic_host == "198.51.100.7" and endpoints.quic_port == 4433
    assert "BEGIN CERTIFICATE" in (endpoints.quic_ca_pem or "")
    assert endpoints.turn_username == "alice"
    assert endpoints.to_dict()["mode"] == "multi_host"
    assert ExternalEndpoints.from_environment({}).to_dict()["mode"] == "local_wan_simulation"


def test_multi_host_mode_drives_a_remote_quic_receiver() -> None:
    async def run() -> None:
        identity = LinkIdentity.generate()
        server = await QuicHiddenStateServer(identity, "127.0.0.1", 0).start()
        try:
            endpoints = ExternalEndpoints(
                quic_host="127.0.0.1", quic_port=server.port, quic_ca_pem=identity.ca_pem
            )
            scenario = FaultScenario(
                scenario_id="unit_multi_host",
                profile=FaultProfile(seed=5),
                nat_rebinds=0,
                frames=4,
                frame_bytes=64,
            )
            report = await run_fault_scenario(scenario, endpoints=endpoints)
            quic = report["quic_0rtt"]
            assert quic["session_resumed"] and quic["early_data_accepted"]
            assert quic["frames_delivered"] == scenario.frames
            assert quic["handshake_latency_ms"] is not None
        finally:
            server.close()

    asyncio.run(run())
