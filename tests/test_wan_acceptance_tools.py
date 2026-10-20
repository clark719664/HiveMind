"""Loopback/mock acceptance checks only; no WAN or two-NAT proof."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import queue
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "wan_acceptance"
sys.path.insert(0, str(TOOLS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frames_are_deterministic_and_tamper_evident():
    frames = _load("wan_frames")
    frame = frames.make_frame(7, 3, 2, 300)
    assert len(frame) == 300 and frame == frames.make_frame(7, 3, 2, 300)
    assert frames.verify_frame(frame, 7, 300) == (True, 3, 2)
    flipped = bytearray(frame)
    flipped[-1] ^= 1
    assert frames.verify_frame(bytes(flipped), 7)[0] is False
    assert frames.verify_frame(frame, 8)[0] is False
    assert frames.verify_frame(frame, 7, 4096)[0] is False
    # Every short prefix, including the former >=64-byte false positive, fails without raising.
    assert all(not frames.verify_frame(frame[:end], 7)[0] for end in range(len(frame)))
    assert frames.verify_frame(frame + b"extra", 7)[0] is False
    with pytest.raises(ValueError):
        frames.make_frame(1, 0, 0, 8)


def test_tracker_rejects_missing_duplicate_unexpected_and_wrong_run_frames():
    frames = _load("wan_frames")
    tracker = frames.FrameTracker(9, 2, 2, 256)
    assert not tracker.summary()["accepted"] and tracker.summary()["missing_frames"] == 4
    # Out-of-order delivery is permitted, but must still include every unique expected frame.
    for batch, index in ((1, 1), (0, 1), (1, 0), (0, 0)):
        assert tracker.observe(frames.make_frame(9, batch, index, 256))["valid"]
    assert tracker.summary()["accepted"] and tracker.summary()["out_of_order_frames"] > 0
    tracker.observe(frames.make_frame(9, 0, 0, 256))
    tracker.observe(frames.make_frame(9, 2, 0, 256))
    tracker.observe(frames.make_frame(8, 0, 0, 256))
    tracker.observe(frames.make_frame(9, 0, 0, 128))
    summary = tracker.summary()
    assert not summary["accepted"] and summary["duplicate_frames"] == 1
    assert summary["invalid_frames"] == 3 and summary["unique_valid_frames"] == 4


@pytest.fixture
def receiver(tmp_path):
    processes = []

    def start(batches=2, frames=3, size=256):
        directory = tmp_path / f"recv-{len(processes)}"
        stop_file = tmp_path / f"stop-{len(processes)}"
        proc = subprocess.Popen(
            [sys.executable, str(TOOLS / "wan_link_receiver.py"), "--bind", "127.0.0.1", "--quic-port", "0",
             "--wss-port", "0", "--out-dir", str(directory), "--duration", "30", "--seed", "7001",
             "--batches", str(batches), "--frames", str(frames), "--size", str(size),
             "--stop-file", str(stop_file)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
        processes.append((proc, stop_file))
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
        try:
            ready = json.loads(lines.get(timeout=20))
        except Exception:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=10)
            pytest.fail(f"receiver did not become ready: {stdout} {stderr}")

        def stop():
            stop_file.touch()
            stdout, stderr = proc.communicate(timeout=10)
            summary = json.loads((directory / "summary.json").read_text())
            records = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
            return proc.returncode, summary, records

        return ready, directory, stop

    yield start
    for proc, stop_file in processes:
        if proc.poll() is None:
            stop_file.touch()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate(timeout=10)


def _send(ready, tmp_path, *extra, quic_port=None):
    out = tmp_path / "send.jsonl"
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "wan_link_sender.py"), "--host", "127.0.0.1",
         "--quic-port", str(quic_port or ready["quic_udp_port"]), "--wss-port", str(ready["wss_tcp_port"]),
         "--ca", ready["ca_pem_path"], "--seed", "7001", "--frames", "3", "--size", "256", "--interval", "0.05",
         "--out", str(out), *extra],
        capture_output=True, text=True, cwd=ROOT, timeout=20)
    records = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    return proc, records


def _assert_evidence(tmp_path, directory, summary, received, sent):
    assert summary["accepted"] and summary["missing_frames"] == 0
    assert summary["invalid_frames"] == summary["duplicate_frames"] == 0
    expected = {(r["batch"], i): digest for r in sent for i, digest in enumerate(r["frame_sha256"])}
    actual = {(r["batch"], r["index"]): r["sha256"] for r in received}
    assert actual == expected and len(received) == len(expected)
    assert all(r["intact"] and r["valid"] and r["bytes"] == 256 for r in received)
    verifier = _load("wan_verify")
    assert verifier.reconcile(tmp_path / "send.jsonl", directory)["accepted"]


def test_loopback_quic_delivery_is_acknowledged_and_verified(receiver, tmp_path):
    ready, directory, stop = receiver()
    proc, sent = _send(ready, tmp_path, "--batches", "2", "--expect-ca-sha256", ready["ca_sha256"])
    assert proc.returncode == 0, proc.stderr
    assert all(r["ok"] and r["transport"] in ("quic", "quic_0rtt") and r["frames_delivered"] == 3 for r in sent)
    code, summary, received = stop()
    assert code == 0
    _assert_evidence(tmp_path, directory, summary, received, sent)
    verifier = _load("wan_verify")
    # A truncated artifact, duplicate identity, or changed digest cannot pass reconciliation.
    for damaged in (received[:-1], received + received[:1],
                    [{**received[0], "sha256": "0" * 64}, *received[1:]]):
        (directory / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in damaged))
        with pytest.raises(ValueError):
            verifier.reconcile(tmp_path / "send.jsonl", directory)


def test_loopback_wss_fallback_when_udp_port_is_dead(receiver, tmp_path):
    ready, directory, stop = receiver(batches=1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        # Keep the port owned by this test while it deliberately ignores QUIC datagrams.
        proc, sent = _send(ready, tmp_path, "--batches", "1", "--timeout", "0.6", quic_port=dead_port)
    assert proc.returncode == 0, proc.stderr
    assert sent[0]["transport"] == "libp2p_websocket" and sent[0]["fallback_reason"]
    assert sent[0]["elapsed_s"] >= 0.6
    code, summary, received = stop()
    assert code == 0 and all(r["transport"] == "wss" for r in received)
    _assert_evidence(tmp_path, directory, summary, received, sent)


def test_sender_refuses_a_ca_with_the_wrong_fingerprint(receiver, tmp_path):
    ready, _, stop = receiver(batches=1)
    proc, records = _send(ready, tmp_path, "--batches", "1", "--expect-ca-sha256", "00" * 32)
    assert proc.returncode == 2 and records == [] and "mismatch" in proc.stderr
    code, summary, received = stop()
    assert code == 1 and not summary["accepted"] and not received


@pytest.mark.parametrize("damage", ["truncated", "corrupt", "wrong_size", "wrong_seed", "missing", "duplicate"])
def test_real_receiver_rejects_incomplete_or_damaged_run(receiver, damage):
    from hivemind.swarm.quic_link import quic_send_frames

    ready, _, stop = receiver(batches=1, frames=2)
    frames = _load("wan_frames")
    payloads = [frames.make_frame(7001, 0, i, 256) for i in range(2)]
    if damage == "truncated":
        payloads[0] = payloads[0][:64]
    elif damage == "corrupt":
        payloads[0] = payloads[0][:-1] + bytes([payloads[0][-1] ^ 1])
    elif damage == "wrong_size":
        payloads[0] = frames.make_frame(7001, 0, 0, 128)
    elif damage == "wrong_seed":
        payloads[0] = frames.make_frame(7002, 0, 0, 256)
    elif damage == "missing":
        payloads.pop()
    else:
        payloads.append(payloads[0])
    report = asyncio.run(quic_send_frames("127.0.0.1", ready["quic_udp_port"], payloads,
                                         Path(ready["ca_pem_path"]).read_text()))
    # The core transport acknowledges counts even for an invalid acceptance payload.
    assert report.delivered == len(payloads)
    code, summary, _ = stop()
    assert code == 1 and not summary["accepted"]
    assert summary["missing_frames"] or summary["duplicate_frames"]


@pytest.mark.parametrize("option,value", [("--batches", "0"), ("--batches", "-1"), ("--frames", "0"),
    ("--frames", "-2"), ("--size", "63"), ("--size", "1048577"), ("--seed", "-1"),
    ("--interval", "nan"), ("--timeout", "inf"), ("--batches", "10001")])
def test_sender_invalid_plan_fails_before_network_or_output(tmp_path, monkeypatch, option, value):
    sender = _load("wan_link_sender")
    monkeypatch.setattr(sender, "amain", lambda args: pytest.fail("invalid arguments reached network code"))
    out = tmp_path / "send.jsonl"
    with pytest.raises(SystemExit) as exc:
        sender.main(["--host", "127.0.0.1", "--quic-port", "1234", "--ca", "nonexistent",
                     "--seed", "7001", "--out", str(out), option, value])
    assert exc.value.code == 2 and not out.exists()


def test_sender_measures_entire_fallback_attempt(tmp_path, monkeypatch):
    sender = _load("wan_link_sender")
    from hivemind.swarm.quic_link import LinkIdentity

    ca = tmp_path / "ca.pem"
    ca.write_text(LinkIdentity.generate().ca_pem)
    clock = [100.0]

    class FakeLink:
        def __init__(self, *args, **kwargs):
            pass

        async def send_frames(self, frames):
            clock[0] += 3.2
            return SimpleNamespace(frames_delivered=len(frames), protocol="libp2p_websocket",
                session_resumed=False, early_data_accepted=False, fallback_reason="TimeoutError", elapsed_s=0.2)

    monkeypatch.setattr(sender, "RemoteHiddenStateLink", FakeLink)
    monkeypatch.setattr(sender.time, "perf_counter", lambda: clock[0])
    out = tmp_path / "send.jsonl"
    assert sender.main(["--host", "127.0.0.1", "--quic-port", "1234", "--ca", str(ca), "--seed", "1",
                        "--batches", "1", "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["elapsed_s"] == 3.2 and record["transport_elapsed_s"] == 0.2


@pytest.mark.parametrize("kind", ["startup_failure", "cancellation"])
def test_receiver_closes_owned_resources_and_writes_failure_summary(tmp_path, monkeypatch, kind):
    receiver = _load("wan_link_receiver")
    closed = []

    class FakeQuic:
        port = 1234
        frames_received = early_frames = 0
        handshakes = []

        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return self

        def close(self):
            closed.append("quic")

    class FakeWss:
        port = 1235
        frames_received = 0

        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            if kind == "startup_failure":
                raise OSError("simulated occupied WSS port")
            raise asyncio.CancelledError()

        async def close(self):
            closed.append("wss")

    monkeypatch.setattr(receiver, "QuicHiddenStateServer", FakeQuic)
    monkeypatch.setattr(receiver, "WebSocketHiddenStateServer", FakeWss)
    out = tmp_path / "recv"
    argv = ["--quic-port", "0", "--wss-port", "0", "--seed", "1", "--batches", "1",
            "--frames", "1", "--size", "256", "--duration", "1", "--out-dir", str(out)]
    if kind == "startup_failure":
        with pytest.raises(OSError, match="occupied WSS"):
            receiver.main(argv)
    else:
        assert receiver.main(argv) == 1
    assert closed == ["quic", "wss"]
    assert json.loads((out / "summary.json").read_text())["accepted"] is False
    # Opening for append also catches a leaked live handle on Windows.
    with (out / "frames.jsonl").open("a") as evidence:
        evidence.write("")


def test_sender_closes_evidence_on_cancel_without_replaying(tmp_path, monkeypatch):
    sender = _load("wan_link_sender")
    from hivemind.swarm.quic_link import LinkIdentity

    ca = tmp_path / "ca.pem"
    ca.write_text(LinkIdentity.generate().ca_pem)
    opened = []
    real_open = open

    def capture_open(*args, **kwargs):
        file = real_open(*args, **kwargs)
        opened.append(file)
        return file

    class CancelledLink:
        def __init__(self, *args, **kwargs):
            pass

        async def send_frames(self, frames):
            raise asyncio.CancelledError()

    monkeypatch.setattr(sender, "open", capture_open, raising=False)
    monkeypatch.setattr(sender, "RemoteHiddenStateLink", CancelledLink)
    with pytest.raises(asyncio.CancelledError):
        sender.main(["--host", "127.0.0.1", "--quic-port", "1234", "--ca", str(ca), "--seed", "1",
                     "--batches", "1", "--out", str(tmp_path / "send.jsonl")])
    assert len(opened) == 1 and opened[0].closed


# ---------------------------------------------------------------- NAT probe (loopback servers only)


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("wan_nat_probe_under_test", TOOLS / "wan_nat_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


def test_probe_stun_reports_mapping_and_needs_two_servers_to_classify(probe, capsys):
    from hivemind.swarm import nat_traversal as nt

    with nt.StunServer("127.0.0.1", 0) as first, nt.StunServer("127.0.0.1", 0) as second:
        assert probe.main(["stun", "--server", first.url, "--server", second.url]) == 0
        two = json.loads(capsys.readouterr().out)
        assert two["behaviour"] == "no_nat" and two["hole_punchable"] is True
        assert probe.main(["stun", "--server", first.url]) == 0
        one = json.loads(capsys.readouterr().out)
        assert one["behaviour"].startswith("unknown") and one["mapped"][0] == "127.0.0.1"
    assert one["socket_closed_after_result"] and not one["mapping_reusable_for_punch"]


def test_probe_redaction_masks_ipv4(probe, capsys):
    probe.emit({"mapped": ["203.0.113.77", 4000], "note": "198.51.100.9"}, redact=True)
    text = capsys.readouterr().out
    assert "203.0.x.x" in text and "113.77" not in text and "198.51.x.x" in text


def test_probe_hole_punch_between_two_local_sockets(probe, monkeypatch):
    results = []
    monkeypatch.setattr(probe, "emit", lambda payload, redact: results.append(payload))

    def free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port_a, port_b = free_port(), free_port()
    codes = {}

    def side(name, local, peer):
        argv = ["punch", "--local-port", str(local), "--peer", f"127.0.0.1:{peer}", "--session", "acceptance-test",
                "--punch-timeout", "5"]
        codes[name] = probe.main(argv)

    threads = [threading.Thread(target=side, args=("a", port_a, port_b)), threading.Thread(target=side, args=("b", port_b, port_a))]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert codes == {"a": 0, "b": 0}
    assert all(r["succeeded"] for r in results) and len(results) == 2
    assert all(r["wan_acceptance"] == "UNSUPPORTED" for r in results)


def test_probe_turn_relay_between_two_allocations(probe, monkeypatch):
    from hivemind.swarm import nat_traversal as nt

    emitted = []
    monkeypatch.setattr(probe, "emit", lambda payload, redact: emitted.append(payload))
    monkeypatch.setenv("TURN_USERNAME", "acceptance")
    monkeypatch.setenv("TURN_PASSWORD", "local-test-only")
    with nt.TurnRelayServer({"acceptance": "local-test-only"}, host="127.0.0.1") as server:
        code = {}
        listener = threading.Thread(target=lambda: code.update(listen=probe.main(
            ["turn-listen", "--server", server.url, "--peer-ip", "127.0.0.1", "--wait", "10"])))
        listener.start()
        for _ in range(100):
            if emitted:
                break
            threading.Event().wait(0.1)
        relay = emitted[0]["relayed_address"]
        sent = probe.main(["turn-send", "--server", server.url, "--relay-peer", f"{relay[0]}:{relay[1]}", "--payload", "through-turn"])
        listener.join(30)
    assert sent == 0 and code["listen"] == 0
    received = [e for e in emitted if e.get("received")]
    assert received and received[0]["bytes"] == len(b"through-turn")


def test_probe_turn_requires_credentials_in_environment(probe, monkeypatch):
    monkeypatch.delenv("TURN_USERNAME", raising=False)
    monkeypatch.delenv("TURN_PASSWORD", raising=False)
    with pytest.raises(SystemExit):
        probe.main(["turn-send", "--server", "127.0.0.1:3478", "--relay-peer", "127.0.0.1:9"])
