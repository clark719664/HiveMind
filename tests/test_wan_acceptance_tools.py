"""Loopback-only tests for tools/wan_acceptance. They prove the helper scripts drive the real
quic_link APIs correctly; they say nothing about any WAN, NAT or second environment."""
from __future__ import annotations

import importlib.util
import json
import queue
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "wan_acceptance"


def _load_frames():
    spec = importlib.util.spec_from_file_location("wan_frames_under_test", TOOLS / "wan_frames.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frames_are_deterministic_and_tamper_evident():
    frames = _load_frames()
    frame = frames.make_frame(7, 3, 2, 300)
    assert len(frame) == 300 and frame == frames.make_frame(7, 3, 2, 300)
    assert frames.verify_frame(frame, 7) == (True, 3, 2)
    flipped = bytearray(frame)
    flipped[-1] ^= 1
    assert frames.verify_frame(bytes(flipped), 7)[0] is False
    assert frames.verify_frame(frame, 8)[0] is False
    assert frames.verify_frame(b"short", 7) == (False, -1, -1)
    with pytest.raises(ValueError):
        frames.make_frame(1, 0, 0, 8)


@pytest.fixture
def receiver(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, str(TOOLS / "wan_link_receiver.py"), "--bind", "127.0.0.1", "--quic-port", "0",
         "--wss-port", "0", "--out-dir", str(tmp_path / "recv"), "--duration", "120"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
    try:
        ready = json.loads(lines.get(timeout=120))
    except Exception:
        proc.kill()
        raise
    yield ready, tmp_path
    proc.kill()
    proc.wait(timeout=20)


def _send(ready, tmp_path, *extra, quic_port=None):
    out = tmp_path / "send.jsonl"
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "wan_link_sender.py"), "--host", "127.0.0.1",
         "--quic-port", str(quic_port or ready["quic_udp_port"]), "--wss-port", str(ready["wss_tcp_port"]),
         "--ca", ready["ca_pem_path"], "--frames", "3", "--size", "256", "--interval", "0.05",
         "--out", str(out), *extra],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    return proc, records


def test_loopback_quic_delivery_is_acknowledged_and_verified(receiver):
    ready, tmp_path = receiver
    proc, records = _send(ready, tmp_path, "--batches", "2", "--expect-ca-sha256", ready["ca_sha256"])
    assert proc.returncode == 0, proc.stderr
    assert [r["ok"] for r in records] == [True, True]
    assert all(r["transport"] in ("quic", "quic_0rtt") and r["frames_delivered"] == 3 for r in records)


def test_loopback_wss_fallback_when_udp_port_is_dead(receiver):
    ready, tmp_path = receiver
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    proc, records = _send(ready, tmp_path, "--batches", "1", "--timeout", "0.6", quic_port=dead_port)
    assert proc.returncode == 0, proc.stderr
    assert records[0]["transport"] == "libp2p_websocket" and records[0]["fallback_reason"]


def test_sender_refuses_a_ca_with_the_wrong_fingerprint(receiver):
    ready, tmp_path = receiver
    proc, records = _send(ready, tmp_path, "--batches", "1", "--expect-ca-sha256", "00" * 32)
    assert proc.returncode == 2 and records == []
    assert "mismatch" in proc.stderr


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
