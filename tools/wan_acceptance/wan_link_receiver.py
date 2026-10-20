#!/usr/bin/env python
"""WAN acceptance RECEIVER: a QUIC (UDP) endpoint and an optional WSS (TCP) endpoint.

Uses the real hivemind.swarm.quic_link servers. Run it ONLY inside an environment you own and are
authorised to test. It generates an ephemeral CA + server certificate, writes the PUBLIC ca.pem for
the sender, and logs one JSON line per received frame (size, full SHA-256 digest, integrity result).

    python tools/wan_acceptance/wan_link_receiver.py --bind 0.0.0.0 --quic-port 4433 --wss-port 4434 \
        --out-dir wan_recv --duration 900 --seed 12345 --batches 20 --frames 8 --size 4096

The default bind is loopback. Binding 0.0.0.0 exposes listeners on all interfaces,
including public ones if present. Use only owner-authorized hosts and ports; router/firewall
changes remain the operator's separate, deliberate step. The private key never touches disk except in a
temporary file that is deleted as soon as the TLS context is loaded.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import ssl
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for extra in (HERE, REPO_ROOT / "src"):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from cryptography.hazmat.primitives import serialization  # noqa: E402

from hivemind.swarm.quic_link import LinkIdentity, QuicHiddenStateServer, WebSocketHiddenStateServer  # noqa: E402
from wan_frames import FrameTracker, validate_plan, validate_seconds  # noqa: E402


def server_tls_context(identity: LinkIdentity) -> ssl.SSLContext:
    with tempfile.TemporaryDirectory(prefix="hm-wan-tls-") as directory:
        cert, key = Path(directory) / "cert.pem", Path(directory) / "key.pem"
        cert.write_bytes(identity.certificate.public_bytes(serialization.Encoding.PEM))
        key.touch(mode=0o600)
        key.write_bytes(identity.private_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
    return context


async def amain(args: argparse.Namespace) -> int:
    # A fresh directory prevents mixing runs or replacing a CA while another run is active.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    tracker = FrameTracker(args.seed, args.batches, args.frames, args.size)
    counts: Counter = Counter()
    quic = wss = None
    started = False
    log = open(out_dir / "frames.jsonl", "x", encoding="utf-8", buffering=1)

    def handler_for(transport: str):
        def handle(frame: bytes) -> None:
            if tracker.received >= args.batches * args.frames + 1024:
                counts["receive_limit_exceeded"] += 1
                return
            counts[f"{transport}_frames"] += 1
            try:
                record = tracker.observe(frame)
                record.update({
                    "t_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                    "t_mono": round(time.monotonic(), 6), "transport": transport})
                log.write(json.dumps(record) + "\n")
            except Exception as exc:  # a handler failure must never produce acceptance
                counts["handler_errors"] += 1
                log.write(json.dumps({"handler_error": type(exc).__name__}) + "\n")
        return handle

    try:
        identity = LinkIdentity.generate(args.server_name, days=args.cert_days)
        (out_dir / "ca.pem").write_text(identity.ca_pem, encoding="ascii")
        ca_sha256 = hashlib.sha256(identity.ca_certificate.public_bytes(serialization.Encoding.DER)).hexdigest()
        quic = QuicHiddenStateServer(identity, args.bind, args.quic_port, on_frame=handler_for("quic"))
        await quic.start()
        if args.wss_port is not None:
            wss = WebSocketHiddenStateServer(args.bind, args.wss_port, on_frame=handler_for("wss"),
                                            ssl_context=server_tls_context(identity))
            await wss.start()
        started = True
        print(json.dumps({"ready": True, "quic_udp_port": quic.port, "wss_tcp_port": wss.port if wss else None,
                          "ca_pem_path": str(out_dir / "ca.pem"), "ca_sha256": ca_sha256,
                          "server_name": args.server_name, "plan": tracker.summary()}), flush=True)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            if counts["receive_limit_exceeded"] or (args.stop_file and Path(args.stop_file).exists()):
                break
            await asyncio.sleep(0.05)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Stop listeners before taking the final snapshot or closing the evidence log.
        if quic is not None:
            quic.close()
        if wss is not None:
            await wss.close()
        summary = {
            **tracker.summary(), "counts": dict(counts),
            "quic_frames_received": quic.frames_received if quic else 0,
            "quic_early_frames": quic.early_frames if quic else 0,
            "quic_handshakes": quic.handshakes if quic else [],
            "wss_frames_received": wss.frames_received if wss else None,
            "note": "Receiver acceptance is separate from transport count ACKs. Out-of-order intact unique frames are allowed. Handshake list capped at 128. No WAN or migration claim.",
        }
        summary["accepted"] = summary["accepted"] and started and not counts["handler_errors"] and not counts["receive_limit_exceeded"]
        log.close()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"summary": summary}), flush=True)
    return 0 if summary["accepted"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="127.0.0.1", help="interface to listen on; 0.0.0.0 only in an environment you own")
    ap.add_argument("--quic-port", type=int, default=4433, help="UDP port (0 = pick one)")
    ap.add_argument("--wss-port", type=int, default=None, help="TCP port for the TLS WebSocket fallback; omit to disable")
    ap.add_argument("--server-name", default="hivemind.local", help="certificate name the sender verifies")
    ap.add_argument("--cert-days", type=int, default=2)
    ap.add_argument("--seed", type=int, required=True, help="fresh non-secret run identity shared with sender")
    ap.add_argument("--batches", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True, help="expected frames per batch")
    ap.add_argument("--size", type=int, required=True, help="expected full frame size in bytes")
    ap.add_argument("--stop-file", help="stop cleanly when this local file appears; must not already exist")
    ap.add_argument("--duration", type=float, default=900.0, help="seconds to stay up")
    ap.add_argument("--out-dir", default="wan_recv")
    args = ap.parse_args(argv)
    try:
        validate_plan(args.seed, args.batches, args.frames, args.size)
        validate_seconds(args.duration, "duration", 86400)
        if not 1 <= args.cert_days <= 7:
            raise ValueError("cert-days must be 1..7")
        for port in (args.quic_port, args.wss_port):
            if port is not None and not 0 <= port <= 65535:
                raise ValueError("receiver ports must be 0..65535")
        if args.stop_file and Path(args.stop_file).exists():
            raise ValueError("stop-file must not already exist")
    except ValueError as exc:
        ap.error(str(exc))
    return asyncio.run(amain(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
