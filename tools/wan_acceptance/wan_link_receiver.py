#!/usr/bin/env python
"""WAN acceptance RECEIVER: a QUIC (UDP) endpoint and an optional WSS (TCP) endpoint.

Uses the real hivemind.swarm.quic_link servers. Run it ONLY inside an environment you own and are
authorised to test. It generates an ephemeral CA + server certificate, writes the PUBLIC ca.pem for
the sender, and logs one JSON line per received frame (size, short digest, integrity result).

    python tools/wan_acceptance/wan_link_receiver.py --bind 0.0.0.0 --quic-port 4433 --wss-port 4434 \
        --out-dir wan_recv --duration 900

Nothing is opened to the internet by this script; router/firewall forwarding is the operator's
separate, deliberate step in their own environment. The private key never touches disk except in a
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
from wan_frames import verify_frame  # noqa: E402


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = LinkIdentity.generate(args.server_name, days=args.cert_days)
    (out_dir / "ca.pem").write_text(identity.ca_pem, encoding="ascii")
    ca_sha256 = hashlib.sha256(identity.ca_certificate.public_bytes(serialization.Encoding.DER)).hexdigest()
    log = open(out_dir / "frames.jsonl", "w", encoding="utf-8", buffering=1)
    counts: Counter = Counter()

    def handler_for(transport: str):
        def handle(frame: bytes) -> None:
            try:
                intact, batch, index = verify_frame(frame, args.seed)
                counts[f"{transport}_frames"] += 1
                counts[f"{transport}_intact" if intact else f"{transport}_corrupt"] += 1
                log.write(json.dumps({
                    "t_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                    "t_mono": round(time.monotonic(), 6), "transport": transport, "bytes": len(frame),
                    "batch": batch, "index": index, "intact": intact,
                    "sha256_16": hashlib.sha256(frame).hexdigest()[:16]}) + "\n")
            except Exception as exc:  # noqa: BLE001 - a raising handler would reset the connection
                counts["handler_errors"] += 1
                log.write(json.dumps({"handler_error": type(exc).__name__}) + "\n")
        return handle

    quic = await QuicHiddenStateServer(identity, args.bind, args.quic_port, on_frame=handler_for("quic")).start()
    wss = None
    if args.wss_port is not None:
        wss = await WebSocketHiddenStateServer(args.bind, args.wss_port, on_frame=handler_for("wss"),
                                               ssl_context=server_tls_context(identity)).start()
    print(json.dumps({"ready": True, "quic_udp_port": quic.port, "wss_tcp_port": wss.port if wss else None,
                      "ca_pem_path": str(out_dir / "ca.pem"), "ca_sha256": ca_sha256,
                      "server_name": args.server_name}), flush=True)
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        summary = {
            "counts": dict(counts), "quic_frames_received": quic.frames_received,
            "quic_early_frames": quic.early_frames, "quic_handshakes": quic.handshakes,
            "wss_frames_received": wss.frames_received if wss else None,
            "note": "handshakes list is capped at the last 128 by the server; early_frames counts frames that arrived before handshake completion (0-RTT)",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        quic.close()
        if wss is not None:
            await wss.close()
        log.close()
        print(json.dumps({"summary": summary}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="127.0.0.1", help="interface to listen on; 0.0.0.0 only in an environment you own")
    ap.add_argument("--quic-port", type=int, default=4433, help="UDP port (0 = pick one)")
    ap.add_argument("--wss-port", type=int, default=None, help="TCP port for the TLS WebSocket fallback; omit to disable")
    ap.add_argument("--server-name", default="hivemind.local", help="certificate name the sender verifies")
    ap.add_argument("--cert-days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=900.0, help="seconds to stay up")
    ap.add_argument("--out-dir", default="wan_recv")
    return asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
