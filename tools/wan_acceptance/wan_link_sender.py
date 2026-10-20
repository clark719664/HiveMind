#!/usr/bin/env python
"""WAN acceptance SENDER: pushes self-verifying frames through hivemind.swarm.quic_link.RemoteHiddenStateLink.

Run it ONLY from an environment you own and are authorised to test, towards a receiver you
control (tools/wan_acceptance/wan_link_receiver.py). One JSON line per batch records which transport
actually carried the batch (quic_0rtt / quic / libp2p_websocket), whether the TLS session was
resumed, the fallback reason, latency and any error. It never retries a batch whose delivery was
uncertain and never claims connection migration (each batch opens a fresh UDP socket).

    python tools/wan_acceptance/wan_link_sender.py --host <receiver address> --quic-port 4433 \
        --wss-port 4434 --ca wan_recv/ca.pem --expect-ca-sha256 <value printed by the receiver> \
        --batches 20 --frames 8 --size 4096 --interval 2 --out wan_send.jsonl

Exit code: 0 every batch acknowledged; 1 at least one batch failed; 2 bad arguments / CA mismatch.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import ssl
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for extra in (HERE, REPO_ROOT / "src"):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from hivemind.swarm.quic_link import (  # noqa: E402
    DeliveryUncertainError, LinkEndpoint, RemoteHiddenStateLink,
)
from wan_frames import make_frame  # noqa: E402


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


async def amain(args: argparse.Namespace) -> int:
    ca_pem = Path(args.ca).read_text(encoding="ascii")
    if args.expect_ca_sha256:
        der = ssl.PEM_cert_to_DER_cert(ca_pem)
        if hashlib.sha256(der).hexdigest() != args.expect_ca_sha256.lower():
            print("CA fingerprint mismatch: refusing to send", file=sys.stderr)
            return 2
    websocket_url = f"wss://{args.host}:{args.wss_port}" if args.wss_port else None
    endpoint = LinkEndpoint(args.host, args.quic_port, ca_pem, server_name=args.server_name, websocket_url=websocket_url)
    link = RemoteHiddenStateLink(endpoint, timeout=args.timeout)
    failures = 0
    out = open(args.out, "w", encoding="utf-8", buffering=1)
    for batch in range(args.batches):
        frames = [make_frame(args.seed, batch, i, args.size) for i in range(args.frames)]
        record = {"batch": batch, "t_utc": utc_now(), "frames_sent": len(frames), "bytes": sum(map(len, frames))}
        started = time.perf_counter()
        try:
            run = await link.send_frames(frames)
            record.update({
                "ok": run.frames_delivered == len(frames), "transport": run.protocol,
                "frames_delivered": run.frames_delivered, "session_resumed": run.session_resumed,
                "early_data_accepted": run.early_data_accepted, "fallback_reason": run.fallback_reason,
                "elapsed_s": round(run.elapsed_s, 4)})
        except DeliveryUncertainError as exc:
            record.update({"ok": False, "error": "DeliveryUncertainError", "detail": str(exc), "not_replayed": True,
                           "elapsed_s": round(time.perf_counter() - started, 4)})
        except Exception as exc:  # noqa: BLE001 - recorded, then the run continues so recovery is visible
            record.update({"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200],
                           "elapsed_s": round(time.perf_counter() - started, 4)})
        failures += 0 if record["ok"] else 1
        out.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if batch + 1 < args.batches:
            await asyncio.sleep(args.interval)
    out.close()
    print(json.dumps({"summary": {"batches": args.batches, "failed": failures}}), flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="receiver address (the environment you are authorised to test)")
    ap.add_argument("--quic-port", type=int, required=True)
    ap.add_argument("--wss-port", type=int, default=None, help="enable the WSS fallback on this TCP port")
    ap.add_argument("--ca", required=True, help="the receiver's PUBLIC ca.pem")
    ap.add_argument("--expect-ca-sha256", default=None, help="fingerprint obtained from the receiver operator out-of-band")
    ap.add_argument("--server-name", default="hivemind.local")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="wan_send.jsonl")
    return asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
