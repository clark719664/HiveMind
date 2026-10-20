#!/usr/bin/env python
"""NAT probes for the two-environment WAN acceptance run, built on hivemind.swarm.nat_traversal.

Run each subcommand ONLY inside an environment you own and are authorised to test. The STUN/TURN
server you point it at must be one you operate or are explicitly allowed to use; this script has no
built-in default server. TURN credentials come from TURN_USERNAME / TURN_PASSWORD, never argv.

    stun   --server HOST:PORT [--server HOST2:PORT]   # server-reflexive address + RFC 5780 mapping class
    punch  --local-port N --peer IP:PORT --session TEXT   # standalone socket probe; NOT WAN acceptance
    turn-listen --server HOST:PORT [--peer-ip IP]     # allocate a relay, print it, wait for one datagram
    turn-send   --server HOST:PORT --relay-peer IP:PORT --payload TEXT

Add --redact to mask IPv4 addresses (first two octets kept) in the printed JSON. Redacted output is
for sharing; keep the raw output privately because the peers need the real addresses to connect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "src").is_dir() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm import nat_traversal as nt  # noqa: E402

IPV4 = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}\b")


def emit(payload: dict, redact: bool) -> None:
    text = json.dumps(payload, default=str)
    print(IPV4.sub(r"\1.\2.x.x", text) if redact else text)


def parse_addr(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host.strip("[]"), int(port)


def cmd_stun(args) -> int:
    sock = nt.udp_socket(("0.0.0.0", args.local_port))
    try:
        if len(args.server) >= 2:
            mapping = nt.classify_nat_mapping(sock, args.server, timeout=args.timeout)
            emit({"command": "stun", "socket_closed_after_result": True, "mapping_reusable_for_punch": False, "behaviour": mapping.behaviour, "nat_type": mapping.nat_type,
                  "hole_punchable": mapping.hole_punchable, "local": mapping.local, "mapped": mapping.mapped,
                  "servers": mapping.servers, "rtt_ms": mapping.rtt_ms}, args.redact)
        else:
            result = nt.stun_binding(sock, args.server[0], timeout=args.timeout)
            emit({"command": "stun", "socket_closed_after_result": True, "mapping_reusable_for_punch": False, "behaviour": "unknown (needs two STUN servers to classify mapping)",
                  "local": result.local, "mapped": result.mapped, "server": result.server,
                  "rtt_ms": result.rtt_ms}, args.redact)
    finally:
        sock.close()
    return 0


def cmd_punch(args) -> int:
    sock = nt.udp_socket(("0.0.0.0", args.local_port))
    try:
        result = nt.udp_hole_punch(sock, [parse_addr(args.peer)], session=args.session.encode("utf-8"),
                                   timeout=args.punch_timeout)
        emit({"command": "punch", "succeeded": result.succeeded, "peer": result.peer, "rtt_ms": result.rtt_ms,
              "probes_sent": result.probes_sent,
              "wan_acceptance": "UNSUPPORTED",
              "note": "standalone socket probe only; do not use a mapping from a closed STUN socket. No same-live-socket discovery/signaling flow is implemented"},
             args.redact)
        return 0 if result.succeeded else 1
    finally:
        sock.close()


def _turn_client(args) -> "nt.TurnClient":
    user, password = os.environ.get("TURN_USERNAME"), os.environ.get("TURN_PASSWORD")
    if not user or not password:
        raise SystemExit("set TURN_USERNAME and TURN_PASSWORD in the environment")
    return nt.TurnClient(args.server, user, password, timeout=args.timeout)


def cmd_turn_listen(args) -> int:
    client = _turn_client(args)
    try:
        relayed = client.allocate()
        if args.peer_ip:
            client.create_permission(args.peer_ip)
        emit({"command": "turn-listen", "relayed_address": relayed, "mapped_address": client.mapped,
              "lifetime_s": client.lifetime, "waiting_s": args.wait}, args.redact)
        sys.stdout.flush()
        peer, data = client.recv_from(timeout=args.wait)
        emit({"command": "turn-listen", "received": True, "from": peer, "bytes": len(data),
              "sha256_16": hashlib.sha256(data).hexdigest()[:16]}, args.redact)
        return 0
    except nt.StunError as exc:
        emit({"command": "turn-listen", "received": False, "error": str(exc)}, args.redact)
        return 1
    finally:
        client.close()


def cmd_turn_send(args) -> int:
    client = _turn_client(args)
    try:
        relayed = client.allocate()
        peer = parse_addr(args.relay_peer)
        client.create_permission(peer[0])
        client.send_to(peer, args.payload.encode("utf-8"))
        emit({"command": "turn-send", "own_relayed_address": relayed, "sent_to": peer,
              "bytes": len(args.payload.encode("utf-8")),
              "sha256_16": hashlib.sha256(args.payload.encode("utf-8")).hexdigest()[:16]}, args.redact)
        return 0
    except nt.StunError as exc:
        emit({"command": "turn-send", "sent": False, "error": str(exc)}, args.redact)
        return 1
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redact", action="store_true")
    ap.add_argument("--timeout", type=float, default=3.0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stun")
    p.add_argument("--server", action="append", required=True)
    p.add_argument("--local-port", type=int, default=0)
    p.set_defaults(fn=cmd_stun)
    p = sub.add_parser("punch")
    p.add_argument("--local-port", type=int, required=True)
    p.add_argument("--peer", required=True, help="the other side's mapped IP:PORT")
    p.add_argument("--session", required=True)
    p.add_argument("--punch-timeout", type=float, default=10.0)
    p.set_defaults(fn=cmd_punch)
    p = sub.add_parser("turn-listen")
    p.add_argument("--server", required=True)
    p.add_argument("--peer-ip", default=None)
    p.add_argument("--wait", type=float, default=60.0)
    p.set_defaults(fn=cmd_turn_listen)
    p = sub.add_parser("turn-send")
    p.add_argument("--server", required=True)
    p.add_argument("--relay-peer", required=True, help="the listener's relayed IP:PORT")
    p.add_argument("--payload", default="hivemind-wan-acceptance")
    p.set_defaults(fn=cmd_turn_send)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
