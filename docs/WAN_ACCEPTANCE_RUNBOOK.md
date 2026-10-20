# WAN acceptance runbook: two environments, two NAT paths

**Status: BLOCKED.** No second, separately owned environment was available to the author of this
runbook, so nothing here has been run across a WAN. Automated coverage uses only local sockets and mocks
(`tests/test_wan_acceptance_tools.py`); the operator procedures below still need actual WAN validation.
The tests exercise payload integrity, evidence reconciliation, QUIC/WSS and protocol helpers locally. They prove nothing about NAT behaviour,
packet loss, ISP filtering, TURN deployments or connection migration.

Authorization rules for whoever runs this:

- Each environment (network, router, host) needs its own explicit authorization from its owner.
- Do not change a router, firewall or OS setting unless that environment's owner has said so. A port
  forward or firewall rule is a deliberate owner action, listed below only as a prerequisite.
- Impairment tools (clumsy, `tc netem`) run only on the test host you own and are removed afterwards.
- Never put credentials, private keys or TURN passwords in a command line, log or this document.

## 1. What the code actually does (read before claiming anything)

| Component | Real behaviour | Source |
|---|---|---|
| QUIC link | aioquic client/server, ALPN `hivemind-hidden-state/1`, one client-initiated stream per batch, length-prefixed frames, receiver replies with the delivered-frame count | `src/hivemind/swarm/quic_link.py:279` (`QuicHiddenStateServer`), `:377` (`quic_send_frames`) |
| Trust | Sender pins a CA you hand it (`LinkEndpoint.ca_pem`) and checks the certificate name (`server_name`, default `hivemind.local`). No public PKI. | `quic_link.py:112` (`LinkIdentity`), `:635` (`LinkEndpoint`) |
| Resumption | TLS 1.3 session tickets, single use on the server. `RemoteHiddenStateLink` keeps one ticket between batches and sends the next batch as 0-RTT early data when the ticket is valid. | `quic_link.py:664` (`RemoteHiddenStateLink`), `:305-315` (ticket store/consume) |
| Connection lifetime | One UDP socket per batch (`local_addr=("0.0.0.0", 0)`), closed after the batch. There is no long-lived connection. | `quic_link.py:409-411`, `:443-458` |
| WSS fallback | If the QUIC attempt fails before any application bytes were queued (`TimeoutError`, `ConnectionError`, `OSError`) and `LinkEndpoint.websocket_url` is set, the batch goes over a libp2p multistream WebSocket. A `DeliveryUncertainError` is never replayed. Non-loopback listeners require TLS. | `quic_link.py:679-712`, `:499-528` |
| STUN / mapping class | RFC 5389 Binding, RFC 5780 style mapping comparison across two servers (`no_nat`, `endpoint_independent`, `address_dependent`, `unknown`). | `src/hivemind/swarm/nat_traversal.py:407` (`stun_binding`), `:452` (`classify_nat_mapping`) |
| Hole punching | `udp_hole_punch(sock, candidates, session)` must run on **both** peers at the same time with each other's mapped address and a shared session id. It uses the caller's still-open socket. The standalone CLI does not implement same-socket STUN discovery/signaling. | `nat_traversal.py:494` |
| TURN | `TurnClient` (RFC 5766 UDP allocations) and a volunteer `TurnRelayServer`. | `nat_traversal.py:541`, `:805` |
| Not wired together | The docstring states STUN/TURN "do not wire a NAT path into this link". `RemoteHiddenStateLink` never uses a punched socket or a TURN relay. | `src/hivemind/swarm/quic_transport.py:1103-1120`, `:769-773` |
| Single-process helpers | `AsyncQUICGeoRingTransport.execute_udp_hole_punch` opens **both** peer sockets in one process. It cannot validate two NAT paths. | `quic_transport.py:1352-1380` |
| Frame token | `deserialize_packet_0rtt` states it does not authenticate its token. Peer authentication comes from the TLS certificate chain, not from the frame. | `quic_transport.py:1031-1034` |

Consequences for the acceptance claims:

- "Direct QUIC", "0-RTT resumption" and "WSS fallback" can be exercised end to end with the real link.
- "TURN relay" can be exercised only as a separate protocol probe. Hole punching across NATs is
  **UNSUPPORTED by these CLI helpers**: the STUN command closes its socket. Neither is a transport
  that carried hidden-state frames.
- **Connection migration must not be claimed.** The link opens a fresh socket per batch and closes
  it; no code path keeps one connection alive across an address change. It cannot be exercised with the
  current design, so its result is `NOT TESTED / NOT IMPLEMENTED`.
- Retransmission and packet-loss counters are not exposed by `quic_link`. You can only observe their
  effect (latency inflation, failures, fallback) or capture your own traffic with a packet tool.

## 2. Topology (fill in; one row per environment)

| | Env A (sender) | Env B (receiver) |
|---|---|---|
| Owner / authorizer | | |
| Network type (home router, cloud VM you own, mobile hotspot) | | |
| Public egress IP (record privately, redact when sharing) | | |
| NAT class from `wan_nat_probe.py stun` | | |
| OS / Python / torch | | |
| Inbound UDP port available to the receiver (owner-provided forward or public IP) | n/a | |
| Inbound TCP port for WSS | n/a | |
| Impairment tool available | clumsy (Windows) or `tc netem` (Linux) | same |

The two environments must egress through different NATs (different public IP and preferably different
ISPs or one mobile path). Two machines behind the same router are one path, not two.

## 3. Helper scripts (all in `tools/wan_acceptance/`)

| Script | Purpose |
|---|---|
| `wan_link_receiver.py` | Starts the real `QuicHiddenStateServer` (UDP) and optional `WebSocketHiddenStateServer` (TLS/TCP), writes the public `ca.pem`, logs one JSON line per received frame (size, full SHA-256, expected identity, duplicate and integrity result) and a `summary.json` with exact-count acceptance, missing/invalid/duplicate counts, handshake flags and early-data counts. |
| `wan_link_sender.py` | Sends batches through `RemoteHiddenStateLink`, logs per batch the transport that carried it (`quic_0rtt`, `quic`, `libp2p_websocket`), resumption flags, fallback reason, total attempt latency (including a failed QUIC attempt before fallback), full per-frame SHA-256 values and any error. `ok` means count ACK only. Verifies the CA fingerprint when given one. |
| `wan_nat_probe.py` | `stun` (reflexive address and mapping class), `punch` (standalone socket probe, **not WAN acceptance**), `turn-listen` / `turn-send` (TURN relay between two allocations). TURN credentials come from `TURN_USERNAME` / `TURN_PASSWORD` environment variables. `--redact` masks IPv4 addresses. |
| `wan_frames.py` | Length-bound deterministic frames plus an independently configured expected run plan. An integrity check for the test, not authentication. |
| `wan_verify.py` | Reconciles all sender frame identities/digests with the stopped receiver's evidence; missing, duplicate, wrong-run, wrong-size, corrupt or unacknowledged frames fail. |

Run everything from the repository root with the project's Python (`python` below).

## 4. Preflight (each environment)

```
python -m pytest tests/test_wan_acceptance_tools.py tests/test_nat_traversal_quic_link.py tests/test_link_endpoint_safety.py -q
```

Record the exit code and the commit (`git rev-parse HEAD`). Live tests in
`tests/test_nat_traversal_quic_link.py` are skipped unless opted in (section 5.1, 5.6).

## 5. Procedure and evidence

For every step record: UTC start/end, environment, exact command (with secrets removed), exit code,
the JSON output, and one sentence of interpretation. Mark a step `BLOCKED` rather than guessing.

### 5.1 Discovery (STUN and NAT class)

Use a STUN server you operate or are explicitly allowed to use. Two servers are needed to classify mapping behaviour.

```
python tools/wan_acceptance/wan_nat_probe.py stun --server STUN1:3478 --server STUN2:3478
```

The repo's opt-in public-server test contacts `stun.l.google.com:19302` and `stun.cloudflare.com:3478`
(`DEFAULT_STUN_SERVERS`, `nat_traversal.py:78`). Run it only if that environment's owner accepts that
those services see the public IP:

```
# PowerShell                                  # POSIX shell
$env:HIVEMIND_LIVE_NET = "1"                  HIVEMIND_LIVE_NET=1 \
python -m pytest tests/test_nat_traversal_quic_link.py -k live_public_stun -v
```

Record: `behaviour`, `nat_type`, `hole_punchable`, mapped address, RTT to each server.
This command closes its socket before exiting. Its mapping is diagnostic only and **must not** be
passed to a later `punch` process, even if the later process reuses the local port. A symmetric
(`address_dependent`) class on either side means a direct hole punch is not expected to work.

### 5.2 Direct QUIC with server authentication and verified payload evidence

Agree on a fresh non-secret unsigned 64-bit `RUN_SEED`, batch count, frames per batch and full frame
size before starting. Reusing a seed can confuse old and new runs. Each receiver process accepts one
plan and one sender run. Use fresh output paths; the helpers refuse to overwrite existing evidence.
The bounded limits are 100,000 frames, 1 MiB per frame, 16 MiB per batch and 512 MiB per run.

Env B (receiver). Its owner arranges the inbound UDP port (and TCP for 5.4). Binding `0.0.0.0`
listens on every IPv4 interface, including public interfaces if present; use only authorized hosts.
The default bind is loopback. A count ACK does not imply that the receiver accepted the payload.

```
python tools/wan_acceptance/wan_link_receiver.py --bind 0.0.0.0 --quic-port 4433 --wss-port 4434 \
    --seed RUN_SEED --batches 20 --frames 8 --size 4096 --out-dir wan_recv_direct --duration 1800
```

It prints `ca_sha256` and the expected plan. Give `wan_recv_direct/ca.pem` and the fingerprint to the
Env A operator over a trusted channel. The receiver's private key stays in memory or a short-lived
owner-only temporary file, and must never be transferred.

Env A (sender), with the public CA copied to its own `receiver-ca.pem`:

```
python tools/wan_acceptance/wan_link_sender.py --host RECEIVER_ADDRESS --quic-port 4433 --wss-port 4434 \
    --ca receiver-ca.pem --expect-ca-sha256 FINGERPRINT --seed RUN_SEED \
    --batches 20 --frames 8 --size 4096 --interval 2 --out wan_send_direct.jsonl
```

After the sender exits, Env B stops its receiver with Ctrl+C (or waits for `--duration`). This writes
`summary.json` after closing listeners. A clean stop with complete, exactly-once valid frames exits 0;
missing, corrupt, wrong-size, unexpected or duplicate frames exit 1. Out-of-order unique intact frames
are allowed and counted, since ordering across batches/transports is not the acceptance criterion.
Never kill the receiver before its summary is written. For scripted local control, optional
`--stop-file PATH` stops when that local file appears; it must not exist at startup.

The operators privately transfer the sender JSONL and the receiver's `frames.jsonl` / `summary.json`
to one authorized operator, then run (using their local artifact paths):

```
python tools/wan_acceptance/wan_verify.py --sender wan_send_direct.jsonl --receiver wan_recv_direct
```

Require exit 0 and `accepted: true`; it checks every full SHA-256 by `(batch, index)` as well as the
independent expected seed, size and counts. Sender `ok: true` is only a transport count ACK and cannot
replace this check. Also require direct-run `transport` values in `quic`/`quic_0rtt`. TLS authenticates
the server certificate against the pinned CA; these test frames do not authenticate the sender.
Retain both sets of raw evidence privately and share only redacted copies with permission.

### 5.3 Session resumption and 0-RTT

From the same run, count batches with `session_resumed: true` and `early_data_accepted: true` on the
sender and compare with the receiver `summary.json` (`quic_handshakes`, `quic_early_frames`). Tickets
are single use and the link keeps only the newest one. On loopback the pattern alternated (a resumed
batch was followed by a non-resumed one), so report the observed ratio instead of assuming every batch
after the first resumes.

### 5.4 Fallback (UDP unusable, WSS carries the batch)

Start a **fresh Env B receiver** as in 5.2, with a new seed, `--batches 5`, `--frames 8`, `--size 4096`,
and `--out-dir wan_recv_wss`. Transfer its new public CA/fingerprint. Env A uses an owner-confirmed
unused UDP port to avoid any firewall change, while keeping the authorized WSS port reachable:

```
python tools/wan_acceptance/wan_link_sender.py --host RECEIVER_ADDRESS --quic-port UNUSED_UDP_PORT --wss-port 4434 \
    --ca receiver-ca.pem --expect-ca-sha256 FINGERPRINT --seed NEW_RUN_SEED \
    --batches 5 --frames 8 --size 4096 --timeout 3 --out wan_send_wss.jsonl
```

Expected: `transport: libp2p_websocket` and a pre-delivery `fallback_reason` such as `TimeoutError`.
`elapsed_s` measures the entire attempt, including failed QUIC setup; `transport_elapsed_s` is the
chosen transport's own narrower measurement. Stop the receiver and reconcile as in 5.2 with these
new artifact paths. An uncertain delivery must never be replayed. If an owner deliberately chose a
temporary firewall rule instead, that owner must restore it afterwards and record the cleanup.

### 5.5 Packet loss, retransmission and reconnects

Run only on your own test host. Apply impairment to the path between the two hosts, start the
sender (`--batches 60 --frames 8 --size 4096 --interval 1`) using a fresh seed,
matching receiver plan and separate artifact paths, and keep notes of the exact impairment and timestamps.

- Windows: clumsy (Drop 5 percent, then Lag 150 ms, then Drop 100 percent for 15 s). Filter to the receiver's UDP port, for example `udp and (ip.DstAddr == RECEIVER_ADDRESS or ip.SrcAddr == RECEIVER_ADDRESS)`.
- Linux: `sudo tc qdisc add dev IFACE root netem loss 5%`, later `... netem delay 150ms`, then `... netem loss 100%` for 15 s, and always finish with `sudo tc qdisc del dev IFACE root`.

Evidence per phase: delivered/failed batches, `elapsed_s` median and p95, the first successful batch
after the outage ends and how long after restore it came (reconnect time), whether the failure was a
`DeliveryUncertainError` (never replayed by design) or a pre-delivery error that fell back to WSS.
Stop the receiver and reconcile as in 5.2. A missing-frame result under deliberate loss is FAIL or
PARTIAL with the counts recorded, never payload acceptance.
Retransmissions themselves are not exposed; if you need them, take a packet capture of your own test
traffic and count them there.

### 5.6 TURN relay (protocol-level only)

Single-host opt-in test against a TURN server you operate (for example coturn). It allocates twice from
one machine, so it proves the server relays, not that two NAT paths work:

```
# PowerShell: $env:TURN_SERVER="host:3478"; $env:TURN_USERNAME="..."; $env:TURN_PASSWORD="..."
python -m pytest tests/test_nat_traversal_quic_link.py -k live_turn -v
```

Two-environment relay probe (credentials in the environment, never on the command line). Env B first:

```
python tools/wan_acceptance/wan_nat_probe.py turn-listen --server TURN_HOST:3478 --peer-ip TURN_RELAY_IP --wait 120
```

It prints its `relayed_address`. Env A then sends to that address:

```
python tools/wan_acceptance/wan_nat_probe.py turn-send --server TURN_HOST:3478 --relay-peer RELAYED_IP:PORT --payload hello-from-A
```

Evidence: the listener prints `received: true` with the same `sha256_16`. Report it as "TURN relay
verified at the datagram level", never as hidden-state transport selection.

### 5.7 Hole punching (WAN acceptance unsupported)

Result: `UNSUPPORTED / NOT TESTED across NATs`. Do not run the previous sequence of standalone
`stun` discovery followed by `punch --local-port 5555`: the discovered ephemeral socket has already
closed, so its mapped address cannot identify the new punch socket. Rebinding the same local port
also does not guarantee the same NAT mapping.

The `punch` command remains a low-level socket probe and is tested on loopback only. Its exit 0 means
only that a probe/answer crossed between the supplied endpoints; it explicitly reports WAN acceptance
as unsupported. It is not part of the two-NAT acceptance procedure.

A future supported flow must keep the **same UDP socket open** through STUN discovery, bounded
out-of-band peer endpoint exchange, mapping refresh and simultaneous punch, then close it on every
success/failure path. That lifecycle needs local socket-identity regression tests before a two-owner
WAN test. Even then, it would prove only datagram probing: the QUIC link opens separate sockets.

### 5.8 Transport selection

Build the table from the per-batch `transport` field across 5.2 to 5.5. A batch's transport is what
`RemoteHiddenStateLink` actually used; the only selectable paths today are QUIC and the WSS fallback.

### 5.9 Connection migration

Result: `NOT TESTED / NOT IMPLEMENTED`. Do not write "migration works" anywhere. To claim it a future
change must keep a single QUIC connection open, rebind or switch the client network mid-stream, and show
the server accepting the new path on the same connection id (aioquic path validation). Capture that
evidence at the packet level and in the server log.

## 6. Redaction

Keep raw artifacts private. Share only redacted copies.

- Mask IPv4/IPv6 addresses, hostnames, MAC addresses, SSIDs and ISP names. Use `ISP-A`, `ISP-B`,
  `ENV-A`, `ENV-B`. `wan_nat_probe.py --redact` keeps the first two octets, which is still identifying
  on a small ISP: mask further by hand before publishing.
- Ports may stay. Router model, public hostnames and DDNS names must go.
- Never include TURN usernames/passwords, private keys, or the `key.pem` temporary file. `ca.pem` is public;
  share its SHA-256 fingerprint rather than relying on the file alone.
- `frames.jsonl` and the sender logs contain no addresses, but check any text you paste into the results.
- Do not publish packet captures. If one is needed, keep only handshake metadata and counts.
- Remove timestamps that reveal usage patterns if the owners ask; keep relative offsets.

## 7. Results template (copy into `docs/` or an issue after redaction)

```
WAN acceptance result: <date UTC>
Commit: <sha>   Operator A: <role>   Operator B: <role>   Authorization: <yes/no per environment>

Topology: ENV-A (<type>, NAT <class>)  <->  ENV-B (<type>, NAT <class>)
Overall: PASS | PARTIAL | BLOCKED | FAIL

| Check | Command / test | Exit | Result | Evidence file |
|---|---|---|---|---|
| Discovery / NAT class A | wan_nat_probe stun | | | |
| Discovery / NAT class B | wan_nat_probe stun | | | |
| Direct QUIC + ACK | wan_link_sender (5.2) | | n/20 batches ok | |
| Payload integrity | wan_verify (5.2) + stopped receiver summary | | complete identities/full SHA-256; missing/invalid/duplicate counts | |
| Resumption / 0-RTT | sender + summary.json | | resumed k/n, early frames m | |
| WSS fallback | 5.4 | | | |
| Loss 5% / delay / outage | 5.5 | | median/p95 s, reconnect s | |
| TURN relay (datagram) | 5.6 | | | |
| Hole punch across NATs | 5.7 | n/a | UNSUPPORTED / NOT TESTED | |
| Transport selection table | 5.8 | | quic a, quic_0rtt b, wss c | |
| Connection migration | 5.9 | n/a | NOT TESTED / NOT IMPLEMENTED | |

Deviations, owner approvals for any port forward or firewall rule, and the cleanup performed
(netem rule removed, receiver stopped, temporary rules reverted): <text>
Claims NOT supported by this run: <text>
```
