# WAN acceptance runbook: two environments, two NAT paths

**Status: BLOCKED.** No second, separately owned environment was available to the author of this
runbook, so nothing here has been run across a WAN. Every command below was exercised only over
loopback (`tests/test_wan_acceptance_tools.py`). Loopback results prove that the helper scripts drive
the real `quic_link` / `nat_traversal` APIs correctly. They prove nothing about NAT behaviour,
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
| Hole punching | `udp_hole_punch(sock, candidates, session)` must run on **both** peers at the same time with each other's mapped address and a shared session id. It uses its own socket. | `nat_traversal.py:494` |
| TURN | `TurnClient` (RFC 5766 UDP allocations) and a volunteer `TurnRelayServer`. | `nat_traversal.py:541`, `:805` |
| Not wired together | The docstring states STUN/TURN "do not wire a NAT path into this link". `RemoteHiddenStateLink` never uses a punched socket or a TURN relay. | `src/hivemind/swarm/quic_transport.py:1103-1120`, `:769-773` |
| Single-process helpers | `AsyncQUICGeoRingTransport.execute_udp_hole_punch` opens **both** peer sockets in one process. It cannot validate two NAT paths. | `quic_transport.py:1352-1380` |
| Frame token | `deserialize_packet_0rtt` states it does not authenticate its token. Peer authentication comes from the TLS certificate chain, not from the frame. | `quic_transport.py:1031-1034` |

Consequences for the acceptance claims:

- "Direct QUIC", "0-RTT resumption" and "WSS fallback" can be exercised end to end with the real link.
- "Hole punch" and "TURN relay" can be exercised only as separate protocol probes. Do not report them
  as a transport that carried hidden-state frames.
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
| `wan_link_receiver.py` | Starts the real `QuicHiddenStateServer` (UDP) and optional `WebSocketHiddenStateServer` (TLS/TCP), writes the public `ca.pem`, logs one JSON line per received frame (size, short digest, integrity result) and a `summary.json` with server-side handshake flags and early-data counts. |
| `wan_link_sender.py` | Sends batches through `RemoteHiddenStateLink`, logs per batch the transport that carried it (`quic_0rtt`, `quic`, `libp2p_websocket`), resumption flags, fallback reason, latency and any error. Verifies the CA fingerprint when given one. |
| `wan_nat_probe.py` | `stun` (reflexive address and mapping class), `punch` (simultaneous hole punch), `turn-listen` / `turn-send` (TURN relay between two allocations). TURN credentials come from `TURN_USERNAME` / `TURN_PASSWORD` environment variables. `--redact` masks IPv4 addresses. |
| `wan_frames.py` | Deterministic self-verifying frames. An integrity check for the test, not authentication. |

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

Record: `behaviour`, `nat_type`, `hole_punchable`, mapped address, RTT to each server. A symmetric
(`address_dependent`) class on either side means a direct hole punch is not expected to work.

### 5.2 Direct QUIC with authenticated payload and acknowledgements

Env B (receiver). The owner of Env B arranges an inbound UDP port (and TCP port for 5.4) to this host;
the script only listens.

```
python tools/wan_acceptance/wan_link_receiver.py --bind 0.0.0.0 --quic-port 4433 --wss-port 4434 --out-dir wan_recv --duration 1800
```

It prints one JSON line with `ca_sha256`. Give `wan_recv/ca.pem` and the fingerprint to the Env A
operator over a channel you trust (the fingerprint is how Env A detects a swapped CA).

Env A (sender):

```
python tools/wan_acceptance/wan_link_sender.py --host RECEIVER_ADDRESS --quic-port 4433 --wss-port 4434 \
    --ca wan_recv/ca.pem --expect-ca-sha256 FINGERPRINT --batches 20 --frames 8 --size 4096 --interval 2 --out wan_send_direct.jsonl
```

Evidence: every batch has `ok: true`, `frames_delivered == frames_sent`, `transport` in
`quic`/`quic_0rtt`; the receiver's `frames.jsonl` shows the same frame count with `intact: true` and
matching `sha256_16` digests. "Authenticated" here means the TLS server certificate chained to the CA
Env A pinned; say that, not "application-layer authenticated".

### 5.3 Session resumption and 0-RTT

From the same run, count batches with `session_resumed: true` and `early_data_accepted: true` on the
sender and compare with the receiver `summary.json` (`quic_handshakes`, `quic_early_frames`). Tickets
are single use and the link keeps only the newest one. On loopback the pattern alternated (a resumed
batch was followed by a non-resumed one), so report the observed ratio instead of assuming every batch
after the first resumes.

### 5.4 Fallback (UDP unusable, WSS carries the batch)

On Env A only, with Env A's owner permission, make UDP to the receiver unusable for the test (for
example an outbound rule on the test host, or point `--quic-port` at a closed UDP port), then:

```
python tools/wan_acceptance/wan_link_sender.py --host RECEIVER_ADDRESS --quic-port 4999 --wss-port 4434 \
    --ca wan_recv/ca.pem --expect-ca-sha256 FINGERPRINT --batches 5 --timeout 3 --out wan_send_wss.jsonl
```

Expected: `transport: libp2p_websocket` and `fallback_reason` such as `TimeoutError`. The aioquic
library may print "Future exception was never retrieved" noise on this path; note it, it is not a
delivery failure. Remove the temporary rule afterwards.

### 5.5 Packet loss, retransmission and reconnects

Run only on your own test host. Apply impairment to the path between the two hosts, start the
sender (`--batches 60 --interval 1`), and keep notes of the exact impairment and timestamps.

- Windows: clumsy (Drop 5 percent, then Lag 150 ms, then Drop 100 percent for 15 s). Filter to the receiver's UDP port, for example `udp and (ip.DstAddr == RECEIVER_ADDRESS or ip.SrcAddr == RECEIVER_ADDRESS)`.
- Linux: `sudo tc qdisc add dev IFACE root netem loss 5%`, later `... netem delay 150ms`, then `... netem loss 100%` for 15 s, and always finish with `sudo tc qdisc del dev IFACE root`.

Evidence per phase: delivered/failed batches, `elapsed_s` median and p95, the first successful batch
after the outage ends and how long after restore it came (reconnect time), whether the failure was a
`DeliveryUncertainError` (never replayed by design) or a pre-delivery error that fell back to WSS.
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

### 5.7 Hole punching (probe only)

Both sides run at the same time with the other's mapped address from 5.1 and the same session text:

```
python tools/wan_acceptance/wan_nat_probe.py punch --local-port 5555 --peer OTHER_MAPPED_IP:PORT --session run-001 --punch-timeout 10
```

Exit 0 means a probe and its answer crossed. This does not mean QUIC will use that hole; the link opens
its own sockets.

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
| Payload integrity | receiver frames.jsonl | | n intact / n received | |
| Resumption / 0-RTT | sender + summary.json | | resumed k/n, early frames m | |
| WSS fallback | 5.4 | | | |
| Loss 5% / delay / outage | 5.5 | | median/p95 s, reconnect s | |
| TURN relay (datagram) | 5.6 | | | |
| Hole punch (probe) | 5.7 | | | |
| Transport selection table | 5.8 | | quic a, quic_0rtt b, wss c | |
| Connection migration | 5.9 | n/a | NOT TESTED / NOT IMPLEMENTED | |

Deviations, owner approvals for any port forward or firewall rule, and the cleanup performed
(netem rule removed, receiver stopped, temporary rules reverted): <text>
Claims NOT supported by this run: <text>
```
