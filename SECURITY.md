# Security Policy & Cryptographic Vulnerability Disclosure

## Supported Versions

Only immutable tagged releases (`v1.0.0` and subsequent `v*.*.*` tags protected by the GitHub Immutable Tag Ruleset) are supported for production swarm deployment.

| Version | Supported | Lean 4 Formal Spec | Cryptographic SHA-256 Attestation |
| :--- | :---: | :---: | :---: |
| **`v1.0.0+`** | ✅ | `17/17` Theorems (`0 sorry`) | [`artifacts/SHA256SUMS.txt`](artifacts/SHA256SUMS.txt) |
| `< v1.0.0` | ❌ | Partial | Unattested |

---

## Scope of Security-Critical Components

1. **Deterministic Seeded Spot-Check Validator** ([`src/hivemind/verification/spot_check.py`](src/hivemind/verification/spot_check.py)) — Gumbel-Max seeded token reproduction (`< 1e-5` tolerance) and SHA-256 + Ed25519 Parquet shard signatures.
2. **Byzantine-Resilient Multi-Krum Outer Aggregator** ([`src/hivemind/compression/multikrum.py`](src/hivemind/compression/multikrum.py)) — Rejects gradient poisoners under $(n=6, f=1)$ and $(n=10, f=2)$ adversarial assumptions.
3. **QUIC 0-RTT Session Resumption & Replay Defense** ([`src/hivemind/swarm/quic_transport.py`](src/hivemind/swarm/quic_transport.py)) — RFC 9001 `ZeroRTTSessionTicket` expiry and anti-replay bounds.
4. **L2 Micro-Fee State-Channel Vouchers** ([`src/hivemind/boinc/coordinator_api.py`](src/hivemind/boinc/coordinator_api.py)) — Strictly monotonic nonce enforcement and HMAC-SHA256 voucher signatures.
5. **Lean 4 Formal Specification** ([`lean4/HiveMindSpec.lean`](lean4/HiveMindSpec.lean)) — Any soundness bug, unintended axiom introduction, or theorem bypass.

---

## Reporting a Vulnerability

Please report suspected vulnerabilities via **GitHub Private Vulnerability Reporting** on [`https://github.com/xaviercallens/HiveMind/security/advisories/new`](https://github.com/xaviercallens/HiveMind/security/advisories/new) rather than public issues.
