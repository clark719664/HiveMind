# 🔐 Security & Byzantine Vulnerability Policy

Project HiveMind operates across untrusted consumer gaming GPUs (**8 GB–12 GB VRAM**) and residential P2P networks. Cryptographic verification, VRAM sandboxing, and Byzantine resistance are core security boundaries.

---

## 🛡️ Security Architecture & Trust Boundaries

1. **Zero Arbitrary Code Execution (`safetensors` Only):**
   - Worker nodes and the Content-Addressed Swarm Store (`cid://sha256/...`) **never** load Python `pickle` (`.bin` / `.pt`) weights from untrusted peers. All base models and LoRA expert adapters are verified via SHA-256 content addressing and `safetensors` header validation.
2. **Deterministic Micro-Checkpoint Spot-Check ($\epsilon \le 10^{-4}$):**
   - Any worker submitting a DiLoCo pseudo-gradient delta ($\Delta_k^{(t)}$) must include SHA-256 micro-checkpoint commitments every $s$ steps under `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
   - `DeterministicSpotCheckValidator` replays a random segment and slashes any node whose L-infinity tensor drift exceeds $10^{-4}$.
3. **Multi-Krum $2\sigma$ Gradient Poisoner Defense:**
   - `DiLoCoOuterOptimizer` computes pairwise cosine similarities across all worker deltas and rejects outliers falling outside $2\sigma$ of the median before applying the Nesterov outer step.
4. **Optimistic Proof-of-Inference (OPoI) 5% Audits:**
   - P2P Sparse MoE inference providers sign HMAC commitments over hidden-state activations and are probabilistically audited (5% rate) via cosine similarity ($\ge 0.99$).
5. **Steamworks Anti-Sybil Gate (`AppID: 3133700`):**
   - Nodes must present a valid non-limited Steam session ticket ($\ge \$5.00$ spend, `0` VAC/Game bans) via `SteamworksSDKBridge` before joining the `$HIVE` PoUW reward pool.

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability, a bypass in `DeterministicSpotCheckValidator` or `MultiKrum`, a Merkle Proof-of-Storage forgery vector, or a remote code execution bug in the P2P transport layer:

- **Do NOT open a public GitHub Issue** with exploit details.
- Report privately via **[GitHub Private Vulnerability Reporting](https://github.com/xaviercallens/HiveMind/security/advisories/new)** or email the maintainer (`callensxavier@gmail.com`).
- Validated cryptographic or Byzantine defense vulnerabilities are eligible for our highest-tier **Bug Bounty (`5,000 $HIVE`)** and Hall-of-Fame credit.
