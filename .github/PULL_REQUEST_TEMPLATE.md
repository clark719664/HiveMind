## 🧠 Project HiveMind — Peer Contribution Pull Request

> **Before submitting:** Ensure your branch follows the naming convention in [`CONTRIBUTING.md`](../CONTRIBUTING.md) (`feat/m1-*`, `feat/m2-*`, `feat/m3-*`, `feat/m4-*`, `expert/<domain>`, `game/steam-*`, or `fix/*`) and that `pytest -v` passes all unit & integration checks.

### 📌 Summary of Changes
<!-- Briefly describe what this PR implements, fixes, or optimizes, and why it matters for the decentralized MoE swarm. -->

- **Linked Issue(s):** Closes #<!-- issue number -->
- **Branch Name:** `<!-- e.g., feat/m2-double-buffered-dma -->`

---

### 🎯 Swarm Architecture Track
Select all tracks touched by this PR:

- [ ] **Pillar 1 — Zero-Copy 4-Bit NF4 & DiLoCo-QLoRA** (`src/hivemind/core/quantization.py`, `diloco_inner.py`, `compression.py`)
- [ ] **Pillar 2 — Byzantine Fault Tolerance (Spot-Check + Multi-Krum)** (`src/hivemind/boinc/validator.py`, `diloco_outer.py`)
- [ ] **Pillar 3 — Factored In-Place Sparse MoE Upcycling (BTM Phase C)** (`src/hivemind/core/btm_upcycle.py`)
- [ ] **Pillar 4 — SETI-Swarm Lifecycle, CAS Vault & `$HIVE` PoUW Ledger** (`src/hivemind/client/daemon.py`, `dht_storage.py`, `ledger.py`)
- [ ] **Milestone M1 (Swarm Fabric) — Kademlia Geo-Ring & QUIC/Libp2p Transport** (`src/hivemind/swarm/swarm_fabric.py`)
- [ ] **Milestone M2 (Swarm Fabric) — 3-Tier Pinned RAM DMA & Macro/Micro Router** (`src/hivemind/swarm/swarm_fabric.py`)
- [ ] **Milestone M3 (Swarm Fabric) — Eagle/Medusa Token Tree Speculation ($\gamma=6$) & FP8 Wire** (`src/hivemind/swarm/swarm_fabric.py`)
- [ ] **Milestone M4 (Swarm Fabric) — OpenAI `/v1/chat/completions` Gateway, OPoI 5% Audit & L2 State Channels** (`src/hivemind/boinc/coordinator_api.py`)
- [ ] **Milestone M5 — `HiveMind: Neural Breach` Free Steam Edition & Hugging Face Hack-Miner** (`src/hivemind/game/`)

---

### 🖥️ Consumer GPU Hardware & 8 GB VRAM Telemetry
<!-- All worker and MoE changes MUST fit comfortably inside an 8 GB–12 GB consumer gaming GPU envelope. -->

| Metric | Value | Target / Ceiling |
| :--- | :--- | :--- |
| **Test GPU / Device** | <!-- e.g., NVIDIA RTX 2070 (8 GB) / CPU CI --> | 8 GB–12 GB Consumer GPU |
| **Peak Allocated VRAM (`measure_vram_footprint`)** | <!-- e.g., 1,527.03 MiB --> | $\le 6,500\text{ MiB}$ on 8 GB GPU |
| **Deterministic Spot-Check Drift ($\max \|\Delta\|$)** | <!-- e.g., 0.000000 --> | $\epsilon \le 10^{-4}$ |
| **Wire Compression Ratio (if applicable)** | <!-- e.g., 3.33x (Top-k + FP8) --> | $\ge 2.0\text{x}$ |

---

### ✅ Peer Review & Verification Checklist
- [ ] **16/16+ Test Suite Passing:** Ran `pytest tests/ -v` locally with zero failures.
- [ ] **8 GB VRAM Budget Respected:** No full-model FP16/FP32 host-RAM or VRAM materialization introduced (`stream_load_hf_causal_lm` / NF4 invariants preserved).
- [ ] **Deterministic CUDA Execution:** Compatible with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- [ ] **Local Disk Cache Isolation:** No files written outside workspace `.cache/` or `artifacts/` during tests.
- [ ] **Documentation Updated:** Relevant docstrings, `README.md`, or `roadmap.md` updated.

---

### 🏆 Optional: `$HIVE` PoUW Contributor & Steam Leaderboard Attribution
- **Steam Persona / Gamer Tag (optional):** `<!-- e.g., CyberValkyrie_FR -->`
- **SteamID64 / Node ID (optional):** `<!-- e.g., 76561198000000001 -->`
