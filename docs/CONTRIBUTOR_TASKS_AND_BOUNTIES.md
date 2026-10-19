# 🛠️ Project HiveMind — Large Contributor Tasks, Architecture Roadmap & `$HIVE` Bounties

> **Steam status:** The Hack-Miner and RTX Steam Tool package run locally from this repository. The default test App ID `3133700` belongs to another product on the public Steam Store; see [Steam integration status](STEAM_STATUS.md).
>
> **Status:** Phase 1 & Local RTX 2070 PoC (`20/20` Passing Tests) + Free Steam RTX Tool & Hack-Miner Prototype v1.0.0 (Season 1 & Season 2 6-Mission Campaign + 12-Expert Sparse MoE) + Phase 2 Swarm Fabric (M1–M5: `HM-M1-01` through `HM-M5-02`) are **LIVE & VERIFIED**.
> All 6 community Pull Requests (`#6`–`#10` by `@clark719664` closing Issues `#1`–`#5`, and `#11` by `@radialmonster` adding `docs/STEAM_STATUS.md`) have been reviewed and integrated!

---

## 📊 Contributor Task Matrix & Merged PR Registry

| Issue ID | Milestone Track | Task Title | Status & Merged PR | Target Branch Prefix | Bounty Pool |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **HM-M1-01** (`#1`) | **M1 (Transport)** | Native `AsyncQUICGeoRingTransport` HTTP/3 0-RTT & Libp2p WebSocket NAT Hole-Punching (`STUN/TURN/ICE`) | ✅ **MERGED (`PR #9` by `@clark719664`)** | `feat/m1-quic-nat-traversal` | `2,500 $HIVE` |
| **HM-M1-02** | **M1 (Transport)** | Kademlia DHT GossipSub Live Expert Table & Automatic Regional Geo-Ring Failover (`<25ms`) | ✅ **COMPLETED** | `feat/m1-kademlia-gossipsub` | `1,500 $HIVE` |
| **HM-M2-01** (`#2`) | **M2 (3-Tier Memory)** | Double-Buffered CUDA Stream Lookahead Prefetcher (`slot_A` / `slot_B` Overlap with Attention) | ✅ **MERGED (`PR #7` by `@clark719664`)** | `feat/m2-double-buffer-dma` | `3,000 $HIVE` |
| **HM-M2-02** | **M2 (3-Tier Memory)** | Bounded LRU NVMe-to-Pinned-RAM Cache (`max_pinned_ram_bytes`) for 8 GB GPUs | ✅ **COMPLETED** | `feat/m2-nvme-mmap-cache` | `2,000 $HIVE` |
| **HM-M3-01** (`#3`) | **M3 (Tree Speculation)** | Triton `float8_e4m3fn` Group-Quantization Kernel & Prefix-Trie 2D Causal Tree-Attention Mask ($\gamma=6..8$) | ✅ **MERGED (`PR #8` by `@clark719664`)** | `feat/m3-triton-fp8-tree` | `3,000 $HIVE` |
| **HM-M4-01** (`#4`) | **M4 (OpenAI & OPoI)** | Streaming SSE (`stream=True`) `/v1/chat/completions` Proxy for Cursor, Continue.dev & Cline | ✅ **MERGED (`PR #6` by `@clark719664`)** | `feat/m4-openai-sse-stream` | `1,200 $HIVE` |
| **HM-M4-02** | **M4 (OpenAI & OPoI)** | Merkle-Proof Fraud Dispute Game for Optimistic Proof-of-Inference (OPoI) & Layer-2 `$HIVE` Refund | ✅ **COMPLETED** | `feat/m4-opoi-dispute-game` | `2,500 $HIVE` |
| **HM-M5-01** (`#5`) | **M5 (Steam Game)** | Native `steam_api64.dll` Ctypes Bridge, Live Steam Overlay & Rich Presence | ✅ **MERGED (`PR #10` by `@clark719664`)** | `game/steamworks-native-dll` | `1,800 $HIVE` |
| **HM-M5-02** (`#5`) | **M5 (HF Domains)** | Season 2 Hugging Face Hack-Miner Missions: `Rust_Systems_Security`, `CyberSecurity_Vulnerability_Recon`, `Multilingual_FR_Synthesis` (`E=2 -> E=6` MoE) | ✅ **MERGED (`PR #10` by `@clark719664`)** | `expert/<domain-name>` | `1,000 $HIVE` / expert |
| **HM-DOC-01** | **M5 (Steam Docs)** | Steam AppID `3133700` Public Store Audit & Local Hack-Miner Prototype Status (`docs/STEAM_STATUS.md`) | ✅ **MERGED (`PR #11` by `@radialmonster`)** | `fix/steam-release-documentation` | `1,000 $HIVE` |

---


## 🌐 Track M1: P2P Transport & Kademlia Geo-Rings (`src/hivemind/swarm/swarm_fabric.py`)

### Task `HM-M1-01`: Native `aioquic` HTTP/3 0-RTT & Libp2p WebSocket NAT Hole-Punching
- **Problem:** Consumer gaming PCs sit behind residential NATs and CGNATs. While `RegionalGeoRingTransport` implements 160-bit Kademlia XOR distance routing and `<20 ms` regional admission control, we need real UDP hole-punching (`STUN/ICE`) over `aioquic` with automatic fallback to `libp2p` WebSockets over TLS.
- **Scope:**
  1. Implement `AsyncQUICGeoRingTransport` in `src/hivemind/swarm/quic_transport.py`.
  2. Support 0-RTT session resumption for micro-batch hidden-state frames (`CompressedHiddenStatePacket`).
  3. Add loopback multi-socket latency benchmark in `tests/test_seti_swarm_v2.py`.
- **Acceptance Criteria:**
  - Sustains $\ge 100$ hidden-state frames/sec on localhost UDP loopback with `< 1.5 ms` serialization overhead.
  - Gracefully falls back to WebSocket framing when UDP is blocked.

### Task `HM-M1-02`: Kademlia DHT GossipSub Live Expert Table & Regional Failover
- **Problem:** When a gamer launches a fullscreen game on Steam, `ZeroIntrusionDaemon` yields 100% of VRAM in `< 1.5s`. The Regional Geo-Ring must reroute active Top-2 MoE token streams to a standby replica peer in `< 25 ms` without dropping an active `/v1/chat/completions` request.
- **Scope:**
  1. Add heartbeat health probes and hot-standby peer promotion in `RegionalGeoRingTransport`.
  2. Broadcast `PREEMPTING_FOR_GAME` gossip frames immediately when `ZeroIntrusionDaemon.poll_and_preempt()` triggers.

---

## ⚡ Track M2: 3-Tier NVMe $\rightarrow$ Pinned RAM $\rightarrow$ CUDA VRAM Memory (`src/hivemind/swarm/swarm_fabric.py`)

### Task `HM-M2-01`: Double-Buffered CUDA Stream Lookahead Prefetcher
- **Problem:** An 8 GB GPU cannot hold 64+ domain experts simultaneously in VRAM, but a gaming PC typically has 16 GB–32 GB of System RAM and PCIe 3.0/4.0 bandwidth (`12–25 GB/s`). Streaming a 16 MB LoRA expert from Pinned Host RAM (`cudaHostAlloc`) takes only `~0.8 ms`.
- **Scope:**
  1. Extend `PinnedRAMExpertDMAStreamer` with a 2-slot double buffer (`slot_A`, `slot_B`) on a dedicated non-blocking `torch.cuda.Stream`.
  2. Use `TwoLevelMacroMicroRouter` predictions at Transformer layer $L$ to prefetch the Top-2 micro-experts for layer $L+1$ while layer $L$ executes self-attention.
  3. Record CUDA event timing (`torch.cuda.Event(enable_timing=True)`) to prove `0.0 ms` exposed stall latency on 8 GB RTX cards.

### Task `HM-M2-02`: Zero-Copy `mmap` NVMe-to-Pinned-RAM LRU Cache
- **Scope:**
  1. Integrate `safetensors` memory-mapped (`mmap`) loading from the Content-Addressed Swarm Store (`cid://sha256/...`) into the Pinned Host RAM tier.
  2. Implement an LRU eviction policy capped at a user-configurable System RAM budget (default `4.0 GB`).

---

## 🌲 Track M3: Eagle/Medusa Token Tree Speculation & FP8 Kernels (`src/hivemind/swarm/swarm_fabric.py`)

### Task `HM-M3-01`: Custom Triton `float8_e4m3fn` Quantization Kernel & Dynamic Speculative Tree Attention
- **Problem:** Standard autoregressive P2P MoE inference pays network RTT on every single token. `TokenTreeSpeculativeVerifier` ($\gamma=6$) + `compress_hidden_states_fp8` reduces RTT by up to `6.0x`, and can be further accelerated with fused Triton kernels and tree-attention masks.
- **Scope:**
  1. Write a fused Triton kernel (with PyTorch native fallback for Windows WDDM / CPU CI) for per-token absmax `float8_e4m3fn` activation compression + decompression.
  2. Extend `TokenTreeSpeculativeVerifier` to construct a 2D causal tree-attention mask so the Top-2 Sparse MoE ring verifies multiple speculative branches `[root -> (b1, b2) -> ...]` in a single forward pass.

---

## 🔐 Track M4: OpenAI Gateway, OPoI Audits & Layer-2 `$HIVE` State Channels (`src/hivemind/boinc/coordinator_api.py`)

### Task `HM-M4-01`: Streaming SSE (`stream=True`) `/v1/chat/completions` Proxy for Cursor & Continue.dev
- **Problem:** AI coding assistants (`Cursor`, `Continue.dev`, `Cline`, `OpenWebUI`) expect Server-Sent Events (`text/event-stream`) with `chat.completion.chunk` deltas when `"stream": true` is passed to `/v1/chat/completions`.
- **Scope:**
  1. Upgrade `POST /v1/chat/completions` in `src/hivemind/boinc/coordinator_api.py` to support both synchronous JSON and `StreamingResponse` (`text/event-stream`) mode.
  2. Emit real-time speculative tree verification batches (4–6 tokens per SSE chunk) and attach the final OPoI audit + Layer-2 `$HIVE` voucher in the closing usage chunk.

### Task `HM-M4-02`: Merkle-Proof Fraud Dispute Game for Optimistic Proof-of-Inference (OPoI)
- **Scope:**
  1. Extend `OptimisticProofOfInferenceAuditor` and `HiveLayer2StateChannel` so a challenged inference provider can submit a compact Merkle commitment of layer activations.
  2. Automatically slash dishonest inference hosts in `HiveSwarmLedger` and refund the consumer's Layer-2 state channel escrow.

---

## 🎮 Track M5: `HiveMind: Neural Breach` Free Steam Edition (`AppID: 3133700`) & Hugging Face Domains (`src/hivemind/game/`)

### Task `HM-M5-01`: Native `steam_api64.dll` Bridge & Live Steam Rich Presence
- **Scope:**
  1. Extend `SteamworksSDKBridge` (`src/hivemind/client/steamworks.py`) to dynamically load `steam_api64.dll` / `libsteam_api.so` when present alongside `steam_appid.txt` (`3133700`), while keeping deterministic HMAC fallback for headless/CI runs.
  2. Broadcast live Steam Friends Rich Presence: `"Hack-Mining PubMedQA on RTX 2070 | Rank V: Singularity Core | +45.0 $HIVE"`.

### Task `HM-M5-02`: Season 2 Hugging Face Domain Expert Bounties
- **Scope:** Add new playable missions to `src/hivemind/game/hf_hack_miner.py` and verify their QLoRA convergence on an 8 GB GPU:
  - **Mission 4 (`Rust_Systems_Security`):** `anon/rust-code-instructions`
  - **Mission 5 (`CyberSecurity_Vulnerability_Recon`):** CVE / secure coding dataset
  - **Mission 6 (`Multilingual_FR_Synthesis`):** French instruction & reasoning dataset

## 🚀 Next Wave — Season 3 & Phase 3 Global WAN Scale-Out Tasks (OPEN FOR CONTRIBUTORS)

With all Wave 1 milestones (`HM-M1-01` through `HM-M5-02`) merged and verified (`20/20` tests passing), the following **5 isolated parallel tracks** are now open for community PRs:

| Task ID | Track | Task Description | Difficulty | Target Branch Prefix | Bounty Pool |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **HM-S3-01** | **Track 1 (WAN P2P)** | Cross-Continent Geo-Ring Bridge & Dynamic Relay Selection (`EU-West` $\leftrightarrow$ `US-East` $\leftrightarrow$ `AP-Tokyo`) | 🔴 Hard | `feat/s3-wan-georing-bridge` | `3,500 $HIVE` |
| **HM-S3-02** | **Track 2 (CUDA/DMA)** | Multi-GPU / Peer-to-Peer PCIe NVLink & Vulkan/ROCm Consumer Fallback for 8GB–12GB Nodes | 🔴 Hard | `feat/s3-rocm-vulkan-backend` | `3,500 $HIVE` |
| **HM-S3-03** | **Track 3 (MoE Core)** | 12-to-32 Expert Hierarchical BTM Upcycling & Dynamic Load-Balancing Auxiliary Loss | 🟡 Medium | `feat/s3-hierarchical-moe32` | `2,500 $HIVE` |
| **HM-S3-04** | **Track 4 (HF Experts)** | Season 3 Hugging Face Hack-Miner Missions: `Lean4_Formal_Math`, `Verilog_RTL_Hardware`, `Biochem_Protein_Design` | 🟢 Starter | `expert/s3-<domain>` | `1,500 $HIVE` / expert |
| **HM-S3-05** | **Track 5 (Gateway/UI)** | WebSocket Live Telemetry Stream & Multi-Node Swarm Topology Map in Cyberpunk Web HUD (`:8090`) | 🟢 Starter | `game/s3-live-topology-hud` | `1,800 $HIVE` |

---

## 🤝 How to Claim a Task & Submit a Parallel PR

1. Open or claim an issue using the **[Swarm Fabric Milestone Task](https://github.com/xaviercallens/HiveMind/issues/new?template=01_swarm_fabric_milestone.yml)** or **[Domain Expert Bounty](https://github.com/xaviercallens/HiveMind/issues/new?template=02_domain_expert_lora_bounty.yml)** template (or comment `/claim` on an existing open issue).
2. Create an isolated topic branch off `main` following the track prefix (e.g., `git checkout -b feat/s3-wan-georing-bridge` or `git checkout -b expert/s3-lean4-math`).
3. Run `uv run --no-sync pytest tests/ -v` to verify all `20/20+` tests pass and `verify_lean4_specification()` reports `0 sorry`.
4. Open a Pull Request against `main` using our [PR Template](../.github/PULL_REQUEST_TEMPLATE.md)!

