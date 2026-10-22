# 🗺️ Season 7 (`Wave 6`) Roadmap & Contributor Guide (`v1.8.0`)

All **48 Pull Requests across Waves 1, 2, 3, 4, and 5 (`Seasons 1–6`) + Dependabot** have been reviewed, verified, and merged into `main`. We have now opened **8 Season 7 / Wave 6 Community Tasks (`HM-S7-01` through `HM-S7-08` — `24,000 PTS` Compute-Credit Pool)** on **[GitHub Discussions #83](https://github.com/xaviercallens/HiveMind/discussions/83)**, **[GitHub Discussions #85](https://github.com/xaviercallens/HiveMind/discussions/85)**, and **[GitHub Issues](https://github.com/xaviercallens/HiveMind/issues)**!

---

## ⚖️ Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / USD`)

> **Important Notice for All Human & AI Contributors**:
> - **No Financial Payment (`0.00 EUR / USD`)**: Project HiveMind is a non-commercial, open-science engineering initiative. There is **no cash payment, salary, fiat payout, or speculative cryptocurrency value** attached to any task, PR, or benchmark submission.
> - **Compute-Reciprocity Points Only (`PTS` / `$HIVE`)**: Earned points (`PTS` / `$HIVE`) are tracked in [`docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md`](https://github.com/xaviercallens/HiveMind/blob/main/docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md) and the deterministic ledger (`src/hivemind/economy/ledger.py`). They confer **only reciprocal computing processing benefits**—priority distributed Sparse MoE inference routing, higher QLoRA fine-tuning job quotas, and multi-guild co-mining bandwidth—when the **live production HiveMind Swarm** is deployed and used by a large community.

---

## 1. Open Season 7 (`Wave 6`) Tasks (`HM-S7-01` through `HM-S7-08` — `24,000 PTS` Pool)

| Task ID | Track & Target Scope | Deliverable Summary | Suggested Contributors | Compute-Credit Points (`PTS`) |
| :---: | :--- | :--- | :--- | :---: |
| **`HM-S7-01`** | **Track 1: Sparse Activation Checkpointing (`Issue #66`)** (`src/hivemind/core/btm_moe.py`, `src/hivemind/core/qlora_4bit.py`) | Selective Top-2 Expert activation recomputation reducing 8K-context QLoRA backward activation memory by `>= 35%` while keeping `< 2.2 GB` peak VRAM on 8 GB GPUs | `@clark719664`, `@pedromineo`, Open | **`+3,500 PTS`** |
| **`HM-S7-02`** | **Track 2: Live Multi-Region WAN GameDay & UPnP/PCP Nat Mapping** (`src/hivemind/p2p/nat_traversal.py`, `src/hivemind/p2p/quic_link.py`) | Opt-in multi-host WAN benchmark across real residential NATs with automatic UPnP IGD / RFC 6887 PCP port mapping fallback before TURN `ChannelBind` | `@clark719664`, Open | **`+3,000 PTS`** |
| **`HM-S7-03`** | **Track 3: Asynchronous Pipeline-Parallel Expert Sharding** (`src/hivemind/core/swarm_fabric.py`, `src/hivemind/core/btm_moe.py`) | Micro-batch token pipelining across 2–4 consumer GPUs over QUIC 0-RTT with automatic straggler bypass when a peer enters game-preemption mode | `@pedromineo`, `@clark719664`, Open | **`+3,500 PTS`** |
| **`HM-S7-04`** | **Track 4: Zero-Copy Safetensors MMap & Delta-Adapter Streaming** (`src/hivemind/swarm/hf_community_hub.py`, `src/hivemind/swarm/dht_storage.py`) | Memory-mapped `safetensors` differential LoRA patch streaming (`Delta-LoRA`) cutting peer hot-swap disk I/O and RAM footprint by `50%+` | `@pedromineo`, Open | **`+2,500 PTS`** |
| **`HM-S7-05`** | **Track 5: Verifiable Compute-Credit Redemption & Quota Scheduler** (`src/hivemind/economy/ledger.py`, `src/hivemind/cli/swarm_dashboard.py`) | Production Priority Compute Queue (`Deficit Round-Robin + Compute-Points Weighting`) translating contributor `PTS` into priority inference & QLoRA job slots | `@clark719664`, `@radialmonster`, Open | **`+3,000 PTS`** |
| **`HM-S7-06`** | **Track 6: Community GPU Hardware Telemetry & Profiling Suite** (`src/hivemind/cli/community_gpu_benchmark.py`, `scripts/run_community_gpu_benchmark.py`) | Extend the opt-in GPU benchmark with DirectML/ROCm/Metal backend detection, thermal throttling telemetry, and anonymized JSON leaderboard export | `@radialmonster`, `@pedromineo`, Open | **`+2,500 PTS`** |
| **`HM-S7-07`** | **Track 7: Hack-Miner Season 7 (`Missions 25–28`) & Live Telemetry Replay** (`src/hivemind/cli/hf_hack_miner.py`, `src/hivemind/cli/georing_webgl.py`) | Missions 25–28 (`Distributed RAG Synthesis`, `CUDA Warp-Specialized MoE`, `Cross-Guild ZK Audit`, `64-Node WAN Chaos`) + timeline scrubber in `:8090` | `@pedromineo`, `@radialmonster`, Open | **`+2,500 PTS`** |
| **`HM-S7-08`** | **Track 8: Lean 4 Season 7 Formal Proofs (`Theorems 55–62`)** (`lean4/HiveMindSpec.lean`, `paper/hivemind_whitepaper.tex`) | 8 machine-checked Lean 4 theorems covering selective activation checkpointing bounds, deficit round-robin compute-point fairness, and pipeline straggler safety (`0 sorry`) | `@clark719664`, Open | **`+3,500 PTS`** |

---

## 2. Historical Wave Summary (`Waves 1–5`: `48/48` PRs Merged)

- **Waves 1–3 (`Seasons 1–4`)**: 25 PRs merged — 4-bit NF4 QLoRA, Sparse 32-Expert MoE, Kademlia DHT, Multi-Krum & ZK-OPoI v1–v2, Steam Background Overlay, 3D WebGL Globe, 40 Lean 4 proofs.
- **Wave 4 (`Season 5`, `PRs #52–#62`)**: 64-Expert 2-Tier MoE, FP8 KV-Cache Paging, Interactive $O(\log L)$ Bisection Fraud Game v3, Live HF Hub API + Multi-Peer 64KB Streaming, Guild Wars Missions 17–20, Multi-Host WAN QUIC+TURN Benchmark, Lean 4 Theorems 41–48.
- **Wave 5 (`Season 6`, `PRs #63–#82`)**: EF21-DiLoCo Stateful Error-Feedback (`3.2x` speedup), 128-Expert 3-Tier Hierarchical MoE (`16x` sparsity), Multi-Round Parallel 4-Way Fraud Game (`ZK-OPoI v4`) + 5% Guild Insurance Pool, BitTorrent Rarest-First Chunk Scheduler, Elo Guild Matchmaking (`Missions 21–24`), RFC 5766 TURN `ChannelBind` (`0x4000`), Lean 4 `77` machine-checked `theorem` declarations (`0 sorry`).

---

## 3. Conflict-Free Pull Request Workflow (Auto-Reviewed Every 3 Hours)

1. **Claim a Task in GitHub Discussions:** Comment `/claim HM-S7-XX` in **[Discussion #85](https://github.com/xaviercallens/HiveMind/discussions/85)** or **[Discussion #83](https://github.com/xaviercallens/HiveMind/discussions/83)**.
2. **Create an Isolated Topic Branch:**
   ```bash
   git checkout -b feat/s7-activation-checkpointing
   ```
3. **Run the Full Verification Gate Locally (`300+` Tests + `77` Lean 4 Theorems):**
   ```bash
   PYTHONPATH=. pytest tests/ -q
   cd lean4 && lake build
   ```
4. **Open a Pull Request against `main`:** Reference `Closes #<issue>` or `Implements HM-S7-XX`. The **3-Hour Autonomous Community Orchestrator** (`scripts/hivemind_autonomous_community_orchestrator.py`) automatically scans, verifies, and merges clean PRs every 3 hours (`0 */3 * * *`).
