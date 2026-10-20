# 🧠⚡ Project HiveMind Wiki — Decentralized Sparse MoE Training & Inference on Consumer Gaming GPUs (`v1.7.0`)

Welcome to the official **Project HiveMind** Wiki! This knowledgebase is synchronized between the [`wiki/`](https://github.com/xaviercallens/HiveMind/tree/main/wiki) directory in the main repository and the live **[GitHub Wiki (`github.com/xaviercallens/HiveMind/wiki`)](https://github.com/xaviercallens/HiveMind/wiki)**.

---

## 🧭 Wiki Sitemap & Core Guides (`8` Pages)

1. **[🏛️ Architecture & 5-Layer Stack](Architecture-and-5-Layer-Stack)**
   - End-to-End Decentralized Training, Upcycling & Speculative Swarm Inference Topology
   - Real `aioquic` 0-RTT Transport (`quic_link.py`), RFC 5389/5766 STUN/TURN NAT Traversal (`nat_traversal.py`), and Multi-Host WAN Fault-Injection (`wan_fault_benchmark.py`)
   - 3-Tier Memory Hierarchy: NVMe CAS Vault $\rightarrow$ Bounded LRU Pinned Host RAM $\rightarrow$ 2-Slot Async DMA (`cuda_stream`, `hip_stream`, `mps_unified`)
2. **[🎮 Consumer GPU Node Setup & Free Steam RTX Tool](Consumer-GPU-Node-Setup-and-Steam-Tool)**
   - Hardware Requirements (`8 GB – 12 GB VRAM` on NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU fallback)
   - Autonomous 3-State Win32 Idle Daemon (`<150 ms` Game Launch Preemption)
   - Local OpenAI `/v1/chat/completions` SSE Proxy (`:8000`) & Raw WebGL 3D Geo-Ring HUD (`:8090`) with 3D Guild Raid Arcs
3. **[🧬 Sparse MoE Upcycling & SVD Rank Projection](Sparse-MoE-Upcycling-and-SVD-Rank-Projection)**
   - Zero-Copy 4-Bit NF4 Safetensors Stream Loader (`175.96 MB` base footprint)
   - 64-Expert 2-Level Hierarchical Sparse MoE (`HierarchicalMoERouter`) & Block-Scaled FP8 Paged KV-Cache (`< 2.2 GB` VRAM)
   - Dynamic Fisher Spectral Rank Allocation (`r_e in {8, 12, 16, 20, 24, 32}`) & Truncated SVD Normalization
4. **[🛡️ Byzantine Defense, ZK-OPoI v3 & 48 Lean 4 Proofs](Byzantine-Defense-ZK-OPoI-and-Lean4-Proofs)**
   - 10% Deterministic CUDA Spot-Check (`CUBLAS_WORKSPACE_CONFIG=:4096:8`) & $2\sigma$-Clipped Multi-Krum
   - Interactive $O(\log_2 L)$ Bisection Fraud Proof Game (`ZK-OPoI v3`) with L2 Prover Bond Slashing (`50%` Challenger / `50%` Consumer)
   - Complete Inventory of the **48 Machine-Checked Lean 4 Theorems (`0 sorry`, `0` custom axioms)**
5. **[🌐 HF Community Hub, Guilds & Joined Co-Op Campaigns](HF-Community-Hub-Guilds-and-Joined-Campaigns)**
   - Live/Offline Hugging Face Hub Push/Pull Sync & Resumable 64 KB Multi-Peer CAS Chunk Seeding (`HM-S5-01`)
   - Scientific Swarm Knowledge Transfer (`100%` peer reuse efficiency & zero redundant retraining)
   - Guild Wars Co-Op Tournament, Missions 17–20 (`HM-S5-04`) & Swarm Mining Guild Alliances (`1.30x`–`1.40x` synergy)
6. **[🗺️ Season 5 & 6 Roadmap & Contributor Guide](Season-5-Roadmap-and-Contributor-Guide)**
   - Collaborator Co-Ownership Matrix (`@xaviercallens`, `@clark719664`, `@pedromineo`, `@radialmonster`)
   - Completed Waves 1–4 (`29/29` PRs Merged) & Season 6 (`Wave 5`) Autonomous Swarm Bounties
7. **[⚡ Swarm Strategy Improvements & Pareto Optimizer (`SI-01`..`SI-05`)](Swarm-Strategy-Improvements-and-Pareto-Optimizer)**
   - `SI-01`: `DynamicFisherSVDSelector` (`95.25% -> 99.51%` spectral energy at `0 B` extra VRAM)
   - `SI-02`: `UCB1CurriculumShardScheduler` (Perplexity-Velocity Multi-Armed Bandit)
   - `SI-03`: `AdaptiveDiLoCoCadenceController` (BBRv1 + RTT/Loss-Aware `<= 8%` Comm SLA)
   - `SI-04`: `reputation_weighted_ties_merge_lora` (`-62.10%` holdout loss vs. uniform TIES under noisy nodes)
   - `SI-05`: `ParetoGuildCampaignPlanner` (`4.65x` speedup, `198.5 $HIVE/GPU-hr`)

---

## 🏆 Collaborator Hall of Fame (`29/29` PRs Merged)

| Collaborator | Role & Track Co-Ownership | Merged PRs | Lean 4 Theorems | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`@xaviercallens`](https://github.com/xaviercallens)** | Founding Protocol Architect & Principal Maintainer | `29` | `25` | 4 Core Pillars, RTX 2070 Stress Suite, Scientific Knowledge Transfer, Strategy Optimizer (`SI-01..05`) & Bisection v3 |
| **[`@clark719664`](https://github.com/clark719664)** | Grandmaster Core, Networking & Formal Architect | `19` | `18` | Real `aioquic` 0-RTT & RFC 5766 TURN (`PR #40`), Raw WebGL 3D Globe (`PR #42`), Non-Circular Lean 4 Proofs (`PRs #21, #39, #41`) |
| **[`@pedromineo`](https://github.com/pedromineo)** | Multi-Vendor GPU, Super-Ring, WAN Faults & Guild Wars Architect | `5` (`PRs #33, #52, #53, #54, #55`) | `5` | Season 4 Omnibus (`PR #33`), WAN QUIC/TURN Fault Benchmark (`PR #52`), Guild Wars Missions 17–20 (`PR #53`), 64-MoE (`PR #54`), 64KB CAS Seeding (`PR #55`) |
| **[`@radialmonster`](https://github.com/radialmonster)** | Steam Ecosystem, Local Prototype & Wiki Co-Owner | `1` (`PR #11`) | — | Public Steam Store AppID Audit (`PR #11`), Local Hack-Miner Prototype Status & Community Docs |
