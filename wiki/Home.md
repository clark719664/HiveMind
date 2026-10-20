# 🧠⚡ Project HiveMind Wiki — Decentralized Sparse MoE Training & Inference on Consumer Gaming GPUs (`v1.6.0`)

Welcome to the official **Project HiveMind** Wiki! This knowledgebase is synchronized between the [`wiki/`](https://github.com/xaviercallens/HiveMind/tree/main/wiki) directory in the main repository and the live **[GitHub Wiki (`github.com/xaviercallens/HiveMind/wiki`)](https://github.com/xaviercallens/HiveMind/wiki)**.

---

## 🧭 Wiki Sitemap & Core Guides

1. **[🏛️ Architecture & 5-Layer Stack](Architecture-and-5-Layer-Stack)**
   - End-to-End Decentralized Training, Upcycling & Speculative Swarm Inference Topology
   - Real `aioquic` 0-RTT Transport (`quic_link.py`), RFC 5389/5766 STUN/TURN NAT Traversal (`nat_traversal.py`), and 2-Tier Continental Super-Ring (`quic_transport.py`)
   - 3-Tier Memory Hierarchy: NVMe CAS Vault $\rightarrow$ Bounded LRU Pinned Host RAM $\rightarrow$ 2-Slot Async DMA (`cuda_stream`, `hip_stream`, `mps_unified`)
2. **[🎮 Consumer GPU Node Setup & Free Steam RTX Tool](Consumer-GPU-Node-Setup-and-Steam-Tool)**
   - Hardware Requirements (`8 GB – 12 GB VRAM` on NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU fallback)
   - Autonomous 3-State Win32 Idle Daemon (`<150 ms` Game Launch Preemption)
   - Local OpenAI `/v1/chat/completions` SSE Proxy (`:8000`) & Raw WebGL 3D Geo-Ring HUD (`:8090`)
3. **[🧬 Sparse MoE Upcycling & SVD Rank Projection](Sparse-MoE-Upcycling-and-SVD-Rank-Projection)**
   - Zero-Copy 4-Bit NF4 Safetensors Stream Loader (`175.96 MB` base footprint)
   - DiLoCo-QLoRA Inner Loop + Top-$k$ (`k=0.05`) + FP8 (`float8_e4m3fn`) Pseudo-Gradient Compression
   - Triton INT4+FP8 Fused MoE Router & Truncated SVD Rank-Adaptive 32-Expert Scaling (`r=8..64 -> r=16` in `< 2.0 GB` VRAM)
4. **[🛡️ Byzantine Defense, ZK-OPoI v2 & 40 Lean 4 Proofs](Byzantine-Defense-ZK-OPoI-and-Lean4-Proofs)**
   - 10% Deterministic CUDA Spot-Check (`CUBLAS_WORKSPACE_CONFIG=:4096:8`) & $2\sigma$-Clipped Multi-Krum
   - Cryptographic Activation Spot-Check (`ZK-OPoI v2`) with $O(\log L)$ Merkle Inclusion Proofs
   - Complete Inventory of the **40 Machine-Checked Lean 4 Theorems (`0 sorry`, `0` custom axioms)**
5. **[🌐 HF Community Hub, Guilds & Joined Co-Op Campaigns](HF-Community-Hub-Guilds-and-Joined-Campaigns)**
   - Scientific Swarm Knowledge Transfer (`100%` peer reuse efficiency & zero redundant retraining)
   - 4 Joined Co-Op Swarm Campaigns (`3.84x` faster & `+94.12%` lower holdout loss via TIES-Merging)
   - Portable Hugging Face PEFT Adapter Bundles & Swarm Mining Guild Alliances (`1.30x`–`1.40x` synergy)
6. **[🗺️ Season 5 Roadmap & Contributor Guide](Season-5-Roadmap-and-Contributor-Guide)**
   - Collaborator Co-Ownership Matrix (`@xaviercallens`, `@clark719664`, `@pedromineo`, `@radialmonster`)
   - Open Season 5 / Wave 4 Bounties (`HM-S5-01` through `HM-S5-06` — `15,000 $HIVE` Pool)
   - How to Claim an Issue, Run the `240+` Test Suite, and Submit a Conflict-Free Pull Request

---

## 🏆 Collaborator Hall of Fame (`25/25` PRs Merged)

| Collaborator | Role & Track Co-Ownership | Merged PRs | Lean 4 Theorems | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`@xaviercallens`](https://github.com/xaviercallens)** | Founding Protocol Architect & Principal Maintainer | `25` | `17` | 4 Core Pillars, RTX 2070 Stress Suite, Scientific Knowledge Transfer, HF Community Hub & Guilds |
| **[`@clark719664`](https://github.com/clark719664)** | Grandmaster Core, Networking & Formal Architect | `19` | `23` | Real `aioquic` 0-RTT & RFC 5766 TURN (`PR #40`), Raw WebGL 3D Globe (`PR #42`), 40-Theorem Lean 4 Spec (`PRs #21, #33, #39, #41`) |
| **[`@pedromineo`](https://github.com/pedromineo)** | Multi-Vendor GPU, Super-Ring & ZK-OPoI Co-Owner | `5` | — | AMD ROCm / Apple MPS DMA (`PR #32`), Continental Super-Ring (`PR #37`), ZK-OPoI v2 (`PR #35`), SVD 32-MoE (`PR #34`) |
| **[`@radialmonster`](https://github.com/radialmonster)** | Steam Ecosystem, Local Prototype & Wiki Co-Owner | `1` | — | Public Steam Store AppID Audit (`PR #11`), Local Hack-Miner Prototype Status & Community Docs |
