# 🗺️ Season 5 Roadmap & Contributor Guide

All **25 Pull Requests across Waves 1, 2, and 3 (`Seasons 1–4`)** have been reviewed, verified, and merged into `main`. We have now opened **6 Season 5 / Wave 4 Community Bounties (`HM-S5-01` through `HM-S5-06` — `15,000 $HIVE` Pool)** on **[GitHub Issues](https://github.com/xaviercallens/HiveMind/issues)** and **[GitHub Discussions](https://github.com/xaviercallens/HiveMind/discussions)**!

---

## 1. Open Season 5 / Wave 4 Bounties (`15,000 $HIVE` Pool)

| Milestone ID | Track & Target Scope | Deliverable Summary | Suggested Co-Owners | Bounty |
| :---: | :--- | :--- | :--- | :---: |
| **`HM-S5-01`** | **Track 1: Live HF Hub Push/Pull & P2P Seeding** (`hf_community_hub.py`, `dht_storage.py`) | Live `huggingface_hub` API upload/download + multi-peer 64 KB chunk streaming with SHA-256 verification and offline fallback | `@pedromineo`, `@clark719664` | **`+2,500 $HIVE`** |
| **`HM-S5-02`** | **Track 2: 64-Expert Hierarchical MoE & FP8 KV-Paging** (`btm_moe.py`, `swarm_fabric.py`) | 64-Expert (`8 Macro x 8 Micro`) Sparse MoE + `float8_e4m3fn` paged KV-cache keeping 8 GB consumer GPU VRAM `< 2.2 GB` at 4,096 context | `@clark719664`, `@pedromineo` | **`+3,000 $HIVE`** |
| **`HM-S5-03`** | **Track 3: Interactive Bisection Fraud Game v3** (`validator.py`, `ledger.py`) | $O(\log L)$ interactive binary-search activation dispute game over 32 transformer layers + automatic L2 `$HIVE` bond slashing | `@clark719664`, `@pedromineo` | **`+2,500 $HIVE`** |
| **`HM-S5-04`** | **Track 4: Guild Wars Tournament & Missions 17–20** (`hf_hack_miner.py`, `georing_webgl.py`) | Season 5 Hack-Miner Campaign (`Missions 17–20`), Cross-Guild Co-Op Matchmaking & live 3D WebGL Raid Arcs on `:8090` | `@pedromineo`, `@radialmonster` | **`+2,500 $HIVE`** |
| **`HM-S5-05`** | **Track 5: Multi-Host WAN QUIC + TURN Benchmark** (`quic_link.py`, `nat_traversal.py`, `scripts/`) | Reproducible multi-process WAN benchmark measuring `aioquic` 0-RTT resumption and RFC 5766 TURN relay under 1%–5% packet loss | `@clark719664` | **`+2,000 $HIVE`** |
| **`HM-S5-06`** | **Track 6: Lean 4 Season 5 Proofs (`Theorems 41–46`)** (`lean4/HiveMindSpec.lean`, `paper/`) | Non-circular machine-checked Lean 4 theorems for TIES sign-consensus, Guild synergy conservation, $O(\log L)$ bisection, and FP8 KV bounds | `@clark719664`, `@pedromineo` | **`+2,500 $HIVE`** |

---

## 2. Conflict-Free Pull Request Workflow

1. **Claim an Issue or Join a Discussion:** Comment `/claim` on any open Season 5 issue in [`github.com/xaviercallens/HiveMind/issues`](https://github.com/xaviercallens/HiveMind/issues) or coordinate in [`github.com/xaviercallens/HiveMind/discussions`](https://github.com/xaviercallens/HiveMind/discussions).
2. **Create an Isolated Topic Branch:**
   ```bash
   git checkout -b feat/s5-hf-hub-sync
   ```
3. **Run the Full Verification Gate Locally (`240+` Tests + `40` Lean 4 Proofs):**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
   ```
4. **Open a Pull Request against `main`:** Reference `Closes #<issue>` in your PR description.
