# Project HiveMind — Large-Scale Parallel Contributor & Release Governance (`v1.6.0`)

Project HiveMind is architected so **dozens of independent contributors, Swarm Mining Guilds, and research groups can develop, test, and merge Pull Requests concurrently** without merge conflicts, regressions, or release tampering.

---

## 1. Core Maintainers & Track Co-Owners

| Collaborator Handle | Role & Track Ownership | Merged PRs | Key Contributions |
| :--- | :--- | :---: | :--- |
| **[`@xaviercallens`](https://github.com/xaviercallens)** | Founding Protocol Architect & Principal Maintainer | `20+` | 4 Core Pillars, RTX 2070 Stress Suite, Scientific Swarm Evaluation, HF Community Hub & Guilds |
| **[`@clark719664`](https://github.com/clark719664)** | Grandmaster Core & Formal Architect (`M1`–`M5`, `S3`, `S4`, `PRs #6`–`#10`, `#21`–`#25`, `#33`, `#39`–`#42`) | `19` | Real QUIC 0-RTT & STUN/TURN NAT Traversal, Triton INT4+FP8 MoE, WebGL 3D Geo-Ring, 40-Theorem Lean 4 Spec |
| **[`@pedromineo`](https://github.com/pedromineo)** | Season 4 Systems, Multi-Vendor GPU & ZK-OPoI Co-Owner (`PRs #32`, `#34`–`#37`) | `5` | AMD ROCm / Apple MPS DMA, Continental Super-Ring, ZK-OPoI v2 Merkle Proofs, SVD Rank-Adaptive 32-MoE |
| **[`@radialmonster`](https://github.com/radialmonster)** | Steam Ecosystem, Local Prototype & Documentation Co-Owner (`PR #11`) | `1` | Public Steam Store AppID Audit, Local Hack-Miner Prototype Status & Community Docs |

---

## 2. Subsystem Isolation Matrix (Conflict-Free Parallel PRs)

Every module in `src/hivemind/`, `lean4/`, and `wiki/` maps to an isolated track with dedicated unit tests:

| Track Prefix | Isolated Directory / Module | Scope | Co-Owners | Required CI Gate |
| :--- | :--- | :--- | :--- | :--- |
| **`feat/s5-p2p-*`** | [`src/hivemind/swarm/quic_link.py`](src/hivemind/swarm/quic_link.py), [`nat_traversal.py`](src/hivemind/swarm/nat_traversal.py) | Real QUIC 0-RTT links, STUN/TURN NAT hole-punching, Continental Super-Ring | `@xaviercallens`, `@clark719664`, `@pedromineo` | Full Pytest + Endpoint Safety Suite |
| **`feat/s5-dma-*` / `feat/s5-moe-*`** | [`src/hivemind/swarm/swarm_fabric.py`](src/hivemind/swarm/swarm_fabric.py), [`src/hivemind/core/`](src/hivemind/core/) | Multi-Vendor CUDA/ROCm/MPS DMA, Triton INT4+FP8 MoE, SVD 32/64-Expert Scaling | `@xaviercallens`, `@clark719664`, `@pedromineo` | Full Pytest + `< 2.0 GB` MoE VRAM Cap |
| **`feat/s5-opoi-*`** | [`src/hivemind/boinc/`](src/hivemind/boinc/), [`src/hivemind/swarm/ledger.py`](src/hivemind/swarm/ledger.py) | OpenAI `/v1/chat/completions` SSE Gateway, ZK-OPoI v2 Merkle Disputes, L2 `$HIVE` Vouchers | `@xaviercallens`, `@clark719664`, `@pedromineo` | Full Pytest + Lossless SSE & Merkle Audit |
| **`game/s5-*` / `hub/s5-*`** | [`src/hivemind/game/`](src/hivemind/game/), [`src/hivemind/swarm/hf_community_hub.py`](src/hivemind/swarm/hf_community_hub.py) | HF Hack-Miner Campaigns, Real WebGL 3D Geo-Ring HUD, HF Community Hub & Guilds | `@xaviercallens`, `@clark719664`, `@pedromineo`, `@radialmonster` | Full Pytest + WebGL Renderer & Bundle SHA-256 |
| **`formal/s5-*`** | [`lean4/HiveMindSpec.lean`](lean4/HiveMindSpec.lean), [`paper/`](paper/) | Machine-checked Lean 4 theorems (`40/40`) & Foundation Paper | `@xaviercallens`, `@clark719664`, `@pedromineo` | `0 sorry`, `0` user axioms, real `lean` compiler check |
| **`docs/s5-*` / `wiki/*`** | [`ROADMAP.md`](ROADMAP.md), [`wiki/`](wiki/), [`docs/`](docs/) | Project Wiki, Season 5 Roadmap, Steam Status & Bounties | `@xaviercallens`, `@clark719664`, `@pedromineo`, `@radialmonster` | Markdown & Link Integrity |

---

## 3. Automated Branch Protection on `main`

The `main` branch on `xaviercallens/HiveMind` enforces:
1. **Pull Request Required:** Direct pushes from external contributors are blocked; all changes flow through PRs.
2. **Code Owner Review ([`.github/CODEOWNERS`](.github/CODEOWNERS)):** Requires maintainer/code-owner sign-off and automatic dismissal of stale approvals when new commits are pushed.
3. **Mandatory Status Check ([`.github/workflows/ci_swarm_tests.yml`](.github/workflows/ci_swarm_tests.yml)):**
   - Compiles `lean4/HiveMindSpec.lean` with the pinned Lean 4 compiler (`v4.33.0-rc2`) and requires `require_machine_checked(result)` (`0 sorry`, `0` user axioms, complete `#print axioms` verification for all `40` theorems).
   - Runs the full automated `pytest tests/ -v --tb=short` verification suite (`240+` tests).
4. **Conversation Resolution & History Integrity:** All review threads must be resolved before merge; force-pushes (`allow_force_pushes: false`) and branch deletions (`allow_deletions: false`) on `main` are permanently disabled.

---

## 4. Immutable Release Protocol (`refs/tags/v*`)

1. **Repository Ruleset (`HiveMind Immutable Release Tags (v*)`):**
   - Active GitHub Ruleset targeting `refs/tags/v*` blocks `deletion`, `update`, and `non_fast_forward` rewrites. Once a release tag (`v1.0.0` .. `v1.6.0`) is pushed, its commit SHA is permanently locked.
2. **Cryptographic Attestation ([`artifacts/SHA256SUMS.txt`](artifacts/SHA256SUMS.txt)):**
   - Every release publishes SHA-256 digests for `HiveMind_Steam_FreeEdition_<tag>.zip`, `paper/hivemind_foundation_article.pdf`, `lean4/HiveMindSpec.lean`, and community bundle manifests.
