# Project HiveMind — Large-Scale Parallel Contributor & Release Governance

Project HiveMind is architected so **dozens of independent contributors and research groups can develop, test, and merge Pull Requests concurrently** without merge conflicts, regressions, or release tampering.

---

## 1. Subsystem Isolation Matrix (Conflict-Free Parallel PRs)

Every module in `src/hivemind/` and `lean4/` maps to an isolated track with dedicated unit tests:

| Track Prefix | Isolated Directory / Module | Scope | Required CI Gate |
| :--- | :--- | :--- | :--- |
| **`feat/m1-*`** | [`src/hivemind/swarm/quic_transport.py`](src/hivemind/swarm/quic_transport.py) | UDP/QUIC Geo-Ring, 0-RTT tickets, NAT traversal | `20/20` Pytest + Loopback `<5 ms` |
| **`feat/m2-*` / `feat/m3-*`** | [`src/hivemind/swarm/swarm_fabric.py`](src/hivemind/swarm/swarm_fabric.py) | 2-Slot Async DMA, Triton FP8 (`E4M3`), Prefix-Trie Speculative Tree | `20/20` Pytest + `100%` Tree Acceptance |
| **`feat/m4-*`** | [`src/hivemind/boinc/coordinator_api.py`](src/hivemind/boinc/coordinator_api.py) | OpenAI `/v1/chat/completions` SSE Gateway, L2 vouchers | `20/20` Pytest + Lossless SSE Reconstruction |
| **`game/*`** | [`src/hivemind/game/`](src/hivemind/game/) & [`src/hivemind/client/steamworks.py`](src/hivemind/client/steamworks.py) | HF Hack-Miner Seasons, Web HUD, Steamworks Rich Presence & VDFs | `20/20` Pytest + `<42 ms` Game Preemption |
| **`expert/*`** | [`src/hivemind/core/btm_moe.py`](src/hivemind/core/btm_moe.py) & [`src/hivemind/data/`](src/hivemind/data/) | 8-Branch Sparse MoE, Top-$k$ FP8 (`514x`) LoRA adapters | `20/20` Pytest + `<=5,950 MB` VRAM Cap |
| **`formal/*`** | [`lean4/HiveMindSpec.lean`](lean4/HiveMindSpec.lean) & [`paper/`](paper/) | Machine-checked Lean 4 theorems & Foundation Paper | `0 sorry`, `0` user axioms, `>=17` theorems |

---

## 2. Automated Branch Protection on `main`

The `main` branch on `xaviercallens/HiveMind` enforces:
1. **Pull Request Required:** Direct pushes from external contributors are blocked; all changes flow through PRs.
2. **Code Owner Review ([`.github/CODEOWNERS`](.github/CODEOWNERS)):** Requires maintainer/code-owner sign-off and automatic dismissal of stale approvals when new commits are pushed.
3. **Mandatory Status Check ([`.github/workflows/ci_swarm_tests.yml`](.github/workflows/ci_swarm_tests.yml)):**
   - Job: `Verify 4 Core Pillars, Swarm Fabric (M1-M5), Lean 4 Spec & Steam Hack-Miner (20/20 Tests)`
   - Enforces `0 sorry` and `0` user axioms in `lean4/HiveMindSpec.lean`.
   - Enforces `20/20` passing tests across Pillars 1–4, Phase 2 Swarm Fabric (`M1`–`M5`), and the 6-stage RTX GPU stress suite.
4. **Conversation Resolution & History Integrity:** All review threads must be resolved before merge; force-pushes (`allow_force_pushes: false`) and branch deletions (`allow_deletions: false`) on `main` are permanently disabled.

---

## 3. Immutable Release Protocol (`refs/tags/v*`)

1. **Repository Ruleset (`HiveMind Immutable Release Tags (v*)`):**
   - Active GitHub Ruleset targeting `refs/tags/v*` blocks `deletion`, `update`, and `non_fast_forward` rewrites. Once a release tag (`v1.0.0`, `v1.0.1`, etc.) is pushed, its commit SHA is permanently locked.
2. **Cryptographic Attestation ([`artifacts/SHA256SUMS.txt`](artifacts/SHA256SUMS.txt)):**
   - Every release publishes SHA-256 digests for `HiveMind_Steam_FreeEdition_<tag>.zip`, `paper/hivemind_foundation_article.pdf`, `lean4/HiveMindSpec.lean`, `artifacts/rtx_intense_stress_and_steam_deploy_report.json`, and `steam_dist/HiveMind_Steam_FreeEdition/rtx_gpu_deployment_cert.json`.
