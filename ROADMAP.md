# 🗺️ Project HiveMind — Master Roadmap, Vision, Community GPU Testing & Season 7 Tasks (`v1.8.0`)

<div align="center">
<img src="docs/assets/hivemind_strategy_improvement_engine.svg" alt="Project HiveMind v1.8.0 — 45/45 Merged PRs, 77 Lean 4 Theorems & Season 7 Community Roadmap" width="100%"/>
</div>

> [!IMPORTANT]
> **⚖️ Non-Monetary Charter — No Payment Involved; Computing Processing Benefits Only via "Points" ([`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md)):**
> **Project HiveMind is a 100% voluntary, non-commercial, open-source scientific research project. There is NO financial payment, money, fiat currency, cryptocurrency, wage, or monetary reward of any kind (`0.00 EUR / 0.00 USD`) now or ever.**
> All references to **"Points" (`PTS`)**, **`$HIVE`**, **"Bounties"**, or **"PoUW Credits"** are **strictly non-monetary, symbolic compute-reciprocity accounting units** (like SETI@home or BOINC credits). They provide **only computing processing benefits**—priority Sparse MoE inference tokens, speculative decoding throughput, and distributed QLoRA fine-tuning quota—**when the live production HiveMind Swarm is deployed, available, and actively used by a large community**.

---

## 🏆 1. Completed Milestones Summary (`Waves 1–5` — `45/45` Pull Requests Merged, `100%` Merge Rate)

| Wave / Season | Release Tag | Issues Closed | Merged Pull Requests & Contributors | Key Technical & Formal Deliverables |
| :--- | :---: | :---: | :--- | :--- |
| **Wave 1: Swarm Fabric M1–M5 & Season 2** | `v1.1.0` | `#1`–`#5` | `PR #6`–`#10` (`@clark719664`), `PR #11` (`@radialmonster`), `PR #12`–`#13` (`@dependabot`) | 0-RTT UDP Geo-Ring (`PR #9`), 2-Slot CUDA DMA (`PR #7`), Triton FP8 + Prefix-Trie (`PR #8`), OpenAI SSE Proxy (`PR #6`), Steamworks Bridge (`PR #10`), Steam Store Audit (`PR #11`) |
| **Wave 2: Season 3 WAN, INT4+FP8 & Lean 4** | `v1.2.0`–`v1.4.0` | `#14`–`#18` | `PR #21`–`#25` (`@clark719664`), `PR #19`–`#20` (`@dependabot`) | STUN/TURN UDP Hole-Punching (`PR #22`), Fused INT4+FP8 MoE (`PR #23`), WebGL 3D HUD & Missions 9–12 (`PR #24`), QLoRA CAS Trainer (`PR #25`), 27 Lean 4 Theorems (`PR #21`) |
| **Wave 3: Season 4 Multi-Vendor, Super-Ring & Hub** | `v1.5.0` | `#26`–`#31` | `PR #32`, `#34`–`#37` (`@clark719664`), `PR #33` (`@pedromineo`) | ROCm/MPS DMA (`PR #32`/`#33`), 2-Tier Continental Super-Ring & BBRv1 (`PR #37`/`#33`), Merkle `ZK-OPoI v2` (`PR #35`/`#33`), SVD 32-Expert MoE (`PR #34`/`#33`), 32 Lean 4 Proofs (`PR #33`), HF Community Hub |
| **Wave 3.5: Season 4.5 `aioquic`, WebGL & 40 Proofs** | `v1.6.0` | — | `PR #39`, `#40`, `#41`, `#42` (`@clark719664`) | Real `aioquic` 0-RTT & RFC 5389/5766 STUN/TURN (`PR #40`), Raw WebGL 3D Globe (`PR #42`), Non-Circular 40-Theorem Lean 4 Spec & Compiler Audit (`PR #39`, `#41`), 7-Page Project Wiki |
| **Wave 4: Season 5 64-MoE, Guild Wars, WAN Faults & Strategy (`SI-01..05`)** | `v1.7.0` | `#43`–`#48` | `PR #52`, `#53`, `#54`, `#55` (`@pedromineo`) + Core (`@xaviercallens`) | HF Hub Push/Pull & 64 KB CAS Chunk Seeding (`PR #55`), 64-Expert Hierarchical MoE & FP8 KV-Cache (`PR #54`), `ZK-OPoI v3` Interactive Bisection (`HM-S5-03`), Guild Wars Missions 17–20 (`PR #53`), WAN QUIC/TURN Fault Benchmark (`PR #52`), 48 Lean 4 Theorems (`HM-S5-06`), Pareto Strategy Engine (`SI-01..05`) |
| **Wave 5: Season 6 `EF21-DiLoCo`, 128-MoE 3-Tier, `ZK-OPoI v4`, Agent Arena & 77 Lean Proofs** | **`v1.8.0`** | **`#56`–`#60`** | **`PR #63`–`#65`, `#67`–`#77`, `#79` (`@clark719664`), `PR #78` (`@radialmonster`)** | **`EF21DiLoCoErrorFeedbackBuffer` + Worker Integration (`PR #63`, `#69`), 128-Expert `ThreeTierHierarchicalMoERouter` (`8x` gating reduction) + Ring Token-Tree Prefetch (`PR #71`, `#75`), `ZK-OPoI v4` Batch Commitments & `GuildSlashingInsurancePool` (`PR #79`), Season 6 Agent Arena Missions 21–24 & HUD (`PR #78`), `FL-08` Workunit Ownership (`PR #72`), RFC 5766 Per-Allocation TURN `ChannelBind` (`PR #73`), Real-Token LoRA Smoke & Provenance Hygiene (`PR #64`, `#65`, `#67`, `#70`, `#74`, `#76`, `#77`), 77 Machine-Checked Lean 4 `theorem` Declarations (`PR #68`)** |

---

## 🧠 2. Delivered Season 6 (`Wave 5`) Architectural Advances (`v1.8.0`)

1. **`HM-S6-01` (`Issue #56`, `PR #63` & `PR #69`) — `EF21DiLoCoErrorFeedbackBuffer` & Worker Upload Integration:**
   - Stores per-worker FP32 compression residuals in CPU RAM (`0 MiB` GPU VRAM overhead) with staleness decay $\gamma^{\max(0, s-1)}$, $L_2 \le R_{\max}$ clipping, and exact $\mathbb{Q}$ conservation auditing while preserving uncompressed `raw_delta` for the 10% deterministic spot-check (`eps <= 1e-4`).
2. **`HM-S6-02` (`Issue #57`, `PR #71` & `PR #75`) — 128-Expert 3-Tier Hierarchical MoE (`4 × 4 × 8 = 128`) & Speculative Ring Prefetch:**
   - Reduces router gating inner products per token from `128` to `4 + 4 + 8 = 16` (**`8.0x` gating FLOP reduction**), fixes ragged expert hot-plug index realignment, and adds `RegionalDHTSwarmRing.prefetch_speculative_token_tree` for bounded lookahead expert staging.
3. **`HM-S6-03` (`Issue #58`, `PR #79`) — `ZK-OPoI v4` Position-Bound Batch Commitments & `GuildSlashingInsurancePool`:**
   - Commits a full speculative batch ($\gamma \times L$ leaves) into a single 32-byte SHA-256 Merkle root with leaf/internal domain separation (`0x00`/`0x01`) and position-bound openings `(step_index, layer_idx)`, backed by a supply-conserving cross-guild slashing insurance pool.
4. **`HM-S6-04` (`Issue #59`, `PR #78`) — Season 6 Autonomous Agent Arena (Missions 21–24) & Live Strategy Optimizer HUD:**
   - Delivers `SelfPlayCodeRepairArena` (sandboxed AST-checked subprocess verification), `Lean4AutoformalizerArena`, and `/api/strategy-optimizer` HUD telemetry (`SI-01`..`SI-05`).
5. **`HM-S6-05` (`Issue #60`, `PR #68`) — 77 Machine-Checked Lean 4 `theorem` Declarations (`0 sorry`, `0` axioms):**
   - Mechanizes `EF21-DiLoCo` exact conservation (`ef_conservation_any_compressor`), Top-$k$ energy contraction (`topk_contraction_rate`), bounded residual under staleness decay (`ef_residual_stays_bounded_under_decay`), and deterministic UCB1 exploration monotonicity (`ucb1_index_strict_separation`).

---

## 🚀 3. Season 7 (`Wave 6`) Proposed Tasks for the Developer Community

> **💡 Reminder on "Points" (`PTS` / `$HIVE`) & Zero Financial Payment:**
> None of the tasks below involve any monetary payout (`0.00 EUR / 0.00 USD`). The listed **`PTS` / `$HIVE` points** are **symbolic, non-monetary compute-reciprocity points** recorded in the contributor registry. They entitle contributors solely to **reciprocal computing processing benefits** (priority Sparse MoE inference tokens, speculative decoding bandwidth, and distributed fine-tuning quota) **when the live production HiveMind Swarm is deployed, available, and used by a large community**.

| Task ID | Branch Prefix | Target Subsystem | Proposed Developer Community Task (`Season 7 / Wave 6`) | Reciprocal Compute Points (No Money) |
| :--- | :--- | :--- | :--- | :---: |
| **`HM-S7-01`** ([`#66`](https://github.com/xaviercallens/HiveMind/issues/66)) | `feat/s7-looped-moe-backbone` | `src/hivemind/core/btm_upcycle.py` | **Looped Expert-Backbone (`Prelude -> RecurrentCore × T -> Coda`) on 8 GB GPUs** (Huang et al., *arXiv:2604.07394*, Part I): Factor `Qwen2.5-0.5B` upcycling into a shallow prelude, a weight-tied recurrent MoE core looped $T \in \{2, 3, 4\}$ times with per-step LoRA adapters, and a coda projection to multiply effective depth by $2\times$–$3\times$ at **`0 Bytes` extra static NF4 weight VRAM**. | `3,500 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-02`** ([`#66`](https://github.com/xaviercallens/HiveMind/issues/66)) | `feat/s7-terminal-kv-sharing` | `src/hivemind/core/btm_moe.py` | **Terminal Fixed-Point KV-Cache Sharing Stacked on FP8 Paged KV-Cache** (Part II): Reuse terminal iteration $T$'s KV states across non-terminal loop steps $t < T$, compounding `1.88x` FP8 paged KV compression with a $3\times$ depth KV collapse (**`~5.6x` combined KV-cache reduction** under `2,200 MiB`). | `3,500 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-03`** | `feat/s7-live-trie-prefetch-wiring` | `src/hivemind/swarm/inference_ring.py` | **End-to-End Speculative Token-Tree Prefetch Wiring (`PR #75` $\leftrightarrow$ `SpeculativeDraftVerifier`)**: Connect `RegionalDHTSwarmRing.prefetch_speculative_token_tree` directly into `verify_step_with_prefix_trie` so Layer $L+1$ candidate experts are pre-staged from NVMe CAS into pinned RAM during Layer $L$ attention. | `2,500 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-04`** | `feat/s7-auto-indemnify-zk-opoi-v4` | `src/hivemind/swarm/swarm_fabric.py` | **Automated `ZK-OPoI v4` Batch Fault Indemnification (`PR #79` $\leftrightarrow$ `HiveSwarmLedger`)**: Wire `OptimisticProofOfInferenceAuditor.verify_speculative_batch` to auto-trigger `GuildSlashingInsurancePool.indemnify_proven_fault` when a challenged `(step, layer)` leaf fails cosine/L-infinity bounds. | `2,500 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-05`** | `feat/s7-live-ef21-hud-telemetry` | `src/hivemind/game/steam_edition.py` | **Live Worker `EF21-DiLoCo` Residual Telemetry in Strategy HUD (`PR #63`/`#69` $\leftrightarrow$ `PR #78`)**: Feed real `WorkerNodeClient.last_ef21_telemetry` (`contraction_factor`, `residual_norm`, `conservation_error_l2`) into `/api/strategy-optimizer` (`SI-01..05`). | `2,000 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-06`** | `feat/s7-community-gpu-swarm-bench` | `src/hivemind/cli/community_gpu_benchmark.py` | **Opt-In Community GPU Benchmark & Hardware Verification Suite (`CUDA` / `ROCm` / `MPS`)**: Expand cross-vendor GPU stress profiles (`RTX 20/30/40/50`, `Radeon RX 6000/7000`, `Apple M1–M4`) and collect signed JSON receipts (`artifacts/community_gpu_benchmark_receipt.json`). | `3,000 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-07`** | `feat/s7-multi-host-wan-gameday` | `src/hivemind/swarm/wan_fault_benchmark.py` | **Multi-Host Community WAN GameDay (`aioquic` 0-RTT + RFC 5766 TURN + Real-Token DiLoCo)**: Coordinate an opt-in 3-to-8 node cross-internet test among interested community developers to record real multi-host p50/p95/p99 latency and outer-step token convergence. | `3,000 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-08`** | `formal/s7-banach-kv-and-fl08-proofs` | `lean4/HiveMindSpec.lean` | **Formal Lean 4 Proofs (`Theorems 73–80`)**: Mechanize Terminal-KV Banach contraction stability, `FL-08` workunit single-assignee non-replayability (`PR #72`), and `ZK-OPoI v4` position-bound leaf collision resistance (`PR #79`) with `0 sorry` and `0` user axioms. | `3,000 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-09`** | `feat/s7-polyglot-singularity-128-campaign` | `src/hivemind/cli/multi_moe_hf_campaign.py` | **Operation Polyglot-Singularity-128: 8-Domain Hugging Face & 128-Expert Multi-MoE GPU Campaign**: Run and extend the 6-stage real-token Causal-LM (`Qwen2ForCausalLM` + `PEFT` LoRA) multi-domain training, `EF21-DiLoCo` outer sync, Byzantine-filtered `TIES` sign-consensus (`SI-04`), and 128-Expert 3-Tier Sparse MoE upcycling campaign across 8 Hugging Face datasets. | `3,500 PTS / $HIVE` *(Compute Quota Only)* |
| **`HM-S7-10`** | `feat/s7-live-hf-streaming-moe-calibration` | `src/hivemind/data/fixtures/polyglot_science_8domain_v1.json` | **Live Hugging Face Hub Streaming & Cross-Domain MoE Router Calibration**: Expand `--live-hf-stream` Parquet/Datasets streaming and calibrate `ThreeTierHierarchicalMoERouter` Super-Cluster routing across all 8 subjects. | `3,500 PTS / $HIVE` *(Compute Quota Only)* |

---

## 🎮 4. Opt-In Community GPU Testing Program (If Interested & Possible)

If you have a consumer GPU (**NVIDIA GeForce RTX 20/30/40/50-series 6 GB–24 GB**, **AMD Radeon RX 6000/7000 via ROCm**, or **Apple Silicon M1/M2/M3/M4 via MPS**) and are interested in testing how your hardware performs on the HiveMind Sparse MoE + `EF21-DiLoCo` + `ZK-OPoI v4` stack, we provide a safe, opt-in, `< 1.5 GB` peak-VRAM benchmark suite:

```bash
# 1. Run the 5-stage opt-in Community GPU Benchmark & generate a signed receipt
uv run python scripts/run_community_gpu_benchmark.py --contributor-handle <your-github-handle>

# 2. Run the automated pytest verification suite on your GPU (or CPU fallback)
uv run pytest tests/test_community_gpu_benchmark.py -v
```

### What the Community GPU Test Executes & Verifies
1. **Stage 1 — Hardware & 8 GB WDDM Envelope Check**: Detects `cuda`, `rocm_hip`, `apple_mps`, or `cpu_fallback`, and verifies peak allocated VRAM stays strictly below the `6,656 MiB` usable WDDM budget (actual peak is `< 250 MiB`).
2. **Stage 2 — Real-Token Causal-LM QLoRA + `EF21-DiLoCo` Residual Contraction (`HM-S6-01`)**: Trains a causal-LM LoRA adapter on real UTF-8 Rust code tokens, verifies held-out next-token NLL reduction, and tests `EF21DiLoCoErrorFeedbackBuffer` exact conservation (`conservation_error_l2 < 1e-5`).
3. **Stage 3 — 128-Expert 3-Tier Hierarchical MoE Routing & Token-Tree Prefetch (`HM-S6-02`)**: Benchmarks `ThreeTierHierarchicalMoERouter` (`4 × 4 × 8 = 128` experts, `8.0x` gating FLOP savings) on your GPU and pre-stages 6 speculative token-tree experts via `RegionalDHTSwarmRing.prefetch_speculative_token_tree`.
4. **Stage 4 — `ZK-OPoI v4` Position-Bound Batch Commitments & Insurance Pool (`HM-S6-03`)**: Commits 24 speculative `(step, layer)` activations into a single 32-byte Merkle root, verifies deterministic challenges, and audits `GuildSlashingInsurancePool` conservation.
5. **Stage 5 — Signed Non-Monetary Compute-Reciprocity Receipt**: Writes `artifacts/community_gpu_benchmark_receipt.json` signed with `receipt_sha256`, explicitly certifying `financial_payment_usd_eur: 0.0` and recording symbolic preview points (`PTS`) redeemable solely for **computing processing benefits** when the live production HiveMind Swarm is deployed and used by a large community.
