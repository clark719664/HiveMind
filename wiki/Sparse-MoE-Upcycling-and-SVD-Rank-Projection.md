# 🧬 Sparse MoE Upcycling, 128-Expert 3-Tier Router & SVD Rank Projection (`v1.8.0`)

Project HiveMind enables independent miners to train domain-specialized QLoRA adapters at arbitrary ranks (`r=8, 16, 32, 64`) and seamlessly fuse them into a single **Branch-Train-Mix (BTM) Sparse Mixture-of-Experts (`UpcycledSparseMoEModel`)** within an **8 GB VRAM envelope** (`<= 6,656 MiB` usable WDDM budget).

---

## 1. Factored SwiGLU LoRA Expert Architecture (`src/hivemind/core/btm_moe.py`)

Instead of duplicating dense SwiGLU FFN weights ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) across $E$ experts, HiveMind keeps the **4-bit NF4 base backbone frozen and shared**, and stores each domain expert as a compact **factored low-rank residual** $(A_e, B_e)$ routed via Top-2 gating:

$$y(x) = W_{\text{NF4}}(x) + \sum_{e \in \text{Top-2}(G(x))} g_e(x) \cdot \frac{\alpha_e}{r_e} B_e A_e x$$

- **96 Domain Experts on `Qwen/Qwen2.5-0.5B`:** Fits in **`1,527.03 MiB` peak VRAM** on an 8 GB NVIDIA RTX 2070 (leaving `5.63 GB` free VRAM).
- **Triton INT4 + FP8 Fused MoE Router (`HM-S3-02` / `PR #23`):** Packs expert weights in 2x4-bit `uint8` (`3.88x` compression) and dispatches Top-2 projections in `float8_e4m3fn` with **`0.9963` cosine similarity** and **`31.79 dB` SQNR**.

---

## 2. 128-Expert 3-Tier Hierarchical MoE Router (`HM-S6-02` / `PR #71`)

Implemented in [`ThreeTierHierarchicalMoERouter`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/btm_moe.py) (`PR #71` by `@clark719664`), HiveMind factors a 128-expert pool into a **3-tier hierarchy (`4 Super-Clusters × 4 Macro-Domains × 8 Micro-Experts = 128`)**:

- **Gating FLOP Reduction (`8.0x`):** Evaluates only `4 + 4 + 8 = 16` inner products per token instead of `128` (`gating_reduction_factor = 8.0`).
- **Ragged Hot-Plug Index Realignment:** `hotplug_three_tier_experts` and `SparseMoELayer.forward` use cumulative cluster offsets so uneven cluster growth (`128 -> 129+` experts) never aliases expert indices or leaves a trailing expert unreachable.

---

## 3. Speculative Token-Tree Expert Prefetching (`HM-S6-02` / `PR #75`)

Implemented in [`RegionalDHTSwarmRing.prefetch_speculative_token_tree`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/inference_ring.py) (`PR #75` by `@clark719664`), the inference ring computes branch prefix probabilities (`prefix_trie_branch_probabilities`) over the speculative token tree and pre-stages the highest-probability Layer $L+1$ experts into the lookahead DMA slot (`slot_A` / `slot_B`) of `PinnedRAMExpertDMAStreamer`.

---

## 4. Dynamic Truncated-SVD Rank Projection & Season 7 Looped MoE (`Issue #66`)

- **Truncated-SVD Rank Normalization (`HM-S4-05` / `PR #34`):** `project_lora_rank_svd` normalizes mixed-rank community adapters (`r=8/16/32/64 -> r=16`) via QR-SVD while retaining `>= 99.4%` spectral energy, paired with `SI-01` `DynamicFisherSVDSelector`.
- **Season 7 Open Tasks (`HM-S7-01` & `HM-S7-02` in [`Issue #66`](https://github.com/xaviercallens/HiveMind/issues/66)):**
  - **`HM-S7-01` (`feat/s7-looped-moe-backbone`):** Factor `Qwen2.5-0.5B` upcycling into `Prelude -> RecurrentCore × T -> Coda` with per-step LoRA adapters (`0 Bytes` extra static NF4 weight VRAM).
  - **`HM-S7-02` (`feat/s7-terminal-kv-sharing`):** Stack terminal fixed-point KV-cache sharing across recurrent loop steps on top of `float8_e4m3fn` paged KV-cache (`~5.6x` combined KV-cache reduction under `2,200 MiB`).
