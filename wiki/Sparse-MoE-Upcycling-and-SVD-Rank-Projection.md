# 🧬 Sparse MoE Upcycling & SVD Rank Projection

Project HiveMind enables independent miners to train domain-specialized QLoRA adapters at arbitrary ranks (`r=8, 16, 32, 64`) and seamlessly fuse them into a single **Branch-Train-Mix (BTM) Sparse Mixture-of-Experts (`UpcycledSparseMoEModel`)** within an **8 GB VRAM envelope**.

---

## 1. Factored SwiGLU LoRA Expert Architecture (`src/hivemind/core/btm_moe.py`)

Instead of duplicating the dense SwiGLU FFN weights ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) across $E$ experts—which would immediately exhaust 8 GB of VRAM—HiveMind keeps the **4-bit NF4 base backbone frozen and shared**, and stores each domain expert as a compact **factored low-rank residual** $(A_e, B_e)$ routed via Top-2 gating:

$$y(x) = W_{\text{NF4}}(x) + \sum_{e \in \text{Top-2}(G(x))} g_e(x) \cdot \frac{\alpha_e}{r_e} B_e A_e x$$

- **96 Domain Experts on `Qwen/Qwen2.5-0.5B`:** Fits in **`1,527.03 MiB` peak VRAM** on an 8 GB NVIDIA RTX 2070 (leaving `5.63 GB` free VRAM).
- **Triton INT4 + FP8 Fused MoE Router (`HM-S3-02` / `PR #23`):** Packs expert weights in 2x4-bit `uint8` (`3.88x` compression) and dispatches Top-2 projections in `float8_e4m3fn` with **`0.9963` cosine similarity** and **`31.79 dB` SQNR**.

---

## 2. Dynamic Truncated-SVD Rank-Adaptive Projection (`HM-S4-05` / `PR #33` & `#34`)

Community miners often train adapters with different LoRA ranks (`r=8` on 6 GB laptops, `r=16` on 8 GB RTX 2070, `r=32` or `r=64` on 12 GB–24 GB GPUs). [`project_lora_rank_svd`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/btm_upcycle.py) normalizes any adapter to a canonical rank $r_{\text{target}} = 16$:

1. **Down-Projection ($r_{\text{src}} > r_{\text{target}}$):** Computes the QR-factored singular value decomposition of $\frac{\alpha_{\text{src}}}{r_{\text{src}}} B A = U \Sigma V^\top$ without materializing the full $d_{\text{out}} \times d_{\text{in}}$ matrix, retaining the top $r_{\text{target}}$ singular triples:
   $$B' = U_{[:, :r_{\text{target}}]} \sqrt{\Sigma_{[:r_{\text{target}}]}}, \quad A' = \sqrt{\Sigma_{[:r_{\text{target}}]}} V^\top_{[:r_{\text{target}}, :]}$$
   By the Eckart–Young–Mirsky theorem, this minimizes Frobenius norm reconstruction error and retains **`>= 99.4%` spectral energy**.
2. **Up-Padding ($r_{\text{src}} < r_{\text{target}}$):** Zero-pads $(A, B)$ to $r_{\text{target}}$ while rescaling by $\sqrt{(\alpha_{\text{src}}/r_{\text{src}}) / (\alpha_{\text{target}}/r_{\text{target}})}$ so forward outputs are **bit-for-bit identical (`max_diff = 0.0`)**.
3. **32-Expert Heterogeneous Hot-Plug (`hotplug_rank_adaptive_experts`):** Hot-plugs 32 mixed-rank (`r=8/16/32/64`) experts into `UpcycledSparseMoEModel` in **`0.107 GB` VRAM** (`< 2.0 GB` budget).
