# ⚡ Opt-In Community GPU Benchmark & WAN GameDay (`v1.8.0`)

Have a consumer gaming PC, laptop, or workstation GPU (`RTX 3060`, `RTX 3070`, `RTX 4060`, `RTX 4070`, `RTX 4090`, `ARC`, `Radeon`, or Apple Silicon / CPU fallback)? You can help validate Project HiveMind's **8 GB VRAM invariant**, **128-Expert 3-Tier Hierarchical MoE**, **FP8 KV-Cache Pager**, and **`EF21-DiLoCo` outer sync** on real community hardware!

---

## ⚖️ Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / USD`)

- **100% Opt-In & Local**: Runs in **under 30 seconds** entirely on your local machine; makes **zero external network calls** and uploads **no personal data**.
- **No Financial Payment (`0.00 EUR / USD`)**: Submitting a benchmark receipt (`+500 PTS`) or joining a WAN GameDay (`+1,000 PTS`) provides **no monetary payout**. Points (`PTS` / `$HIVE`) grant **reciprocal computing processing benefits** (priority distributed Sparse MoE inference & QLoRA fine-tuning slots) when the live production HiveMind Swarm is deployed and used by a large community.

---

## 1. Run the Opt-In Community GPU Benchmark (`< 30 Seconds`)

Implemented in `src/hivemind/cli/community_gpu_benchmark.py` and `scripts/run_community_gpu_benchmark.py`, the benchmark executes **5 real PyTorch/HiveMind workloads**:

1. **Workload 1 — 4-Bit NF4 + Double Quantization & QLoRA Forward/Backward (`src/hivemind/core/qlora_4bit.py`)**: Measures quantization compression ratio, reconstruction error, and step throughput (`tokens/sec`).
2. **Workload 2 — 128-Expert 3-Tier Hierarchical MoE + Speculative Token-Tree Prefetch (`src/hivemind/core/btm_moe.py`)**: Measures `4 x 8 x 4` routing latency (`ms`), active parameter sparsity (`16x`), and speculative cache hit rate.
3. **Workload 3 — FP8 (`E4M3FN`) Paged KV-Cache (`src/hivemind/core/swarm_fabric.py`)**: Verifies that peak MoE + KV-cache memory stays strictly **`< 2,252.8 MB` (`2.2 GB`)** at 4,096 context length.
4. **Workload 4 — Stateful `EF21-DiLoCo` Top-2% Sparse Outer Sync (`src/hivemind/p2p/gossip_averaging.py`)**: Measures error-feedback compression throughput and residual contractive decay.
5. **Workload 5 — Interactive `< 1.5s` Game Preemption (`src/hivemind/watchdog/game_detector.py`)**: Measures checkpoint pause-and-yield latency (`ms`) when a foreground game process is detected.

### Execute the Benchmark

```bash
# Quick tier (~5-10 seconds, works on CUDA GPU or CPU fallback)
PYTHONPATH=. python scripts/run_community_gpu_benchmark.py --tier quick \
  --contributor-handle "@your_github_handle"

# Standard tier (~20-30 seconds, deeper batch & context sweep)
PYTHONPATH=. python scripts/run_community_gpu_benchmark.py --tier standard \
  --contributor-handle "@your_github_handle" \
  --output artifacts/community_gpu_benchmark_receipt.json
```

### Share Your Receipt in Discussion #84 (`+500 PTS`)

Copy the Markdown table and JSON receipt printed by `scripts/run_community_gpu_benchmark.py` (including `receipt_sha256`) and post it in **[GitHub Discussion #84 — Opt-In Community GPU Benchmark Receipts Board](https://github.com/xaviercallens/HiveMind/discussions/84)**! The 3-Hour Autonomous Orchestrator will verify your `receipt_sha256` and credit **`+500 PTS`** (`+1,000 PTS` for multi-host WAN GameDay receipts).

---

## 2. Run the Multi-Host WAN QUIC + TURN Benchmark (`scripts/run_wan_quic_turn_benchmark.py`)

To benchmark `aioquic` 0-RTT session resumption, BBRv1 pacing, and RFC 5766 TURN `ChannelBind` (`0x4000` fast-path framing, `PR #63`, `PR #77`) under simulated `1%–5%` packet loss and CGNAT:

```bash
PYTHONPATH=. python scripts/run_wan_quic_turn_benchmark.py
```

 This generates `artifacts/wan_quic_turn_benchmark_report.json` with measured 0-RTT handshake latency, TURN `ChannelBind` header overhead reduction (`77.8%` savings: `36 B -> 8 B`), and `AdaptiveDiLoCoCadenceController` (`SI-03`) cadence adaptation (`H=11` on regional QUIC vs `H=226` on TURN relay).

---

## 3. 🚀 Operation Polyglot-Singularity-128: Large-Scale 8-Domain Hugging Face & 128-Expert Multi-MoE GPU Campaign (`HM-S7-09` / `HM-S7-10`)

To validate the Branch-Train-Merge (`BTM`) + 128-Expert 3-Tier Sparse MoE + `EF21-DiLoCo` + Reputation-Weighted `TIES` (`SI-04`) thesis at scale on **real UTF-8 token cross-entropy (`nats/token`, `perplexity`)**, run `scripts/run_multi_moe_hf_campaign.py` (`src/hivemind/cli/multi_moe_hf_campaign.py`):

### The 8 Curated Hugging Face Datasets & Subjects (`4 Super-Clusters`)

| Super-Cluster | Domain Specialist | Hugging Face Dataset | Scientific / Engineering Subject |
| :--- | :--- | :--- | :--- |
| **`CLUSTER-00-SYSTEMS-FORMAL`** | `Rust_Systems_Kernel` | [`bigcode/the-stack-v2-dedup`](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) (`Rust`) | Zero-copy DMA rings, lock-free atomics & borrow-checker invariants |
| **`CLUSTER-00-SYSTEMS-FORMAL`** | `Lean4_Formal_Mathematics` | [`hoskinson-center/proofnet`](https://huggingface.co/datasets/hoskinson-center/proofnet) | Machine-checked Lean 4 theorems for Byzantine quorums & contraction bounds |
| **`CLUSTER-01-CRYPTO-SECURITY`** | `ZK_Cryptography_And_Audits` | [`ASSERT-KTH/DISL-Solidity-Smart-Contracts`](https://huggingface.co/datasets/ASSERT-KTH/DISL-Solidity-Smart-Contracts) | `ZK-OPoI v4` 4-way Merkle spot checks & L2 state-channel escrow audits |
| **`CLUSTER-01-CRYPTO-SECURITY`** | `Cyber_Vulnerability_Patching` | [`SWE-bench/SWE-bench_Verified`](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | Automated CWE vulnerability remediation & spectral backdoor (`FL-08`) defense |
| **`CLUSTER-02-GPU-ML-KERNELS`** | `Triton_CUDA_Kernel_Synth` | [`ScalingIntelligence/KernelBench`](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) | Fused Triton/CUDA `NF4` dequantization & `FP8` paged KV-cache kernels |
| **`CLUSTER-02-GPU-ML-KERNELS`** | `Distributed_DiLoCo_Protocols` | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Stateful `EF21-DiLoCo` outer sync, Kademlia DHT & QUIC 0-RTT transport |
| **`CLUSTER-03-FRONTIER-SCIENCE`** | `Biomedical_Clinical_Genomics` | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | CRISPR-Cas9 off-target genomics, clinical trial QA & oncology pathways |
| **`CLUSTER-03-FRONTIER-SCIENCE`** | `Astrophysics_JWST_Spectroscopy` | [`MultimodalUniverse/multimodal_universe`](https://huggingface.co/datasets/MultimodalUniverse/multimodal_universe) | JWST transmission spectroscopy biosignatures ($\text{H}_2\text{O} + \text{CH}_4/\text{CO}_2$) & Kepler transits |

### Run the 8-Domain Multi-MoE Campaign on Your GPU

```bash
# Standard 8-domain campaign (12 steps/domain, real UTF-8 token cross-entropy + 128-Expert MoE upcycling)
PYTHONPATH=. python scripts/run_multi_moe_hf_campaign.py \
  --contributor-handle "@your_github_handle" \
  --tier standard

# Optional: stream extra live rows directly from Hugging Face Datasets Server API
PYTHONPATH=. python scripts/run_multi_moe_hf_campaign.py \
  --contributor-handle "@your_github_handle" \
  --tier deep \
  --live-hf-stream
```

Post your generated Markdown table and `campaign_sha256` receipt in **[GitHub Discussions](https://github.com/xaviercallens/HiveMind/discussions)** (`+1,500 PTS` non-monetary compute-reciprocity credits; `0.00 EUR / USD`).

---

## 4. Related Wiki Pages & Discussions

- **[Consumer GPU Node Setup & Steam Tool](https://github.com/xaviercallens/HiveMind/wiki/Consumer-GPU-Node-Setup-and-Steam-Tool)**
- **[Architecture & 5-Layer Stack](https://github.com/xaviercallens/HiveMind/wiki/Architecture-and-5-Layer-Stack)**
- **[Discussion #84 — Community GPU Benchmark Receipts Board](https://github.com/xaviercallens/HiveMind/discussions/84)**
- **[Discussion #85 — Human & AI Agent Coordination Hub](https://github.com/xaviercallens/HiveMind/discussions/85)**
