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

## 3. Related Wiki Pages & Discussions

- **[Consumer GPU Node Setup & Steam Tool](https://github.com/xaviercallens/HiveMind/wiki/Consumer-GPU-Node-Setup-and-Steam-Tool)**
- **[Architecture & 5-Layer Stack](https://github.com/xaviercallens/HiveMind/wiki/Architecture-and-5-Layer-Stack)**
- **[Discussion #84 — Community GPU Benchmark Receipts Board](https://github.com/xaviercallens/HiveMind/discussions/84)**
- **[Discussion #85 — Human & AI Agent Coordination Hub](https://github.com/xaviercallens/HiveMind/discussions/85)**
