# Project HiveMind: Decentralized MoE Training & Inference via Steam, BOINC, and DiLoCo

**System Codename:** `Project HiveMind`  
**Target Architecture:** Consumer-Grade Distributed GPU Cluster (Steam Hardware Ecosystem)  
**Local Hardware Profile Configured:** NVIDIA GeForce RTX 2070 (8 GB VRAM, Turing SM 7.5, Native FP16 Tensor Cores, Driver 591.86)  
**Core Stack:** Base Models (`Qwen2.5`) + Training (`DiLoCo` + `QLoRA NF4` + `Branch-Train-Mix`) + Orchestration (`Modified BOINC` + `Multi-Krum`) + Distribution (`Steamworks SDK` + `Petals DHT Swarm`)

---

## 1. System Architecture & Local RTX 2070 Adaptation

Project HiveMind aggregates idle consumer GPUs from the gaming community to collaboratively train, upcycle, and serve frontier-class **Mixture-of-Experts (MoE)** Large Language Models without datacenter interconnects (InfiniBand/NVLink).

| Layer | Technology / Protocol | Implementation Module |
| :--- | :--- | :--- |
| **Distribution & UI** | Steam Client & Steamworks SDK + Win32 Idle Hook | [`src/hivemind/client/daemon.py`](src/hivemind/client/daemon.py), [`src/hivemind/client/steamworks.py`](src/hivemind/client/steamworks.py) |
| **Orchestration** | Modified BOINC Server + Deterministic Spot-Check | [`src/hivemind/boinc/coordinator_api.py`](src/hivemind/boinc/coordinator_api.py), [`src/hivemind/boinc/validator.py`](src/hivemind/boinc/validator.py) |
| **Optimization** | DiLoCo + 4-bit NF4 QLoRA + Top-$k$/FP8 Compression | [`src/hivemind/core/diloco_inner.py`](src/hivemind/core/diloco_inner.py), [`src/hivemind/core/diloco_outer.py`](src/hivemind/core/diloco_outer.py), [`src/hivemind/core/compression.py`](src/hivemind/core/compression.py) |
| **Model Topology** | Branch-Train-Mix (BTM) 8-Expert Sparse MoE Upcycling | [`src/hivemind/core/btm_upcycle.py`](src/hivemind/core/btm_upcycle.py) |
| **Inference Engine** | Petals-style Regional DHT Swarm + Speculative Decoding | [`src/hivemind/swarm/inference_ring.py`](src/hivemind/swarm/inference_ring.py) |

### Dual Hardware Profiles (`configs/rtx2070_dual_profile.yaml`)

Because the local machine has an **NVIDIA GeForce RTX 2070 (8,192 MiB VRAM total, ~6.7 GB free after Windows WDDM overhead, Compute Capability 7.5)**, the project includes two pre-configured execution profiles plus the 12 GB reference spec:

1. **`rtx2070_moe_e2e` (Default Local End-to-End 8-Branch MoE Profile)**
   - **Backbone:** `Qwen/Qwen2.5-0.5B` or `Qwen/Qwen2.5-1.5B` (or instant local synthetic Qwen-arch checkpoint for zero-network verification)
   - **Purpose:** Executes all 8 domain branches (`Code`, `Mathematics`, `Creative Writing`, `STEM`, `Legal/Medical`, `Multilingual-FR`, `Reasoning`, `General`), runs DiLoCo-QLoRA inner loops with deterministic micro-checkpointing, validates 50-step segments, filters poisoned workers via **Multi-Krum**, upcycles the 8 adapters into a unified **8x Sparse MoE with Top-2 routing**, and benchmarks speculative swarm decoding—all inside ~2.4–4.2 GB VRAM on the RTX 2070.
2. **`rtx2070_7b_worker` (8 GB Worker Limit Benchmark Profile)**
   - **Backbone:** `Qwen/Qwen2.5-7B` (4-bit NormalFloat NF4 + Double Quantization + FP16 LoRA $r=64$ + 8-bit Paged AdamW + Gradient Checkpointing)
   - **Purpose:** Benchmarks single-branch worker execution at the exact 8 GB consumer GPU ceiling (~6.3 GB peak VRAM).
3. **`rtx4070_14b_spec` (12 GB Reference Profile)**
   - **Backbone:** `Qwen/Qwen2.5-14B` (11.6 GB peak VRAM allocation budget from the Technical Specification).

---

## 2. Quickstart Commands (Local RTX 2070)

All dependencies (including CUDA-enabled PyTorch, `bitsandbytes`, `peft`, `transformers`, and `fastapi`) are managed via `uv`:

```powershell
# 1. Inspect local RTX hardware profile & VRAM allocation budgets
uv run hivemind hardware-check

# 2. Run the full End-to-End Project HiveMind Pipeline on the RTX 2070
#    (Phase A Branching -> Phase B DiLoCo-QLoRA -> Spot-Check Validation ->
#     Multi-Krum Outer Step -> Phase C 8-Expert Sparse MoE Upcycling -> Phase 2 Swarm Inference)
uv run hivemind simulate-e2e --profile rtx2070_moe_e2e

# 3. Run the 8 GB VRAM Limit Benchmark (4-bit NF4 QLoRA r=64 + Paged AdamW)
uv run hivemind benchmark-vram --profile rtx2070_7b_worker

# 4. Launch the Modified BOINC Coordinator API Server
uv run hivemind serve-coordinator --host 127.0.0.1 --port 8080

# 5. Launch a Steam/BOINC Worker Node with Win32 Zero-Intrusion Daemon
uv run hivemind start-worker --coordinator-url http://127.0.0.1:8080

# 6. Run the automated test suite
uv run pytest -v
```
