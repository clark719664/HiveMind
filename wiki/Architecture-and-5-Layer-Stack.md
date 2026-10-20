# 🏛️ Architecture & 5-Layer Stack

Project HiveMind overcomes both the **Consumer GPU VRAM Wall** (`8 GB – 12 GB`) and the **Residential WAN Bandwidth Wall** (`10–50 Mbps` upload, `20–150 ms` RTT) through 5 modular layers:

---

## 1. Layer-by-Layer Subsystem Overview

| Layer | Core Modules | Key Responsibilities | Verified Telemetry |
| :--- | :--- | :--- | :--- |
| **Layer 1: Zero-Intrusion Edge Node** | [`src/hivemind/client/daemon.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/client/daemon.py), [`src/hivemind/core/quantization.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/quantization.py) | Streams 4-bit NF4 base weights with `0 bytes` CPU heap duplication; monitors Win32 idle time (`>=300s`) and evicts VRAM in `<150 ms` when a game launches. | `175.96 MB` NF4 base; `135.5 ms` max game eviction |
| **Layer 2: Byzantine-Resilient DiLoCo Outer Loop** | [`src/hivemind/core/compression.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/compression.py), [`src/hivemind/core/multikrum_outer.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/multikrum_outer.py), [`src/hivemind/boinc/validator.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/boinc/validator.py) | Compresses QLoRA pseudo-gradients via `Top-5% + FP8 (float8_e4m3fn)`, audits 10% deterministic micro-segments, and filters poisoners via $2\sigma$ Multi-Krum. | `12.79x`–`64.96x` wire compression; `100%` poisoner rejection |
| **Layer 3: Factored BTM Sparse MoE & Async DMA** | [`src/hivemind/core/btm_moe.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/btm_moe.py), [`src/hivemind/core/btm_upcycle.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/core/btm_upcycle.py), [`src/hivemind/swarm/swarm_fabric.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/swarm_fabric.py) | Upcycles heterogeneous community LoRA adapters (`r=8..64 -> r=16` via truncated SVD) into a Top-2 Sparse MoE with 2-slot double-buffered DMA prefetching. | `32`–`96` experts in `1.53 GB` peak VRAM; `0.0 ms` exposed swap stall |
| **Layer 4: Real QUIC 0-RTT, STUN/TURN & Super-Ring** | [`src/hivemind/swarm/quic_link.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/quic_link.py), [`src/hivemind/swarm/nat_traversal.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/nat_traversal.py), [`src/hivemind/swarm/quic_transport.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/quic_transport.py) | Real `aioquic` TLS 1.3 0-RTT links with strike-register replay defense, RFC 5389 STUN + RFC 5766 authenticated TURN relay, and 2-Tier Continental Super-Ring routing. | `179` transport/NAT security tests passing |
| **Layer 5: OpenAI SSE Gateway, WebGL HUD & HF Hub** | [`src/hivemind/boinc/coordinator_api.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/boinc/coordinator_api.py), [`src/hivemind/game/georing_webgl.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/game/georing_webgl.py), [`src/hivemind/swarm/hf_community_hub.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/hf_community_hub.py) | Serves `/v1/chat/completions` (`stream=True` SSE) with Prefix-Trie speculative decoding ($\gamma=6..8$), raw WebGL 3D Geo-Ring globe HUD (`:8090`), and HF Community Bundles. | `5.60x` RTT speedup; automatic `$HIVE` reuse royalties |

---

## 2. Multi-Vendor GPU Backend Support (`HM-S4-01`)

Configured via [`src/hivemind/config.py`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/config.py) (`detect_hardware_tier_and_vram`) and [`PinnedRAMExpertDMAStreamer`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/swarm_fabric.py):
- **NVIDIA CUDA (`cuda_stream`):** Dedicated secondary `torch.cuda.Stream` + `pin_memory()` double-buffered slots (`slot_A` / `slot_B`).
- **AMD ROCm (`hip_stream`):** Native HIP stream dispatch on RDNA3 (`gfx1100` native `bf16`) and RDNA2 (`gfx1030` `fp16` fallback).
- **Apple Silicon (`mps_unified`):** Unified memory zero-copy tensor staging on M1/M2/M3/M4 GPUs.
- **CPU Fallback (`cpu_copy`):** Deterministic execution for headless CI and verifier nodes.
