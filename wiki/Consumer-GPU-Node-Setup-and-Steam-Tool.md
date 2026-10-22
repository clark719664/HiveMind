# 🎮 Consumer GPU Node Setup & Free Steam RTX Tool (`v1.8.0`)

This guide walks through setting up a consumer gaming PC (**6 GB – 24 GB VRAM**, or CPU fallback) as a HiveMind training miner, storage seeder, and local OpenAI-compatible inference gateway, and running the opt-in **Community GPU Benchmark Suite**.

> **⚖️ Non-Monetary Charter (`0.00 EUR / 0.00 USD`):** Running a node or contributing code involves **zero financial payment**. Accumulated **"Points" (`PTS` / `$HIVE`)** confer **solely computing processing benefits** (priority Sparse MoE inference tokens, speculative decoding bandwidth, and distributed QLoRA fine-tuning slots) **when the live production HiveMind Swarm is deployed, available, and used by a large community**.

---

## 1. Installation & Environment Setup

```bash
# Clone Project HiveMind and create an isolated virtual environment
git clone https://github.com/xaviercallens/HiveMind.git
cd HiveMind
uv venv
uv pip install -e ".[dev]"
```

---

## 2. Opt-In Community GPU Benchmark (`< 1.5 GB` Peak VRAM)

Test your GPU (`NVIDIA RTX 20/30/40/50`, `AMD Radeon RX 6000/7000 ROCm`, `Apple Silicon M1–M4 MPS`, or CPU fallback) on real-token QLoRA convergence, `EF21-DiLoCo` residual contraction, 128-expert 3-tier MoE routing, and `ZK-OPoI v4` batch Merkle commitments:

```bash
# Run the 5-stage benchmark and generate artifacts/community_gpu_benchmark_receipt.json
uv run python scripts/run_community_gpu_benchmark.py --contributor-handle <your-github-handle>

# Run the automated pytest suite
uv run pytest tests/test_community_gpu_benchmark.py -v
```

Share your signed `receipt_sha256` in **[Discussion #84 (Community GPU Benchmark & Hardware Receipts Board)](https://github.com/xaviercallens/HiveMind/discussions/84)**!

---

## 3. CLI Commands & Local Servers

| Command | Description | Default Endpoint / Output |
| :--- | :--- | :--- |
| `python scripts/run_community_gpu_benchmark.py --contributor-handle <user>` | Runs the 5-stage opt-in Community GPU Benchmark (`< 1.5 GB` VRAM) and emits a SHA-256 signed receipt | `artifacts/community_gpu_benchmark_receipt.json` |
| `python -m hivemind.cli.main hardware-check` | Detects GPU vendor (`cuda`, `rocm`, `mps`, `cpu`), free VRAM budget, and Win32 Idle Daemon state | CLI Output |
| `python -m hivemind.cli.main deploy-steam-tool` | Runs the 10-Stage Intense RTX GPU Stress Suite and packages `steam_dist/HiveMind_Steam_FreeEdition/` | `artifacts/` |
| `python -m hivemind.cli.main play-hack-game --season s1-s6` | Executes the Hugging Face Hack-Miner Campaign across Missions 1–24 (`SelfPlayCodeRepairArena` & `Lean4AutoformalizerArena`) | `artifacts/hf_hack_miner_game_report.json` |
| `python -m hivemind.cli.main serve-hack-hud --port 8090` | Launches the Cyberpunk RTX Control Center, Hollow `uRing` WebGL 3D Geo-Ring Globe (`PR #70`), Season 6 Agent Arena & Strategy HUD (`PR #78`) | `http://127.0.0.1:8090` |
| `python -m hivemind.cli.main serve-coordinator --port 8000` | Starts the BOINC Coordinator (`FL-08` hardened, `PR #72`) & OpenAI-compatible `/v1/chat/completions` (`stream=True` SSE) gateway | `http://127.0.0.1:8000/v1` |

---

## 4. Connecting Cursor, Continue.dev & OpenAI Clients

Point any OpenAI-compatible IDE plugin or SDK to `http://127.0.0.1:8000/v1`:

```json
{
  "models": [
    {
      "title": "HiveMind Sparse MoE Swarm (Local P2P Gateway)",
      "provider": "openai",
      "model": "hivemind-qwen2.5-sparse-moe-top2",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "hivemind-local-pouw-credit"
    }
  ]
}
```
