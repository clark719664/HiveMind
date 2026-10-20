# 🎮 Consumer GPU Node Setup & Free Steam RTX Tool

This guide walks through setting up a consumer gaming PC (**8 GB – 12 GB VRAM**) as a HiveMind training miner, storage seeder, and local OpenAI-compatible inference gateway.

---

## 1. Installation & Environment Setup

```powershell
# Clone Project HiveMind and create an isolated virtual environment
git clone https://github.com/xaviercallens/HiveMind.git
cd HiveMind
uv venv
uv pip install -e ".[dev]"
```

---

## 2. CLI Commands & Local Servers

| Command | Description | Default Endpoint |
| :--- | :--- | :--- |
| `python -m hivemind.cli.main hardware-check` | Detects GPU vendor (`cuda`, `rocm`, `mps`, `cpu`), free VRAM budget, and Win32 Idle Daemon state | CLI Output |
| `python -m hivemind.cli.main deploy-steam-tool` | Runs the 10-Stage Intense RTX GPU Stress Suite and packages `steam_dist/HiveMind_Steam_FreeEdition/` | `artifacts/` |
| `python -m hivemind.cli.main play-hack-game --season s1-s4` | Executes the Hugging Face Hack-Miner Campaign across Missions 1–16 and the Sparse MoE Omega Boss Raid | `artifacts/hf_hack_miner_game_report.json` |
| `python -m hivemind.cli.main serve-hack-hud --port 8090` | Launches the Cyberpunk RTX Control Center, Raw WebGL 3D Geo-Ring Globe (`georing_webgl.py`), and `/ws/swarm-telemetry` | `http://127.0.0.1:8090` |
| `python -m hivemind.cli.main serve-coordinator --port 8000` | Starts the BOINC Coordinator & OpenAI-compatible `/v1/chat/completions` (`stream=True` SSE) gateway | `http://127.0.0.1:8000/v1` |

---

## 3. Connecting Cursor, Continue.dev & OpenAI Clients

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

> **Note on Steam AppID (`docs/STEAM_STATUS.md`):** The local Hack-Miner & Steamworks bridge (`src/hivemind/client/steamworks.py`) probes `steam_api64.dll` dynamically with HMAC offline fallback for local testing. See [`docs/STEAM_STATUS.md`](https://github.com/xaviercallens/HiveMind/blob/main/docs/STEAM_STATUS.md) (`PR #11` by `@radialmonster`).
