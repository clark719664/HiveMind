# Steam Integration & Local Prototype Status

Checked against Steam's public store and the repository's current implementation (contributed in PR #11 by `@radialmonster`).

## App ID Clarification

Steam App ID `3133700`, used as a local configuration/test constant in this repository, resolves on the public Steam Store to [Fantasy Grounds - Raiders of the Ghost Zone](https://store.steampowered.com/app/3133700/), a paid DLC published by SmiteWorks (also reported by [Steam's `appdetails` API](https://store.steampowered.com/api/appdetails?appids=3133700)). It is not a public HiveMind Steam Store installation link. A publisher-assigned Valve App ID and public Steam Store listing have not yet been bound to this repository.

## What Runs Locally

- **Local RTX Tool & Hack-Miner Package (`steam_dist/HiveMind_Steam_FreeEdition`):** The `.bat` launchers (`Launch_HiveMind_SteamTool_RTX.bat`, `Launch_HiveMind_HackMiner.bat`, `Launch_HiveMind_WebHUD.bat`) run Python modules from the repository root (`D:\xdev\DecentralizedAI`). The folder requires the configured Python environment (`.venv`) and repository files; it is a local prototype package rather than a standalone Steam store installer.
- **Steamworks Bridge (`src/hivemind/client/steamworks.py`):** `SteamworksSDKBridge` dynamically probes for `steam_api64.dll` / `libsteam_api.so` (`HM-M5-01`) when present and falls back to deterministic HMAC receipts (`steam_lb_hmac_...`, `steam_rp_hmac_...`) and local Python state so all features run offline, headlessly, and in CI without requiring a live Valve partner session.
- **Local Manifests & Cloud Profiles (`src/hivemind/game/steam_edition.py`):** The package builder and RTX deployer (`src/hivemind/game/rtx_steam_tool_deployer.py`) write local VDF/JSON manifests (`app_build_3133700.vdf`, `depot_build_3133701.vdf`, `tool_manifest_3133700.vdf`, `steam_remote_storage_profiles.json`, `rtx_gpu_deployment_cert.json`) as local SteamPipe templates and hardware certification artifacts.
- **Swarm & PoUW Ledger (`src/hivemind/cli/demo_hf_domain_swarm.py`):** The domain swarm and `$HIVE` ledger run locally on your consumer GPU (`cuda:0`) and local disk.

See the [Steam bridge](../src/hivemind/client/steamworks.py), [package builder](../src/hivemind/game/steam_edition.py), [RTX stress & tool deployer](../src/hivemind/game/rtx_steam_tool_deployer.py), and [swarm demo](../src/hivemind/cli/demo_hf_domain_swarm.py).

## Trying the Local RTX Tool & Hack-Miner Prototype

Follow the [development setup](../CONTRIBUTING.md#3-local-development-setup-8-gb-gpu-or-cpu), then run from the repository root:

```powershell
# 1. Play the 6-Mission Hugging Face Hack-Miner + 6-Expert Sparse MoE Boss Raid
hivemind play-hack-game --season all --inner-steps 8

# 2. Run the 6-Stage Intense RTX GPU Stress Suite & package the local Steam Tool prototype
hivemind deploy-steam-tool

# 3. Launch the Interactive Cyberpunk Web HUD (http://127.0.0.1:8090)
hivemind serve-hack-hud --port 8090
```

Opening `web_ui/index.html` directly without the FastAPI server inspects the offline HUD snapshot; use `hivemind serve-hack-hud` or `Launch_HiveMind_SteamTool_RTX.bat` to run live GPU QLoRA micro-forging and RTX diagnostics.

The legacy `3133700` App ID and `3133701` Depot ID constants remain in runtime code and test fixtures as local development defaults so all unit and integration tests pass deterministically. Do not use `3133700` to publish to SteamPipe; replace it with your own partner-assigned Valve App ID before uploading to Steamworks.
