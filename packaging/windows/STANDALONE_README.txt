HiveMind HUD - standalone Windows build (CPU only)
==================================================

Run HiveMindHUD.exe. It starts a local web server on http://127.0.0.1:8090/
(or a free port if 8090 is taken) and opens your browser. Close the window or
press Ctrl+C to stop it, or run:  HiveMindHUD.exe --stop

Only one HUD may use a data folder at a time. A second launch using that
folder exits with code 8 without changing the running instance. Use a different
--data-dir to run another instance; only then can a busy default port fall back.
Use the same --data-dir with --stop to stop the intended instance. Stale status
files after a crash do not block restart, and stop requests never signal a PID
or apply to a replacement instance. Keep the data folder on a local drive and
do not manually delete its run\hud-server.lock file while a HUD is running.

What this build is
- Self-contained: no Python, git or source checkout is needed.
- Loopback only by default. --host <addr> --allow-network exposes the
  unauthenticated HUD to your network; do not do this on untrusted networks.
- Offline by default: dataset REST, Hugging Face model/tokenizer and dataset
  downloads are disabled; local caches and curated fallback text still work.
  --online allows downloads unless an HF offline environment flag is already
  set. Offline operation does not resolve or transmit a saved Hugging Face token.
- Writes only to %LOCALAPPDATA%\HiveMind (override with --data-dir or the
  HIVEMIND_DATA_DIR environment variable). Logs: %LOCALAPPDATA%\HiveMind\logs.
- Installs nothing: no services, no auto-start, no registry entries.

What this build is NOT
- No GPU acceleration. PyTorch is the CPU build; CUDA/ROCm, fused NF4 kernels,
  VRAM preemption timing and DMA paths are not available or exercised.
- No Steamworks. steam_api64.dll is not included; Steam achievements,
  leaderboards, cloud saves and receipts shown are local simulation.
- Geo-ring and swarm telemetry are demo fixtures, not live peers.
- No Lean compiler; formal proof checking is unavailable.

Useful flags
  --check       verify the bundle and print detected capabilities as JSON
  --port N      fixed port (fails clearly if busy); --port 0 picks any free port
  --no-browser  do not open a browser
  --data-dir D  use another per-user data folder

Integrity: SHA256SUMS.txt lists every file; BUILD_MANIFEST.json records the
source commit, dependency versions and build command. Third-party license
texts are in licenses\.
