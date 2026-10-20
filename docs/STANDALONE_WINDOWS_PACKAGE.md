# Standalone Windows package (HiveMind HUD, CPU-only)

## Current source versus built artifacts

The latest checked-in Windows build evidence is for `d36e654acefc` (2026-10-19),
which predates the integrated offline, ownership, GPU and WAN fixes. Its 26/27
build-machine smoke result and browser failures remain historical evidence.
No rebuilt EXE or clean-machine pass is claimed for the current source.

The source now blocks offline network fallbacks and gives one process exclusive
ownership of each data directory. The smoke harness tests port conflicts using
**different** data directories (exit 3), rejects a shared-directory contender
(exit 8), checks that the original owner/status survive, and verifies `--stop`
leaves an independent owner alive. Portable harness regressions also run in the
native Windows/Ubuntu lifecycle CI jobs. These tests do not substitute for a new
frozen-EXE run; follow [the final PC checklist](FINAL_PC_ACCEPTANCE.md).

Clean-machine acceptance, owner-selected project licensing, signing/SmartScreen,
and the intermittent full-sequence browser restore investigation remain open.

## What it is

A PyInstaller one-folder, console build of the HUD (`HiveMindHUD\HiveMindHUD.exe` plus `_internal\`),
shipped as `HiveMindHUD-<sha12>-win64-cpu.zip`. The entry point is `src/hivemind/cli/standalone.py`
(via `packaging/windows/hivemind_hud_entry.py`). The user does not need Python, git, or a source checkout.

## Building

Requires Windows x64 and Python 3.11 (the validated baseline). Tracked files must be committed;
`--allow-dirty` builds anyway and records `source_dirty: true` in the manifest.

```
py -3.11 packaging\windows\build_standalone.py                # fresh venv at build\standalone\venv
py -3.11 packaging\windows\build_standalone.py --reuse-venv   # reuse it (the project is always force-reinstalled)
```

Steps: create the venv, then `pip install -c packaging/windows/constraints-windows-cpu.txt` with the
PyTorch CPU index, then `pip install --no-deps --force-reinstall <repo>` so a reused venv always
bundles the current source, then `pip check`, then PyInstaller with `HiveMindHUD.spec`. After that it
collects third-party licenses, writes `SHA256SUMS.txt` and `BUILD_MANIFEST.json`, and creates the zip.
Outputs (all under the gitignored `build/standalone/`):

- `dist\HiveMindHUD\` is the unpacked app.
- `HiveMindHUD-<sha12>-win64-cpu.zip` is the release archive.
- `HiveMindHUD-<sha12>-win64-cpu.manifest.json` is the manifest plus the zip hash and size.

Neither the zip nor `dist\` is committed. A redacted copy of the manifest is kept under `docs/evidence/standalone/<sha12>/`.

## Supported vs unsupported

| Area | In this build |
|---|---|
| OS | Windows 10/11 x64. Only tested on the build machine (Windows 11, 16 GB RAM, AMD RX 7600). |
| Compute | CPU PyTorch `2.6.0+cpu`. `--check` reports `compute_backend: cpu`. |
| CUDA / ROCm acceleration | **Unsupported**. No CUDA build of torch. `bitsandbytes` and `triton` are excluded. |
| NF4 fused kernels, VRAM preemption timing, DMA paths | **Unsupported / not exercised**. |
| Steamworks | **Unsupported**. `steam_api64.dll` is not bundled, and Steam features are a local HMAC simulation only. |
| Lean proof checking | **Unsupported**. No Lean compiler is bundled. |
| Geo-Ring topology and WebSocket telemetry | Demo fixtures, labelled `source: demo_fixture`, with `live_peer_discovery: false`. |
| Forge `real_training: true` | Runs for real on CPU: LoRA on a randomly initialised tiny GPT-2 over bundled fixture rows. This is not evidence of model quality, and rewards are simulated. |
| WAN / multi-peer swarm | Not tested and not claimed. |

## Runtime behaviour

- **Per-user paths.** Data lives in `%LOCALAPPDATA%\HiveMind`. You can override this with `--data-dir` or `HIVEMIND_DATA_DIR`.
  The launcher sets `HIVEMIND_CACHE_DIR=<data>\cache`, `HIVEMIND_ARTIFACTS_DIR=<data>\artifacts` and
  `HF_HOME=<data>\cache\huggingface` before `hivemind.config` is imported. The log is at `<data>\logs\hud-launcher.log`.
  A frozen build refuses (exit 5) any data directory inside the install directory. The smoke test confirms
  the install directory is byte-for-byte unchanged after a full run.
- **Loopback by default.** It binds `127.0.0.1` through an exclusive (`SO_EXCLUSIVEADDRUSE`) socket handed to uvicorn.
  A non-loopback `--host` is refused (exit 2) unless you pass `--allow-network`, because the HUD has no authentication.
  An explicit `--port` that is busy exits 3. The default port 8090 falls back to a free loopback port.
- **Offline by default.** It sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, and `--online` re-enables
  Hub downloads unless an inherited HF offline flag remains set. Direct dataset REST fallbacks
  also honor offline mode, and offline operation does not resolve or transmit a saved HF token.
- **Stopping.** `HiveMindHUD.exe --stop` writes a stop request to the data directory and waits for the server to exit
  (exit 0, or 7 on timeout). The status file is `<data>\run\hud-server.json`. A kernel-held
  lock permits only one owner per directory; a contender exits 8 before imports/binding. Stop
  requests target the recorded generation and do not signal a PID or stop a replacement owner.
- **`--check`** prints a JSON capability report and exits. Exit codes: 2 usage, 3 port, 4 missing deps,
  5 unsafe paths, 6 startup failure, 7 stop timeout, 8 data directory already owned.

## Smoke test

`packaging/windows/smoke_test_standalone.py` uses only the standard library. It extracts the zip under a path with spaces and
non-ASCII characters (`Hive Mind Smoke Ünïcødé 测试`) and uses a fake `LOCALAPPDATA` (`User Ü\...`). It strips `PATH` to
the Windows system directories and points `HTTP(S)_PROXY` at a local recorder that logs and drops each outbound attempt.

```
py -3.11 packaging\windows\smoke_test_standalone.py --zip build\standalone\HiveMindHUD-<sha12>-win64-cpu.zip ^
    --out docs\evidence\standalone\<sha12>\smoke_results.json --workdir %TEMP%\hm-smoke-runs [--keep] [--skip-real-training]
```

This is a smoke test on the build machine. It is **not** clean-machine acceptance: system-wide installs, such as Python and
the VC++ runtime, are still on disk even though `PATH` is stripped.

### Results for `ee6202c68378` (`docs/evidence/standalone/ee6202c68378/smoke_results.json`)

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | no python/git/py on PATH | PASS | PATH = System32, Windows, Wbem |
| 2 | install path has spaces + non-ASCII | PASS | |
| 3 | `--check` exits 0 with JSON | PASS | exit 0 |
| 4 | `--check` reports frozen CPU build | PASS | `compute_backend: cpu` |
| 5 | data dir resolves to per-user LOCALAPPDATA | PASS | |
| 6 | writable roots outside install dir | PASS | cache/artifacts under data dir |
| 7 | non-loopback bind refused without `--allow-network` | PASS | exit 2 |
| 8 | server starts and announces URL | PASS | 5.0 s to URL |
| 9 | only loopback listeners | PASS | `127.0.0.1:<port>` only |
| 10 | `GET /` serves HUD HTML (`id="webgl-georing-canvas"`) | PASS | 200, 48,014 bytes |
| 11 | topology labelled demo fixture | PASS | `source: demo_fixture`, 4 nodes |
| 12 | `/api/rtx-tool-status` reports no CUDA | PASS | `cuda_available: false`, torch `2.6.0+cpu` |
| 13 | WebSocket telemetry frame labelled demo fixture | PASS | |
| 14 | offline: shard scan on supplied rows | PASS | 200 |
| 15 | offline: dataset scan responds without hanging | PASS | 200 (records behaviour only) |
| 16 | offline: no outbound HTTP(S) attempt reached the recording proxy | **FAIL** | `CONNECT datasets-server.huggingface.co:443` (known issue 1) |
| 17 | forge, synthetic micro-forge | PASS | 200 |
| 18 | forge `real_training: true` returns a structured result | PASS | 200 `trained`, 8 steps, 0.1 s (details below) |
| 19 | second instance on the same explicit port exits 3 | PASS | exit 3 |
| 20 | default port busy, falls back to a free loopback port | PASS | test held 8090 |
| 21 | fallback instance stops via `--stop` | PASS | stop 0, server 0 (separate data dir) |
| 22 | graceful shutdown via `--stop` | PASS | stop 0, server 0, 1.1 s |
| 23 | status file removed after shutdown | PASS | |
| 24 | no processes left running from install dir | PASS | |
| 25 | port released after shutdown | PASS | |
| 26 | install directory unchanged by running the app | PASS | 0 changed files |
| 27 | app wrote its files under the per-user data dir | PASS | |
| obs | two instances share one data dir, then one `--stop` | OBSERVED | B stopped (exit 0), A kept running, `--stop` printed "HiveMind HUD stopped." (known issue 2) |

The forge `real_training` response for check 18 was recorded verbatim in `real_training_response`. Its key fields:

- `model_kind: randomly_initialized_tiny_gpt2` and `pretrained_model_verified: false`.
- `training_sample_count: 2`, `heldout_sample_count: 1` and `heldout_disjoint_verified: true`.
- Heldout loss went from 5.5242 to 5.5211 (0.06%), with `heldout_improved: true`.
- `validation_status: not_run`, `rewards_simulated: true` and `total_pts_awarded: 0`.
- `adapter_reloaded_and_verified: true` (exact state-dict round trip).

None of this is evidence of model quality.

Earlier pipeline dry-run: the build of `7e69514` (before the training fix was merged) exited 0. Its first smoke run failed
because of a harness bug, not a packaging bug: the env allow-list was case-sensitive, so it dropped `SYSTEMROOT` and Winsock failed with
WinError 10106. The frozen app then reported fastapi/uvicorn/torch as missing with a secondary `NameError: base_events`.
After the fix (`ee6202c`), the dry-run zip passed its 26 checks at the time, before the proxy check and observation existed.

## Manifest fields

`BUILD_MANIFEST.json` (inside the app) and `<zip>.manifest.json` (next to the zip, which also records the zip hash) contain:

- `source_repo`, `source_sha` and `source_dirty`.
- `build_command`, `built_at_utc` and `build_seconds`.
- `build_host` (OS, Python, machine).
- `dependencies`: the full `pip freeze --all` of the build venv.
- `excluded_from_bundle`, `supported` and `unsupported`.
- `project_license_file`.
- `third_party_licenses`: name, version, original license metadata, copied files and per-file source/hash provenance.
- `license_text_gaps`: distributions for which no license text was collected; an explicit review list, not legal clearance.
- `exe_sha256`, `file_count` and `total_bytes`.
- `zip` (`file`, `sha256`, `bytes`).

For `ee6202c68378` (redacted copy: `docs/evidence/standalone/ee6202c68378/HiveMindHUD-ee6202c68378-win64-cpu.manifest.json`):

| Field | Value |
|---|---|
| source_sha | `ee6202c68378e606849054a0566e97dd3451b2a9` (clean) |
| build_command | `py -3.11 packaging\windows\build_standalone.py --reuse-venv` (284 s with a reused venv) |
| key deps | torch 2.6.0+cpu, transformers 4.57.6, peft 0.17.1, numpy 2.1.3, accelerate 1.15.0, tokenizers 0.22.2, safetensors 0.8.0, fastapi 0.150.0, uvicorn 0.54.0, pyinstaller 6.22.3 |
| exe SHA-256 | `a5a94168daa15f47c79cdc655f4ebe1a8d2999b1296fcec5dc9d6992ef42559c` |
| zip | `HiveMindHUD-ee6202c68378-win64-cpu.zip`, SHA-256 `6618e1e6dfebca40069da567cbe049d77ddb3a0e48b31bc2fb1dacd4443c2def`, 217,261,400 bytes (207 MiB) |
| unpacked | 4,937 files, 588,596,284 bytes (561 MiB) |

Both hashes were re-checked independently with `sha256sum` against the built files.

## Historical build issues (fixed in source; rebuild pending)

1. **"Offline mode" still has a direct Hugging Face network fallback.** With `HF_HUB_OFFLINE=1` set,
   `POST /api/scan-hf-shard` with a `dataset_id` tried `CONNECT datasets-server.huggingface.co:443`. The URL is built in
   `src/hivemind/cli/demo_hf_domain_swarm.py:232`, and the `HF_*_OFFLINE` variables do not cover a direct HTTP call.
   The endpoint still returned 200 because the smoke proxy dropped the connection. On a networked machine the request
   would go out. This was fixed by the integrated offline/ownership patch. The historical smoke result
   is unchanged; the final EXE must pass the proxy check after rebuilding.
2. **Several launches can share the stop/status files ambiguously.** Two instances that use one data directory both start, on
   different ports. A single `--stop` stopped only the later instance and reported success, and the earlier one kept running.
   This historical behavior is fixed in source with kernel-held directory ownership and generation-specific
   stop files. The current smoke harness asserts the corrected behavior instead of merely observing it.

## Blockers

- **Clean-machine acceptance: BLOCKED.** Windows Sandbox is not present (`C:\Windows\System32\WindowsSandbox.exe` does
  not exist) and no VM is available. Nothing was enabled or installed to get one. Until the zip runs on a machine
  without Python or the VC++ runtime, "runs on any Windows PC" is unproven.
- **No project LICENSE file.** The repository has no LICENSE/COPYING file and `pyproject.toml` declares no license. The manifest
  records `project_license_file: NOT FOUND`. A maintainer must settle redistribution terms for HiveMind itself before
  anyone publishes the zip.
- **Third-party license bundle needs rebuilt verification.** The historical EXE omitted tokenizers 0.22.2
  license text. The source collector now uses a version-specific, hash-verified copy of its official
  upstream license when the installed distribution omits it, records provenance, and lists remaining
  `license_text_gaps`. Inspect the rebuilt archive. This is not comprehensive native/Rust dependency
  license clearance; project licensing remains an owner decision.
- **Unsigned executable.** The build has no Authenticode signing step. SmartScreen and antivirus behaviour on other machines is untested.
- **Generated shortcuts need frozen validation.** Frozen startup now passes its actual data directory
  into package generation. Both shortcuts invoke the original standalone EXE with that directory,
  preserve quoted/Unicode paths, and no longer require Python or a checkout. The campaign shortcut
  explains that this build opens the HUD, because it does not bundle the CLI campaign. Source
  checkout launchers are unchanged. Verify both shortcuts against the rebuilt EXE.
- **Import diagnostics need frozen validation.** Dependency probing now logs each full traceback in
  `hud-launcher.log`, preserving the first native/import failure before later secondary errors.
  Portable regressions cover this; verify diagnostics in a fresh frozen build if startup fails.
