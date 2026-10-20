# Final Windows acceptance handoff

This checklist separates source/CI verification from actual Windows artifact and
hardware evidence. Use the exact commit supplied with the final checkpoint; do
not reuse the historical `d36e654` EXE or overwrite an active checkout.

## 1. Fresh checkout and native tests

In PowerShell, from a directory where `HiveMind-final-acceptance` does not exist:

```powershell
git clone --single-branch --branch fix/final-packaging-readiness https://github.com/clark719664/HiveMind.git HiveMind-final-acceptance
Set-Location HiveMind-final-acceptance
$sha = (git rev-parse HEAD).Trim()
$sha12 = $sha.Substring(0, 12)
# Compare $sha with the exact reviewed checkpoint before building.
if (git status --porcelain) { throw 'Working tree must be clean before building.' }
py -3.11 -m venv .acceptance-venv
& .\.acceptance-venv\Scripts\python.exe -m pip install 'pytest>=8.3,<10'
$env:PYTHONPATH = 'src;.'
& .\.acceptance-venv\Scripts\python.exe -m pytest tests/test_standalone_launcher.py tests/test_standalone_smoke_harness.py tests/test_standalone_license_collection.py -q -k 'not check_reports' --tb=short
if ($LASTEXITCODE -ne 0) { throw 'Native regression checks failed.' }
```

A GitHub Windows lifecycle pass is useful but does not validate a frozen EXE.
Do not reuse a failing test result or suppress a failed assertion.

## 2. Build the exact source and smoke-test the EXE

```powershell
py -3.11 packaging\windows\build_standalone.py
if ($LASTEXITCODE -ne 0) { throw 'Standalone build failed.' }
$zip = "build\standalone\HiveMindHUD-$sha12-win64-cpu.zip"
$evidence = "docs\evidence\standalone\$sha12"
New-Item -ItemType Directory -Force $evidence | Out-Null
py -3.11 packaging\windows\smoke_test_standalone.py --zip $zip --out "$evidence\smoke_results.json" --keep
if ($LASTEXITCODE -ne 0) { throw 'Frozen smoke checks failed; inspect preserved logs.' }
Copy-Item "build\standalone\HiveMindHUD-$sha12-win64-cpu.manifest.json" $evidence
Get-FileHash $zip -Algorithm SHA256
```

Require the manifest's `source_sha` to equal `$sha` and `source_dirty` to be false.
Verify the tokenizers license text/provenance is bundled. The smoke test must
pass its offline proxy, explicit-port conflict, shared-directory rejection,
owner preservation, targeted stop, independent-owner survival, shutdown and
install-directory-unchanged checks. Read any failure logs before changing code.
Test both generated `.bat` launchers from the per-user cache and verify they
invoke the same EXE/data directory without Python/source-checkout assumptions.

This is still a build-machine test. Keep the archive local until project
redistribution terms are resolved. Do not upload EXEs, select a project license,
or sign/publish a release as part of this checklist.

## 3. Full browser sequence against the new EXE

Use an installed Edge or Chrome. The harness needs `playwright`, `pillow`,
`numpy`, and `ffmpeg` on PATH. If ffmpeg is absent, report that prerequisite
before claiming recording evidence. The following creates an isolated tooling
environment from the official package registry; it does not download a browser.
Do not substitute software rendering for a hardware-acceleration claim.

```powershell
py -3.11 -m venv build\browser-evidence-venv
$browserPython = (Resolve-Path 'build\browser-evidence-venv\Scripts\python.exe').Path
& $browserPython -m pip install playwright pillow numpy
if ($LASTEXITCODE -ne 0) { throw 'Browser-evidence dependency installation failed.' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'ffmpeg is required for recording evidence.' }
$exe = (Resolve-Path 'build\standalone\dist\HiveMindHUD\HiveMindHUD.exe').Path
$data = Join-Path $env:TEMP "HiveMind-browser-$sha12"
$p = Start-Process -FilePath $exe -ArgumentList "--no-browser --port 0 --data-dir `"$data`"" -PassThru
try {
    $deadline = (Get-Date).AddSeconds(240)
    $status = $null
    do {
        if ($p.HasExited) { throw "HUD exited during startup: $($p.ExitCode)" }
        $statusPath = Join-Path $data 'run\hud-server.json'
        if (Test-Path $statusPath) {
            $status = Get-Content $statusPath -Raw | ConvertFrom-Json
        }
        if ($status.state -eq 'running') { break }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if ($status.state -ne 'running') { throw 'HUD startup timed out; inspect launcher logs.' }
    & $browserPython tools\browser_evidence\hud_browser_evidence.py --url $status.url --channel msedge --out "docs\evidence\browser-frozen\$sha12-run1"
    & $browserPython tools\browser_evidence\hud_browser_evidence.py --url $status.url --channel msedge --trace-events --out "docs\evidence\browser-frozen\$sha12-run2-traced"
} finally {
    & $exe --stop --data-dir $data
    if (-not $p.WaitForExit(60000)) { throw 'HUD did not exit after --stop; preserve logs.' }
}
```

Inspect both runs, including the bounded `interaction_trace.json` from the traced
run, screenshots/recordings, console/network errors,
412px viewport DPR 1/2/3, and the complete repeated context-loss/restore sequence.
Tracing can affect timing, so retain both ordinary and traced runs. The
historical intermittent cycle-2 drag mismatch is **unresolved** until the
full-sequence pointer/state/context capture establishes what happened. A
simplified drag-only pass is insufficient. If a failure recurs, preserve the
unmodified evidence and diagnose event delivery, capture, renderer state and
frame scheduling before changing the renderer.

The bounded pretrained CPU LoRA execution/reload check has already completed in
the cloud; see [the recorded result](PRETRAINED_CPU_VALIDATION.md). There is no
need to rerun it on the PC merely to complete this checklist. Broad model quality
is not established by its three similar heldouts.

## 4. Work that needs more than a routine PC run

These remain separately gated; do not turn a missing capability into a pass:

- **Clean Windows machine:** test the new archive on an authorized clean Windows
  VM or another machine without development dependencies. Build-machine PATH
  stripping is not equivalent. Do not enable OS security/virtualization features
  or provision paid machines without permission.
- **Actual supported GPU:** follow [GPU acceptance reporting](GPU_ACCEPTANCE_REPORTING.md)
  and [preemption validation](GPU_PREEMPTION_VALIDATION.md). A CPU-only EXE and an
  unsupported accelerator cannot prove CUDA kernels, NF4 memory, or preemption
  latency. Retain RAM gates and report unsupported/missing hardware honestly.
- **Real WAN:** [WAN acceptance runbook](WAN_ACCEPTANCE_RUNBOOK.md) needs two
  authorized environments on different networks, authenticated transfer and
  receiver artifacts. Do not open firewall/router ports, buy relay service, or
  transmit private data without permission. Unsupported hole punching remains
  unsupported.
- **Steam:** [Steam acceptance runbook](STEAM_ACCEPTANCE_RUNBOOK.md) requires the
  owner's authorized AppID, SDK and account rights. The repository's simulated
  Steam features and unrelated test AppID are not live Steam acceptance.
- **Release decisions:** project license choice belongs to the owner. Signing,
  SmartScreen/antivirus validation and redistribution review remain outstanding;
  never buy a certificate, sign an agreement or publish by inference.

Return the exact source SHA, test commands/counts, build hashes, manifest,
smoke/browser evidence paths, and an explicit pass/fail/blocked list. Preserve
existing user work and never force-push, merge upstream or retry the denied
upstream-PR action.
