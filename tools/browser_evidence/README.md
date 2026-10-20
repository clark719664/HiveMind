# HUD browser checks

Run the existing full browser sequence against the source HUD or the rebuilt
frozen executable, using an installed Edge or Chrome. The evidence runner does
not install a browser. It needs `playwright`, `pillow`, `numpy`, and `ffmpeg`.

```powershell
python tools/browser_evidence/hud_browser_evidence.py --url http://127.0.0.1:8090/ --channel msedge --out docs/evidence/browser-frozen/final-normal
python tools/browser_evidence/hud_browser_evidence.py --url http://127.0.0.1:8090/ --channel msedge --trace-events --out docs/evidence/browser-frozen/final-traced
```

Use the URL printed by the launcher, which can differ from port 8090. Source
server runs can instead omit `--url` and supply `--server-python <venv python>`.
The source and rebuilt frozen binary both need validation; historical results do
not establish the behavior of the new build.

## Full-sequence restoration diagnostic

`--trace-events` adds `interaction_trace.json` while preserving the same full
sequence, including reload, synthetic context-loss cycles, and recorder
screenshots between drag segments. It records:

- Pointer coordinates, pointer IDs, buttons, capture/bubble phase, and pointer
  capture status, including cancellation and lost capture.
- Actual Geo-Ring `angles` shader uniform submissions (yaw/pitch), plus the last
  submitted angles at each event, rAF/draw counters, and context-loss state.
- Focus, visibility, scroll, canvas bounds, and context/drag segment markers.

The trace is bounded to 10,000 records per document; dropped records are reported.
The angles are the last submitted/rendered values, not a direct probe of private
in-memory renderer state. Logging can affect timing, so keep an ordinary run as
well as a traced run. Neither extra observer handlers nor instrumentation are
shipped in the HUD. The historical intermittent post-restore mismatch remains
unresolved until a full-sequence reproduction or sufficient new evidence
explains it. A simplified drag loop alone does not resolve it.

## Fast layout regression

The dependency-light template/launcher suite includes two opt-in browser tests:

```powershell
$env:HIVEMIND_BROWSER_EXECUTABLE = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
python -m pytest tests/test_hud_distribution_assets.py -q
```

They serve the real generated and shipped HTML with minimal test-only API
fixtures, check 320/412/800/1280 CSS-pixel widths at DPR 1/2/3, exercise keyboard
scrolling of the leaderboard region, and reject fallback `/favicon.ico` requests.
These headless checks validate layout/assets only. They do not establish
hardware acceleration, frozen-app startup, or renderer restoration correctness.
Without the environment variable, the browser tests are explicitly skipped.

For acceptance on the actual rebuilt binary, also verify that all nine table
columns can be reached at 412px, the page itself does not scroll sideways, and
there are no favicon resource errors. Exercise both generated `.bat` files from
the per-user cache after stopping the running HUD. They should reopen the same
standalone executable/data directory without requiring Python or a source
checkout; the HackMiner shortcut explains that the standalone package supports
the HUD rather than the CLI campaign. Repeat with a custom data directory and
paths containing spaces and non-ASCII characters.
