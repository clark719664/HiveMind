# Frozen-exe HUD browser evidence (`d36e654`)

`tools/browser_evidence/hud_browser_evidence.py --url` was run against `HiveMindHUD.exe`. That is the PyInstaller build of `fix/opus-integration` at `d36e654acefc104cdb70149a0dc1a7c47fe96f46`, exe SHA-256 `a5636490d4d2841ee6446ef5a6f9a4f4daf03194d2fc1725e1111900c28c34c6`. The exe was extracted outside the checkout to a path with spaces and non-ASCII characters and served on `127.0.0.1:8471`. The browser was headed Edge on an AMD Radeon RX 7600 (D3D11 ANGLE).

| Run | Pass | Fail | Failures |
|---|---|---|---|
| `d36e654-run1` | 18 | 5 | `console_clean` (favicon 404), `page_no_horizontal_overflow`, `no_duplicate_loops`, `no_duplicate_handlers`, `post_restore_interaction` |
| `d36e654-run2` | 21 | 2 | `console_clean` (favicon 404), `page_no_horizontal_overflow` |

## Run 1: intermittent cycle-2 mismatch, not reproduced

The three extra run-1 failures all come from context-loss cycle 2:

- Cycle 1 matched exactly: the post-restore drag was pixel-identical to the pre-loss drag.
- In cycle 2 the post-restore drag image differed from the pre-loss drag by 4.26% of pixels.
- The rAF and draw counters were identical (20 rAF callbacks and 60 draws per drag, both before and after).
- Listener registrations were unchanged.
- The image did not match the 2x-drag control either.

So this is not evidence of a duplicated handler or loop; the more likely cause is a different starting rotation. The mismatch did not recur in run 2 or in the source-server run (`docs/evidence/browser/f52deca2ffbc`). Its cause is **unresolved**: either a timing race in the harness or a real intermittent state issue. Treat the restore checks as passing 2 of 3 observed runs, not as proven deterministic.

## What this does not prove

- The context loss is synthetic (`WEBGL_lose_context`), not a driver reset.
- Telemetry is a localhost WebSocket of `demo_fixture` frames, not live peers.
- Hardware acceleration is established only for this PC and browser.
- This build still contains the two launcher issues pending the maintainer's fix: the offline-mode HF network fallback, and shared stop/status files.
