# Frozen-exe HUD browser evidence (`d36e654`)

`tools/browser_evidence/hud_browser_evidence.py --url` was run against `HiveMindHUD.exe`. That is the PyInstaller build of `fix/opus-integration` at `d36e654acefc104cdb70149a0dc1a7c47fe96f46`, exe SHA-256 `a5636490d4d2841ee6446ef5a6f9a4f4daf03194d2fc1725e1111900c28c34c6`. The exe was extracted outside the checkout to a path with spaces and non-ASCII characters and served on `127.0.0.1:8471`. The browser was headed Edge on an AMD Radeon RX 7600 (D3D11 ANGLE).

| Run | Pass | Fail | Failures |
|---|---|---|---|
| `d36e654-run1` | 18 | 5 | `console_clean` (favicon 404), `page_no_horizontal_overflow`, `no_duplicate_loops`, `no_duplicate_handlers`, `post_restore_interaction` |
| `d36e654-run2` | 21 | 2 | `console_clean` (favicon 404), `page_no_horizontal_overflow` |

## Run 1: intermittent cycle-2 mismatch, not reproduced

The three extra run-1 failures all come from context-loss cycle 2:

- Cycle 1 matched exactly: the post-restore drag was pixel-identical to the pre-loss drag, with 20 rAF / 60 draws both before and after.
- **Cycle 2 did not:** the pre-loss drag ran 20 rAF / 60 draws, but the post-restore drag ran only **5 rAF / 15 draws**. Its image differed from the pre-loss drag by 4.26% of pixels and did not match the 2x-drag control either.
- Listener registrations were unchanged, and idle rAF stayed 0.

(An earlier version of this README said the frame counts were identical in both cycles. That was wrong: it was true only for cycle 1.)

The harness's `drag()` sends 4 segments x 5 pointer-move steps, so 20 moves. Each move schedules one frame. Only 5 frames means the HUD stopped processing the drag after the first segment; there is no evidence of a duplicated handler or loop. The post-restore probe is the only drag that takes recorder screenshots between segments. The mismatch did not recur in run 2 or in the source-server run (`docs/evidence/browser/f52deca2ffbc`).

**Cause: unresolved, under investigation.** It could be a harness artifact or a real HUD pointer-capture issue, and nothing is assumed either way. Treat the restore checks as passing 2 of 3 observed runs, not as proven deterministic.

## What this does not prove

- The context loss is synthetic (`WEBGL_lose_context`), not a driver reset.
- Telemetry is a localhost WebSocket of `demo_fixture` frames, not live peers.
- Hardware acceleration is established only for this PC and browser.
- This build still contains the two launcher issues pending the maintainer's fix: the offline-mode HF network fallback, and shared stop/status files.

## Diagnostic 1: mid-drag screenshots (not reproduced)

`tools/browser_evidence/diagnose_drag_truncation.py` repeated the harness's 4x5-step drag 60 times against the same frozen exe:
- Conditions were a fresh context and a post-restore context, each with no screenshot, card element screenshots, or full-page screenshots between segments, 10 trials each.
- Every pointer, focus and scroll event was logged in the capture phase.
- **0 of 60 drags were truncated.** Every trial delivered 20 pointermoves to the canvas, with no `pointercancel` or lost capture. Results: `d36e654-drag-diagnostic.json`.

Mid-drag screenshots alone do not explain the run-1 failure. Two caveats:
- The run-1 counters cannot tell lost moves apart from coalesced frames: `schedule()` skips a new request while a frame is pending.
- The image mismatch with the full-drag result suggests lost moves. Rotation is the sum of deltas, so coalesced frames alone would not change the final image.

The full harness sequence (resize contexts, then cycle 1 with interaction while lost, then the recorder) has not yet been re-run with pointer-event instrumentation. The cause remains **unresolved**.
