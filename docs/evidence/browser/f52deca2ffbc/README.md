# Geo-Ring WebGL HUD: real-browser evidence (f52deca2ffbc)

The HUD was generated from source at commit `f52deca2ffbc`. It was served by
`python -m hivemind.cli.main serve-hack-hud --host 127.0.0.1 --port <free port>` and driven by
`tools/browser_evidence/hud_browser_evidence.py`, which was uncommitted at run time.
Machine-readable results are in `results.json` and every console event is in `console.json`.
For comparison, `../6b68958a7cf0/` holds the same harness run against the pre-fix source.

## Environment (measured)

| Item | Value |
|---|---|
| Browser | Microsoft Edge 155.0.4283.62 (Chromium 155.0.8059.85), installed copy, `channel='msedge'`, **headed**, no extra launch flags |
| Driver | Playwright for Python 1.63.0 (no browser download) |
| User agent | `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0` |
| WebGL (HUD's own context, `WEBGL_debug_renderer_info`) | `ANGLE (AMD, AMD Radeon RX 7600 (0x00007480) Direct3D11 vs_5_0 ps_5_0, D3D11)`, WebGL 1.0 |
| `edge://gpu` | `WebGL: Hardware accelerated`, `Canvas: Hardware accelerated`, `Compositing: Hardware accelerated`; GPU0 = AMD Radeon RX 7600 *ACTIVE* |
| `failIfMajorPerformanceCaveat` probe | context created (no software fallback) |

**Hardware acceleration is established for this run.** The renderer is not SwiftShader, WARP or Basic Render,
and the GPU page reports WebGL as hardware accelerated. This covers this browser on this PC only.

## Checks (22 pass, 2 fail)

| Check | Result | Observable evidence |
|---|---|---|
| hardware_acceleration | pass | see above |
| initial_render | pass | canvas element screenshot: 2.4% non-background pixels; grid/link/node colours present (5325/152/275 px) |
| node_selection_selector | pass | `#georing-terminal` gains `seti-node-paris-01 ... Example RTT` after `#georing-node` change |
| node_selection_click | pass | click at projected node position selects `seti-node-tokyo-04`; selector value follows |
| drag_rotation | pass | 150x40 px drag changes 4.2% of pixels; 20 rAF callbacks, 60 drawArrays |
| keyboard_rotation | pass | implemented (canvas `keydown`, arrow keys); 4.6% pixels change; page did not scroll |
| reset_view | pass | after reset, 0 pixels differ from the default view |
| reload_rerender | pass | after reload, 0 pixels differ from the initial render |
| resize (6 cases) | pass | 1280x800, 1024x768, 800x600, 412x915 at DPR 1, and 412x915 at DPR 2 and 3: backing = content box x min(DPR,2) exactly; globe bbox aspect 0.989-0.992 |
| context_loss_pause (2 cycles) | pass | `isContextLost()` true; terminal `[WEBGL CONTEXT LOST] Drawing paused...`; drag, arrow key, reset and select during loss gave **0 rAF callbacks and 0 drawArrays**, and the canvas stayed unchanged |
| context_restore_resume | pass | after `restoreContext()`, 0 pixels differ from the pre-loss view |
| no_duplicate_loops | pass | idle 1.5 s: 0 rAF callbacks before and after restore. Same drag: 20 callbacks / 60 draws before and after |
| no_duplicate_handlers | pass | listener registrations identical before and after restore (each canvas handler registered once). The same drag after restore gives a pixel-identical image (0 px diff), and a 2x-drag control differs by 4.1% |
| post_restore_interaction (+ 2 click checks) | pass | selector and canvas-click selection both work after each restore |
| telemetry_ws_fixture | pass | 22 WS frames from `/ws/swarm-telemetry`, all `source: demo_fixture` |
| rest_topology_fixture | pass | `/api/georing-topology`: `source: demo_fixture`, `live_peer_discovery: false`, 4 nodes / 4 links |
| ui_fixture_labels | pass | heading `... // DEMO FIXTURE`, terminal `[DEMO FIXTURE] ... no live peer discovery`, telemetry `[SIMULATED TELEMETRY]` |
| no_script_errors | pass | 0 pageerror, 0 window errors, 0 unhandled rejections, 0 script console errors |
| no_webgl_console_messages | pass | 0 (was 8 before the fix) |
| server_shutdown_clean | pass | CTRL_BREAK, graceful uvicorn shutdown lines, port closed, 0 leftover processes (exit code 3 is the CRT default after uvicorn re-raises SIGBREAK) |
| **console_clean** | **fail** | 1 console error: `favicon.ico` 404 (the HUD has no icon link and the server has no route) |
| **page_no_horizontal_overflow** | **fail** | whole page scrolls sideways at 800 px (138 px) and 412 px (526 px): the leaderboard `th` cells and `span.pill` elements overflow. The Geo-Ring card itself fits |

## Defects found and fixed in f52deca (Geo-Ring script only, regression test in `tests/test_webgl_hud.py`)

1. **Stale WebGL objects deleted after restore.** `release()` deleted buffers and the program from the lost
   context, which logged 4 `WebGL: INVALID_OPERATION: delete: object does not belong to this context` warnings
   per restore (8 in the pre-fix run). The lost handler now drops those references.
2. **Backing store sized from the border box.** It was 2 CSS px (x DPR) too large on each axis, so every frame
   was resampled. Pre-fix measurements: backing 1187x320 for a 1185x318 content box, and 1028 unique colours
   in the default view. After the fix: an exact match and 25 colours. `pick()` uses the same content box.

## Not fixed (outside the Geo-Ring script): favicon 404 console error; page horizontal overflow at 800 px and narrower

## What this does NOT prove

- **Context loss was synthetic** (`WEBGL_lose_context`). A real GPU-process crash, driver reset (TDR) or device
  removal was not tested.
- **The WebSocket is not evidence of live distributed peers.** It is a localhost connection to the HUD server,
  which generates fixture frames. Nothing here shows peer discovery, WAN behaviour or measured RTT.
- Only this browser, GPU and OS were tested. The frozen exe's HUD was not run; use `--url` for that.
  Steam and its overlay were not tested, and neither were frame rate under load or accessibility beyond keyboard and aria-label.
- The renderer is on-demand, with no continuous loop. So "loop rate" means callbacks per interaction and while idle.
- Window listener counts include Playwright's own instrumentation listeners. They stay constant and are not HUD handlers.

## Observations

- Selecting a node changes only the terminal text. The canvas does not highlight the node.
- While the context is lost, Edge paints its own light placeholder with a sad-face glyph inside the canvas.
- DPR is capped at 2 by design, as the 3x case confirms.

## Reproduce

```
python -m venv build/evidence-venv
build/evidence-venv/Scripts/python -m pip install playwright==1.63.0 pillow==12.3.0 numpy==2.4.6
build/evidence-venv/Scripts/python tools/browser_evidence/hud_browser_evidence.py --server-python <python with HiveMind deps>
```

Running the source server regenerates `steam_dist/HiveMind_Steam_FreeEdition/web_ui/index.html`. That file was
restored with `git checkout` and not committed.

Files: `01_initial_canvas.png`, `02_after_drag_canvas.png`, `03_before_context_loss_canvas.png`,
`04_context_lost_card.png`, `05_after_restore_canvas.png`, `06_resize_412x915_dpr{1,2,3}_canvas.png`, and
`hud_key_sequence.mp4` (29 frames at 3 fps, 9.7 s, 163 KB, Geo-Ring card only).
