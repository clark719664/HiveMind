# Geo-Ring WebGL HUD: pre-fix comparison run (6b68958a7cf0)

This is the same harness as `../f52deca2ffbc/`, run on the same Edge 155 headed browser with the same ANGLE D3D11
AMD Radeon RX 7600 renderer (hardware accelerated). It served a `git archive` export of `6b68958` from
`build/pre_fix_src` with `--source-tree build/pre_fix_src --source-commit 6b68958...`. The export wrote its
generated files inside `build/`, so no tracked files changed.

Result: 20 pass, 4 fail. Both renderer defects that f52deca fixes are reproduced here:

- **no_webgl_console_messages: fail.** There are 8 `WebGL: INVALID_OPERATION: delete: object does not belong to this
  context` warnings, 4 per `restoreContext()` cycle (3 buffers + 1 program from the lost context).
- **resize: fail at all 6 sizes.** The backing store follows the border box, not the content box. Examples: 1187x320 for
  1185x318 at DPR 1, and 724x640 for 360x318 at DPR 2. The resampling also shows up as 1028 unique colours in the
  default view, against 25 after the fix.
- console_clean and page_no_horizontal_overflow fail as they do after the fix: favicon 404 and leaderboard
  overflow, both outside the Geo-Ring script.

Everything else matched the post-fix run, including the context-loss pause and resume, the duplicate-loop and
duplicate-handler checks, selection, drag, keys, reset, reload and the fixture labels. The limits stated in
`../f52deca2ffbc/README.md` apply here too.
