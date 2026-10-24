# Vendored GWAYA zero-stub auditor

- Source repository: https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel (owner: xaviercallens; same owner as HiveMind)
- Commit SHA (tip of default branch `main` at retrieval): `9c0001e8307d079ac834a86bc37e2d56703a4fb5`
- Retrieved: 2026-10-24, from `raw.githubusercontent.com/xaviercallens/GWAYA-GwenLayaOpenModel/<sha>/<path>`

| File (here) | Upstream path | sha256 |
|---|---|---|
| `ast_audit.py` | `gwaya/ast_audit.py` | `af8f382c5a670bfc2e3632bf91d942d78810c54bc3f8f7014162ebf22b37dd7b` |
| `LICENSE` | `LICENSE` | `fa48d88d74dc5b33192cd559ae291b0d07ef3acbc17fbb776226ad3e3d2233a0` |

## Modifications

No modifications. Both files are byte-identical to upstream at the pinned commit.

## License

MIT, Copyright (c) 2026 xaviercallens (see `LICENSE` in this directory). The MIT notice must be kept with any copy.

## Why vendored

`ast_audit.py` imports only the standard library (`ast`, `re`, `dataclasses`, `typing`) and only parses source text; it never executes it
(checked by reading the whole file before vendoring). Vendoring one pinned file gives E15 the exact GWAYA zero-stub rule
without installing the `gwaya` package or any extra dependency, and without drift. `tests/test_e15_gate.py` verifies the sha256 above.
