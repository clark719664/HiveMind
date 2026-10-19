## Summary & Parallel Track

- **Task / Bounty ID:** `HM-S3-__` (or `HM-M_-__` / Custom)
- **Closes Issue:** `#__`
- **Subsystem Track:**
  - [ ] `feat/*` — Core Swarm Fabric (`src/hivemind/swarm/`)
  - [ ] `expert/*` — Sparse MoE & Compression (`src/hivemind/core/`, `src/hivemind/compression/`, `src/hivemind/data/`)
  - [ ] `game/*` — Steam Hack-Miner, Web HUD & Steamworks (`src/hivemind/game/`, `src/hivemind/client/`)
  - [ ] `formal/*` — Lean 4 Formal Specification & Paper (`lean4/`, `paper/`)
  - [ ] `docs/*` / `fix/*` — Documentation, CI/CD, or Bug Fix

---

## Architectural Changes

<!-- Describe the technical implementation, algorithmic complexity, and wire/VRAM impact -->

---

## Mandatory Verification Checklist

- [ ] **Full 20/20 Test Suite Green:** `python -m pytest tests/ -v --tb=short` passes with `0` failures.
- [ ] **Lean 4 Formal Invariants Intact:** `lean4/HiveMindSpec.lean` contains `0 sorry` and `0` user-defined `axiom` declarations (`>= 17` machine-checked theorems).
- [ ] **8 GB Consumer GPU VRAM Envelope:** Peak active VRAM remains strictly `<= 5,950 MB` (`<= 5.81 GB`) with CPU fallback when CUDA/Triton/Steamworks DLLs are absent.
- [ ] **Single-Subsystem Isolation:** Changes are scoped to the target module + test suite to enable conflict-free parallel PR merges.
