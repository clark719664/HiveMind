# Changelog

## v2.5.1 — 2026-10-24 — preprint v3.7.1: E12 pilot complete (bf16 cell)

- E12-pilot P12d supported: bf16 autocast replayed by an fp32 validator diverges to 3.3e-3 with 96/96 tensors above ε, 2.7× the tampered control; dtype is a class boundary (validators assigned within dtype class; bf16 ε to be measured on GPU, `HM-S10-08`). PROV-34; ledger P12d; paper §4.1, §E12 and limitations updated. No other numbers changed.


## v2.5.0 — 2026-10-23 — preprint v3.7.0: E12 replay-divergence pilot; hourly PR gatekeeper

- E12-pilot (pre-registered): the Byzantine spot-check's ε = 1e-4 replay rule fails on CPU for a thread-count change alone (2.6e-4) and for a second host (1.5e-4); bit-identical execution reproduces exactly; tampered control 1.25e-3 with 96/96 tensors above ε. Rule re-specified as execution-class matched with a per-class ε and a tensor-count term. bf16 cell pending at build time.
- `scripts/pr_gatekeeper.py` (hourly, owner-only merge authority; PR text is data); branch protection on `main`.


## v2.4.0 — 2026-10-23 — preprint v3.6.0 after an external review of v3.5.0

- Review recorded verbatim (`paper/review/external_review_v3.5.0_2026-10-23.md`, provenance unstated, no score) with a point-by-point response (`response_to_external_review_v3.5.0.md`).
- Paper v3.6.0: architecture figure (worker inner step; outer WAN round), hardware-heterogeneity caveat and architecture-matched rule for the ε = 1e-4 spot-check (new E12 / RQ11), compounding-approximations and PoC-scale limitations, Skip-Token-as-dropout note, PA-CGAD-under-quorum note. Numbers unchanged.
- Kept in the manuscript, on purpose: the correction that versions 2.9–3.1 labelled a scripted audit a "peer review" (the error is on public records; so is its correction).
- Zenodo: new version of the concept record; Lean: `lean4/mathlib/RDAMultiplier.lean` (3 Mathlib theorems, independent gate).

