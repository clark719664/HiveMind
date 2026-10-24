# Changelog

## Unreleased — 2026-10-24 — 10-hour T4 benchmark campaign (E11, E12, E10b, E11b)

- Pre-registered experiments run on one Tesla T4 with Hugging Face data and models (Qwen2.5-0.5B, Qwen2.5-Coder-7B in fp16 and Gemma-4-E4B in bf16 on a 16 GB card): E11 refutes P11a on all three models and P11b; E12 supports P12-same and P12-cross; E10b at 0.5B is invalid by its positive control. Report: `docs/benchmarks/T4_10h_campaign_report.md`.
- New code: `hivemind.compress.kv_lattice` (GPU ℤ⁸ / A₂ / E₈ codecs, bit-identical to the reference decoders), `hivemind.core.kv_page_codecs` (pluggable KV page codec, FP8 default unchanged), `hivemind.boinc.validator_assignment` (architecture-matched spot-check validators), `ComputeVisibleSparseLoCoCompressor(value_codec=...)`, experiment runners E10b / E11 / E12, the E11b aggregator and a resumable GPU job queue with NVML sampling.
- Fix: `hivemind.core.quantization` imports `peft` lazily, so the spot-check replay path and its tests run without peft.

## v2.4.0 — 2026-10-23 — preprint v3.6.0 after an external review of v3.5.0

- Review recorded verbatim (`paper/review/external_review_v3.5.0_2026-10-23.md`, provenance unstated, no score) with a point-by-point response (`response_to_external_review_v3.5.0.md`).
- Paper v3.6.0: architecture figure (worker inner step; outer WAN round), hardware-heterogeneity caveat and architecture-matched rule for the ε = 1e-4 spot-check (new E12 / RQ11), compounding-approximations and PoC-scale limitations, Skip-Token-as-dropout note, PA-CGAD-under-quorum note. Numbers unchanged.
- Kept in the manuscript, on purpose: the correction that versions 2.9–3.1 labelled a scripted audit a "peer review" (the error is on public records; so is its correction).
- Zenodo: new version of the concept record; Lean: `lean4/mathlib/RDAMultiplier.lean` (3 Mathlib theorems, independent gate).

