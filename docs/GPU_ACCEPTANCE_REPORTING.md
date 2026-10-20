# GPU acceptance report states

The scripts in `tools/gpu_acceptance/` emit schema version 2. CPU tests and
synthetic CUDA mocks test the scripts' control flow; they do not measure GPU
latency, VRAM release, CUDA numerical accuracy, or a real Triton launch.

## NF4 dense reference

`nf4_dense_reference.py` always runs CPU correctness unless CUDA acceptance is
blocked before work starts. On a CUDA host it also attempts the requested
shape/dtype matrix and the direct Triton FP8 parity probe. The default command
therefore may allocate on CUDA; `--require-cuda` additionally refuses a CPU-only
run. The NF4 execution path still reconstructs dense weights.

- Exit 0, `PASS_CORRECTNESS_ONLY_NO_CUDA`: CPU correctness passed; no GPU evidence.
- Exit 0, `PASS_CORRECTNESS_AND_CUDA_MEASURED`: CPU correctness passed, every
  requested measurement completed, and the direct Triton parity probe passed.
  This is not a fused-NF4 or memory-residency claim. CUDA NF4 timing rows are
  measurements; the independent NF4 correctness suite remains CPU-based.
- Exit 1, `FAIL_CORRECTNESS`, `FAIL_CUDA_MEASUREMENTS`, or `FAIL_CUDA_PARITY`:
  a required check or attempted measurement failed. CPU success cannot override
  a CUDA exception. Partial measurements remain in the report for diagnosis.
- Exit 2, `BLOCKED_NO_CUDA`, `BLOCKED_NON_CUDA_BACKEND`,
  `BLOCKED_INCOMPLETE_CUDA_MEASUREMENTS`, or `BLOCKED_CUDA_PARITY_UNAVAILABLE`:
  acceptance evidence is incomplete. An unsupported dtype, omitted measurement,
  or unavailable Triton probe cannot become a successful measurement.
- Invalid shapes, negative warmup counts, and nonpositive iteration counts are
  rejected by argument parsing before measurement work (exit 2).

Each CUDA matrix row is `MEASURED`, `SKIPPED_UNSUPPORTED`, or `ERROR`; the
`measurement_summary` records requested, measured, skipped, error, and complete
counts/state; malformed or incomplete success payloads are errors. The requested matrix includes float16, bfloat16, and float32.
A hardware capability skip remains visible and blocks a complete-matrix verdict.

The Triton probe launches `_triton_fp8_quant_kernel` directly and synchronizes
before declaring that it ran. It cannot pass through the production wrapper's
PyTorch fallback. Its threshold is zero mismatched FP8 bytes and zero difference
in the emitted FP16 scales. Unsupported/unavailable and launch-error results are
reported separately from a completed parity comparison. This probe says nothing
about the unused fused INT4/FP8 router kernel.

## Preemption

`preemption_measure.py --functional-dry-run-cpu` always selects CPU, including
when CUDA is visible. It uses the CPU default model sizes, skips the live CUDA
bystander scenario, emits no preemption latency/VRAM measurements, and exits 3
on functional success, 1 on failure, or 2 if host-RAM admission is blocked.
A dry run is never acceptance evidence.

Without the dry-run flag, missing CUDA and ROCm/HIP are blocked (exit 2). A CUDA
run exits 0 only when its functional checks pass; scenario exceptions emit
`FAIL_EXECUTION` and exit 1. Successful main, bystander, and offload-rejection
scenarios must produce a readable, matching emergency checkpoint without an
error, matching LoRA keys, and valid CPU/CUDA RNG state fields. Only the explicitly injected unwritable-directory scenario expects a
missing checkpoint with a visible error; an unexpected success also fails it.

The manual hardware/session checks and limitations in
[GPU_PREEMPTION_VALIDATION.md](GPU_PREEMPTION_VALIDATION.md) still apply. A passing
script does not establish a fixed deadline, complete VRAM release, reliable
Windows hooks, or trajectory-equivalent optimizer resume.

### Host-RAM admission

Before any scenario model/bystander allocation, and before the initial CUDA
property probe, the tool requires known, valid available-host-RAM telemetry and
a conservative budget. Missing/invalid telemetry returns
`BLOCKED_HOST_RAM_UNKNOWN`; insufficient memory returns
`BLOCKED_INSUFFICIENT_HOST_RAM`. Both exit 2 and do not begin the blocked model
allocation. Each later scenario and the bare-offload reference recheck admission.
No force/ignore-RAM override is provided.

The budget is four times an arithmetic upper bound on model storage, plus
256 MiB headroom and any bystander allocation. The bound uses eight-byte elements,
LoRA parameters, and deliberately overestimates Qwen2's tied embeddings and
grouped-query projections. It constructs no model or tensor. Reports record the
estimate, budget, telemetry snapshot, and admission outcome. Small real CPU
profiles are tested against the bound; unknown/insufficient-memory regressions
forbid model allocation and GPU probes.

This is an admission estimate, not measured peak RAM or a memory reservation.
Other processes can change availability after a check. Free memory or reduce the
explicit model profile before retrying a blocked run; do not treat missing RAM
telemetry as permission to continue. It does not establish physical GPU results.
