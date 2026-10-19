# Local state and restart behavior

## HiveSwarmLedger

When `ledger_path` names an existing file, the ledger loads its chain, balances,
reputation, slashed-node registry, and inference settlement history. It checks
schema and value types, finite nonnegative amounts, canonical genesis, contiguous
block heights, block hashes and links, and the checksum of versioned snapshots.
Invalid, truncated, unsupported, or unreadable files stop initialization. They are
not replaced with a new genesis ledger.

The original unversioned five-field JSON format is supported. Loading a valid
legacy file leaves it untouched; the next successful write produces schema
version 1 with a snapshot checksum. Missing legacy fields are treated as corrupt
state, rather than filled with empty defaults. Keep a backup when migrating older
files. The checksum detects accidental corruption; it is not authentication
against someone who can rewrite the file and its checksum.

Writes use a temporary file and atomic replacement. Inference settlement rolls
back in-memory balances and history when persistence fails. The file is intended
for one local writer. Concurrent ledger instances sharing a path are not
coordinated, and this implementation is not a distributed consensus database.

## HiveLayer2StateChannel

Channel IDs, keys, nonces, vouchers, dispute status, and settlement watermarks are
process-local. They are not stored in the L1 ledger file and cannot be restored
by reopening that ledger. Repeated settlement is idempotent within the lifetime
of the same channel object. Cross-process or crash/restart redemption guarantees
are not implemented.

After a restart, create a new channel. Do not reconstruct a previous channel by
replaying its cumulative vouchers: the restored L1 ledger cannot infer that
channel's previous settlement watermark.

## BOINCCoordinatorService

Assignments, their initial-state snapshots, consumed-workunit protection, pending
deltas, branch state, and reputation are held in memory. A newly constructed
coordinator rejects previous workunit IDs as unknown. Workers must request new
assignments; constructing a new coordinator does not resume a training campaign.

Within a running coordinator, an assignment is bound to its issued metadata,
worker, account, and initial state. It is reserved during validation and consumed
once a verdict is recorded, preventing repeated credit issuance for that
assignment. A server-side validation failure restores it for retry.

## Training preemption and recovery

The daemon enforces OS idle detection by default. On unsupported platforms or
when input telemetry is unavailable, background training pauses conservatively.
An explicitly supplied simulated idle signal or `enforce_os_idle_hook=False`
remains available for intentional simulations and benchmarks.

When the training loop preempts, it stops taking optimizer steps, clears its
optimizer references, and attempts to move the model's registered parameters and
buffers to CPU. The model stays on CPU. A controller must explicitly restore its
training device after its idle policy permits further work; the worker does not
automatically move it back. Check `model_offloaded_to_cpu` and
`model_offload_error` before treating an offload as successful.

Emergency checkpoints use CPU tensor storage and atomic file replacement. A
failed write is reported by `emergency_checkpoint_error` and does not stop the
CPU-offload attempt. They contain the initial/final adapter state, optimizer
state, and PyTorch RNG state for recovery work. There is no automatic exact-resume
implementation: calling the inner loop again creates a fresh optimizer and
reseeds its RNG. Do not present that as continuing the saved optimizer trajectory.

The legacy `emergency_dump_latency_sec` field measures the complete preemption
operation, including offload and saving. No deadline or percentage of released
VRAM is guaranteed. Optional CUDA allocator readings are measurements for the
original model device; CPU-only runs leave them unset. External tensors, caches,
other model owners, the CUDA context, and other processes can retain GPU memory.
