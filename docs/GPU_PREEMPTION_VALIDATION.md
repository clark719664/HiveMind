# Hardware validation handoff: training preemption

This is an unexecuted validation plan for Claude or another authorized developer
on a suitable Windows/CUDA machine. The cloud implementation and regression
suite were tested on CPU. Do not infer GPU latency, complete VRAM release, or
Windows input-hook reliability from those results.

## Bounded task

Validate the preemption path in `core/diloco_inner.py`, `client/daemon.py`, and
`client/worker_node.py`. Use the target GPU and model profile with the owner's
permission. Do not change drivers, OS security settings, credentials, or machine
configuration as a side effect.

1. Record OS, GPU, driver, Python, Torch, model profile, parameter dtype, and the
   actual initial CUDA allocated/reserved memory. Keep the original model owner
   alive throughout the test; deleting a test-local model is not sufficient proof.
2. Run several inner steps so AdamW state exists, then trigger the supplied
   preemption callback at a known step boundary. Verify there are no later
   optimizer steps, the final checkpoint matches the stopped weights, registered
   parameters/buffers are on CPU, and returned offload/error fields are accurate.
3. Measure wall time and actual CUDA allocated/reserved memory after offload.
   Compare with process-level GPU memory, recording CUDA-context and external
   allocations separately. Include a test with an unrelated live CUDA tensor;
   the implementation must not claim it released that tensor's memory.
4. Exercise failed disk writes and a model that rejects CPU offload. Training
   must stop, the failure must remain visible, and no partial file or false
   offload-success claim should appear. Check available host RAM before large
   model offloads and report partial/failed offloads explicitly.
5. With permission for interactive testing, verify real Windows idle readings
   change after input and that recent input or a watched game requests preemption.
   Confirm missing telemetry pauses background work. Explicit benchmark opt-outs
   should remain explicit, rather than silently becoming production defaults.
6. Validate the intended controller's recovery: the model remains on CPU after
   preemption. It must not return to CUDA until the idle policy permits. Any
   controller that relies on an old cached device field must reconcile it with
   the model's actual device before its next forward pass.
7. Load emergency checkpoints with CPU mapping and inspect the saved optimizer
   and RNG state. Exact optimizer continuation is currently unsupported. If
   implementing it, add a deterministic interrupted-versus-uninterrupted test
   before claiming trajectory-equivalent resume.

Report measured outcomes and remaining failures. Do not claim “100% VRAM
release” or a fixed 1.5-second deadline unless the defined target workload and
measurements support that specific claim.
