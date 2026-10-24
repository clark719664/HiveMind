# A GWAYA-gated coding assistant that trains and runs in under 8 GB of VRAM (experiment E15)

> Project HiveMind is a voluntary, non-commercial research project. Nothing here involves money. Any recognition for contributions is non-monetary
> reciprocal compute accounting, as described in `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`.

## What this is

[GWAYA](https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel) (MIT) is a fail-closed verification layer between a code-generating model and execution.
A candidate is accepted only if (1) a zero-stub AST audit is clean (no `pass`, `...`, `raise NotImplementedError`-only bodies, `mock_*`/`fake_*`/`dummy_*`/`stub_*` names)
and (2) an oracle passes (syntax plus sandboxed execution of the task's tests). Otherwise the verdict is `UNVERIFIED` and the code is never accepted.
GWAYA does not train anything. E15 adds the training around it:

* **At inference time** the gate picks which candidate is returned (greedy, two samples, then up to three repair rounds fed with the gate's feedback).
* **At training time** the gate decides which model-generated solutions become fine-tuning data (rejection-sampling fine-tuning, RFT): the base model
  samples 8 solutions per MBPP training problem, only gate-accepted ones (all tests pass, audit clean) are kept, at most 2 distinct ones per problem.

Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` in fp16 with a hand-rolled LoRA (rank 16, alpha 32, all attention and MLP projections; no `peft`, no `bitsandbytes`).
A hard cap `torch.cuda.set_per_process_memory_fraction(7.5 GiB / total)` makes every stage run inside an 8 GB card budget even on a larger GPU.

Files: `scripts/e15_gate.py` (gate and sandbox), `scripts/e15_data.py` (MBPP/HumanEval loaders), `scripts/e15_lib.py`, `scripts/run_experiment_E15_gwaya_rft.py`,
`third_party/gwaya/` (the vendored auditor, unmodified, pinned commit, see `PROVENANCE.md`), tests `tests/test_e15_*.py`.

## Run it on an 8 GB GPU

Prerequisites: the model and datasets in the Hugging Face cache (`Qwen/Qwen2.5-Coder-1.5B-Instruct`, `google-research-datasets/mbpp` incl. `sanitized/test`
and `full/{train,validation}`, `openai/openai_humaneval`), Linux with unprivileged user namespaces (`unshare --user --map-root-user --mount --net --pid` must work),
a system `python3` outside your home directory. Nothing needs to be pip-installed for the gate.

```bash
python3 scripts/run_experiment_E15_gwaya_rft.py --estimate                      # planned generations / tokens / time (assumptions, not measurements)
python3 scripts/run_experiment_E15_gwaya_rft.py --smoke --out /tmp/e15_smoke    # tiny random model on CPU, ~15 s, plumbing check only
python3 -m pytest tests/test_e15_gate.py tests/test_e15_lib.py tests/test_e15_smoke.py   # CPU, about 45 s
python3 scripts/run_experiment_E15_gwaya_rft.py --out artifacts/experiments --scratch ~/e15_scratch --resume   # the registered run
```

Lower `--sample-bs` / `--eval-bs` (default 16) if you hit the cap on a smaller card; the generator halves the batch on an out-of-memory error. The full run refuses to
start without `artifacts/experiments/E15_gwaya_rft_prereg.json`. `--resume` continues from the per-problem journals. The LoRA file is written to `--scratch`, not to `artifacts/`.
Stages: `baseline` (C0) -> `loop_base` (C1) -> `rft` -> `train` -> `tuned` (C2, C3) -> `controls`; the report holds pass@1 with Wilson 95% intervals, paired differences
with exact McNemar p-values and discordant counts, cost accounting (generated tokens, GPU seconds, correct solutions per GPU-minute, peak reserved GiB, NVML energy proxy in Wh) and the controls.
Exit codes: 2 refused to start, 3 registered fail-fast (fewer than 150 training problems with an accepted solution, or projected time above `--max-hours`).

## How the gate and the sandbox work

`gate(code, tests, setup, timeout_s, mem_mb)` returns `ACCEPTED`, `REJECTED_SYNTAX`, `REJECTED_STUB`, `REJECTED_TEST` or `UNVERIFIED`:

1. extract the last ```` ```python ```` block, `ast.parse`;
2. GWAYA's `ZeroStubAudit.audit_python_code` (loaded by path after a sha256 check against the pin);
3. run `setup + code + tests` in the sandbox; every test must pass and the harness must print a one-time marker, so an early `sys.exit(0)` / `os._exit(0)` is not an acceptance.

If the sandbox cannot be established, errors, or returns garbled output, the verdict is `UNVERIFIED`; `gate_available()` self-tests the sandbox once (it must run `print(1)`,
must not see the repository file, `/home`, the network or any capability) and, if false, makes every verdict `UNVERIFIED`.

The sandbox is a small launcher started with `unshare --user --map-root-user --mount --net --pid --ipc --uts --fork --kill-child`. Inside it: private mounts, a fresh tmpfs over
`/home`, `/root`, `/tmp`, `/var/tmp`, `/run`, `/dev/shm`, `/mnt`, `/media`, `/srv` and the real home and repository directories, a fresh `/proc`, a minimal `/dev` (null, zero, full, random, urandom; no GPU device nodes), all capabilities dropped (otherwise the
candidate could unmount the tmpfs) with `no_new_privs`, then the system `python3 -I -S -B` with an empty environment (PATH only), a clean cwd on the tmpfs, `RLIMIT_AS`
(2 GiB), `RLIMIT_CPU`, `RLIMIT_NPROC` (64), `RLIMIT_FSIZE` (10 MB), `RLIMIT_NOFILE`, a wall-clock kill of the process group, output truncated to 10 KB. The network namespace has no
usable interface. Candidate source reaches the launcher through a pipe, never through a host file.

The containment tests plant hostile candidates (secret-file read incl. a random canary inside the real home, TCP to 127.0.0.1 and 8.8.8.8, 4 GiB allocation, infinite loop, fork bomb,
100 MB disk write, `cat /proc/1/environ`, tampering with the vendored auditor, `curl`, `umount /home`, early exits) and require that none is accepted and none sees the canary.

### What is NOT equivalent to GWAYA's own Bubblewrap oracle

* No seccomp filter and no cgroup limits: memory is limited per process (`RLIMIT_AS`), and `RLIMIT_NPROC` counts processes of the whole real user, so on a busy host forks simply fail.
* The kernel is shared. A kernel or namespace vulnerability would defeat this sandbox as it would defeat Bubblewrap; this is defence in depth for LLM-generated code, not a security product.
* Candidates can still create nested user namespaces (they gain nothing over the parent mounts), and read world-readable files outside the hidden directories (for example `/etc`, `/usr`).
* The pass marker is printed by in-process code; a candidate that deliberately inspects the harness could forge it. Accidental gaming (early exit, printing) is covered, deliberate attacks on the
  verdict are not. Run only code from models you are evaluating, never untrusted third-party code.
* Python only: no Lean/Rust/C++/Go paths of GWAYA are used. GWAYA's repair prompts are not used; the repair loop here is our own.

## Honest expected results and limitations

No result is claimed before the run. The registration fixes the hypotheses in advance (H15a RFT improves greedy pass@1 by at least 3 points with McNemar p < 0.05; H15b the inference loop improves the base by at
least 5 points; stacking and HumanEval transfer are descriptive). For context, GWAYA's own README reports +4.7 to +7.8 points for the same model family with best-of-3 plus repair and misses its own +8 point gate;
the plausible outcomes include "no significant change". Limitations: one 1.5B model; MBPP is small and may have been seen in the pre-training of the base model, so contamination of the base cannot be excluded (E15 only guarantees
no overlap between its own train and eval splits; two MBPP training problems with the same prompt text as eval problems are dropped, 462 remain); the visible test is part of the prompt, so the inference gate
mostly checks syntax, stubs and that one example; sampling seeds are per batch; energy is whole-GPU power while other jobs may share the GPU; paired tests on 257 problems detect only differences of several points.

## Five small contribution ideas

1. Add a second visible-test policy (for example gate with 2 of the 3 MBPP tests) and report how much the inference loop gains when the gate sees more.
2. Add `seccomp` (a small BPF denylist via `prctl`) to the launcher and extend the containment probes (`ptrace`, `unshare`, `mount`, `io_uring`).
3. Port the gate to another benchmark (for example a small class of Lean or Rust tasks through GWAYA's other audit functions) with the same fail-closed contract.
4. Try a smaller `rft` budget (n=4 samples, 1 kept solution) and report the pass@1 versus GPU-minute curve.
5. Add a `--lora-only-attention` ablation (q,k,v,o only) and measure peak reserved memory and pass@1.
