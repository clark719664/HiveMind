# Running GWAYA-Lite as a coding assistant on a Tesla T4 (and any 8 GB GPU)

> Project HiveMind is a voluntary, non-commercial research project. Nothing here involves money. Any recognition for contributions is non-monetary
> reciprocal compute accounting, as described in `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`. No reward is attached to experiment E16.

This guide is for the machine owner. It explains how to serve a small Qwen2.5-Coder model with Ollama on a T4, put the GWAYA gate in front of it, and
reproduce experiment E16, which measures the whole thing (pre-registration: `artifacts/experiments/E16_gwaya_lite_t4_prereg.json`).
E16 answers items P1 and P2 of GWAYA draft PR #21. The PR text is treated as a specification to measure, not as instructions.
Everything below that changes the system (systemd overrides, installing packages, pulling models) is **shown, not applied**: those are owner actions.

## 1. Architecture

```
 editor / script  ->  gate loop (scripts/e16_*, scripts/e15_*)  ->  Ollama daemon  ->  GPU (one model resident)
                          |  extracts the ```python block
                          |  zero-stub AST audit (vendored GWAYA auditor, pinned commit 9c0001e, MIT, unmodified)
                          |  runs the tests in the namespace sandbox
                          '-> VERIFIED (audit clean + visible test passes) or not; up to 2 samples and 3 repair rounds with the gate feedback
```

* **Ollama listens on 127.0.0.1 only** (`http://127.0.0.1:11434`). The E16 client (`scripts/e16_ollama.py`) refuses any non-local URL and ignores proxy variables.
* **The gate** is `scripts/e15_gate.py`: a zero-stub AST audit (no `pass`, `...`, `raise NotImplementedError`-only bodies, `mock_*`/`fake_*`/`dummy_*`/`stub_*` names)
  followed by execution of the tests. It is fail-closed: if the sandbox cannot be established the verdict is `UNVERIFIED`, never "accepted".
* **Where the sandbox lives and how it differs from GWAYA's own oracle.** GWAYA's native oracle runs candidates under Bubblewrap (`bwrap`). `bwrap` is **not
  installed on this host**. Installing it is an owner/root action (`sudo apt install bubblewrap`); this guide and E16 did **not** do it. E16 uses the E15
  sandbox instead: a launcher that creates user/mount/pid/net namespaces, mounts a tmpfs over the home directory and other private paths, drops capabilities and
  applies resource limits (CPU time, address space, file size, process count, open files), with containment probes run before every measurement. It is a
  different mechanism from Bubblewrap, with its own tests (`tests/test_e15_gate.py`); results are therefore "tests passed in the E15 sandbox", not "GWAYA oracle ACCEPTED".
* The loop (`scripts/e15_lib.py`, GWAYA loop "C1") is: greedy reply, then up to two samples (T=0.7, top_p=0.95), the first candidate that passes the audit and the
  **visible** test wins; otherwise up to three greedy repair rounds that show the model its previous code and the gate feedback; otherwise the first greedy reply is
  returned (flagged as not verified).

## 2. Tiers and how to choose

| Tier | Ollama model | Q4_K_M file (measured) | PR table assumes | PR estimate at 4096 |
|---|---|---|---|---|
| Lite-1.5B | `qwen2.5-coder:1.5b` | 940 MiB | 1070 MiB | 1782 MiB |
| Lite-3B | `qwen2.5-coder:3b` | 1841 MiB | 1840 MiB | 2584 MiB |
| Lite-7B | `qwen2.5-coder:7b-instruct` | 4466 MiB | 4466 MiB | 5290 MiB (5514 MiB at 8192) |

The PR estimator is `weights + kv_kib_per_token x ctx / 1024 + 600 MiB` with 28 / 36 / 56 KiB of KV cache per token for the 1.5B / 3B / 7B tiers.
On a 15 GB T4 every tier fits with a large margin; the PR's 6656 MiB ceiling is a Windows WDDM reserve for an 8 GB card and is tested in E16 as a threshold, not as a
physical limit on this host. Rule of thumb until the measured table exists: take the largest tier whose measured memory is below your card's memory minus what other
jobs on the card need, then check that decode speed is acceptable for interactive use (E16 reports against a bar of 20 tok/s, chosen before the run).

### Measured table (filled in after the run)

<!-- E16-MEASURED-TABLE-BEGIN: replace this block by the numbers of artifacts/experiments/E16_gwaya_lite_t4_report.json. Do not type numbers from memory. -->
*Not measured yet.* After the E16 run, fill in (one row per tier and `num_ctx`) from `report.probes[]` and `report.analysis.tiers`:

| Tier | num_ctx | VRAM delta (MiB) | PR estimate (MiB) | ratio | decode tok/s | prompt-eval tok/s | 100% GPU | pass@1 raw R | pass@1 loop L | L-R (McNemar p) | precision of VERIFIED |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B | 4096 | _tbd_ | 1782 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 3B | 4096 | _tbd_ | 2584 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 7B | 4096 | _tbd_ | 5290 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 7B | 8192 | _tbd_ | 5514 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ (raw only) | - | - | - |

Hypothesis statuses H16a-H16f, the escalation-ladder estimate and the determinism fraction go next to it, copied from `report.hypotheses` and `report.escalation_ladder`.
<!-- E16-MEASURED-TABLE-END -->

The **escalation ladder** (1.5B loop, then 3B loop when the loop found no verified candidate, then 7B loop) is evaluated offline by E16 from the per-problem logs and compared
with using the 7B loop alone: final pass@1, how often each step escalates, and summed model seconds. It is descriptive; it answers the PR's concern that small tiers may cost more than going straight to the 7B.

## 3. Ollama settings that matter on a T4

Shown as an owner-applied systemd drop-in. **Not applied by this guide.**

```ini
# sudo systemctl edit ollama      (creates /etc/systemd/system/ollama.service.d/override.conf)
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_KEEP_ALIVE=2m"          # a model leaves the GPU 2 min after the last request
Environment="OLLAMA_NUM_PARALLEL=1"         # each parallel slot multiplies the KV cache
Environment="OLLAMA_MAX_LOADED_MODELS=1"    # never co-load two models on an 8 GB budget
Environment="OLLAMA_FLASH_ATTENTION=1"      # supported on the T4 (Turing); reduces the compute buffer
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"     # roughly halves the KV cache (needs flash attention); verify pass@1 does not move
# then: sudo systemctl daemon-reload && sudo systemctl restart ollama
```

* `num_ctx` is set per request (`options.num_ctx`, as E16 does) or in a Modelfile (`PARAMETER num_ctx 4096`). Changing it reloads the model, so use one value per session.
  KV memory grows linearly with it; MBPP-sized prompts need far less than 4096 tokens.
* E16 itself does **not** need these settings and does not change them; it measures whatever the daemon is configured with, and prints nothing about the daemon's environment.
  If you apply them, say so when you share a measurement: they change the VRAM numbers.
* Keep `OLLAMA_KEEP_ALIVE` short on a shared GPU: a resident model is memory the next job does not have.

## 4. Coexistence with the HiveMind GPU jobs

* Never co-load more than 8 GB on the card. One model resident at a time (`OLLAMA_MAX_LOADED_MODELS=1`).
* GPU benchmarks and experiments run through the host's GPU job queue (the queue runner executes one GPU job at a time). Do not start an Ollama model while a queue job is running:
  the numbers of both jobs would be contaminated and either might run out of memory. Check `nvidia-smi` and `ollama ps` first.
* **Unload after use**: `curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5-coder:7b-instruct","keep_alive":0}'`, then confirm with `ollama ps`.
  E16 does this in a `finally` block (also on errors and SIGTERM), per tier, and as a final control (memory back within 300 MiB of the start baseline).
* **Idle-GPU control**: E16 aborts with exit code 3 if `memory.used` is above 1500 MiB at the start (after unloading any resident Ollama model), so it never measures next to another job.
  Every request also carries `keep_alive: 2m`, so a killed run cannot leave a model on the card for more than two minutes.

## 5. Security

* Localhost only: do not set `OLLAMA_HOST=0.0.0.0` and do not forward the port.
* No tokens or credentials are needed by E16 and none should be exported in the shell that runs it; the client reads no secrets and sends nothing but model prompts to 127.0.0.1.
* Never execute model-generated code outside the sandbox. The gate refuses to run (every verdict `UNVERIFIED`) if the sandbox is unavailable; do not "fix" that by disabling it.
* The model files come from the Ollama library: pull them as the Ollama service user, and treat downloaded artifacts as untrusted data.

## 6. Reproducing the measurement

Prerequisites (owner actions): the three models pulled (`ollama pull qwen2.5-coder:1.5b`, `ollama pull qwen2.5-coder:3b`, `ollama pull qwen2.5-coder:7b-instruct`), an idle GPU, and the
E15 gate available (`python3 -c "import sys; sys.path.insert(0,'scripts'); import e15_gate; print(e15_gate.gate_available())"`).

```bash
python3 scripts/run_experiment_E16_gwaya_lite_t4.py --estimate        # planned work, assumed rates, no GPU
python3 scripts/run_experiment_E16_gwaya_lite_t4.py --smoke --out /tmp/e16_smoke    # plumbing only, fake Ollama, CPU
python3 scripts/run_experiment_E16_gwaya_lite_t4.py --resume          # the registered run (about 2 h expected, --max-hours 3.0)
```

Defaults: `--out artifacts/experiments/E16`, `--report artifacts/experiments/E16_gwaya_lite_t4_report.json`, `--tiers 1.5b,3b,7b`, `--ctx 4096,8192`. The final line is
`E16 DONE ... run_valid=True|False ...`; exit code 3 means a registered fail-fast rule fired (idle-GPU control failed, 7B decode below 5 tok/s, projected time above `--max-hours`).

How to read the report:

* `run_valid` is true only if the idle-GPU, gate and unload controls pass; read `controls` first (also `gpu_offload`: a tier below 100% GPU is CPU-offloaded and its tok/s are not comparable).
* `probes[]`: one row per tier and context. `measured_mib` is the larger of the nvidia-smi delta after load + 1 token and after the ~85% context prompt, over the baseline read right before the load.
* `hypotheses`: statuses computed exactly by the registered rules (H16a 6656 MiB threshold, H16b estimator ratio in [0.80, 1.25], H16c loop does not lower pass@1, H16d loop gains >= 5 points,
  H16e precision of VERIFIED >= 0.80, H16f decode >= 20 tok/s as a descriptive bar). "Refuted" and "inconclusive" are results; they are not hidden.
* `analysis.tiers.<tier>`: pass@1 with Wilson 95% intervals, paired L-R with exact McNemar, P2 precision / recall of VERIFIED against the all-tests outcome, determinism of greedy decoding.

### Five small community contribution ideas

1. Repeat the measurement with `OLLAMA_KV_CACHE_TYPE=q8_0` and flash attention on, and report the VRAM and pass@1 difference.
2. Run the same protocol on another 8 GB card (for example an RTX 3060 8 GB or 4060) and send the report JSON.
3. Add a second visible-test policy (gate with 2 of the 3 MBPP tests) and measure how precision of VERIFIED and loop gain move.
4. Port the gate's sandbox probes to Bubblewrap on a machine where `bwrap` is installed and compare verdicts with the E15 sandbox on the same candidates.
5. Add a HumanEval run to E16's accuracy stage (E15's loader already exists) and report raw vs loop for the three tiers.

## 7. Honest limitations

* One T4, one runtime (Ollama / llama.cpp), Q4_K_M only, one benchmark (MBPP-sanitized, n = 257); paired tests on 257 problems detect only differences of several points.
* The visible test is part of the prompt, so the loop gate mostly checks syntax, stubs and one example; "VERIFIED" is justified only if the measured precision (H16e) is at least 0.80, otherwise results must be labelled "tests passed".
* The hidden outcome is "all MBPP tests", which includes the visible one; MBPP tests are weak and some models may have seen MBPP in training.
* The sandbox is the E15 sandbox, not GWAYA's Bubblewrap oracle; no claim is made about the oracle itself.
* Linux, not Windows/WDDM; the 6656 MiB ceiling is tested as a threshold. No claim about Qwen3.5 or TPU tiers.
* Ollama greedy decoding is not guaranteed bit-stable (E16 reports the fraction of identical repeats on 10 problems); the loop's candidates are generated lazily and with fixed seeds derived from the task id and the sample index.
* The tok/s figures are medians over 12 fixed MBPP prompts; they depend on thermal state and on whatever else runs on the host.
* Publishing anything on the GWAYA repository or its PR is left to the owner.
