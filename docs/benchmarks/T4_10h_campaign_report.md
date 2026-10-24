# HiveMind — 10-hour T4 benchmark campaign (2026-10-23/24)

Hardware: Tesla T4 (15 GB, sm_75), 8 vCPU, 31.5 GB RAM; torch 2.7.1+cu118, transformers 5.17.0. Data: Hugging Face WikiText-2, fineweb-edu, codeparrot-clean-valid, GSM8K (local cache). Models: Qwen2.5-0.5B-Instruct (fp16), Qwen2.5-Coder-7B-Instruct (fp16, embeddings on CPU), Gemma-4-E4B (bf16, per-layer embeddings on CPU).

Every experiment below was pre-registered before it ran (files in `artifacts/experiments/*prereg*.json`; deviations and the one amendment are recorded there). Numbers are copied from the receipts listed at the end; 95 % intervals are df = 2 t-intervals over three seeds. A refuted prediction is reported as refuted.

## 1. Campaign timeline

- Started 2026-10-23 20:50 UTC, budget 10.0 h, wall time 10.0 h.
- GPU utilisation (NVML, 30 s samples, 1200 samples): mean 64.3 %, peak memory 14.81 GiB.

| job | status | wall (min) | mean GPU util % | what it was |
| :-- | :-- | --: | --: | :-- |
| `E11_qwen0.5b` | exit 1 | 1.2 | 26.0 | crashed at the end (a code bug: KLT table is a dict); fixed and retried |
| `E11_qwen7b` | exit 1 | 1.8 | 31.3 | same crash; retried |
| `E11_gemma4-e4b` | exit 1 | 5.3 | 26.8 | same crash; retried |
| `E11_qwen0.5b_retry` | ok | 3.3 | 72.9 | registered E11 run (amendment 1) |
| `E11_gemma4-e4b_retry` | ok | 51.9 | 97.7 | registered E11 run |
| `E11_qwen7b_retry` | ok | 15.7 | 73.2 | registered E11 run |
| `E11_aggregate` | exit 1 | 0.2 | - | report written; exit 1 = a negative control failed on the Qwen models (reported) |
| `E12_replay` | exit 1 | 0.3 | 0.0 | crashed in 20 s (wrong torch API path); fixed and retried |
| `E10b_lattice_05b` | exit -15 | 23.9 | 99.7 | first execution, stopped by the lead after one cell (not a registered rule) |
| `E12_replay_retry` | ok | 109.9 | 10.6 | registered E12 run |
| `E10b_amended` | exit 1 | 92.1 | 98.5 | amendment-2 run; exit 1 = positive control failed, run INVALID by its registered rule |
| `E11b_qwen0.5b_fineweb-edu_w1024` | ok | 5.6 | 64.5 | E11b exploratory |
| `E11b_qwen7b_fineweb-edu_w1024` | ok | 21.8 | 76.6 | E11b exploratory |
| `E11b_gemma4-e4b_fineweb-edu_w1024` | ok | 78.3 | 90.6 | E11b exploratory |
| `E13_qwen0.5b` | ok | 41.8 | 86.1 | registered E13 (Qwen2.5-0.5B, 18 cells); all controls passed |
| `E13_granite1b` | exit 3 | 3.9 | 28.4 | stopped by the registered time fail-fast (2.69 s/step with eager experts + checkpointing) |
| `E13_granite3b` | ok | 79.3 | 34.1 | registered E13 feasibility run (Granite-3B, 1 seed) |
| `E13_granite_step_bench` | ok | 2.3 | 34.0 | Granite step-time microbenchmark (decided amendment 4) |
| `E13_qwen0.5b_m2proc` | exit 3 | 1.0 | 43.5 | stopped by the learning gate (M=2 learns less); waiver registered (amendment 5) |
| `E13c_qwen0.5b_rotstraggle` | exit 1 | 15.4 | 89.7 | exploratory diagnostic; exit 1 only because the identity control is absent from its S0+S2 grid |
| `E13_qwen0.5b_m2proc_v2` | exit 1 | 3.9 | 72.4 | two real OS processes: S1 == S0 bit-identical; exit 1 only because the codec-rate control is n/a |
| `E13_granite1b_v2` | killed at deadline | 41.1 | 47.6 | Granite-1B with grouped_mm (amendment 4); cut by the deadline, resumed in the extension queue |
| `E11b_qwen0.5b_code_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_code_w1024` | not started (deadline reached) | - | - |  |
| `E11b_gemma4-e4b_code_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_gsm8k_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_gsm8k_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_news_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_news_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_fiction_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_fiction_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_pubmed_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_pubmed_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_wiki-fr_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen7b_wiki-fr_w1024` | not started (deadline reached) | - | - |  |
| `E11b_gemma4-e4b_gsm8k_w1024` | not started (deadline reached) | - | - |  |
| `E11b_qwen0.5b_wikitext2_w4096` | not started (deadline reached) | - | - |  |
| `E11b_gemma4-e4b_wikitext2_w4096` | not started (deadline reached) | - | - |  |
| `E11b_aggregate` | not started (deadline reached) | - | - |  |

Successful jobs used 6.83 GPU-hours. Failed attempts are listed and explained below; they are not hidden.

## 2. E11 — lattice KV-cache codecs vs FP8 / scalar (RQ10, HM-S10-05)

**Decision (pre-registered, amendment 1):** P11a **refuted (on qwen0.5b, qwen7b, gemma4-e4b)**; P11b **refuted (on gemma4-e4b)**. Controls all passed: False. Receipt `d47e0956eef3a831`.

| model | dtype | peak GiB | baseline NLL | ΔNLL FP8 | ΔNLL INT4 | P11a | P11b | controls |
| :-- | :-- | --: | --: | --: | --: | :-- | :-- | :-- |
| qwen0.5b | float16 | 1.295 | 2.785 | +0.0413 ± 0.0095 | +2.410 ± 0.461 | refuted | inconclusive | failed: negative_A2at8_equals_fp8 |
| qwen7b | float16 | 14.228 | 2.396 | +0.4725 ± 0.1975 | +7.707 ± 0.840 | refuted | supported | failed: negative_A2at8_equals_fp8 |
| gemma4-e4b | bfloat16 | 9.484 | 2.092 | +0.0017 ± 0.0013 | +0.026 ± 0.008 | refuted | refuted (E8 already beats Z without the transform) | all pass |

Paired differences at matched measured rate (ΔNLL(Z) − ΔNLL(lattice), nats/token, positive = lattice better):

| model | R | Z − A₂ (none) | Z − E₈ (none) | Z − E₈ (KLT) |
| :-- | --: | :-- | :-- | :-- |
| qwen0.5b | 2 | -2.5171 ± 1.3890 (invalid (rate not matched)) | +0.3725 ± 0.4861 (CI includes 0) | +0.8861 ± 0.4717 (E8 better) |
| qwen0.5b | 3 | -3.0592 ± 0.1803 (invalid (rate not matched)) | -0.5382 ± 0.0795 (E8 worse) | -1.1913 ± 0.6485 (E8 worse) |
| qwen0.5b | 4 | -0.0939 ± 0.1785 (CI includes 0) | -0.4394 ± 0.0857 (E8 worse) | +0.0648 ± 0.2290 (CI includes 0) |
| qwen7b | 2 | +0.5279 ± 0.9803 (invalid (rate not matched)) | -1.6564 ± 2.1149 (invalid (rate not matched)) | +0.2847 ± 0.1609 (E8 better) |
| qwen7b | 3 | +0.8436 ± 0.4334 (invalid (rate not matched)) | -0.1955 ± 0.5939 (invalid (rate not matched)) | -0.0860 ± 0.2658 (invalid (rate not matched)) |
| qwen7b | 4 | +0.1301 ± 0.6358 (CI includes 0) | +0.6133 ± 0.7446 (CI includes 0) | -0.1614 ± 0.2405 (invalid (rate not matched)) |
| gemma4-e4b | 2 | +0.0057 ± 0.0115 (CI includes 0) | +0.0271 ± 0.0058 (E8 better) | +0.0061 ± 0.0114 (invalid (rate not matched)) |
| gemma4-e4b | 3 | +0.0039 ± 0.0065 (CI includes 0) | +0.0051 ± 0.0068 (CI includes 0) | -0.0008 ± 0.0118 (invalid (rate not matched)) |
| gemma4-e4b | 4 | +0.0054 ± 0.0091 (CI includes 0) | +0.0028 ± 0.0024 (E8 better) | +0.0015 ± 0.0068 (invalid (rate not matched)) |

Descriptive only (decides nothing): does a 4-bit codec reach FP8 (paired ΔNLL difference CI includes 0) at about half the bytes?

| model | A₂@4 | E₈@4 + KLT | Z@4 | INT4 |
| :-- | :-- | :-- | :-- | :-- |
| qwen0.5b | +0.3602 ± 0.0660 | +0.2751 ± 0.1615 | +0.2662 ± 0.1919 | +2.3686 ± 0.4698 |
| qwen7b | +2.2622 ± 0.7772 | +0.4337 ± 0.1773 | +2.3923 ± 0.7239 | +7.2341 ± 0.9948 |
| gemma4-e4b | +0.0012 ± 0.0079 ✓ | +0.0003 ± 0.0030 ✓ | +0.0066 ± 0.0014 | +0.0240 ± 0.0078 |

## 3. E12 — spot-check replay divergence across hardware (RQ11, HM-S10-08)

**Decision (pre-registered):** P12-same **supported** (cell B (same T4, separate process) - a proxy for same class, not a second T4); P12-cross **supported** (cell C (T4 vs this host's x86 CPU, float32 both)); fixed-point clause: not decided (no fixed-point kernel); fp64 + 2^-20 snapping proxy reported in cell D. Controls all passed: True. ε = 0.0001.

| cell | max ‖Δθ‖∞ | min ‖Δθ‖∞ | segments above ε |
| :-- | --: | --: | --: |
| A_same_process_T4 | 0.000e+00 | 0.000e+00 | 0/9 |
| B_separate_process_T4 | 0.000e+00 | 0.000e+00 | 0/9 |
| C_cross_T4_vs_cpu | 6.260e-04 | 2.050e-04 | 9/9 |
| D_proxy_fp64_snap_T4_vs_cpu | 6.342e-04 | 1.984e-04 | 9/9 |
| E_tamper | 2.961e+00 | 2.619e+00 | 9/9 |

Not run on this host: RTX 2070 (sm_75, other host), TPU-VM CPU, a second T4.
Determinism warnings captured per configuration (count): {'cuda_fp32': 0, 'cuda_fp32_subprocess': 0, 'cpu_fp32': 0, 'cuda_fp64_snap': 0}.

## 4. E10b — lattice outer-delta codec at 0.5B (RQ9, HM-S10-01)

**Decision (pre-registered, amendment 2):** P10a **invalid (controls failed: positive_sign_hurts); computed: supported**; P10b **invalid (controls failed: positive_sign_hurts); computed: inconclusive (CI includes 0)**; P10c **invalid (controls failed: positive_sign_hurts); computed: supported**. Controls all passed: False.

Distortion on real 0.5B LoRA deltas (descriptive whatever the validity): relMSE(E₈)/relMSE(ℤ⁸) = 0.869 ± 0.003 per seed [0.8685, 0.8685, 0.8703] (Gaussian prediction 0.86; E10 toy 0.81).

| arm | final held-out loss | rel. MSE | rate (bits/dim) |
| :-- | :-- | --: | --: |
| none | +2.9459 ± 0.0017 | +0.0000 | +32.000 |
| Z8@1+EF | +2.9460 ± 0.0011 | +0.5153 | +0.991 |
| E8@1+EF | +2.9459 ± 0.0009 | +0.4478 | +1.001 |
| sign | +2.9459 ± 0.0025 | +0.3553 | +1.000 |
| E8@6 | +2.9459 ± 0.0016 | +0.0003 | +6.014 |

## 5. E11b — exploratory sweep over Hugging Face domains and context length

E11b report not found (sweep not aggregated).

## 6. E13 — single-GPU swarm simulation (Qwen2.5-0.5B dense; Granite-3.1 MoE 1B and 3B)

Single-GPU swarm over real localhost HTTP (E8's localhost negative control): **no WAN claim**; latencies are GPU time-sharing artefacts. Peers are separate OS processes only where the memory preflight allowed it (see the hosting-fallback count).

### qwen0.5b

Controls all passed: True; failed controls: []; cells failed: []. Hosting fallback cells: 15.

| cell | seeds | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |
| :-- | --: | :-- | --: | --: | :-- | --: |
| S0 sync_baseline | 3 | 2.2334, 2.2312, 2.2308 | 1.00 | 0 | - | 19.8 |
| S1 swarm_identity | 3 | 2.2334, 2.2312, 2.2308 | 1.00 | 0 | 1.00x / 1.00x | 20.2 |
| S2 swarm_all_on | 3 | 2.3856, 2.3801, 2.3506 | 0.50 | 27 | 11.03x / 16.05x | 16.8 |
| S3 no_codec | 3 | 2.3419, 2.3441, 2.3647 | 0.50 | 25 | 1.00x / 1.00x | 16.5 |
| S4 no_rda | 3 | 2.3249, 2.3481, 2.3175 | 0.50 | 26 | 11.03x / 16.05x | 16.8 |
| S5 no_quorum | 3 | 2.2437, 2.2377, 2.2370 | 0.96 | 0 | 11.00x / 16.05x | 24.1 |

Registered predictions: P13a **supported**; P13b **refuted** (retention -0.051 ± 0.350, bar 0.5); P13c **supported** (upload 0.0907 of raw, ideal 0.0623, bar 0.30).

Each mechanism off (descriptive; retention = share of the synchronous improvement kept):

| cell | mean retention | admitted | late |
| :-- | --: | --: | --: |
| S3 no_codec | 0.114 | 0.50 | 25 |
| S4 no_rda | 0.264 | 0.50 | 26 |
| S5 no_quorum | 0.942 | 0.96 | 0 |

### granite3b

Controls all passed: True; failed controls: []; cells failed: []. Hosting fallback cells: 0.

| cell | seeds | final held-out loss (per seed) | admitted | late | upload ratio (actual / ideal) | round latency p50 (s) |
| :-- | --: | :-- | --: | --: | :-- | --: |
| S0 sync_baseline | 1 | 1.7185 | 1.00 | 0 | - | 187.1 |
| S1 swarm_identity | 1 | 1.7185 | 1.00 | 0 | 1.00x / 1.00x | 186.3 |
| S2 swarm_all_on | 1 | 1.8934 | 0.50 | 8 | 11.04x / 16.03x | 153.0 |

Registered predictions: P13a **supported**; P13b **descriptive (n=1)** (per seed {'1': -0.6975575286211182}); P13c **supported** (upload 0.0906 of raw, ideal 0.0624, bar 0.30).

### E13c — rotating straggler (exploratory diagnostic, decides nothing)

Same as S2 but the delayed worker rotates (3, 2, 1) instead of always being the GSM8K worker. Retention of the synchronous improvement: +0.402 ± 0.304 (per seed {1: 0.5332002329341676, 2: 0.29108089995191744, 3: 0.3812219410475062}), against the registered persistent-straggler value in the table above.

### E13 with M=2 separate OS processes (identity control only)

S1 (2 OS processes over HTTP) equals its S0: passed = True, rows = [{'available': True, 'max_abs_diff': 0.0, 'tolerance': 1e-06, 'passed': True, 'bit_identical': True, 'identity_noise_floor': {'available': True, 'max_abs_diff': 0.0, 'bit_identical': True, 'twin_final_loss': 2.3666575253009796, 'note': 'diagnostic only; the identity rule stays S1 vs S0 <= 1e-6'}}].


## 7. Receipts

| artifact | sha256 (first 12) |
| :-- | :-- |
| `E11_kv_lattice_prereg.json` | `f79e06aeb2f1` |
| `E11_kv_lattice_prereg_amendment_1.json` | `2794f8884932` |
| `E11_kv_lattice_report.json` | `be45db7d9106` |
| `E12_replay_divergence_prereg.json` | `9d039cca758d` |
| `E12_replay_divergence_report.json` | `d3c112bf71a3` |
| `E10b_lattice_05b_prereg.json` | `e5102af84574` |
| `E10b_lattice_05b_report.json` | `b65920ad8b78` |
| `E11b_kv_lattice_sweep_prereg.json` | `5c68ffed8f88` |
| `E11b_kv_lattice_sweep_report.json` | `-` |
| `E13_swarm_local_prereg.json` | `4e9a2d800204` |
| `E13_swarm_local_prereg_amendment_1.json` | `9e9ae076864a` |
| `E13_swarm_local_prereg_amendment_2.json` | `820e6ba8a24c` |
| `E13_swarm_local_prereg_amendment_3.json` | `5ed406d70bd4` |
| `E13_swarm_local_report.json` | `8fe42e241e55` |
| `E13_swarm_local_predictions.json` | `b1cbb42e336d` |
| `T4_benchmark_queue_log.json` | `a98d675de634` |
