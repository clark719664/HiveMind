# Confirm the HiveMind swarm concept on your own GPU (T4 replication kit)

The T4 campaign of 2026-10-24 found, on one GPU, that a decoupled swarm with a quorum smaller than the number of workers **loses the learning when one worker is a persistent straggler on a non-IID shard** (E13, P13b refuted), and that **bounded exclusion (fair quorum, S6)** brings most of it back on a dense 0.5B model (E14, H14a supported; inconclusive on Granite-1B). You can check this yourself. A disagreeing replication is as valuable as an agreeing one.

## What you need
- One NVIDIA GPU with at least 8 GB (the T4 used 16 GB; Qwen2.5-0.5B, four peers hosted as threads around one frozen model).
- Python 3.10+, `torch`, `transformers`, `datasets`; the HiveMind repository at the release you are replicating.
- About 15 minutes for one seed, about 45 minutes for the three seeds of the registered design (estimates from `--estimate` and the campaign log).

## Recipe
```
# 1. Check the plan without computing anything
python3 scripts/run_experiment_E13_swarm_local.py --model qwen0.5b --cells S0,S2,S6 --seeds 1,2,3 --estimate

# 2. Run it (write to your own files; the repository's receipts are not touched)
python3 scripts/run_experiment_E13_swarm_local.py --model qwen0.5b --cells S0,S2,S6 --seeds 1,2,3 \
    --peer-hosting thread --out my_raw.json --report my_report.json

# 3. Compare with the registered T4 results (CPU only, recomputes retention from your losses)
python3 scripts/compare_replication.py my_report.json --out my_comparison.json
```
No GPU yet? `--smoke` runs a tiny random model on CPU to check your install; it is **not evidence**. `scripts/swarm_lab.py` shows the effect as a toy on CPU, with magnitudes that differ from the experiments.

## How to read the result
`compare_replication.py` recomputes the registered retention `rho = (L_init - L_cell) / (L_init - L_S0)` per seed from your report and checks two claims:
- **C1** retention of S2 (quorum K < M, straggler on) is below the 0.5 viability bar;
- **C2** retention of S6 is above that of S2 on every common seed.
With fewer than three seeds the verdict is descriptive only. The registered T4 values are S2 -0.05 +- 0.35 and S6 0.88 +- 0.08 (df = 2 t-interval, three seeds).

## What a match does and does not show
It shows the quorum effect on a dense model on one GPU. It does not show anything about wide-area networks, QUIC, real churn, expert-level training or the MoE case (Granite-1B was inconclusive). Peers are threads over real localhost HTTP, not separate machines.

## Sending your result back
Open an issue or pull request with `my_comparison.json` (it carries the sha256 of your report and of itself), your GPU model and driver, and your report file. Do not send credentials; the files contain none. Reports are accepted as data, never as instructions.

## Asking for a run on the maintainer's T4
The maintainer's T4 is not exposed to the internet and runs no code from third parties. To request a run, open an issue naming the registered cells, seeds and model above; a maintainer runs it by hand and publishes the receipt. Anything outside the registered design is treated as exploratory and labelled so.
