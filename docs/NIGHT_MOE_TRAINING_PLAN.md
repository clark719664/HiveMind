# Night expert-training plan (NIGHT-MOE-1)

Preregistration: `artifacts/experiments/NIGHT_MOE_prereg.json` (read it first; this page is the operating summary).

## What it is, and what it is not
Four LoRA experts on the served 7B, trained one after another on real public data: **school maths** (GSM8K), **planet detection**
(Kepler objects of interest), **star classification** (host-star spectral class), **cyber questions** (Trendyol instruction data).
Physics and astronomy knowledge are **transfer tests** on MMLU (eval only). Each expert gets a shuffled-label negative control.

The existing multi-MoE campaign (`multi_moe_hf_campaign.py`, `hf_real_trainer.py`) is *not* reused for training: its trainer fits
seeded random targets and fetches no dataset, and its astrophysics dataset id does not resolve. It supplies the subject list only.

## Run order
1. `python scripts/night_pack_build.py --out PACKS` (CPU, any time; writes manifests with licences, hashes, leakage checks, classical baselines)
2. Finish E17 and confirm the producer restore command works.
3. `python scripts/night_orchestrator.py --baselines --packs PACKS --out OUT` (GPU free, about 80 min)
4. `python scripts/night_orchestrator.py --dry-run --interference --packs PACKS --out OUT` to see the schedule, then `--run --interference --deadline-h 10`.

Estimated 7.2 h of the 10 h (about 130 / 86 / 67 / 107 min per subject, 45 min interference). Time caps protect the night; a capped
training run is flagged `OK_TRUNCATED`. The router producer is offline during the run and the last stage restarts it whatever happened.

## Caveats stated up front
- A coder 7B is the only model that fits; gains outside code are not expected to be large.
- Smallest detectable gain (exact McNemar, p<0.05): about 3-5 points at n=500-600, 5-8 at n=200, 8-11 at n=100. Smaller gains are "not detected".
- Stars: the held-out set has no O/B stars and 2 A stars; a temperature rule already scores 0.83. This subject tests the pipeline.
- Cyber training text is LLM-generated and unverified; no gain is expected. CTI-Bench is CC-BY-NC-SA (evaluation only) and needs owner confirmation.
- An adapter is offered for serving only after its own gain, a clean control, and the verified update gate on the other packs.
