# NIGHT-MOE-1 results

Base: Qwen2.5-Coder-7B-Instruct Q4_K_M (QLoRA r16 a32 adapters). Exact paired McNemar vs the base model on identical items; alpha 0.05, Bonferroni alpha 0.00294 over 17 transfer/interference cells.

## Subjects

| Subject | Own pack | Base | Adapter | Delta | p | Control | Control p | Verdict |
|---|---|---|---|---|---|---|---|---|
| maths | maths (n=200) | 0.785 | 0.715 | -0.07 | 0.07039 | 0.0 (HARM) | 0.0 | **NO_GAIN** |
| exoplanet | exoplanet (n=600) | 0.3617 | 0.8733 | +0.5117 | 0.0 | 0.3867 (GAIN) | 0.0237 | **VOID** |
| stars | stars (n=317) | 0.489 | 0.8927 | +0.4038 | 0.0 | 0.306 (HARM) | 5e-05 | **VALID** |
| cyber | cyber (n=500) | 0.65 | 0.66 | +0.01 | 0.38331 | 0.658 (NS) | 0.4545 | **NO_GAIN** |

## Transfer and interference cells

| Adapter | Pack | Kind | n | Base (same items) | Adapter | Delta | p | Flag | Bonferroni | Unparsable |
|---|---|---|---|---|---|---|---|---|---|---|
| maths | physics_transfer | transfer | 488 | 0.541 | 0.3422 | -0.1988 | 0.0 | HARM | yes | 155 |
| maths | astro_transfer | interference | 100 | 0.73 | 0.28 | -0.45 | 0.0 | HARM | yes | 54 |
| maths | cyber | interference | 100 | 0.79 | 0.55 | -0.24 | 0.0 | HARM | yes | 30 |
| maths | exoplanet | interference | 100 | 0.38 | 0.58 | +0.2 | 0.0 | GAIN | yes | 0 |
| maths | stars | interference | 100 | 0.46 | 0.6 | +0.14 | 0.00258 | GAIN | yes | 0 |
| exoplanet | astro_transfer | transfer | 152 | 0.7105 | 0.7171 | +0.0066 | 1.0 | NS | no | 0 |
| exoplanet | cyber | interference | 100 | 0.79 | 0.79 | +0.0 | 1.0 | NS | no | 0 |
| exoplanet | physics_transfer | interference | 100 | 0.45 | 0.41 | -0.04 | 0.125 | NS | no | 1 |
| exoplanet | stars | interference | 100 | 0.46 | 0.52 | +0.06 | 0.07031 | NS | no | 0 |
| stars | astro_transfer | transfer | 152 | 0.7106 | 0.7303 | +0.0197 | 0.25 | NS | no | 0 |
| stars | cyber | interference | 100 | 0.79 | 0.79 | +0.0 | 1.0 | NS | no | 0 |
| stars | exoplanet | interference | 100 | 0.38 | 0.38 | +0.0 | 1.0 | NS | no | 0 |
| stars | physics_transfer | interference | 100 | 0.45 | 0.44 | -0.01 | 1.0 | NS | no | 0 |
| cyber | astro_transfer | interference | 100 | 0.73 | 0.73 | +0.0 | 1.0 | NS | no | 1 |
| cyber | exoplanet | interference | 100 | 0.38 | 0.41 | +0.03 | 0.25 | NS | no | 0 |
| cyber | physics_transfer | interference | 100 | 0.45 | 0.46 | +0.01 | 1.0 | NS | no | 1 |
| cyber | stars | interference | 100 | 0.46 | 0.54 | +0.08 | 0.00781 | GAIN | no | 0 |

## Reasons and gaps

- **maths**: NO_GAIN, own-pack change -0.0700, p=0.07039
  - control: the shuffled control scored 0 correct, which is expected for exact-match numeric answers (a shuffled answer is almost never right), so this control cannot gain and the VOID rule is not testable for this subject
- **exoplanet**: VOID, shuffled-label control gained significantly over base (p=0.0237)
  - no result for: maths (stage timed out or was not run)
- **stars**: VALID, own-pack gain +0.4038 (p=0.0); control HARM
  - no result for: maths (stage timed out or was not run)
- **cyber**: NO_GAIN, own-pack change +0.0100, p=0.38331
  - no result for: maths (stage timed out or was not run)
