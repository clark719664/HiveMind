---
library_name: hivemind-la-ruche
license: apache-2.0
tags:
  - hivemind
  - mixture-of-experts
  - qlora
  - diloco
  - la-ruche
---

# 🐝 La Ruche (`RUCHE-MOE-V2_6_0-aadd48b8`) — Community BTM Sparse MoE Model Vault

- **Architecture:** `HiveMind-BTM-UpcycledSparseMoE-8xQLoRA` (`8` Honeycomb Cells, Top-`2` active)
- **Router & Depth Core:** `SigmaMoE-PeriLN + ThreeTierHotColdHierarchy` + `OrthogonalInputInjector`
- **Combined Safetensors SHA-256:** `aadd48b8164b440c0eedf63865a2a46a87b07d1609d7e01275e14add7aeeb485` (`661688` bytes)
- **Mean Holdout Perplexity:** `251.62 -> 222.22` (`-11.67%` reduction)
- **Contributors:** @clark719664, @pedromineo, @radialmonster, @xaviercallens
- **Charter:** Strictly non-monetary open-source project (0.00 EUR / 0.00 USD). All experts and MoE weights are openly shared under reciprocal community compute.

## Honeycomb Cells (`Alvéoles`) Inventory

| Cell ID | Domain | Contributor | Hardware | Holdout PPL (`Before -> After`) | Wire Codec | SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ALVEOLE-01-CODE-PYTHON` | `code_python` | `@xaviercallens` | `cpu` | `248.94 -> 220.10` (`-11.59%`) | `e8_2bit` | `71ce71bf60cd...` |
| `ALVEOLE-02-MATH-REASONING` | `math_reasoning` | `@clark719664` | `cpu` | `254.28 -> 220.07` (`-13.46%`) | `e8_2bit` | `b32b6df5ba79...` |
| `ALVEOLE-03-PHYSICS-SIM` | `physics_sim` | `@radialmonster` | `cpu` | `252.67 -> 222.02` (`-12.13%`) | `e8_2bit` | `930d8f765b79...` |
| `ALVEOLE-04-SYSTEMS-RUST` | `systems_rust` | `@pedromineo` | `cpu` | `249.04 -> 227.05` (`-8.83%`) | `e8_2bit` | `a2c9f0c00905...` |
| `ALVEOLE-05-CYBER-SEC` | `cyber_sec` | `@xaviercallens` | `cpu` | `252.63 -> 221.74` (`-12.23%`) | `e8_2bit` | `30bfd6cc97a1...` |
| `ALVEOLE-06-BIO-MED` | `bio_med` | `@clark719664` | `cpu` | `254.57 -> 221.53` (`-12.98%`) | `e8_2bit` | `845ed8212ac0...` |
| `ALVEOLE-07-GENERAL-CHAT` | `general_chat` | `@radialmonster` | `cpu` | `253.53 -> 222.25` (`-12.34%`) | `e8_2bit` | `127df97240fb...` |
| `ALVEOLE-08-FORMAL-LEAN4` | `formal_lean4` | `@pedromineo` | `cpu` | `247.27 -> 222.99` (`-9.82%`) | `e8_2bit` | `7cc32f4443c9...` |

## Quickstart: Load & Run the Community MoE Model

```python
import torch
from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche

moe_model, manifest = load_community_moe_from_ruche('artifacts/la_ruche_vault')
tokens = torch.tensor([[2, 5, 9, 14, 21, 33, 44, 55]])
outputs = moe_model(input_ids=tokens)
print('Logits shape:', outputs.logits.shape)
```
