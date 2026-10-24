# 📡 HiveMind@home, GCP Cloud Command Deck & 🐝 La Ruche Community MoE Model Vault (`v2.6.0`)

> **Strictly Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / 0.00 USD`):**
> All curated dataset shards, trained QLoRA domain experts, and merged Branch-Train-Mix (BTM) Sparse Mixture-of-Experts (MoE) checkpoints in **La Ruche** are 100% open-source and freely available to the community.

---

## 1. Overview: From Distributed Dataset Mining to `La Ruche` Community MoE

Project HiveMind (`v2.6.0`) unifies three community pillars into a single 1-command workflow:

1. **⛏️ Distributed Scientific Dataset Mining Campaign (`CAMP-S10-RUCHE-DATASET-MINING` / `src/hivemind/data/dataset_mining_campaign.py`)**:
   - Curates, deduplicates (exact normalized string check + cross-split 4-gram Jaccard verification), and computes Shannon byte entropy across the **8 HiveMind Specialist Domains**:
     - `code_python` (`@xaviercallens`)
     - `math_reasoning` (`@clark719664`)
     - `physics_sim` (`@radialmonster`)
     - `systems_rust` (`@pedromineo`)
     - `cyber_sec` (`@xaviercallens`)
     - `bio_med` (`@clark719664`)
     - `general_chat` (`@radialmonster`)
     - `formal_lean4` (`@pedromineo`)
   - Pins every dataset shard into `ContentAddressedSwarmStore` (`cid://sha256/...`) and leases each shard to a GPU contributor via **Heartbeat-Backed CAS Data Shard Leasing (`HM-S9-06`)** with zero duplicate shard commits (`Theorem 124`).
   - Trains real byte-tokenized `Qwen2` QLoRA/LoRA adapters on GPU (`cuda` when available, deterministic CPU fallback) and compresses outer pseudo-gradient deltas via **Conway-Sloane Gosset $E_8$ 2-Bit Lattice Quantization (`HM-S10-01`)**.

2. **🐝 `La Ruche` ("The Beehive") Community MoE Model Vault (`src/hivemind/swarm/la_ruche_vault.py`)**:
   - Organized as 8 hexagonal **Honeycomb Cells (`Alvéoles`, `ALVEOLE-01`..`ALVEOLE-08`)** in axial Conway-Sloane $A_2$ lattice coordinates `(q, r)` surrounding the central **Sigma-MoE + 3-Tier Hot/Cold Router Core (`(0, 0)`)**.
   - **Gatekeeper Verification**: Every contributed QLoRA adapter must pass finite-tensor checks (`torch.isfinite`), non-zero Frobenius norm checks, SVD spectral energy analysis (`project_lora_rank_svd_with_report`), and a strict **Holdout Perplexity Improvement Gate** (`holdout_nll_after < holdout_nll_before`) before entering **La Ruche**.
   - **Community BTM Sparse MoE Assembly & Release**: Merges all 8 verified Honeycomb Cells into a single `UpcycledSparseMoEModel` (`hivemind_la_ruche_community_moe.safetensors`), generates a Hugging Face Model Card (`README.md`) and `ruche_manifest.json`, and exposes a 1-line Python loader:
     ```python
     from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche

     moe_model, manifest = load_community_moe_from_ruche("artifacts/la_ruche_vault")
     ```

3. **📡 `HiveMind@home` SETI-Style & GCP Cloud Command Dashboard (`src/hivemind/boinc/hivemind_at_home_dashboard.py`)**:
   - **SETI@home Real-Time 60 FPS Spectrum & Lattice Waterfall Analyzer**: Visualizes the 8 specialist MoE frequency channels, Gosset $E_8$ (`12.66 dB` SQNR) & Conway-Sloane $A_2$ quantization envelopes, asynchronous $k$-of-$N$ DiLoCo quorum waves, and 5% two-statistic spot-check verification pulses (`95.36%` compute savings).
   - **GCP Cloud Control Plane + Local CUDA Telemetry**: Displays live status of the GCP Cloud TPU VM (`gwenlaya-tpu-1`, `us-west4-a`, `v5litepod-1`, `READY`), the local CUDA node (`NVIDIA GeForce RTX 5060 Laptop GPU`, `8 GB VRAM`), the Season 9 & 10 GPU Campaign receipts (`CAMP-S9/S10 1..4`), and the **128/128 zero-`sorry` Lean 4 proof shield**.
   - **Interactive SVG Hexagonal Beehive (`La Ruche` Explorer)**: Click any hexagonal `Alvéole` cell to inspect its `safetensors` SHA-256 digest, CAS `cid://`, contributor handle, and holdout perplexity reduction.

---

## 2. Quickstart Commands

### Run the Complete Dataset Mining, `La Ruche` Vault & `HiveMind@home` Dashboard Pipeline
```bash
uv run python scripts/run_hivemind_at_home_and_ruche_campaign.py
```

### Launch the Live FastAPI `HiveMind@home` Web Portal (`http://localhost:8090`)
```bash
uv run python scripts/run_hivemind_at_home_and_ruche_campaign.py --serve --port 8090
```

### Load & Run the Community BTM Sparse MoE Model from `La Ruche`
```python
import torch
from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche

moe_model, manifest = load_community_moe_from_ruche("artifacts/la_ruche_vault")
tokens = torch.tensor([[2, 5, 9, 14, 21, 33, 44, 55]], dtype=torch.long)
outputs = moe_model(input_ids=tokens)
print("Release ID:", manifest["release_id"], "| Logits shape:", outputs.logits.shape)
```
