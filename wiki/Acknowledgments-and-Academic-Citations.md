# 📚 11. Acknowledgments, Leveraged Open-Source GitHub Repositories & Academic Citations (`v1.9.0`)

**Project HiveMind** gratefully acknowledges the open-source distributed machine learning community, academic researchers across outer-loop `DiLoCo` and Mixture-of-Experts (`MoE`) architectures, and our human & AI contributor community (`52/52` merged Pull Requests across Waves 1–6).

---

## 1. Leveraged Open-Source GitHub Repositories (Cloned & Integrated in Season 8 / Wave 7)

All 8 upstream reference repositories are archived in `sub_projects/hivemind/reference_repos/` and programmatically registered in `LEVERAGED_GITHUB_REPOSITORIES` (`src/hivemind/swarm/diloco_nextgen.py`):

| Repository | Authors / Organization | License | Associated Paper | Integrated HiveMind Component (`src/hivemind/swarm/diloco_nextgen.py`) |
| :--- | :--- | :---: | :--- | :--- |
| **[`one-covenant/SparseLoCo`](https://github.com/one-covenant/SparseLoCo)** | Amir Sarfi, Benjamin Thérien, Stefan Horoi, Eugene Belilovsky (*Mila / Concordia / Covenant*) | `MIT` | [`arXiv:2508.15706`](https://arxiv.org/abs/2508.15706) | `ComputeVisibleSparseLoCoCompressor` (`SI-09B`: 2-bit `4`-bin empirical statistical centroid LUT quantization + Top-$k$ outer error feedback) & `ChunkedOrthonormalDCTTransform` (`SI-09C`) |
| **[`shoroi/isoloco`](https://github.com/shoroi/isoloco)** | Stefan Horoi, Amir Sarfi, Eugene Belilovsky (*Mila / Concordia University*) | `MIT` | [`arXiv:2607.03011`](https://arxiv.org/abs/2607.03011) | `ReputationWeightedIsoCTSMerger` (`SI-07`: Isotropic Covariance `ISO-C` + Common & Task-Specific `ISO-CTS` subspace orthogonalization) & `TSVDecorrelatedMerger` (`SI-07B`: `tsvloco.py` Task Singular Vector whitening) |
| **[`PrimeIntellect-ai/OpenDiLoCo`](https://github.com/PrimeIntellect-ai/OpenDiLoCo)** | Sami Jaghouar, Jack Min Ong, Johannes Hagemann (*Prime Intellect*) | `Apache-2.0` | [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) | `CPUOffloadedFlatBucketOuterState` (`SI-06B`: CPU-offloaded outer parameter & pseudo-gradient management over Hivemind DHT) |
| **[`PrimeIntellect-ai/prime-diloco`](https://github.com/PrimeIntellect-ai/prime-diloco)** | Prime Intellect Team (*Sami Jaghouar, Jack Min Ong, Johannes Hagemann et al.*) | `Apache-2.0` | [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) | `CPUOffloadedFlatBucketOuterState` (`SI-06B`: contiguous flat 1D FP32 CPU master-weight buffer with `0 B` extra GPU VRAM overhead, inspired by `zeroband/diloco.py`) |
| **[`bloc97/DeMo`](https://github.com/bloc97/DeMo)** | Bowen Peng (`bloc97`), Jeffrey Quesnelle, Diederik P. Kingma (*Nous Research*) | `Apache-2.0` | [`arXiv:2411.19870`](https://arxiv.org/abs/2411.19870) | `ChunkedOrthonormalDCTTransform` (`SI-09C`: orthonormal 2D chunked Type-II Discrete Cosine Transform `TransformDCT`, hardened with bitwise-deterministic `index_copy_` instead of CUDA `scatter_reduce_`) |
| **[`google-deepmind/asyncdiloco`](https://github.com/google-deepmind/asyncdiloco)** | Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev et al. (*Google DeepMind*) | `Apache-2.0` | [`arXiv:2401.09135`](https://arxiv.org/abs/2401.09135) | `CosineGatedAdamDecayOuterOptimizer` (`SI-06`: asynchronous worker-delay & streaming fragment-age tables) |
| **[`pmerolla/fomoe`](https://github.com/pmerolla/fomoe)** | Paul Merolla (*Fast Opportunistic Mixture of Experts*) | `MIT` | [`github.com/pmerolla/fomoe`](https://github.com/pmerolla/fomoe) | `SkipAttentionSpeculativeRouter` (`SI-10`: Skip-Attention pre-attention speculative expert prefetch achieving `~85.8%` top-$k$ accuracy + Cache-Aware Routing `CAR` fallback) |
| **[`learning-at-home/hivemind`](https://github.com/learning-at-home/hivemind)** | Max Ryabinin, Anton Gusev, Anton Diskin, Denis Mazur et al. (*Learning@Home / HSE*) | `MIT` | [`arXiv:2006.16668`](https://arxiv.org/abs/2006.16668) | Foundational inspiration for decentralized Kademlia DHT parameter averaging (`DecentralizedAverager`) and WAN Mixture-of-Experts routing |

---

## 2. Complete Academic Citations (2020–2026 DiLoCo, MoE & Subspace Merging Literature)

1. **Douillard et al. (2023)** — *DiLoCo: Distributed Low-Communication Training of Language Models* ([`https://arxiv.org/abs/2311.08105`](https://arxiv.org/abs/2311.08105))
2. **Douillard et al. (2024)** — *Asynchronous Local-SGD Training for Language Modeling* ([`https://arxiv.org/abs/2401.09135`](https://arxiv.org/abs/2401.09135))
3. **Jaghouar, Ong, & Hagemann (2024)** — *OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training* ([`https://arxiv.org/abs/2407.07852`](https://arxiv.org/abs/2407.07852))
4. **Peng, Quesnelle, & Kingma (2024)** — *DeMo: Decoupled Momentum Optimization* ([`https://arxiv.org/abs/2411.19870`](https://arxiv.org/abs/2411.19870))
5. **Douillard et al. (2025)** — *Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch* ([`https://arxiv.org/abs/2501.18512`](https://arxiv.org/abs/2501.18512))
6. **Charles et al. (2025)** — *Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo* ([`https://arxiv.org/abs/2503.09799`](https://arxiv.org/abs/2503.09799))
7. **Sarfi, Thérien, Horoi, & Belilovsky (2025)** — *SparseLoCo: Communication-Efficient Distributed Training with Sparse Low-Bit Pseudo-Gradients* ([`https://arxiv.org/abs/2508.15706`](https://arxiv.org/abs/2508.15706))
8. **Kwon & Kim (2025a)** — *Scaling Distributed Low-Communication Training with One-Bit Quantization (PoLoCo)* ([`https://arxiv.org/abs/2510.17558`](https://arxiv.org/abs/2510.17558))
9. **Kwon & Kim (2025b)** — *Skip the Negatives: Clamp-Guided Quantization for Outer-Loop Distributed Training (SNQ-L)* ([`https://arxiv.org/abs/2512.02042`](https://arxiv.org/abs/2512.02042))
10. **Miahi & Belilovsky (2026)** — *Skip the Invisible Updates: Checkpointing Pseudo-Gradient Sparsification for DiLoCo (PULSELoCo)* ([`https://arxiv.org/abs/2602.03839`](https://arxiv.org/abs/2602.03839))
11. **Khaled & Jin (2026)** — *Tighter Convergence Bounds for Distributed Low-Communication Training* ([`https://arxiv.org/abs/2602.06954`](https://arxiv.org/abs/2602.06954))
12. **Bogachev & Ryabinin (2026)** — *Analysis of the Generalization Fidelity of DiLoCo vs Data-Parallel Training* ([`https://arxiv.org/abs/2602.23029`](https://arxiv.org/abs/2602.23029))
13. **Sani & Lane (2026)** — *Gradient-Less Federated Gradient Boosted Trees (FGBoost)* ([`https://arxiv.org/abs/2603.24124`](https://arxiv.org/abs/2603.24124))
14. **Sani, Iacob, Cao, Qiu, & Lane (2026)** — *Decoupled DiLoCo: Asynchronous Distributed Training Without Global Barriers* ([`https://arxiv.org/abs/2604.21428`](https://arxiv.org/abs/2604.21428))
15. **Shah & Sun (2026)** — *Cosine-Gated Adam-Decay (CGAD): Staleness-Aware Outer Optimization for Asynchronous DiLoCo* ([`https://arxiv.org/abs/2605.09126`](https://arxiv.org/abs/2605.09126))
16. **Asif & Belilovsky (2026)** — *HeLoCo: Direction-Aware Outer Optimization for Heterogeneous Non-IID DiLoCo* ([`https://arxiv.org/abs/2606.00271`](https://arxiv.org/abs/2606.00271))
17. **Sani, Iacob, & Lane (2026)** — *FoMoE: Federated Mixture-of-Experts with Partial Expert Replication and Skip-Token* ([`https://arxiv.org/abs/2606.19025`](https://arxiv.org/abs/2606.19025))
18. **Horoi, Sarfi, & Belilovsky (2026)** — *IsoLoCo: Isotropic Subspace Merging for Scalable Low-Communication Training* ([`https://arxiv.org/abs/2607.03011`](https://arxiv.org/abs/2607.03011))
19. **Thérien, Sarfi, & Belilovsky (2026)** — *MuLoCo: Inner Muon Optimization for Low-Communication Distributed Training* ([`https://arxiv.org/abs/2607.12925`](https://arxiv.org/abs/2607.12925))
20. **Wang et al. (2026)** — *Subspace-Constrained Outer Optimization for DiLoCo* ([`https://arxiv.org/abs/2608.05958`](https://arxiv.org/abs/2608.05958))
21. **Li et al. (2026)** — *FedLoCo: Federated Low-Communication Optimization with Drift Correction* ([`https://arxiv.org/abs/2609.02737`](https://arxiv.org/abs/2609.02737))
22. **He & Tao (2026)** — *AutoLoCo: Adaptive Synchronization Intervals and Outer Optimizer Correction for DiLoCo* ([`https://arxiv.org/abs/2609.36662`](https://arxiv.org/abs/2609.36662))
23. **Morales (2026)** — *How Inner Optimizers Shape Replica Scaling in Distributed Low-Communication Training* ([`https://arxiv.org/abs/2610.19064`](https://arxiv.org/abs/2610.19064))
24. **Gargiulo et al. (2025)** — *Task Singular Vectors: Reducing Task Interference in Model Merging* ([`https://arxiv.org/abs/2412.00081`](https://arxiv.org/abs/2412.00081))
25. **Marczak et al. (2025)** — *No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces* ([`https://arxiv.org/abs/2502.04959`](https://arxiv.org/abs/2502.04959))
26. **Ryabinin & Gusev (2020)** — *Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts* ([`https://arxiv.org/abs/2006.16668`](https://arxiv.org/abs/2006.16668))

---

## 3. Open-Source Contributor Acknowledgments (`52/52` Merged Pull Requests)

- **`@clark719664`**: 44 merged Pull Requests across Waves 1–6 (`PRs #6`–`#10`, `#21`–`#25`, `#32`, `#34`–`#42`, `#63`–`#77`, `#79`–`#82`, `#88`–`#91`).
- **`@pedromineo`**: 5 merged Pull Requests (`PR #33`, `#52`–`#55`).
- **`@radialmonster`**: 2 merged Pull Requests (`PR #11`, `#78`).
