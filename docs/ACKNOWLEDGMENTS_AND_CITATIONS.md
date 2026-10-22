# Acknowledgments, Open-Source Attribution & Academic Citations (`v1.9.0`)

**Project HiveMind** stands on the shoulders of the open-source distributed machine learning community, academic researchers across decentralized optimization and Mixture-of-Experts (`MoE`) architectures, and our human & AI contributor community. This document formally acknowledges every external repository, algorithm, dataset, and research paper integrated into or leveraged by **Project HiveMind** (`src/hivemind/swarm/diloco_nextgen.py`, `src/hivemind/swarm/strategy_optimizer.py`, `src/hivemind/core/`, and `lean4/HiveMindSpec.lean`).

---

## 1. Leveraged Open-Source GitHub Repositories (Cloned & Integrated in Season 8 / Wave 7)

All 8 upstream reference repositories are archived locally in `/home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/reference_repos/` and programmatically registered in [`LEVERAGED_GITHUB_REPOSITORIES`](../src/hivemind/swarm/diloco_nextgen.py):

| Repository | Authors / Organization | License | Associated Paper | Integrated HiveMind Component (`src/hivemind/swarm/diloco_nextgen.py`) |
| :--- | :--- | :---: | :--- | :--- |
| **[`one-covenant/SparseLoCo`](https://github.com/one-covenant/SparseLoCo)** | Amir Sarfi, Benjamin Thérien, Stefan Horoi, Eugene Belilovsky (*Mila / Concordia University / Covenant*) | `MIT` | [`arXiv:2508.15706`](https://arxiv.org/abs/2508.15706) | [`ComputeVisibleSparseLoCoCompressor`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-09B`: 2-bit `4`-bin empirical statistical centroid LUT quantization + Top-$k$ outer error feedback) & [`ChunkedOrthonormalDCTTransform`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-09C`) |
| **[`shoroi/isoloco`](https://github.com/shoroi/isoloco)** | Stefan Horoi, Amir Sarfi, Eugene Belilovsky (*Mila / Concordia University*) | `MIT` | [`arXiv:2607.03011`](https://arxiv.org/abs/2607.03011) | [`ReputationWeightedIsoCTSMerger`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-07`: Isotropic Covariance `ISO-C` + Common & Task-Specific `ISO-CTS` subspace orthogonalization) & [`TSVDecorrelatedMerger`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-07B`: `tsvloco.py` Task Singular Vector whitening) |
| **[`PrimeIntellect-ai/OpenDiLoCo`](https://github.com/PrimeIntellect-ai/OpenDiLoCo)** | Sami Jaghouar, Jack Min Ong, Johannes Hagemann (*Prime Intellect*) | `Apache-2.0` | [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) | [`CPUOffloadedFlatBucketOuterState`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-06B`: CPU-offloaded outer parameter & pseudo-gradient management over Hivemind DHT) |
| **[`PrimeIntellect-ai/prime-diloco`](https://github.com/PrimeIntellect-ai/prime-diloco)** | Prime Intellect Team (*Sami Jaghouar, Jack Min Ong, Johannes Hagemann et al.*) | `Apache-2.0` | [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) | [`CPUOffloadedFlatBucketOuterState`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-06B`: contiguous flat 1D FP32 CPU master-weight buffer with `0 B` extra GPU VRAM overhead, inspired by `zeroband/diloco.py`) |
| **[`bloc97/DeMo`](https://github.com/bloc97/DeMo)** | Bowen Peng (`bloc97`), Jeffrey Quesnelle, Diederik P. Kingma (*Nous Research*) | `Apache-2.0` | [`arXiv:2411.19870`](https://arxiv.org/abs/2411.19870) | [`ChunkedOrthonormalDCTTransform`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-09C`: orthonormal 2D chunked Type-II Discrete Cosine Transform `TransformDCT`, hardened with bitwise-deterministic `index_copy_` instead of CUDA `scatter_reduce_`) |
| **[`google-deepmind/asyncdiloco`](https://github.com/google-deepmind/asyncdiloco)** | Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen (*Google DeepMind*) | `Apache-2.0` | [`arXiv:2401.09135`](https://arxiv.org/abs/2401.09135) | [`CosineGatedAdamDecayOuterOptimizer`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-06`: asynchronous worker-delay & streaming fragment-age tables) |
| **[`pmerolla/fomoe`](https://github.com/pmerolla/fomoe)** | Paul Merolla (*Fast Opportunistic Mixture of Experts*) | `MIT` | [`github.com/pmerolla/fomoe`](https://github.com/pmerolla/fomoe) | [`SkipAttentionSpeculativeRouter`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-10`: Skip-Attention pre-attention speculative expert prefetch achieving `~85.8%` top-$k$ accuracy + Cache-Aware Routing `CAR` fallback) |
| **[`learning-at-home/hivemind`](https://github.com/learning-at-home/hivemind)** | Max Ryabinin, Anton Gusev, Anton Diskin, Denis Mazur et al. (*Learning@Home / Yandex Research / HSE*) | `MIT` | [`arXiv:2006.16668`](https://arxiv.org/abs/2006.16668) | Foundational inspiration for decentralized Kademlia DHT parameter averaging (`DecentralizedAverager`) and WAN Mixture-of-Experts routing |

---

## 2. Complete Academic Citations (2020–2026 DiLoCo, MoE & Subspace Merging Literature)

### 2.1 Core Outer-Loop `DiLoCo` & Asynchronous Scaling Papers
1. **Douillard, A., Feng, Q., Rusu, A. A., Chhaparia, R., Donchev, Y., Kuncoro, A., Ranzato, M., Szlam, A., & Shen, J. (2023).** *DiLoCo: Distributed Low-Communication Training of Language Models.* [`https://arxiv.org/abs/2311.08105`](https://arxiv.org/abs/2311.08105)
2. **Douillard, A., Feng, Q., Rusu, A. A., Chhaparia, R., Donchev, Y., Kuncoro, A., Ranzato, M., Szlam, A., & Shen, J. (2024).** *Asynchronous Local-SGD Training for Language Modeling.* [`https://arxiv.org/abs/2401.09135`](https://arxiv.org/abs/2401.09135) — Code: [`https://github.com/google-deepmind/asyncdiloco`](https://github.com/google-deepmind/asyncdiloco)
3. **Jaghouar, S., Ong, J. M., & Hagemann, J. (2024).** *OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training.* [`https://arxiv.org/abs/2407.07852`](https://arxiv.org/abs/2407.07852) — Code: [`https://github.com/PrimeIntellect-ai/OpenDiLoCo`](https://github.com/PrimeIntellect-ai/OpenDiLoCo) & [`https://github.com/PrimeIntellect-ai/prime-diloco`](https://github.com/PrimeIntellect-ai/prime-diloco)
4. **Douillard, A., Donchev, Y., Rush, K., Kale, S., Charles, Z., Garrett, Z., Testuggine, D., Ramasesh, V., & Shen, J. (2025).** *Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch.* [`https://arxiv.org/abs/2501.18512`](https://arxiv.org/abs/2501.18512)
5. **Charles, Z., Testuggine, D., Garrett, Z., Rush, K., Donchev, Y., Shen, J., & Douillard, A. (2025).** *Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo.* [`https://arxiv.org/abs/2503.09799`](https://arxiv.org/abs/2503.09799)

### 2.2 Staleness-Aware, Barrier-Free & Heterogeneous Outer Optimizers (2026)
6. **Sani, L., Iacob, A., Cao, Z., Qiu, X., & Lane, N. D. (2026).** *Decoupled DiLoCo: Asynchronous Distributed Training Without Global Barriers.* [`https://arxiv.org/abs/2604.21428`](https://arxiv.org/abs/2604.21428) (CaMLSys / University of Cambridge & Flower Labs)
7. **Shah, N., & Sun, M. (2026).** *Cosine-Gated Adam-Decay (CGAD): Staleness-Aware Outer Optimization for Asynchronous DiLoCo.* [`https://arxiv.org/abs/2605.09126`](https://arxiv.org/abs/2605.09126) — Integrated as `SI-06` (`CosineGatedAdamDecayOuterOptimizer`) and proven in Lean 4 (`Theorem 73`).
8. **Asif, M., & Belilovsky, E. (2026).** *HeLoCo: Direction-Aware Outer Optimization for Heterogeneous Non-IID DiLoCo.* [`https://arxiv.org/abs/2606.00271`](https://arxiv.org/abs/2606.00271) — Integrated as `SI-08A` (`HeLoCoDirectionCorrector`).
9. **He, Z., & Tao, L. (2026).** *AutoLoCo: Adaptive Synchronization Intervals and Outer Optimizer Correction for DiLoCo.* [`https://arxiv.org/abs/2609.36662`](https://arxiv.org/abs/2609.36662) — Integrated as `SI-08B` (`AutoLoCoIntervalCorrector`).
10. **Li, H., et al. (2026).** *FedLoCo: Federated Low-Communication Optimization with Drift Correction.* [`https://arxiv.org/abs/2609.02737`](https://arxiv.org/abs/2609.02737)
11. **Khaled, A., & Jin, C. (2026).** *Tighter Convergence Bounds for Distributed Low-Communication Training.* [`https://arxiv.org/abs/2602.06954`](https://arxiv.org/abs/2602.06954) (Princeton University)
12. **Bogachev, N., & Ryabinin, M. (2026).** *Analysis of the Generalization Fidelity of DiLoCo vs Data-Parallel Training.* [`https://arxiv.org/abs/2602.23029`](https://arxiv.org/abs/2602.23029)

### 2.3 Spectral Isotropy, Subspace Merging & Inner Preconditioning (2025–2026)
13. **Horoi, S., Sarfi, A., & Belilovsky, E. (2026).** *IsoLoCo: Isotropic Subspace Merging for Scalable Low-Communication Training.* [`https://arxiv.org/abs/2607.03011`](https://arxiv.org/abs/2607.03011) — Code: [`https://github.com/shoroi/isoloco`](https://github.com/shoroi/isoloco). Integrated as `SI-07` (`ReputationWeightedIsoCTSMerger`) & `SI-07B` (`TSVDecorrelatedMerger`).
14. **Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., & Rodolà, E. (2025).** *Task Singular Vectors: Reducing Task Interference in Model Merging (CVPR 2025).* [`https://arxiv.org/abs/2412.00081`](https://arxiv.org/abs/2412.00081)
15. **Marczak, D., Twardowski, B., Trzciński, T., & Cygert, S. (2025).** *No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces (`ISO-C` / `ISO-CTS`).* [`https://arxiv.org/abs/2502.04959`](https://arxiv.org/abs/2502.04959)
16. **Thérien, B., Sarfi, A., & Belilovsky, E. (2026).** *MuLoCo: Inner Muon Optimization for Low-Communication Distributed Training.* [`https://arxiv.org/abs/2607.12925`](https://arxiv.org/abs/2607.12925)
17. **Morales, L. (2026).** *How Inner Optimizers Shape Replica Scaling in Distributed Low-Communication Training.* [`https://arxiv.org/abs/2610.19064`](https://arxiv.org/abs/2610.19064) — Integrated as `SI-11` (`MuonNewtonSchulzLoRAPreconditioner`).
18. **Wang, Y., et al. (2026).** *Subspace-Constrained Outer Optimization for DiLoCo.* [`https://arxiv.org/abs/2608.05958`](https://arxiv.org/abs/2608.05958)

### 2.4 Ultra-Low-Bit Pseudo-Gradient Compression & Compute-Visible Sparsification
19. **Peng, B., Quesnelle, J., & Kingma, D. P. (2024).** *DeMo: Decoupled Momentum Optimization.* [`https://arxiv.org/abs/2411.19870`](https://arxiv.org/abs/2411.19870) — Code: [`https://github.com/bloc97/DeMo`](https://github.com/bloc97/DeMo). Integrated as `SI-09C` (`ChunkedOrthonormalDCTTransform`).
20. **Sarfi, A., Thérien, B., Horoi, S., & Belilovsky, E. (2025).** *SparseLoCo: Communication-Efficient Distributed Training with Sparse Low-Bit Pseudo-Gradients.* [`https://arxiv.org/abs/2508.15706`](https://arxiv.org/abs/2508.15706) — Code: [`https://github.com/one-covenant/SparseLoCo`](https://github.com/one-covenant/SparseLoCo). Integrated as `SI-09B` (`ComputeVisibleSparseLoCoCompressor`).
21. **Miahi, E., & Belilovsky, E. (2026).** *Skip the Invisible Updates: Checkpointing Pseudo-Gradient Sparsification for DiLoCo (PULSELoCo).* [`https://arxiv.org/abs/2602.03839`](https://arxiv.org/abs/2602.03839) — Integrated as `SI-09B` and proven in Lean 4 (`Theorem 76`).
22. **Kwon, K., & Kim, J. (2025).** *Scaling Distributed Low-Communication Training with One-Bit Quantization (PoLoCo).* [`https://arxiv.org/abs/2510.17558`](https://arxiv.org/abs/2510.17558)
23. **Kwon, K., & Kim, J. (2025).** *Skip the Negatives: Clamp-Guided Quantization for Outer-Loop Distributed Training (SNQ-L).* [`https://arxiv.org/abs/2512.02042`](https://arxiv.org/abs/2512.02042)

### 2.5 Federated Mixture-of-Experts (`FoMoE`) & Speculative Expert Routing
24. **Sani, L., Iacob, A., & Lane, N. D. (2026).** *FoMoE: Federated Mixture-of-Experts with Partial Expert Replication and Skip-Token.* [`https://arxiv.org/abs/2606.19025`](https://arxiv.org/abs/2606.19025) — Integrated as `SI-09A` (`FoMoEPartialReplicaPlanner`) and proven in Lean 4 (`Theorem 74` & `Theorem 75`).
25. **Merolla, P. (2026).** *fomoe: Fast Opportunistic Mixture of Experts with Skip-Attention Speculative Expert Routing (`85.8%` Top-4 Accuracy) and Cache-Aware Routing (`CAR`).* Code: [`https://github.com/pmerolla/fomoe`](https://github.com/pmerolla/fomoe) — Integrated as `SI-10` (`SkipAttentionSpeculativeRouter`) and proven in Lean 4 (`Theorem 77`).
26. **Sani, L., & Lane, N. D. (2026).** *Gradient-Less Federated Gradient Boosted Trees (FGBoost).* [`https://arxiv.org/abs/2603.24124`](https://arxiv.org/abs/2603.24124)
27. **Ryabinin, M., & Gusev, A. (2020).** *Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts (Learning@Home Hivemind).* [`https://arxiv.org/abs/2006.16668`](https://arxiv.org/abs/2006.16668) — Code: [`https://github.com/learning-at-home/hivemind`](https://github.com/learning-at-home/hivemind)

---

## 3. Community Contributors & HuggingFace Datasets Acknowledgments

- **Project HiveMind Open-Source Contributors**: Special thanks to **`@clark719664`** for 52 merged pull requests across Seasons 5–7 (`PR #52..#82`, `#86..#91`), including `EF21DiLoCoErrorFeedbackBuffer` (`HM-S6-01`), 3-tier `128x7B` `HierarchicalMacroMicroRouter` (`HM-S6-02`), `StrategyOptimizerTelemetryEngine` (`HM-S6-04`), real WAN UDP socket tests (`HM-S6-03`), and Season 6 Lean 4 proofs (`Theorems 65–72`).
- **Curated HuggingFace Datasets (`Operation Polymath-128`)**:
  - `nvidia/OpenMathReasoning` & `AI-MO/NuminaMath-CoT`
  - `open-r1/OpenR1-Math-220k` & `bespokelabs/Bespoke-Stratos-17k`
  - `bigcode/the-stack-v2-dedup` (Rust & Lean 4 subsets)
  - `Camel-AI/physics` & `Camel-AI/biology`

---

## 4. BibTeX Citation Block

```bibtex
@article{douillard2023diloco,
  title={DiLoCo: Distributed Low-Communication Training of Language Models},
  author={Douillard, Arthur and Feng, Qixuan and Rusu, Andrei A and Chhaparia, Rachita and Donchev, Yani and Kuncoro, Adhiguna and Ranzato, Marc'Aurelio and Szlam, Arthur and Shen, Jiajun},
  journal={arXiv preprint arXiv:2311.08105},
  year={2023}
}

@article{jaghouar2024opendiloco,
  title={OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training},
  author={Jaghouar, Sami and Ong, Jack Min and Hagemann, Johannes},
  journal={arXiv preprint arXiv:2407.07852},
  year={2024}
}

@article{peng2024demo,
  title={DeMo: Decoupled Momentum Optimization},
  author={Peng, Bowen and Quesnelle, Jeffrey and Kingma, Diederik P},
  journal={arXiv preprint arXiv:2411.19870},
  year={2024}
}

@article{sarfi2025sparseloco,
  title={SparseLoCo: Communication-Efficient Distributed Training with Sparse Low-Bit Pseudo-Gradients},
  author={Sarfi, Amir and Th{\'e}rien, Benjamin and Horoi, Stefan and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2508.15706},
  year={2025}
}

@article{miahi2026pulseloco,
  title={Skip the Invisible Updates: Checkpointing Pseudo-Gradient Sparsification for DiLoCo},
  author={Miahi, Erfan and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2602.03839},
  year={2026}
}

@article{sani2026decoupleddiloco,
  title={Decoupled DiLoCo: Asynchronous Distributed Training Without Global Barriers},
  author={Sani, Lorenzo and Iacob, Alex and Cao, Zeyu and Qiu, Xinchi and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2604.21428},
  year={2026}
}

@article{shah2026cgad,
  title={Cosine-Gated Adam-Decay (CGAD): Staleness-Aware Outer Optimization for Asynchronous DiLoCo},
  author={Shah, Neil and Sun, Ming},
  journal={arXiv preprint arXiv:2605.09126},
  year={2026}
}

@article{asif2026heloco,
  title={HeLoCo: Direction-Aware Outer Optimization for Heterogeneous Non-IID DiLoCo},
  author={Asif, Muhammad and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2606.00271},
  year={2026}
}

@article{sani2026fomoe,
  title={FoMoE: Federated Mixture-of-Experts with Partial Expert Replication and Skip-Token},
  author={Sani, Lorenzo and Iacob, Alex and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2606.19025},
  year={2026}
}

@article{horoi2026isoloco,
  title={IsoLoCo: Isotropic Subspace Merging for Scalable Low-Communication Training},
  author={Horoi, Stefan and Sarfi, Amir and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2607.03011},
  year={2026}
}

@article{he2026autoloco,
  title={AutoLoCo: Adaptive Synchronization Intervals and Outer Optimizer Correction for DiLoCo},
  author={He, Zexi and Tao, Lin},
  journal={arXiv preprint arXiv:2609.36662},
  year={2026}
}

@article{morales2026inner,
  title={How Inner Optimizers Shape Replica Scaling in Distributed Low-Communication Training},
  author={Morales, Lucas},
  journal={arXiv preprint arXiv:2610.19064},
  year={2026}
}

@software{merolla2026fomoe,
  title={fomoe: Fast Opportunistic Mixture of Experts with Skip-Attention Speculative Routing and Cache-Aware Routing},
  author={Merolla, Paul},
  url={https://github.com/pmerolla/fomoe},
  year={2026}
}
```
