# External review of preprint v3.5.0 (received 2026-10-23)

**Provenance (recorded by the scientific lead, not by a script):** this text was pasted into the working
session by the project owner on 2026-10-23 with the words "record the peer review and improve accordingly".
The owner did not state who or what wrote it. It is therefore recorded as an *external review of unstated
provenance*: it is not a human peer review of record, it was not produced through the project's logged
model-review path (`scripts/logged_model_review.py` / `APIExtractor`), and it carries no numeric score.
It does **not** by itself satisfy gate G4 of `docs/RESEARCH_PROGRAM.md`. The text below is verbatim.
The point-by-point response is `response_to_external_review_v3.5.0.md`.

---

Here is a comprehensive peer review report for the manuscript **"Project Hive Mind: A System Design for Decentralized Sparse Mixture-of-Experts Training with DiLoCo-Style Outer Optimization on 8 GB Consumer GPUs" (Version 3.5.0)**.

---

### **1. Overall Assessment & Recommendation**

**Recommendation: Accept as a System Design and Pre-Registered Research Protocol** *(with minor revisions)*

**Summary:**
This paper outlines an exceptionally ambitious, meticulously documented system design aimed at democratizing the training of Large Language Models (LLMs) by distributing the workload across heterogeneous consumer GPUs (specifically capped at an 8 GB VRAM budget) connected via standard residential internet. To overcome the dual bottlenecks of VRAM limitations and low bandwidth, the author synthesizes a vast array of recent techniques: DiLoCo outer-loop optimization, 4-bit NF4 + QLoRA, 2-bit DCT outer-gradient compression (DeMo + SparseLoCo) with EF21 error feedback, and partial expert replication (FoMoE).

The manuscript is highly unique in its structure and epistemic rigor. It serves simultaneously as a systems engineering blueprint, a mathematical bookkeeping ledger (verified via Lean 4), a report on single-host simulation experiments, and a rigorously pre-registered protocol for future multi-host evaluations.

---

### **2. Major Strengths**

* **Radical Epistemic Transparency:** The paper sets a gold standard for open science in ML systems. The clear demarcation of claims via the "Epistemic Stratification" (Table 8)—cleanly separating what is formally verified, what is empirically simulated on a single host, and what remains an open multi-node question—prevents the overclaiming that plagues the distributed systems literature.
* **Methodological Rigor & Pre-Registration:** Applying clinical-trial-style pre-registration to ML systems research is a phenomenal contribution. The frank discussion regarding the failure of the `E5-toy` experiment (realizing the testbed was too simple because learner deltas were collinear) and the subsequent design of `E5-het` to test Radial-Directional Averaging (RDA) exemplifies mature scientific practice.
* **Formal Verification of Systems Code (Lean 4):** Using Lean 4 to formally verify 120 lemmas regarding discrete arithmetic invariants (e.g., EF21 telescoping sum conservation, flat-bucket partition logic, byte-budgets) is highly novel. It successfully bridges the gap between formal methods and empirical deep learning systems, providing guarantees against common off-by-one errors and memory leaks.
* **Theoretical Elegance:** Proposition 7.1 concisely re-frames Radial-Directional Averaging (RDA) as an adaptive outer learning-rate multiplier under common-norm conditions. This provides excellent intuition that is immediately validated by the empirical $\bar{r}$ measurements in the `E5-het` experiments.

---

### **3. Critical Weaknesses & Technical Concerns**

* **Hardware Heterogeneity vs. Byzantine Spot-Checks (Section 4.1):**
The Byzantine Fault Tolerance (BFT) pillar relies on a 10% micro-checkpoint spot-check, where a validator replays a segment and checks if $\vert{}\vert{}\theta_{validator} - \theta_k\vert{}\vert{}_\infty < 10^{-4}$. However, floating-point non-determinism across different GPU architectures (e.g., NVIDIA Turing vs. Ada Lovelace) or vendors (NVIDIA vs. AMD ROCm vs. Apple Metal) makes tight $\epsilon$-bounded reproducibility practically impossible. Even with `torch.use_deterministic_algorithms(True)`, a validator on an RTX 4090 will likely reject an honest worker on an RTX 2070 due to differing FMA kernel implementations at the hardware level. The paper must address how this scheme survives actual hardware heterogeneity in a consumer swarm.
* **The "Compounding Approximations" Risk:**
The architecture aggressively stacks lossy compression and approximation techniques: 4-bit base weights + 2-bit DCT outer gradients + FoMoE Skip-Token dropping + speculative Skip-Attention + PA-CGAD staleness truncation + fixed-point early loop exits. While unit tests (H1-H10) prove these work in isolation, neural networks at scale are highly sensitive to interacting noise. The non-linear compounding of these approximations over a chaotic WAN poses a severe risk of representation collapse or training divergence. This compound risk should be explicitly highlighted as a core threat to validity.
* **Meaningless Toy Scales for the PoC (Section 8.3):**
The RTX 2070 proof-of-concept trains Qwen2.5-0.5B on just *8 rows of data per worker*. While this successfully proves the software compiles and obeys the 1.5 GB VRAM budget, a neural network can trivially memorize 8 rows even with heavily corrupted gradients. It does not validate the mathematical correctness of the learning trajectory.
* **Lack of Visual Architectural Aids:**
The text is incredibly dense with acronyms and code references. Reading through the integration of a dozen complex mechanisms requires the reader to hold massive amounts of system state in their head. The paper desperately needs a full-page **Architecture Flowchart** demonstrating (a) a single forward/backward pass on a worker constrained by VRAM, and (b) a single outer-loop gradient sync across the WAN.

---

### **4. Specific Section Feedback**

* **Section 6.2 (FoMoE & Skip-Token):** Dropping tokens routed to non-resident Ghost Experts locally is a clever bandwidth-saving trick, but what is the theoretical penalty on downstream zero-shot reasoning? Since the local expert adapters never see those tokens, how does this affect representation continuity? Skipping tokens essentially creates a dynamic, routing-dependent dropout.
* **Section 7.3 & E5 (PA-CGAD under Quorum):** The observation that PA-CGAD *hurt* performance under a quorum setting (P4 refuted) is highly insightful. It suggests that down-weighting stale but directionally accurate gradients throws away vital signal when effective batch sizes are constrained by the quorum. This tension between staleness-aware optimization and asynchronous quorum systems warrants a brief highlight directly in Section 7.3 to temper theoretical expectations.
* **Section 10.1 (Fake Peer Review Confession):** The author's admission that versions 2.9-3.1 included an AI-generated fake peer review is startlingly candid. While cleaning the academic record is the correct ethical action, the author should be aware that admitting to fabricating peer reviews in previous preprints could trigger automatic desk rejections or ethical investigations at major academic venues (e.g., NeurIPS, MLSys). It is highly recommended to place this note strictly in the repository's `CHANGELOG.md` or a submission cover letter rather than the main text of the academic manuscript.

---

### **5. Minor Corrections & Typos**

* **Abstract, Page 1:** Typo: "open-source **ource** decentralized..." $\rightarrow$ Remove the duplicate "ource".
* **Abstract, Page 1:** Typo: "sparse **arse** MoE upcycling..." $\rightarrow$ Remove "arse".
* **Page 16, Theorem 7.3 Proof:** There is a typo in the inequality chain: "$\int \le \tau\sigma(\tau) \le \phi(\tau) := \tau e^{-\alpha\tau}$". The integral sign $\int$ appears to be a transcription/typographical error for $0$. It should read "$0 \le \tau\sigma(\tau) \le \phi(\tau)$".
* **Page 25, Section 9.6:** Typo: "Proposition 7.1 predicts **$s~$** $r = (1/M + ...$". There is a stray "$s~$" before the $r$.

---

### **Conclusion**

This manuscript represents a tour-de-force of modern distributed LLM systems engineering. The strict adherence to pre-registered methodologies and the utilization of Lean 4 for formal bounds sets a new gold standard for reproducible ML systems research. If the cryptographic/floating-point BFT tensions are addressed, and the planned multi-host E8/E9 WAN experiments eventually yield positive results, this framework will serve as a foundational blueprint in the field of decentralized AI.
