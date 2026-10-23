# Review of the MoE / Google DeepMind / OpenDiLoCo roadmap and paper (2026-10-23)

Scope: `ROADMAP.md` Seasons 7–8, `paper/hivemind_foundation_article.tex` v3.1.0 (Sections 6–7),
`docs/DILOCO_ARXIV_LITERATURE_AND_IMPROVEMENTS.md`, `docs/ACKNOWLEDGMENTS_AND_CITATIONS.md`,
`CITATION.cff`, `paper/autoevolve_peer_review_pipeline.py`, `src/hivemind/swarm/diloco_nextgen.py`
and its tests. Method: every arXiv identifier cited anywhere in the repository was opened on
arXiv/alphaXiv and compared with the title and authors the repository attributes to it; the
implementation was read against the four locally archived PDFs in
`sub_projects/hivemind/reference_papers/`; the test suite and Lean build were run.

## 1. Findings

### 1.1 The "peer review" was a script with constant scores (blocking)

`paper/autoevolve_peer_review_pipeline.py` imports only `hashlib, json, os, re, shutil,
subprocess, time`. It calls no model and no reviewer. It wrote `PEER_REVIEW_REPORT.json` with
`lead_peer_review_model: "claude-opus-5-5"`, `co_auditor_model: "gemini-4-argon"`, three
"referees" with hard-coded `score_out_of_10` values 9.4 / 9.5 / 9.45, a hard-coded
`calibrated_epistemic_confidence: 0.9250`, and `decision: ACCEPT_AS_FOUNDATION_ARTICLE`. The
paper (line 603) and the Lean file header (Season 13) repeated the "Opus 5.5 5-round peer review"
label; commit `a2995fd` carries it in its subject. This is the same defect already recorded for
two other sub-projects in this workspace (scripted "reviews" that call no model).

**Fixed in this revision.** The script now states `is_peer_review: false`, carries a disclosure
string, has no model names and no numeric scores, and its decision reads
`SCRIPTED_GATES_PASSED_NOT_A_PEER_REVIEW`. The paper says so in Section 9 and in the
reproducibility list; the Lean namespace `Season13_Opus55...` was renamed. What the script still
does is useful and unchanged: Lean build + `#print axioms`, JSON-to-LaTeX number lock, LaTeX lint.

### 1.2 Fabricated and mis-attributed citations (blocking for the docs, minor for the paper)

Checked against arXiv on 2026-10-23:

| Where | Claimed | Actual |
| :-- | :-- | :-- |
| `ROADMAP.md` HM-S8-09, `README.md`, `DILOCO_ARXIV…`, `CITATION.cff`, wiki, `community_orchestrator.py` | MuLoCo = `arXiv:2607.12925` | 2607.12925 is a spin-1 parent-Hamiltonian physics paper. MuLoCo is `arXiv:2505.23725` (Thérien, Huang, Defazio, Rish, Belilovsky). The paper's own bibliography had the right id. |
| same files, `diloco_nextgen.py` docstrings | `arXiv:2610.19064` by "Lucas Morales" | Belilovsky, Horoi, Oyallon & Thérien, *How Inner Optimizers Shape Replica Scaling*. No author Morales. |
| `ACKNOWLEDGMENTS`, `CITATION.cff` | 2602.23029, 2602.06954, 2608.05958, 2609.02737, 2510.17558, 2512.02042, 2603.24124 as DiLoCo papers | WISER (image retrieval), an astrophysics disk paper, an AHP pairwise-comparison paper, *Language Models Can Control Their Own Attention*, *The Free Transformer*, a Fermat-type PDE paper, *The Alignment Tax*. None concerns DiLoCo. |
| `ACKNOWLEDGMENTS`, `CITATION.cff`, `ROADMAP.md` | Decoupled DiLoCo (2604.21428) by Sani et al. / CaMLSys | Decoupled DiLoCo Team, Google DeepMind & Google Research. |
| `ROADMAP.md`, `ACKNOWLEDGMENTS` | CGAD by "Neil Shah & Ming Sun"; HeLoCo by "Muhammad Asif & Belilovsky"; AutoLoCo by "Zexi He & Lin Tao" | Vatsal Shah & Jiahao Sun (FLock.io); Al Asif, Diem, Muñoz, Wolf, Jannesari & Mazaheri; He, Zhang, Li & Yang. |
| `ACKNOWLEDGMENTS`, `CITATION.cff`, `DILOCO_ARXIV…` | Titles of FoMoE, IsoLoCo, PULSE, SparseLoCo | All four titles were invented paraphrases; real titles now used. |
| docs, `diloco_nextgen.py` | Hivemind paper = `arXiv:2006.16668` | 2006.16668 is GShard. The Hivemind paper is `arXiv:2002.04013`. |
| paper §6.4 | 2610.23036 = "Scaling Latent Reasoning via Looped Language Models"; 2610.06833 = "Implicit Language Models are Fixed-Point Learners" | *Towards Looped Models Done Right*, Part I and Part II (Huang, Shi, Chen, Wen, Liu, Xing, Ma). The bibliography had the right titles; the body text did not. |

All 22 identifiers in the DiLoCo literature table are real papers with the stated topics; the
problem was concentrated in the acknowledgments/citation files, which padded the list with
invented entries. **Fixed**: `docs/ACKNOWLEDGMENTS_AND_CITATIONS.md`, the wiki copy and
`CITATION.cff` are regenerated from one verified list of 30 papers, with the removed identifiers
listed explicitly; every other occurrence was corrected in place. Every entry of the paper's
bibliography was checked individually.

### 1.3 What the MoE / DiLoCo implementation actually covers

Positive: `diloco_nextgen.py` (1,187 lines) implements the eleven `SI-06`..`SI-11` components,
and they match the source papers where I compared them (CGAD gate and bound; FoMoE
ring factor `R(O_e) = 2(O_e-1)/O_e`; DeMo/SparseLoCo codec; Muon quintic coefficients). The
126 tests in the six DiLoCo/MoE test files pass; the Lean spec builds with the whitelisted axioms
only. The paper's "honest scope" statements (no multi-host run, largest MoE trained has 4
experts, H5's 95.83 % is a noise-level artefact) are accurate and should stay.

Gaps against the cited literature:

1. **Decoupled DiLoCo was cited but not implemented.** Section 7 is titled "OpenDiLoCo & Google
   DeepMind Outer-Loop Ecosystem", the PDF is archived locally, yet nothing in the code did
   minimum-quorum aggregation, an adaptive grace window, token-weighted learner weights or
   Radial-Directional Averaging. These are precisely the mechanisms a volunteer swarm needs.
   **Added** (`SI-12`, see §2).
2. **FoMoE's σ-MoE routing and its "measure the ghost fraction" advice were not implemented.**
   `FoMoEPartialReplicaPlanner` uses only the ideal `N_le / N_e` scaling; FoMoE §3.3.3 says the
   measured ghost-assignment frequency must replace it under imbalance. **Added** (see §2).
3. **Looped backbone**: Part II's learned depth prior and orthogonal input injection are absent;
   terminal-KV sharing is listed as HM-S7-02 but the Part II result that fixed-depth training breaks
   KV sharing (GSM8K 50.6 % → 21.2 %) is not reflected anywhere. → `HM-S9-03`.
4. **E2 (PA-CGAD straggler experiment)** compares against un-gated outer AdamW only; the natural
   baseline from the same literature, Nesterov with periodic restart (2605.28585), is missing.
   → `HM-S9-05`.
5. **Multi-host WAN measurement** is still absent; every Category III claim depends on it.
   → `HM-S9-07` (carried over, unchanged in substance).
6. `README.md` already advertised "Season 9 tasks HM-S9-01..08" that did not exist in
   `ROADMAP.md`. Section 3.7 now exists with nine tasks.

## 2. Extensions delivered in this revision (working tree, uncommitted)

| Item | File | Evidence |
| :-- | :-- | :-- |
| `SI-12` Decoupled DiLoCo syncer: `token_weighted_quality_weight`, `select_quorum`, `AdaptiveGraceWindowController`, `radial_directional_average`, `DecoupledDiLoCoQuorumSyncer` | `src/hivemind/swarm/decoupled_diloco.py` | `tests/test_decoupled_diloco_quorum.py` (13 tests, 4 Hypothesis properties) |
| FoMoE σ-MoE routing with measured skip fraction and Eq. 3 cost model | `src/hivemind/core/fomoe_sigma_routing.py` | `tests/test_fomoe_sigma_routing.py` (7 tests, 1 Hypothesis property) |
| Check `H11` (1 positive arm, 2 negative controls) | `scripts/verify_decoupled_diloco_quorum.py` → `artifacts/decoupled_diloco_quorum_check.json` | 3/3 arms pass; RDA/direct ratio at M=8 is 2.8284 (=√8); grace 3.35 s on 6.7 s slack; straggler leak 0.0 |
| Lean Theorems 117–120 (Season 14) | `lean4/HiveMindSpec.lean` (+ `../formal/HiveMindSpec.lean` copy) | `lake build` ok, 120 theorems, axioms ⊆ {propext, Classical.choice, Quot.sound} |
| Paper v3.2.0: §7.2 Decoupled DiLoCo, σ-MoE paragraph in §6.2, H11 paragraph in §8, limitation item 6, bibliography entry, counts 116→120, honest audit wording | `paper/hivemind_foundation_article.tex` | compiles (pdflatex ×2, 23 pages, no undefined references) |
| Provenance checks PROV-15/16 for the H11 numbers | `paper/autoevolve_peer_review_pipeline.py` | pipeline run recorded below |
| Season 9 roadmap (`HM-S9-01`..`HM-S9-09`) | `ROADMAP.md` §3.7 | — |

Measured outputs quoted above come from running the scripts in this session; nothing was
estimated. The one failing test in the full suite
(`tests/test_cov_core_and_swarm.py::test_async_quic_geo_ring_transport_methods`) is in the QUIC
transport and is unrelated to these changes; see the session report for its error.

## 3. Recommendations beyond Season 9

- Never let a script emit a score or a verdict labelled as a model's or a referee's. If model
  review is wanted, route it through `anse.core.api_extractor.APIExtractor` so the call is logged,
  and commit the transcript.
- Keep one machine-readable list of references (the generator in this session built the three
  citation files from a single Python list); regenerate the Markdown/CFF from it instead of
  editing them by hand, and re-open every new arXiv id before it is cited.
- The `116`/`120` theorem count appears in nine places in the paper; the provenance lock already
  checks it (PROV-01). Consider generating that number into the LaTeX from the Lean inventory.
