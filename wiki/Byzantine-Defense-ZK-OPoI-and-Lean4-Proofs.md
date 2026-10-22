# 🛡️ Byzantine Defense, `ZK-OPoI v4` & 77 Machine-Checked Lean 4 Proofs (`v1.8.0`)

Because Project HiveMind runs over untrusted residential nodes, every training outer step, inference token batch, and PoUW ledger block is protected by defense-in-depth cryptography and **77 machine-checked Lean 4 `theorem` declarations (`Theorems 1–72` + 5 helper theorems, `0 sorry`, `0` custom axioms)** in [`lean4/HiveMindSpec.lean`](https://github.com/xaviercallens/HiveMind/blob/main/lean4/HiveMindSpec.lean).

---

## 1. Four-Layer Byzantine & Protocol Defense Stack (`Waves 1–5`)

1. **10% Deterministic CUDA Micro-Segment Spot-Check (`src/hivemind/boinc/validator.py`) & `FL-08` Ownership (`PR #72`):**
   - Workers commit a Merkle root of 10 inner-loop micro-checkpoints ($H=500 \implies s=50$ steps/segment) under `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
   - Even when `EF21-DiLoCo` (`PR #63, #69`) is enabled, workers upload their uncompressed `raw_delta` alongside the EF21-compressed wire packet so the 10% spot-check (`eps <= 1e-4`) never false-slashes honest EF21 peers.
   - `BOINCProjectServer` (`PR #72`) enforces single-assignee workunit binding (`assigned_host_id`) and post-verdict consumption (`consumed_workunits`) to prevent workunit hijacking and replay.
2. **$2\sigma$-Clipped Multi-Krum Outer Aggregation (`src/hivemind/core/multikrum_outer.py`):**
   - Filters out gradient poisoners (`-15x` sign-flip, Gaussian noise, Sybil collusion) before applying the DiLoCo Nesterov outer momentum step ($\mu=0.9, \eta=0.7$).
3. **`ZK-OPoI v4` Position-Bound Batch Commitments & `GuildSlashingInsurancePool` (`HM-S6-03` / `PR #79`):**
   - [`OptimisticProofOfInferenceAuditor`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/swarm_fabric.py) commits an entire speculative batch ($\gamma \times L$ leaves) into a single **32-byte SHA-256 Merkle root** with leaf/internal domain separation (`0x00` vs `0x01`) and position-bound leaf payloads `(step_index, layer_idx, tensor_sha256)`.
   - [`GuildSlashingInsurancePool`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/ledger.py) provides supply-conserving cross-guild indemnification (`paid_from_peer_stake + paid_from_reserve`, single-use `(merkle_root, step, layer)` fault claims) so consumers are refunded immediately when a challenged leaf fails.
4. **RFC 5766 Per-Allocation TURN `ChannelBind` Isolation (`PR #73`):**
   - [`EmbeddedStunTurnServer`](https://github.com/xaviercallens/HiveMind/blob/main/src/hivemind/swarm/nat_traversal.py) scopes TURN channel bindings (`0x4000–0x7FFE`) per authenticated client allocation so one peer cannot overwrite or hijack another peer's relay channel.

---

## 2. Fail-Closed Lean 4 Verification Gate (`77` `theorem` Declarations, `0 sorry`, `0` Axioms)

[`paper/autoevolve_peer_review_pipeline.py`](https://github.com/xaviercallens/HiveMind/blob/main/paper/autoevolve_peer_review_pipeline.py) (`verify_lean4_specification` & `require_machine_checked`, hardened in `PR #77`) compiles [`lean4/HiveMindSpec.lean`](https://github.com/xaviercallens/HiveMind/blob/main/lean4/HiveMindSpec.lean) via `lake build` and inspects `#print axioms` for every theorem in the specification:

| Namespace | Verified Theorems (`77` Total `theorem` Declarations) | Key Formal Guarantees |
| :--- | :--- | :--- |
| `Pillar1_NF4_DiLoCo` | `nf4_double_quant_compression_bound`, `rtx2070_8gb_vram_envelope_verified`, `diloco_topk_fp8_bandwidth_reduction`, etc. | `>3.87x` NF4 compression, `1,527 MiB <= 6,656 MiB` 8 GB VRAM bound, `>1,666x` DiLoCo+FP8 wire savings |
| `Pillar2_Byzantine_Defense` | `spot_check_10pct_and_20x_boinc_savings`, `multikrum_neighborhood_contains_honest_worker`, etc. | 20x BOINC verification savings & honest worker inclusion under $n \ge 2f + 3$ |
| `Pillar3_BTM_Sparse_MoE` | `top2_sparse_moe_quarter_active_compute`, `speculative_tree_rtt_speedup`, etc. | 25% active FFN compute under Top-2 routing & speculative RTT speedup |
| `Pillar4_PoUW_Tokenomics` | `pouw_reward_pools_conserve_100pct`, `hive_geometric_halving_respects_21m_cap`, etc. | Exact `60/15/15/10%` reward conservation & `< 21,000,000 $HIVE` supply cap |
| `Phase2_SwarmFabric` & `Season3_PrefixTrie_And_0RTT` | `dma_double_buffer_zero_exposed_stall`, `speculative_rejection_sampling_target_invariance`, `rfc9001_0rtt_no_nonce_accepted_twice`, etc. | Zero exposed PCIe swap stall, lossless speculative rejection sampling, and monotonic strike-register replay defense (`PR #41`) |
| `Season4_DAOGovernance` & `Season5_StrategyAndBisection` | `quadratic_voting_cost_bound`, `ties_sign_consensus_majority_bound`, `interactive_bisection_interval_halving_bound`, `fisher_spectral_rank_budget_conservation`, `adaptive_diloco_communication_sla_bound` | Quadratic DAO cost, TIES majority sign consensus, $O(\log_2 L)$ bisection, Fisher rank conservation & `<= 8%` DiLoCo comm SLA |
| `Season6_ErrorFeedback_And_128MoE` & `Season6_ErrorFeedbackModel` (`PR #68`) | `ef_conservation_any_compressor`, `topk_contraction_rate`, `ef_residual_stays_bounded_under_decay`, `ucb1_ExplorationBonus_antitone_pulls`, `ucb1_index_strict_separation`, `hierarchical_128_expert_3tier_gating_savings`, `slashing_insurance_pool_solvency_bound` + 5 helper theorems | Constructive `Int`/`Rat` vector proofs of `EF21-DiLoCo` exact conservation for any compressor, Top-$k$ contraction rate ($d \cdot \text{kept} \le (d - k) \cdot \text{total}$), bounded residual under staleness decay, UCB1 separation, and 128-expert `1/8` gating ratio |
