# 🤖 Season 6 (`HM-S6-04`) — Autonomous Agent Arena, Missions 21–24 & Live `SI-01`..`SI-05` HUD Tab

> **Status:** Implemented on `feat/s6-agent-arena`; targets Issue `#59` (`HM-S6-04`).
> All `$HIVE` figures are non-monetary compute-reciprocity units per
> [`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](../LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md).

## 1. What Ships

| Surface | Module | Description |
| :--- | :--- | :--- |
| Missions 21–24 | `src/hivemind/game/hf_hack_miner.py` | `SEASON_6_HF_BOUNTY_CONTRACTS`, `SEASON_6_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS`, `SEASON_6_EXTENDED_HF_BOUNTY_CONTRACTS`, curated offline corpora, `build_season6_mission_catalog()` |
| Self-play code repair | `SelfPlayCodeRepairArena` | Four agents learn per-task mixed policies over four repair styles; curated assertions verify each submitted patch |
| Lean 4 autoformalization | `Lean4AutoformalizerArena`, `audit_lean4_proof()` | Multi-agent proof search with a structural theorem/`by`/`sorry`/axiom/tactic/delimiter audit |
| Combined arena | `build_agent_arena_state()` | Deterministic champion, validated repairs, zero-`sorry` proofs and round telemetry |
| Live strategy telemetry | `StrategyOptimizerTelemetryEngine` in `src/hivemind/swarm/strategy_optimizer.py` | Computes `SI-01`..`SI-05` from real optimizer components and posts link/curriculum updates |
| HUD | `src/hivemind/game/steam_edition.py` | Season 6 panel, arena buttons, live strategy terminal, `GET/POST /api/agent-arena`, `GET/POST /api/strategy-optimizer`, `GET /api/season6-missions` |

## 2. Missions

1. **`CONTRACT-HF-21-SELFPLAY-CODE-REPAIR`** — `SelfPlay_Code_Repair` (`hivemind/selfplay-code-repair-arena`).
2. **`CONTRACT-HF-22-LEAN4-AGENT-FORMALIZER`** — `Lean4_Agent_Autoformalizer` (`hivemind/lean4-agent-autoformalizer`).
3. **`CONTRACT-HF-23-PARETO-STRATEGY-OPTIMIZER`** — `Pareto_Guild_Strategy_Optimizer` (`hivemind/pareto-guild-strategy-optimizer`).
4. **`CONTRACT-HF-24-AUTONOMOUS-AGENT-ARENA`** — `Autonomous_Agent_Arena` (`hivemind/autonomous-agent-arena`).

Each mission ships a curated offline corpus, so the HUD and `--season 6` campaign remain playable
without a Hugging Face Hub round-trip. Forging a Season 6 mission updates
`season6_missions_forged`, unlocks `S6_LEAN4_AUTOFORMALIZER` for Mission 22 and
`S6_AGENT_ARENA_CHAMPION` after all four are forged.

## 3. Agent Arena Design

### 3.1 Self-play code repair

- Four agents (`agent-01`..`agent-04`) maintain per-task weights over four repair policies:
  `test_driven`, `static_analysis`, `mutation_search`, `reflection`.
- Each round, every agent samples a policy, submits the corresponding curated patch, and the verifier
  parses the patch with `ast`, rejects imports and dangerous builtins, then runs the task's assertions
  in an **isolated Python subprocess** (`-I`, 2 s timeout, results file).
- Rewards are the fraction of tests passed; weights update multiplicatively from relative reward.
- Verified `(patch, tests)` pairs are LRU-cached, so a full arena run costs ~1 s cold and a few
  milliseconds once warm.
- The report includes per-round telemetry, a learning curve, per-task validated patches and the
  champion agent/policy. Runs are deterministic for a fixed `seed`.

**Safety boundary:** only patches shipped in `SELF_PLAY_CODE_REPAIR_TASKS` reach the subprocess.
Imports, `open`, `eval`/`exec`, `getattr`, and similar sinks are rejected by the AST screen, and the
child process is timeout-bounded. Arbitrary player or network-supplied code is never executed.

### 3.2 Lean 4 autoformalizer

`audit_lean4_proof()` checks theorem declaration, `:= by`, absence of `sorry`/`admit`, absence of
custom `axiom`, delimiter balance, and use of a known tactic. Agents compete over tactic candidates
(`simp`, `omega`, `ring`, `rfl`, `sorry`) and learn per-task weights.

**Honesty boundary:** this is a *structural lint*, not kernel verification. The arena returns
`kernel_checked: false`; CI's pinned `lake build` (via `verify_lean4_specification()` /
`require_machine_checked`) remains the machine-checking authority.

## 4. Live `SI-01`..`SI-05` Telemetry

`GET /api/strategy-optimizer` returns:

| Field | Computation |
| :--- | :--- |
| `si01_dynamic_fisher_svd` | SVD spectra of seeded adapters → entropy/Fisher rank allocation (budget conserved) |
| `si02_ucb1_curriculum_bandit` | UCB1 scores over recorded holdout-loss observations |
| `si03_adaptive_diloco_cadence` | Cadence/overhead from the current posted link profile (`rtt_ms`, pacing, loss, TURN) |
| `si04_reputation_weighted_ties` | Uniform vs. reputation-weighted TIES holdout probe |
| `si05_pareto_strategy_frontier` | Frontier over holdout loss, speedup, `$HIVE`/GPU-hr and overhead |

`POST /api/strategy-optimizer` accepts link and curriculum updates:

```json
{
  "ring_rtt_ms": 145.0,
  "bbr_pacing_mbps": 18.0,
  "packet_loss_rate": 0.03,
  "turn_relay_active": true,
  "domain_name": "Rust_Systems_Security",
  "loss_before": 1.0,
  "loss_after": 0.4,
  "gpu_seconds": 0.5
}
```

`SI-05` speedup (`3.84x`, `4.65x`) and `$HIVE`/GPU-hour baselines are **modeled planning
assumptions**, not measured RTX 2070 throughput. The payload carries an `honesty` block that lists
which coordinates are measured and which are modeled.

## 5. Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_season6_agent_arena.py -v
```

The suite covers mission catalogs/corpora, micro-forge completion, AST import rejection, arena
determinism/champions, Lean `sorry`/axiom/unbalanced rejection, all `SI-01`..`SI-05` components,
POST update validation, achievements, telemetry frames and the HUD markup.

## 6. Known Environment Limits

- Full-suite Lean tests require the pinned Lean toolchain (`lake build`); without it those tests
  report `tier_a_machine_checked == False` and are unrelated to this arena's structural lint.
- The arena's `SI-04` holdout probe uses seeded adapters and the curated Rust scientific suite; it is
  a deterministic component check, not a claim of trained-model accuracy.
