"""End-to-end tests for Season 8 Track 2 (`HM-S8-01` / `HM-S8-02`, issue #94).

These tests wire the pre-existing `SI-06` PA-CGAD optimizer and `SI-06B` CPU
flat-bucket state into the real `DiLoCoOuterOptimizer` aggregation path, including
Multi-Krum Byzantine rejection and staleness gating.
"""

from __future__ import annotations

import pytest
import torch

from hivemind.core.diloco_outer import DiLoCoOuterOptimizer
from hivemind.swarm.diloco_nextgen import (
    CPUOffloadedFlatBucketOuterState,
    CosineGatedAdamDecayOuterOptimizer,
)


def _honest_plus_byzantine_deltas() -> dict[str, dict[str, torch.Tensor]]:
    """Four honest workers at ``+0.5`` and one sign-flipped, 100x poisoner."""
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    for i in range(1, 5):
        deltas[f"honest-{i}"] = {
            "lora_A": torch.full((4, 4), 0.5, dtype=torch.float32),
            "lora_B": torch.full((2, 2), -0.25, dtype=torch.float32),
        }
    deltas["byzantine-poisoner"] = {
        "lora_A": torch.full((4, 4), -50.0, dtype=torch.float32),
        "lora_B": torch.full((2, 2), 50.0, dtype=torch.float32),
    }
    return deltas


def _theta() -> dict[str, torch.Tensor]:
    return {
        "lora_A": torch.full((4, 4), 2.0, dtype=torch.float32),
        "lora_B": torch.full((2, 2), -1.0, dtype=torch.float32),
    }


def test_pa_cgad_multi_krum_filters_poisoner_and_emits_telemetry() -> None:
    """`HM-S8-01`: Multi-Krum runs before PA-CGAD, stale fragments are frozen, and telemetry is recorded."""
    theta_0 = _theta()
    opt = DiLoCoOuterOptimizer(
        max_byzantine_f=1,
        cosine_sigma=2.0,
        outer_optimizer_mode="cosine_gated_adam_decay",
        cgad_lr=1e-2,
        cgad_alpha=0.2,
        cgad_tau_cut=8.0,
        cgad_betas=(0.9, 0.99),
        cgad_weight_decay=0.01,
    )

    res = opt.step(
        branch_name="Mathematics",
        outer_step=1,
        theta_prev=theta_0,
        worker_deltas=_honest_plus_byzantine_deltas(),
    )

    assert res.outer_optimizer_mode == "cosine_gated_adam_decay"
    assert "byzantine-poisoner" in res.filter_report.rejected_workers
    assert "byzantine-poisoner" not in res.filter_report.accepted_workers
    assert len(res.filter_report.accepted_workers) == 4
    assert res.mean_cosine_gate == 1.0
    assert res.extra_gpu_vram_bytes == 0
    assert len(res.cgad_telemetry) == 2
    assert all(t.decay_gate_sigma == 1.0 for t in res.cgad_telemetry)
    assert set(res.theta_updated) == set(theta_0)
    for name in theta_0:
        assert torch.isfinite(res.theta_updated[name]).all()
        assert not torch.allclose(res.theta_updated[name], theta_0[name])

    # Round 2: lora_A is fresh, lora_B is beyond the cosine cutoff and must be frozen.
    res2 = opt.step(
        branch_name="Mathematics",
        outer_step=2,
        theta_prev=res.theta_updated,
        worker_deltas=_honest_plus_byzantine_deltas(),
        worker_delays={"lora_A": 0.0, "lora_B": 9.0},
    )
    telemetry = {t.fragment_name: t for t in res2.cgad_telemetry}
    assert telemetry["lora_B"].dropped_by_cutoff is True
    assert telemetry["lora_A"].dropped_by_cutoff is False
    assert torch.equal(res2.theta_updated["lora_B"], res.theta_updated["lora_B"])
    assert not torch.equal(res2.theta_updated["lora_A"], res.theta_updated["lora_A"])
    assert 0.0 < res2.mean_cosine_gate < 1.0


def test_cpu_flat_bucket_nesterov_matches_reference_math_and_zero_vram() -> None:
    """`HM-S8-02`: the CPU flat bucket produces the exact Nesterov update and reports `0 B` VRAM."""
    opt = DiLoCoOuterOptimizer(
        outer_lr_eta=0.7,
        outer_momentum_mu=0.9,
        max_byzantine_f=1,
        cosine_sigma=2.0,
        use_cpu_flat_bucket=True,
    )
    res = opt.step(
        branch_name="Code",
        outer_step=1,
        theta_prev=_theta(),
        worker_deltas=_honest_plus_byzantine_deltas(),
    )

    assert res.outer_optimizer_mode == "nesterov"
    assert res.mean_cosine_gate is None
    assert res.extra_gpu_vram_bytes == 0
    expected = torch.full((4, 4), 2.0 - 0.7 * (0.9 * 0.5 + 0.5), dtype=torch.float32)
    assert torch.allclose(res.theta_updated["lora_A"], expected, atol=1e-5)
    assert torch.allclose(
        res.velocity_updated["lora_A"], torch.full((4, 4), 0.5, dtype=torch.float32), atol=1e-5
    )

    bucket = opt.cpu_buckets["Code"]
    memory = bucket.memory_telemetry()
    assert memory["extra_gpu_vram_bytes"] == 0
    assert memory["is_contiguous"] is True

    # Round 2 refreshes the flat master from theta_prev, so the recursive momentum math holds.
    res2 = opt.step(
        branch_name="Code",
        outer_step=2,
        theta_prev=res.theta_updated,
        worker_deltas=_honest_plus_byzantine_deltas(),
    )
    expected_round2 = 1.335 - 0.7 * (0.9 * (0.9 * 0.5 + 0.5) + 0.5)
    assert torch.allclose(
        res2.theta_updated["lora_A"],
        torch.full((4, 4), expected_round2, dtype=torch.float32),
        atol=1e-5,
    )


def test_pa_cgad_with_cpu_flat_bucket_keeps_master_state_off_gpu() -> None:
    """`HM-S8-01` + `HM-S8-02`: PA-CGAD and Multi-Krum run behind a CPU flat-bucket mirror."""
    opt = DiLoCoOuterOptimizer(
        max_byzantine_f=1,
        cosine_sigma=2.0,
        outer_optimizer_mode="cosine_gated_adam_decay",
        use_cpu_flat_bucket=True,
        cgad_lr=1e-2,
        cgad_tau_cut=8.0,
    )
    theta_0 = _theta()
    res = opt.step(
        branch_name="Physics",
        outer_step=1,
        theta_prev=theta_0,
        worker_deltas=_honest_plus_byzantine_deltas(),
    )

    assert "byzantine-poisoner" in res.filter_report.rejected_workers
    assert res.extra_gpu_vram_bytes == 0
    assert res.mean_cosine_gate == 1.0
    bucket = opt.cpu_buckets["Physics"]
    assert bucket.memory_telemetry()["extra_gpu_vram_bytes"] == 0
    assert bucket.flat_master_cpu.numel() == 4 * 4 + 2 * 2

    # A fully stale round is a no-op for parameters and master state.
    stale = opt.step(
        branch_name="Physics",
        outer_step=2,
        theta_prev=res.theta_updated,
        worker_deltas=_honest_plus_byzantine_deltas(),
        worker_delays={"lora_A": 12.0, "lora_B": 12.0},
    )
    assert stale.mean_cosine_gate == 0.0
    assert all(t.dropped_by_cutoff for t in stale.cgad_telemetry)
    for name in theta_0:
        assert torch.equal(stale.theta_updated[name], res.theta_updated[name])
    master = bucket.flat_master_cpu.clone()
    assert torch.equal(master, torch.cat([stale.theta_updated[k].reshape(-1) for k in bucket.keys]))


def test_cgad_weight_decay_shrinks_updated_fragment() -> None:
    """The optional Adam-Decay weight decay shrinks the parameter without touching the moments."""
    params = {"w": torch.full((2, 2), 1.0, dtype=torch.float32)}
    grads = {"w": torch.full((2, 2), 0.1, dtype=torch.float32)}
    no_decay = CosineGatedAdamDecayOuterOptimizer(lr=1e-2, weight_decay=0.0)
    with_decay = CosineGatedAdamDecayOuterOptimizer(lr=1e-2, weight_decay=0.1)
    plain, _ = no_decay.step_fragments(params, grads)
    decayed, _ = with_decay.step_fragments(params, grads)
    assert (decayed["w"] < plain["w"]).all()
    assert torch.allclose(no_decay.m_buffers["w"], with_decay.m_buffers["w"])


def test_outer_configuration_validation() -> None:
    """Unknown outer modes and invalid CGAD hyperparameters are rejected up front."""
    with pytest.raises(ValueError, match="outer_optimizer_mode"):
        DiLoCoOuterOptimizer(outer_optimizer_mode="not-a-real-optimizer")
    with pytest.raises(ValueError, match="weight_decay"):
        CosineGatedAdamDecayOuterOptimizer(weight_decay=-1e-3)
    for bad_kwargs in (
        {"lr": 0.0},
        {"alpha": 0.0},
        {"tau_cut": 0.0},
        {"betas": (1.0, 0.9)},
    ):
        with pytest.raises(ValueError):
            CosineGatedAdamDecayOuterOptimizer(**bad_kwargs)


def test_cpu_flat_bucket_rejects_malformed_outer_steps() -> None:
    """The flat bucket validates hyperparameters, fragments and wire shapes."""
    params = {"w": torch.zeros(2, dtype=torch.float32)}
    bucket = CPUOffloadedFlatBucketOuterState(params)

    with pytest.raises(ValueError, match="lr"):
        bucket.outer_nesterov_step({"w": torch.ones(2)}, lr=0.0)
    with pytest.raises(ValueError, match="momentum"):
        bucket.outer_nesterov_step({"w": torch.ones(2)}, momentum=1.0)
    with pytest.raises(KeyError, match="missing flat-bucket fragments"):
        bucket.outer_nesterov_step({"w": torch.ones(2)}, theta_prev={"other": torch.zeros(2)})
    with pytest.raises(ValueError, match="elements"):
        bucket.outer_nesterov_step({"w": torch.ones(3)}, theta_prev=params)

    bucket.flat_velocity_cpu = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="velocity buffer"):
        bucket.outer_nesterov_step({"w": torch.ones(2)}, theta_prev=params)


def test_pa_cgad_streaming_synced_fragments_only_updates_the_synced_subset() -> None:
    """`synced_fragments` flows through the outer optimizer to PA-CGAD's streaming path."""
    opt = DiLoCoOuterOptimizer(
        outer_optimizer_mode="cosine_gated_adam_decay",
        cgad_lr=1e-2,
        cgad_tau_cut=8.0,
    )
    theta_0 = _theta()
    res = opt.step(
        branch_name="Streaming",
        outer_step=1,
        theta_prev=theta_0,
        worker_deltas=_honest_plus_byzantine_deltas(),
        synced_fragments=["lora_A"],
    )
    assert [t.fragment_name for t in res.cgad_telemetry] == ["lora_A"]
    assert torch.equal(res.theta_updated["lora_B"], theta_0["lora_B"])
    assert opt.cgad_optimizers["Streaming"].fragment_ages["lora_B"] == 1
