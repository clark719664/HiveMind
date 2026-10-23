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


def _params():
    return {"a": torch.full((2, 2), 2.0), "b": torch.full((2,), 4.0)}


def _workers(a=0.5):
    workers = {
        f"honest-{i}": {"a": torch.full((2, 2), a), "b": torch.full((2,), 0.25)}
        for i in range(4)
    }
    workers["poisoner"] = {"a": torch.full((2, 2), -50.0), "b": torch.full((2,), -50.0)}
    workers["nonfinite"] = {"a": torch.full((2, 2), float("nan")), "b": torch.ones(2)}
    return workers


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(
    "mode,first,second,velocity",
    [
        ("nesterov", (1.335, 3.6675), (0.3865, 3.19325), 0.95),
        ("cosine_gated_adam_decay", (1.286, 3.272), (0.576998, 2.549096), 0.095),
    ],
)
def test_hardened_outer_modes_two_rounds_and_branch_isolation(flat, mode, first, second, velocity):
    opt = DiLoCoOuterOptimizer(outer_optimizer_mode=mode, use_cpu_flat_bucket=flat)
    original = _params()
    initial = {k: v.clone() for k, v in original.items()}
    result = opt.step("STEM", 1, original, _workers())
    for name, want in zip(("a", "b"), first):
        torch.testing.assert_close(result.theta_updated[name], torch.full_like(original[name], want))
    assert set(result.filter_report.rejected_workers) == {"poisoner", "nonfinite"}
    assert len(result.federated_security_report.accepted_workers) == 4
    assert result.outer_optimizer_mode == mode
    assert result.mean_cosine_gate == (1.0 if mode == "cosine_gated_adam_decay" else None)
    assert result.extra_gpu_vram_bytes == 0
    saved_result = {k: v.clone() for k, v in result.theta_updated.items()}

    # A different branch must start fresh even with the same parameter names.
    other = opt.step("Code", 1, _params(), _workers())
    torch.testing.assert_close(other.theta_updated, saved_result)
    again = opt.step("STEM", 2, result.theta_updated, _workers())
    for name, want in zip(("a", "b"), second):
        torch.testing.assert_close(again.theta_updated[name], torch.full_like(original[name], want))
    torch.testing.assert_close(again.velocity_updated["a"], torch.full((2, 2), velocity))
    torch.testing.assert_close(result.theta_updated, saved_result)
    torch.testing.assert_close(original, initial)
    assert all(t.device.type == "cpu" and t.dtype == torch.float32
               for t in [*again.theta_updated.values(), *again.velocity_updated.values()])
    if flat:
        bucket = opt.cpu_flat_buckets["STEM"]
        assert bucket.flat_master_cpu.is_contiguous()
        assert bucket.memory_telemetry()["extra_gpu_vram_bytes"] == 0
        torch.testing.assert_close(bucket.flat_master_cpu, torch.cat([
            again.theta_updated["a"].reshape(-1), again.theta_updated["b"].reshape(-1)
        ]))

    # Existing callers reset outer state by clearing the branch's velocity entry.
    opt.velocities.pop("STEM")
    reset = opt.step("STEM", 1, _params(), _workers())
    torch.testing.assert_close(reset.theta_updated, saved_result)
    torch.testing.assert_close(reset.velocity_updated["a"], torch.full(
        (2, 2), 0.05 if mode == "cosine_gated_adam_decay" else 0.5
    ))


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("mode", ["nesterov", "cosine_gated_adam_decay"])
def test_accepted_spike_is_clipped_before_the_outer_update(flat, mode):
    opt = DiLoCoOuterOptimizer(outer_optimizer_mode=mode, use_cpu_flat_bucket=flat)
    result = opt.step("STEM", 1, {"a": torch.full((4,), 2.0)}, {
        "honest": {"a": torch.full((4,), 0.5)},
        "spike": {"a": torch.full((4,), 50.0)},
    })
    # Both pass the K <= 2 filter; FL clipping limits the spike to 1.0 per coordinate.
    assert set(result.filter_report.accepted_workers) == {"honest", "spike"}
    assert result.federated_security_report.clipped_workers == ["spike"]
    assert result.aggregated_delta_norm == 1.5
    torch.testing.assert_close(result.velocity_updated["a"], torch.full(
        (4,), 0.075 if mode == "cosine_gated_adam_decay" else 0.75
    ))


@pytest.mark.parametrize("flat", [False, True])
def test_cgad_honors_custom_betas_and_can_disable_weight_decay(flat):
    opt = DiLoCoOuterOptimizer(
        outer_optimizer_mode="cosine_gated_adam_decay", use_cpu_flat_bucket=flat,
        beta1=0.5, beta2=0.25, weight_decay=0.0,
    )
    first = opt.step("STEM", 1, _params(), _workers())
    second = opt.step("STEM", 2, first.theta_updated, _workers(a=0.25))
    # m_hat = 1/3, v_hat = 0.1: 2 - 0.7 - 0.7*(1/3)/sqrt(0.1).
    torch.testing.assert_close(second.theta_updated["a"], torch.full((2, 2), 0.5621352))


def test_flat_bucket_nesterov_reuses_contiguous_cpu_momentum():
    bucket = CPUOffloadedFlatBucketOuterState(_params())
    grads = {"a": torch.full((2, 2), 0.5), "b": torch.full((2,), 0.25)}
    first, _ = bucket.outer_nesterov_step(grads, lr=0.7, momentum=0.9)
    second, velocity = bucket.outer_nesterov_step(grads, lr=0.7, momentum=0.9)
    torch.testing.assert_close(first["a"], torch.full((2, 2), 1.335))
    torch.testing.assert_close(second["a"], torch.full((2, 2), 0.3865))
    torch.testing.assert_close(velocity["a"], torch.full((2, 2), 0.95))
    assert bucket.flat_velocity_cpu.device.type == "cpu"
    assert bucket.flat_velocity_cpu.is_contiguous()


def test_default_mode_preserves_nesterov():
    result = DiLoCoOuterOptimizer().step("STEM", 1, _params(), _workers())
    torch.testing.assert_close(result.theta_updated["a"], torch.full((2, 2), 1.335))
    assert result.outer_optimizer_mode == "nesterov"
    assert result.mean_cosine_gate is None


@pytest.mark.parametrize("kwargs", [
    {"outer_optimizer_mode": "adam"},
    {"beta1": 1.0}, {"beta2": -0.1}, {"beta2": float("nan")},
    {"weight_decay": -0.1}, {"weight_decay": float("inf")},
])
def test_invalid_new_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DiLoCoOuterOptimizer(**kwargs)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("lr", [0.0, float("nan"), float("inf")])
def test_invalid_cgad_learning_rate_is_rejected_at_construction(flat, lr):
    with pytest.raises(ValueError):
        DiLoCoOuterOptimizer(outer_optimizer_mode="cosine_gated_adam_decay",
                             use_cpu_flat_bucket=flat, outer_lr_eta=lr)


def test_flat_bucket_rejects_bad_shapes_before_mutating_either_buffer():
    bucket = CPUOffloadedFlatBucketOuterState(_params())
    saved = bucket.flat_master_cpu.clone()
    with pytest.raises(ValueError):
        bucket.outer_nesterov_step({"a": torch.ones(2, 2)})
    with pytest.raises(ValueError):
        bucket.outer_nesterov_step({"a": torch.ones(2, 2), "b": torch.ones(3)})
    torch.testing.assert_close(bucket.flat_master_cpu, saved)
    assert torch.count_nonzero(bucket.flat_velocity_cpu) == 0


def test_flat_outer_rejects_changed_branch_shapes_without_changing_master():
    opt = DiLoCoOuterOptimizer(use_cpu_flat_bucket=True)
    opt.step("STEM", 1, _params(), _workers())
    bucket = opt.cpu_flat_buckets["STEM"]
    saved = bucket.flat_master_cpu.clone()
    with pytest.raises(ValueError):
        opt.step("STEM", 2, {"a": torch.ones(3)}, _workers())
    torch.testing.assert_close(bucket.flat_master_cpu, saved)


@pytest.mark.parametrize("mode,flat", [
    ("nesterov", True), ("cosine_gated_adam_decay", False),
    ("cosine_gated_adam_decay", True),
])
@pytest.mark.parametrize("bad", [
    {"a": torch.ones(2, 2)},
    {"a": torch.ones(4), "b": torch.ones(2)},
])
def test_new_modes_reject_incomplete_or_misshaped_aggregates_before_state_changes(mode, flat, bad):
    opt = DiLoCoOuterOptimizer(outer_optimizer_mode=mode, use_cpu_flat_bucket=flat)
    with pytest.raises(ValueError):
        opt.step("STEM", 1, _params(), {"worker": bad})
    assert not opt.velocities and not opt.cgad_optimizers and not opt.cpu_flat_buckets
    good = opt.step("STEM", 1, _params(), _workers())
    expected = 1.286 if mode == "cosine_gated_adam_decay" else 1.335
    torch.testing.assert_close(good.theta_updated["a"], torch.full((2, 2), expected))
    if mode == "cosine_gated_adam_decay":
        assert good.mean_cosine_gate == 1.0
