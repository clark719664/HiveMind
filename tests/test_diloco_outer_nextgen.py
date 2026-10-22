"""Season 8 outer modes must use the hardened aggregate and isolated CPU state."""

import pytest
import torch

from hivemind.core.diloco_outer import DiLoCoOuterOptimizer
from hivemind.swarm.diloco_nextgen import CPUOffloadedFlatBucketOuterState


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
