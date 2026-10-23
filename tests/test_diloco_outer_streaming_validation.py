"""Streaming CGAD accepts declared partial aggregates without weakening validation."""

import math

import pytest
import torch

from hivemind.core.diloco_outer import DiLoCoOuterOptimizer


def _params():
    return {"a": torch.full((2, 2), 2.0), "b": torch.full((2,), 4.0)}


def _workers(fields):
    values = {"a": torch.full((2, 2), 0.5), "b": torch.full((2,), 0.25)}
    result = {f"honest-{i}": {k: values[k].clone() for k in fields} for i in range(4)}
    result["poisoner"] = {k: -100 * values[k] for k in fields}
    return result


def _optimizer(flat):
    return DiLoCoOuterOptimizer(
        outer_optimizer_mode="cosine_gated_adam_decay", use_cpu_flat_bucket=flat,
        cgad_lr=0.1, cgad_weight_decay=0.0,
    )


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("full_packet", [False, True])
def test_declared_streaming_subset_updates_and_ages_fragments(flat, full_packet):
    opt = _optimizer(flat)
    params = _params()
    saved = {k: v.clone() for k, v in params.items()}
    first = opt.step("STEM", 1, params, _workers(["a", "b"] if full_packet else ["a"]),
                     synced_fragments=["a"])
    torch.testing.assert_close(first.theta_updated["a"], torch.full((2, 2), 1.9))
    torch.testing.assert_close(first.theta_updated["b"], saved["b"])
    assert first.filter_report.rejected_workers == ["poisoner"]
    assert [t.fragment_name for t in first.cgad_telemetry] == ["a"]
    assert opt.cgad_optimizers["STEM"].fragment_ages == {"a": 0, "b": 1}

    second = opt.step("STEM", 2, first.theta_updated, _workers(["b"]),
                      synced_fragments=["b"])
    # b's first update is delayed by one round; bias correction cancels its gradient scale.
    sigma = 0.5 * (1 + math.cos(math.pi / 32)) * math.exp(-0.2)
    torch.testing.assert_close(second.theta_updated["b"], torch.full((2,), 4 - 0.1 * sigma))
    torch.testing.assert_close(second.theta_updated["a"], first.theta_updated["a"])
    assert second.cgad_telemetry[0].fragment_age_rounds == 1
    assert opt.cgad_optimizers["STEM"].fragment_ages == {"a": 1, "b": 0}
    torch.testing.assert_close(params, saved)
    if flat:
        torch.testing.assert_close(opt.cpu_buckets["STEM"].flat_master_cpu,
                                   torch.cat([second.theta_updated[k].flatten() for k in ("a", "b")]))


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("case", ["missing_declared", "unknown_declared", "unknown_aggregate", "shape"])
def test_invalid_streaming_input_does_not_allocate_optimizer_state(flat, case):
    opt = _optimizer(flat)
    delta = {"a": torch.full((2, 2), 0.5)}
    selected = ["a"]
    if case == "missing_declared":
        selected = ["a", "b"]
    elif case == "unknown_declared":
        delta["b"] = torch.full((2,), 0.25)
        selected = ["z"]
    elif case == "unknown_aggregate":
        delta["z"] = torch.ones(2)
    elif case == "shape":
        delta["a"] = torch.ones(4)
    with pytest.raises(ValueError):
        opt.step("STEM", 1, _params(), {"worker": delta}, synced_fragments=selected)
    assert not opt.velocities and not opt.cgad_optimizers and not opt.cpu_buckets


@pytest.mark.parametrize("flat", [False, True])
def test_rejected_partial_update_leaves_existing_moments_and_flat_master_unchanged(flat):
    opt = _optimizer(flat)
    first = opt.step("STEM", 1, _params(), _workers(["a"]), synced_fragments=["a"])
    cgad = opt.cgad_optimizers["STEM"]
    moments = {k: v.clone() for k, v in cgad.m_buffers.items()}
    variances = {k: v.clone() for k, v in cgad.v_buffers.items()}
    steps, ages = dict(cgad.step_count), dict(cgad.fragment_ages)
    master = opt.cpu_buckets["STEM"].flat_master_cpu.clone() if flat else None
    with pytest.raises(ValueError):
        opt.step("STEM", 2, first.theta_updated, _workers(["a"]), synced_fragments=["b"])
    torch.testing.assert_close(cgad.m_buffers, moments)
    torch.testing.assert_close(cgad.v_buffers, variances)
    assert cgad.step_count == steps and cgad.fragment_ages == ages
    if flat:
        torch.testing.assert_close(opt.cpu_buckets["STEM"].flat_master_cpu, master)


@pytest.mark.parametrize("flat", [False, True])
def test_empty_selection_skips_all_fragments_without_claiming_a_gate(flat):
    opt = _optimizer(flat)
    params = _params()
    result = opt.step("STEM", 1, params, _workers(["a", "b"]), synced_fragments=[])
    torch.testing.assert_close(result.theta_updated, params)
    assert result.cgad_telemetry == [] and result.mean_cosine_gate is None
    assert opt.cgad_optimizers["STEM"].fragment_ages == {"a": 1, "b": 1}


@pytest.mark.parametrize("mode,flat", [
    ("cosine_gated_adam_decay", False), ("cosine_gated_adam_decay", True), ("nesterov", True),
])
def test_undeclared_partial_aggregate_remains_rejected(mode, flat):
    opt = DiLoCoOuterOptimizer(outer_optimizer_mode=mode, use_cpu_flat_bucket=flat)
    with pytest.raises(ValueError):
        opt.step("STEM", 1, _params(), _workers(["a"]))
