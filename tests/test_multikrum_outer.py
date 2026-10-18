"""Unit tests for Multi-Krum Byzantine Filtering & DiLoCo Nesterov Outer Step."""

from __future__ import annotations

import torch

from hivemind.core.diloco_outer import DiLoCoOuterOptimizer, filter_byzantine_deltas_multikrum


def test_multikrum_rejects_byzantine_gradient_poisoner() -> None:
    torch.manual_seed(42)
    true_direction = torch.ones((32, 32), dtype=torch.float32) * 0.1

    worker_deltas: dict[str, dict[str, torch.Tensor]] = {}
    for i in range(1, 5):
        noise = torch.randn((32, 32), dtype=torch.float32) * 0.003
        worker_deltas[f"honest-worker-{i}"] = {"lora_w": true_direction + noise}

    # Malicious node reversing gradient direction and scaling magnitude
    worker_deltas["byzantine-poisoner-1"] = {
        "lora_w": -10.0 * true_direction + torch.randn((32, 32)) * 0.5
    }

    report = filter_byzantine_deltas_multikrum(
        worker_deltas=worker_deltas,
        max_byzantine_f=1,
        cosine_sigma=2.0,
    )
    assert "byzantine-poisoner-1" in report.rejected_workers
    assert "byzantine-poisoner-1" not in report.accepted_workers
    assert len(report.accepted_workers) == 4


def test_diloco_nesterov_outer_step_math() -> None:
    theta_0 = {"lora_A": torch.full((4, 4), 2.0, dtype=torch.float32)}
    # Suppose 3 valid workers all report delta = 0.5, and 1 Byzantine worker reports -50.0
    deltas = {
        "w1": {"lora_A": torch.full((4, 4), 0.5, dtype=torch.float32)},
        "w2": {"lora_A": torch.full((4, 4), 0.5, dtype=torch.float32)},
        "w3": {"lora_A": torch.full((4, 4), 0.5, dtype=torch.float32)},
        "byz": {"lora_A": torch.full((4, 4), -50.0, dtype=torch.float32)},
    }

    opt = DiLoCoOuterOptimizer(
        outer_lr_eta=0.7,
        outer_momentum_mu=0.9,
        max_byzantine_f=1,
        cosine_sigma=2.0,
    )
    res = opt.step(branch_name="Mathematics", outer_step=1, theta_prev=theta_0, worker_deltas=deltas)

    assert "byz" in res.filter_report.rejected_workers
    # Expected math at t=1 (with v^(0) = 0):
    # Delta^(1) = 0.5
    # v^(1) = 0.9 * 0 + 0.5 = 0.5
    # theta^(1) = 2.0 - 0.7 * (0.9 * 0.5 + 0.5) = 2.0 - 0.7 * 0.95 = 1.335
    expected = torch.full((4, 4), 2.0 - 0.7 * (0.9 * 0.5 + 0.5), dtype=torch.float32)
    assert torch.allclose(res.theta_updated["lora_A"], expected, atol=1e-5)
