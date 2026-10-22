"""HM-S6-01 error-feedback residual buffer and its synthetic compression benchmark."""

from __future__ import annotations

import json

import pytest
import torch

from hivemind.core.compression import compress_lora_delta, decompress_lora_delta
from hivemind.swarm.ef_compression_benchmark import main, run_benchmark
from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer


def _deltas(seed: int, steps: int) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    return [
        {"lora_A": torch.randn(8, 32, generator=gen), "lora_B": torch.randn(32, 8, generator=gen)}
        for _ in range(steps)
    ]


def _send(buffer: EF21DiLoCoErrorFeedbackBuffer, delta: dict[str, torch.Tensor], step: int):
    return decompress_lora_delta(buffer.compress(delta, "w0", "branch", outer_step=step))


def test_everything_computed_is_sent_or_still_buffered() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.05)
    deltas = _deltas(seed=1, steps=12)
    sent_total = {k: torch.zeros_like(v) for k, v in deltas[0].items()}
    for step, delta in enumerate(deltas, start=1):
        for k, v in _send(buffer, delta, step).items():
            sent_total[k] += v
    for k in sent_total:
        computed = sum(d[k] for d in deltas)
        assert torch.allclose(sent_total[k] + buffer.residuals[k], computed, atol=1e-4)


def test_residual_is_what_the_compressor_dropped() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.1)
    first, second = _deltas(seed=2, steps=2)
    _send(buffer, first, 1)
    plain = decompress_lora_delta(compress_lora_delta(first, "w0", "branch", 1, topk_ratio=0.1))
    for k in first:
        assert torch.equal(buffer.residuals[k], first[k] - plain[k])

    carried = {k: first[k] - plain[k] for k in first}
    sent = _send(buffer, second, 2)
    for k in second:
        assert torch.allclose(buffer.residuals[k], second[k] + carried[k] - sent[k], atol=1e-6)


def test_dense_ratio_leaves_only_fp8_rounding() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=1.0)
    delta = _deltas(seed=3, steps=1)[0]
    _send(buffer, delta, 1)
    delta_norm = float(torch.sqrt(sum(v.pow(2).sum() for v in delta.values())))
    assert 0.0 < buffer.residual_norm() < 0.05 * delta_norm


def test_error_feedback_beats_plain_topk_on_accumulated_signal() -> None:
    delta = {"w": torch.linspace(0.01, 1.0, 400)}
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.01)
    ef_total = torch.zeros(400)
    plain_total = torch.zeros(400)
    for step in range(1, 201):
        ef_total += _send(buffer, delta, step)["w"]
        plain_total += decompress_lora_delta(compress_lora_delta(delta, "w0", "b", step, topk_ratio=0.01))["w"]
    target = 200 * delta["w"]
    assert (plain_total == 0).sum() >= 390
    assert torch.norm(ef_total - target) < 0.5 * torch.norm(plain_total - target)


def test_stale_decay_scales_carried_residual_per_elapsed_step() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.1, stale_decay=0.5)
    first = {"w": torch.randn(64, generator=torch.Generator().manual_seed(4))}
    _send(buffer, first, 1)
    residual = buffer.residuals["w"].clone()
    zero = {"w": torch.zeros(64)}
    sent = _send(buffer, zero, 3)
    assert torch.allclose(buffer.residuals["w"] + sent["w"], 0.25 * residual, atol=1e-6)


def test_residual_norm_is_clipped() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer(topk_ratio=0.01, max_residual_norm=0.5)
    for step, delta in enumerate(_deltas(seed=5, steps=3), start=1):
        _send(buffer, delta, step)
        assert buffer.residual_norm() <= 0.5 + 1e-5
    assert buffer.clip_events == 3


def test_reset_clears_state() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer()
    _send(buffer, _deltas(seed=6, steps=1)[0], 7)
    buffer.reset()
    assert buffer.residuals == {} and buffer.last_outer_step is None and buffer.residual_norm() == 0.0
    _send(buffer, {"other": torch.ones(4)}, 1)


@pytest.mark.parametrize(
    "kwargs",
    [{"topk_ratio": 0.0}, {"topk_ratio": 1.5}, {"stale_decay": 0.0}, {"stale_decay": 1.2}, {"max_residual_norm": 0.0}],
)
def test_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EF21DiLoCoErrorFeedbackBuffer(**kwargs)


def test_rejects_inconsistent_updates() -> None:
    buffer = EF21DiLoCoErrorFeedbackBuffer()
    delta = _deltas(seed=7, steps=1)[0]
    _send(buffer, delta, 2)
    with pytest.raises(ValueError, match="outer_step must increase"):
        _send(buffer, delta, 2)
    with pytest.raises(KeyError):
        _send(buffer, {"lora_A": delta["lora_A"]}, 3)
    with pytest.raises(ValueError, match="shape"):
        _send(buffer, {"lora_A": torch.zeros(4, 4), "lora_B": delta["lora_B"]}, 3)


def test_synthetic_benchmark_meets_issue_56_targets() -> None:
    report = run_benchmark(seed=0)
    results = report["results"]
    assert results["top1_fp8_ef"]["final_holdout_mse"] < results["top1_fp8"]["final_holdout_mse"]
    assert report["comparison_top1_fp8_ef_vs_top5_fp16"]["holdout_mse_gap"] <= 0.02
    assert report["comparison_top1_fp8_ef_vs_top5_fp16"]["wire_bytes_reduction_x"] >= 5.0
    assert results["top1_fp8_ef"]["final_holdout_mse"] < report["initial_holdout_mse"]
    assert "Synthetic" in report["scope"]


def test_benchmark_cli_writes_report(tmp_path, capsys) -> None:
    out = tmp_path / "ef.json"
    assert main(["--outer-rounds", "5", "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["config"]["outer_rounds"] == 5
    assert json.loads(capsys.readouterr().out) == report
    assert main(["--outer-rounds", "1"]) == 0


def test_benchmark_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="unknown schemes"):
        run_benchmark(schemes=("dense", "magic"))
    assert "comparison_top1_fp8_ef_vs_top5_fp16" not in run_benchmark(outer_rounds=2, schemes=("dense",))
