"""CPU verification of preemption control flow; no GPU release claims are inferred."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import hivemind.client.daemon as daemon_module
import hivemind.core.diloco_inner as inner_module
from hivemind.client.daemon import ZeroIntrusionDaemon
from hivemind.client.worker_node import HiveMindWorkerNode


class TinyModel(nn.Module):
    def __init__(self, fail_offload=False):
        super().__init__()
        self.lora_weight = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        self.register_buffer("offset", torch.ones(1))
        self.offload_calls = 0
        self.forward_calls = 0
        self.fail_offload = fail_offload

    def forward(self, input_ids, labels):
        self.forward_calls += 1
        return SimpleNamespace(loss=((self.lora_weight - input_ids.float().mean() / 10) ** 2).mean())

    def to(self, *args, **kwargs):
        self.offload_calls += 1
        if self.fail_offload:
            raise RuntimeError("simulated unsupported CPU offload")
        return super().to(*args, **kwargs)


def run_until_preemption(model, directory=None, **kwargs):
    calls = 0
    def preempt():
        nonlocal calls
        calls += 1
        return calls == 3
    return inner_module.run_diloco_inner_loop(
        model, torch.tensor([[1, 2], [3, 4]]), "worker", "test",
        inner_steps_H=5, checkpoint_interval_s=3, should_preempt_fn=preempt,
        emergency_checkpoint_dir=directory, **kwargs,
    )


def test_preemption_stops_training_and_offloads_registered_storage(tmp_path, monkeypatch):
    optimizers = []
    real_builder = inner_module.build_inner_optimizer
    def track_optimizer(*args, **kwargs):
        optimizer = real_builder(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer
    monkeypatch.setattr(inner_module, "build_inner_optimizer", track_optimizer)
    model = TinyModel()
    result = run_until_preemption(model, tmp_path)
    assert result.preempted_by_daemon and result.steps_completed == 2
    assert model.forward_calls == 2 and model.offload_calls == 1
    assert result.model_offloaded_to_cpu and result.model_offload_error is None
    assert all(t.device.type == "cpu" for t in list(model.parameters()) + list(model.buffers()))
    assert optimizers[0].state == {} and optimizers[0].param_groups == []
    assert result.preemption_original_device == "cpu"
    assert result.cuda_allocated_before_bytes is None and result.cuda_allocated_after_bytes is None
    assert result.emergency_checkpoint_error is None
    checkpoint = torch.load(result.emergency_dump_path, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2
    assert checkpoint["automatic_resume_supported"] is False
    assert checkpoint["optimizer_state"]["state"]
    for state in checkpoint["optimizer_state"]["state"].values():
        assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in state.values())
    assert torch.equal(checkpoint["lora_state"]["lora_weight"], result.theta_final["lora_weight"])
    assert torch.equal(checkpoint["initial_lora_state"]["lora_weight"], result.theta_init["lora_weight"])


def test_checkpoint_save_failure_still_stops_and_offloads(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(inner_module.torch, "save", fail_save)
    model = TinyModel()
    result = run_until_preemption(model, tmp_path)
    assert result.preempted_by_daemon and model.forward_calls == 2
    assert result.model_offloaded_to_cpu
    assert "disk full" in result.emergency_checkpoint_error
    assert result.emergency_dump_path is None
    assert list(tmp_path.iterdir()) == []


def test_offload_failure_is_reported_without_claiming_success(tmp_path):
    model = TinyModel(fail_offload=True)
    result = run_until_preemption(model, tmp_path)
    assert result.preempted_by_daemon and model.forward_calls == 2
    assert not result.model_offloaded_to_cpu
    assert "unsupported CPU offload" in result.model_offload_error
    assert result.emergency_dump_path.exists()
    assert result.cuda_allocated_after_bytes is None


def test_no_checkpoint_directory_does_not_prevent_offload():
    result = run_until_preemption(TinyModel())
    assert result.model_offloaded_to_cpu
    assert result.emergency_dump_path is None
    assert result.emergency_checkpoint_error is None


def test_uninterrupted_training_does_not_move_model():
    model = TinyModel()
    result = inner_module.run_diloco_inner_loop(model, torch.tensor([[1, 2]]), "worker", "test", inner_steps_H=2)
    assert not result.preempted_by_daemon
    assert model.offload_calls == 0
    assert not result.model_offloaded_to_cpu
    assert result.preemption_original_device is None


def test_emergency_filename_stays_inside_requested_directory(tmp_path):
    result = inner_module.run_diloco_inner_loop(
        TinyModel(), torch.tensor([[1, 2]]), "../../unsafe/worker", "test", inner_steps_H=2,
        should_preempt_fn=lambda: True, emergency_checkpoint_dir=tmp_path,
    )
    assert result.emergency_dump_path.parent == tmp_path
    assert result.emergency_dump_path.exists()


@pytest.fixture
def no_games(monkeypatch):
    monkeypatch.setattr(daemon_module, "detect_running_game_process", lambda: None)


def test_default_daemon_honors_real_idle_signal(no_games, monkeypatch):
    monkeypatch.setattr(daemon_module, "_query_windows_idle_seconds", lambda: 0.5)
    daemon = ZeroIntrusionDaemon()
    assert daemon.enforce_os_idle_hook
    assert daemon.poll().should_yield_vram
    monkeypatch.setattr(daemon_module, "_query_windows_idle_seconds", lambda: 120.0)
    assert not daemon.poll().should_yield_vram


@pytest.mark.parametrize("reading", [None, float("nan"), float("inf"), -1])
def test_unavailable_or_invalid_idle_signal_never_grants_background_work(no_games, monkeypatch, reading):
    monkeypatch.setattr(daemon_module, "_query_windows_idle_seconds", lambda: reading)
    telemetry = ZeroIntrusionDaemon().poll()
    assert telemetry.should_yield_vram
    assert not telemetry.os_idle_available
    assert "unavailable" in telemetry.reason


def test_explicit_simulation_and_idle_hook_optout_remain_available(no_games, monkeypatch):
    monkeypatch.setattr(daemon_module, "_query_windows_idle_seconds", lambda: None)
    simulated = ZeroIntrusionDaemon(simulated_idle_seconds=120).poll()
    assert not simulated.should_yield_vram and simulated.os_idle_available
    bypassed = ZeroIntrusionDaemon(enforce_os_idle_hook=False).poll()
    assert not bypassed.should_yield_vram and not bypassed.os_idle_available


def test_legacy_idle_getter_uses_conservative_zero_when_unavailable(monkeypatch):
    monkeypatch.setattr(daemon_module, "_query_windows_idle_seconds", lambda: None)
    assert daemon_module.get_windows_idle_seconds() == 0


def test_game_signal_requests_offload_without_claiming_measured_memory_release():
    telemetry = ZeroIntrusionDaemon(simulated_idle_seconds=120, simulated_active_game="cs2.exe").poll()
    assert telemetry.should_yield_vram
    assert "100%" not in telemetry.reason
    assert "requesting" in telemetry.reason


def test_worker_uses_safe_daemon_default_and_preserves_explicit_simulations():
    model = TinyModel()
    worker = HiveMindWorkerNode("worker", "steam", "tester", model)
    assert worker.daemon.enforce_os_idle_hook
    simulation = ZeroIntrusionDaemon(enforce_os_idle_hook=False)
    worker = HiveMindWorkerNode("worker", "steam", "tester", model, daemon=simulation)
    assert worker.daemon is simulation


def preemption_result_for_report(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"unit-test fixture")
    return SimpleNamespace(
        preempted_by_daemon=True, model_offloaded_to_cpu=True, model_offload_error=None,
        emergency_checkpoint_error=None, emergency_dump_path=path, emergency_dump_latency_sec=0.01,
        cuda_allocated_before_bytes=1024, cuda_allocated_after_bytes=0, preemption_original_device="cuda:0",
    )


@pytest.mark.parametrize("failure", ["offload_flag", "offload_error", "checkpoint_error", "missing_file", "no_preemption", "nan_latency"])
def test_stress_report_cannot_pass_a_fast_failed_preemption(tmp_path, failure):
    from hivemind.game.rtx_steam_tool_deployer import _preemption_report, _summarize_preemption_cycles
    result = preemption_result_for_report(tmp_path)
    if failure == "offload_flag":
        result.model_offloaded_to_cpu = False
    elif failure == "offload_error":
        result.model_offload_error = "offload failed"
    elif failure == "checkpoint_error":
        result.emergency_checkpoint_error = "disk full"
    elif failure == "missing_file":
        result.emergency_dump_path.unlink()
    elif failure == "no_preemption":
        result.preempted_by_daemon = False
    else:
        result.emergency_dump_latency_sec = float("nan")
    report = _preemption_report(result, torch.device("cuda:0"))
    assert not report["preempt_within_1500ms_sla"]
    if failure != "nan_latency":
        assert not report["preemption_succeeded"]
        assert not report["gpu_memory_reduction_observed"]
    report["allocated_after_preempt_mb"] = 0
    summary = _summarize_preemption_cycles([report, report], torch.device("cuda:0"))
    assert not summary["all_preemptions_under_1500ms"]


def test_cpu_stress_report_cannot_certify_gpu_offload(tmp_path):
    from hivemind.game.rtx_steam_tool_deployer import _preemption_report, _summarize_preemption_cycles
    result = preemption_result_for_report(tmp_path)
    report = _preemption_report(result, torch.device("cpu"))
    assert report["preemption_succeeded"]
    assert report["preempt_within_1500ms_sla"]
    assert not report["gpu_memory_reduction_observed"]
    assert report["gpu_validation_status"] == "unverified"
    report["allocated_after_preempt_mb"] = 0
    summary = _summarize_preemption_cycles([report, report], torch.device("cpu"))
    assert not summary["zero_vram_leak_verified"]
    assert not summary["all_gpu_preemptions_verified"]


@pytest.mark.parametrize("before,after", [(None, None), (0, 0), (1024, 1024), (1024, 2048)])
def test_gpu_success_requires_an_observed_allocator_reduction(tmp_path, before, after):
    from hivemind.game.rtx_steam_tool_deployer import _preemption_report
    result = preemption_result_for_report(tmp_path)
    result.cuda_allocated_before_bytes = before
    result.cuda_allocated_after_bytes = after
    assert not _preemption_report(result, torch.device("cuda:0"))["gpu_memory_reduction_observed"]


def test_single_cycle_does_not_establish_repeat_cycle_leak_check(tmp_path):
    from hivemind.game.rtx_steam_tool_deployer import _preemption_report, _summarize_preemption_cycles
    report = _preemption_report(preemption_result_for_report(tmp_path), torch.device("cuda:0"))
    report["allocated_after_preempt_mb"] = 0
    summary = _summarize_preemption_cycles([report], torch.device("cuda:0"))
    assert not summary["zero_vram_leak_verified"]
