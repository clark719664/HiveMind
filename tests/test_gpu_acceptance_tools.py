"""CPU-only tests for tools/gpu_acceptance: correctness checks and the no-CUDA refusal behaviour.

Nothing here measures a GPU. Tests of CUDA reporting and kernel dispatch use SYNTHETIC
counters or CPU-backed mocks to verify control flow, never hardware acceptance results.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "gpu_acceptance"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"gpu_acceptance_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nf4():
    return _load("nf4_dense_reference")


@pytest.fixture(scope="module")
def pre():
    return _load("preemption_measure")


@pytest.fixture(autouse=True)
def synthetic_ample_host_ram(pre, monkeypatch):
    # Functional tests are independent of the test runner's memory pressure.
    # Admission-failure tests override this explicitly. No hardware evidence.
    monkeypatch.setattr(pre, "host_ram", lambda: {"available_bytes": 64 * 2**30, "total_bytes": 64 * 2**30})


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _json_from(capsys):
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------- NF4 dense reference


def test_reference_codebook_matches_repo_constant(nf4):
    from hivemind.core.quantization import NF4_CODEBOOK

    assert [float(v) for v in NF4_CODEBOOK.tolist()] == nf4.REFERENCE_NF4_CODEBOOK
    assert len(nf4.REFERENCE_NF4_CODEBOOK) == 16 and 0.0 in nf4.REFERENCE_NF4_CODEBOOK


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,block", [((1,), 64), ((63,), 64), ((65,), 64), ((3, 5), 33), ((33, 129), 64), ((7, 13), 128)])
def test_repo_nf4_matches_dense_reference_on_odd_shapes(nf4, shape, block, dtype):
    weight = nf4.make_tensor("gaussian", shape, dtype, seed=3)
    result = nf4.check_nf4_case(weight, block, "gaussian")
    assert all(result["checks"].values()), {k: v for k, v in result["checks"].items() if not v}
    assert result["metrics"]["scale_byte_mismatches"] == 0
    assert result["metrics"]["nibble_mismatches"] == result["metrics"]["nibble_mismatches_that_are_near_ties"]


@pytest.mark.parametrize("kind", ["zeros", "constant", "alternating_sign", "single_outlier", "mixed_block_magnitude", "tiny", "huge"])
def test_edge_case_magnitudes_stay_inside_the_analytic_bound(nf4, kind):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        weight = nf4.make_tensor(kind, (32, 40), dtype, seed=1)
        result = nf4.check_nf4_case(weight, 64, kind)
        assert result["checks"]["roundtrip_within_analytic_bound"], (kind, dtype)
        assert result["checks"]["dense_matches_reference_within_tolerance"], (kind, dtype)


def test_double_quantization_dynamic_range_collapse_is_reported_not_hidden(nf4):
    """One large outlier sets the per-tensor super-scale; ordinary blocks flush to zero. The bound still holds."""
    weight = nf4.make_tensor("single_outlier", (64, 64), torch.float32, seed=0)
    metrics = nf4.check_nf4_case(weight, 64, "single_outlier")["metrics"]
    assert metrics["info_blocks_collapsed_to_zero"] >= metrics["info_blocks_total"] - 2


def test_empty_nonfinite_and_bad_block_sizes(nf4):
    assert all(case["passed"] for case in nf4.check_nonfinite_and_invalid())
    result = nf4.check_nf4_case(torch.empty(0, 4), 64, "empty")
    assert all(result["checks"].values())


def _corrupt(monkeypatch, mutate):
    import hivemind.core.quantization as q

    real = q.quantize_tensor_nf4_double_quant

    def corrupted(weight, block_size=64):
        state, _ = real(weight, block_size)
        mutate(state)
        return state, q.dequantize_tensor_nf4(state, dtype=weight.dtype)

    monkeypatch.setattr(q, "quantize_tensor_nf4_double_quant", corrupted)


def test_checker_detects_a_flipped_nibble(nf4, monkeypatch):
    def flip(state):
        state.packed_4bit[0] = (state.packed_4bit[0] ^ 0x0F) & 0xFF

    _corrupt(monkeypatch, flip)
    result = nf4.check_nf4_case(nf4.make_tensor("gaussian", (16, 16), torch.float32, 0), 64, "gaussian")
    assert not result["checks"]["nibbles_match_except_ties"]


def test_checker_detects_a_corrupted_scale_byte(nf4, monkeypatch):
    def bump(state):
        state.q_absmax_uint8[0] = 1 if int(state.q_absmax_uint8[0]) > 128 else 200

    _corrupt(monkeypatch, bump)
    result = nf4.check_nf4_case(nf4.make_tensor("gaussian", (16, 16), torch.float32, 0), 64, "gaussian")
    assert not result["checks"]["scale_bytes_match_except_ties"]


def test_int4_row_path_correct_but_overflow_is_a_finding(nf4):
    ok = nf4.check_int4_row_case(nf4.make_tensor("gaussian", (9, 17), torch.float32, 2), "gaussian")
    assert all(ok["checks"].values())
    huge = nf4.check_int4_row_case(nf4.make_tensor("huge", (4, 8), torch.float32, 2), "huge")
    assert huge["metrics"]["finding_decoded_nonfinite"] is True
    assert huge["metrics"]["scale_fp16_overflow_rows"] > 0


def test_quick_correctness_run_passes_and_has_no_gpu_fields(nf4):
    report = nf4.run_correctness(seed=0, quick=True)
    assert report["passed"], report["summary"]["failed_case_ids"]
    assert report["summary"]["nf4_failed"] == 0 and report["summary"]["int4_row_failed"] == 0


def test_nf4_cli_without_cuda_is_correctness_only(nf4, no_cuda, capsys):
    code = nf4.main(["--quick", "--skip-cpu-timing"])
    captured = capsys.readouterr()
    assert code == 0
    assert nf4.NO_CUDA_MESSAGE in captured.err
    report = json.loads(captured.out)
    assert report["status"] == "PASS_CORRECTNESS_ONLY_NO_CUDA"
    assert report["cuda"] == {"available": False, "message": nf4.NO_CUDA_MESSAGE, "measurements": None}
    assert set(report["cuda"]) == {"available", "message", "measurements"}
    assert "triton_fp8_parity" not in report["cuda"]


def test_nf4_cli_cpu_timings_are_labelled_wall_clock(nf4, no_cuda, capsys):
    assert nf4.main(["--quick"]) == 0
    timings = _json_from(capsys)["cpu_wall_clock_timings"]
    assert "CPU wall-clock" in timings["note"] and "not comparable to GPU" in timings["note"]
    assert all("cpu_wall_clock_ms" in key for key in timings["float32"] if key != "shape")


def test_nf4_require_cuda_blocks_without_cuda(nf4, no_cuda, capsys):
    assert nf4.main(["--quick", "--require-cuda"]) == 2
    report = _json_from(capsys)
    assert report["status"] == "BLOCKED_NO_CUDA" and report["cpu_correctness"] is None


# ---------------------------------------------------------------- preemption measurement


def test_preemption_blocked_without_cuda_emits_no_numbers(pre, no_cuda, capsys):
    assert pre.main([]) == pre.EXIT_BLOCKED == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "BLOCKED_NO_CUDA"
    assert report["preemption_latency_seconds"] is None and report["vram_release"] is None
    assert report["measurements"] is None and "scenarios" not in report
    assert "BLOCKED" in captured.err
    assert "wall_seconds" not in captured.out and "allocated_bytes" not in captured.out


@pytest.mark.skipif(torch.cuda.is_available(), reason="subprocess cannot be forced CUDA-less on a CUDA machine")
def test_preemption_script_exit_code_is_nonzero_when_run_for_real():
    proc = subprocess.run([sys.executable, str(TOOLS / "preemption_measure.py")], capture_output=True, text=True,
                          cwd=ROOT, timeout=300)
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["status"] == "BLOCKED_NO_CUDA"


def test_cpu_dry_run_checks_control_flow_but_never_reports_gpu_numbers(pre, no_cuda, capsys):
    code = pre.main(["--functional-dry-run-cpu"])
    text = capsys.readouterr().out
    report = json.loads(text)
    assert code == pre.EXIT_DRY_RUN == 3, report.get("status")
    assert report["status"] == "DRY_RUN_CPU_FUNCTIONAL_ONLY_NOT_ACCEPTANCE"
    assert report["preemption_latency_seconds"] is None and report["vram_release"] is None
    assert {s["scenario"] for s in report["scenarios"]} == {"main", "checkpoint_dir_unwritable", "model_rejects_cpu_offload"}
    assert all(s["measurements"] is None for s in report["scenarios"])
    for forbidden in ("trigger_to_return_wall_seconds", "allocated_bytes_released_by_preemption", "device_free_bytes"):
        assert forbidden not in text
    by_name = {s["scenario"]: s for s in report["scenarios"]}
    assert by_name["main"]["functional_passed"] and by_name["main"]["optimizer_steps_counted"] == 3
    assert by_name["main"]["checkpoint"]["automatic_resume_supported"] is False
    assert by_name["main"]["restart"]["continuation_is_fresh_optimizer_not_exact_resume"] is True
    assert by_name["checkpoint_dir_unwritable"]["result_fields"]["emergency_checkpoint_error"].startswith("Could not save")
    assert by_name["checkpoint_dir_unwritable"]["checkpoint"]["leftover_temp_files"] == []
    rejected = by_name["model_rejects_cpu_offload"]["result_fields"]
    assert rejected["model_offloaded_to_cpu"] is False and "simulated model that rejects" in rejected["model_offload_error"]
    assert report["manual_checks_not_run"] and "100% VRAM release" in report["claims_not_supported_by_this_tool_alone"]


def test_cuda_reporting_branch_runs_with_synthetic_counters(pre, monkeypatch, tmp_path):
    """Smoke test only: proves the CUDA-branch code executes. The counters below are fabricated."""
    sequence = iter([0, 500_000_000, 600_000_000, 100_000_000])

    def fake_mem(dev):
        value = next(sequence, 100_000_000)
        return {"allocated_bytes": value, "reserved_bytes": value, "device_free_bytes": 10**10 - value, "device_total_bytes": 10**10}

    monkeypatch.setattr(pre, "_is_cuda", lambda dev: True)
    monkeypatch.setattr(pre, "cuda_mem", fake_mem)
    monkeypatch.setattr(pre, "nvidia_smi_process_memory", lambda: {"available": False})
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    report = pre.run_scenario("synthetic", torch.device("cpu"), args, bystander_mib=1, checkpoint_dir=tmp_path / "ckpt", restart=True)
    measured = report["measurements"]
    assert measured["owner_model_kept_alive"] is True
    assert isinstance(measured["trigger_to_return_wall_seconds"], float)
    assert measured["allocated_bytes_released_by_preemption"] == 600_000_000 - 100_000_000
    assert measured["bystander_bytes"] == 2**20
    assert "after_cleanup" in report and "restart" in report


# ---------------------------------------------------------------- reporting regressions
# All CUDA results below are synthetic branch tests, never hardware evidence.


def _synthetic_nf4_row(shape, dtype):
    snapshot = {"allocated_bytes": 0, "reserved_bytes": 0, "device_free_bytes": 1024, "device_total_bytes": 1024}
    timing = {"unit": "ms (torch.cuda.Event)", "warmup_iters": 0, "iters": 1, "median": 1.0, "p95": 1.0, "min": 1.0}
    return {"shape": list(shape), "dtype": str(dtype).replace("torch.", ""), "status": "MEASURED",
            "quantize_peak_allocated_over_baseline_bytes": 16,
            "dequantize_peak_allocated_over_pre_call_bytes": 16, "dense_output_bytes": 16,
            "memory_bytes_torch_allocator": {
                "baseline": snapshot, "original_dense_weight_resident": snapshot,
                "plus_packed_buffers_on_gpu": snapshot, "plus_dequantized_dense_output": snapshot,
                "dense_weight_alone_bytes": 16, "packed_buffers_alone_bytes": 8,
            }, "latency": {name: timing for name in (
                "dequantize_gpu_resident_packed", "dequantize_cpu_resident_packed_includes_h2d_copy",
                "dense_linear_reference", "dequantize_then_linear_gpu_resident",
            )}, "synthetic_test_only": True}


@pytest.fixture
def synthetic_nf4_cuda(nf4, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(nf4, "environment", lambda: {"synthetic_test_only": True})
    monkeypatch.setattr(nf4, "run_correctness", lambda *a: {"passed": True})
    monkeypatch.setattr(nf4, "cuda_triton_parity", lambda: {"ran": True, "passed": True, "status": "PASSED"})

    def measured(shape, dtype, *args):
        return _synthetic_nf4_row(shape, dtype)

    monkeypatch.setattr(nf4, "_measure_one_cuda", measured)


def _nf4_synthetic_main(nf4):
    return nf4.main(["--require-cuda", "--skip-cpu-timing", "--cuda-shapes", "2x2", "--iters", "1", "--warmup", "0"])


def test_nf4_all_requested_synthetic_measurements_must_succeed(nf4, synthetic_nf4_cuda, capsys):
    assert _nf4_synthetic_main(nf4) == 0
    report = _json_from(capsys)
    assert report["status"] == "PASS_CORRECTNESS_AND_CUDA_MEASURED"
    assert report["cuda"]["measurement_summary"] == {
        "requested": 3, "measured": 3, "errors": 0, "skipped": 0, "invalid_rows": [], "complete": True,
    }


@pytest.mark.parametrize("failure_dtype", [torch.float16, torch.bfloat16, torch.float32, None])
def test_nf4_caught_measurement_exceptions_cannot_pass(nf4, synthetic_nf4_cuda, monkeypatch, capsys, failure_dtype):
    original = nf4._measure_one_cuda

    def broken(shape, dtype, *args):
        if failure_dtype is None or dtype == failure_dtype:
            raise RuntimeError("injected CUDA measurement failure")
        return original(shape, dtype, *args)

    monkeypatch.setattr(nf4, "_measure_one_cuda", broken)
    assert _nf4_synthetic_main(nf4) == 1
    report = _json_from(capsys)
    assert report["cpu_correctness"]["passed"] is True
    assert report["status"] == "FAIL_CUDA_MEASUREMENTS"
    assert report["cuda"]["measurement_summary"]["errors"] == (3 if failure_dtype is None else 1)
    assert not report["cuda"]["measurement_summary"]["complete"]


def test_nf4_top_level_measurement_error_emits_failed_json(nf4, synthetic_nf4_cuda, monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(nf4, "measure_cuda", fail)
    assert _nf4_synthetic_main(nf4) == 1
    assert _json_from(capsys)["status"] == "FAIL_CUDA_MEASUREMENTS"


def test_nf4_unsupported_dtype_is_visible_and_blocks_full_acceptance(nf4, synthetic_nf4_cuda, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _nf4_synthetic_main(nf4) == 2
    report = _json_from(capsys)
    assert report["status"] == "BLOCKED_INCOMPLETE_CUDA_MEASUREMENTS"
    assert report["cuda"]["measurement_summary"]["skipped"] == 1
    assert report["cuda"]["measurements"][1]["status"] == "SKIPPED_UNSUPPORTED"


@pytest.mark.parametrize("measurements", [[], [{"shape": [2, 2], "dtype": "float32"}], [
    _synthetic_nf4_row((2, 2), torch.float32),
] * 3])
def test_nf4_missing_unknown_or_duplicate_measurements_never_pass(nf4, synthetic_nf4_cuda, monkeypatch, capsys, measurements):
    monkeypatch.setattr(nf4, "measure_cuda", lambda *args: measurements)
    assert _nf4_synthetic_main(nf4) == 2
    assert _json_from(capsys)["status"] == "BLOCKED_INCOMPLETE_CUDA_MEASUREMENTS"


@pytest.mark.parametrize("parity,expected_code,expected_status", [
    ({"ran": True, "passed": False, "status": "FAILED"}, 1, "FAIL_CUDA_PARITY"),
    ({"ran": False, "status": "ERROR", "error": "launch failed"}, 1, "FAIL_CUDA_PARITY"),
    ({"ran": False, "status": "SKIPPED_UNAVAILABLE"}, 2, "BLOCKED_CUDA_PARITY_UNAVAILABLE"),
])
def test_nf4_parity_failure_and_unavailable_are_not_passes(nf4, synthetic_nf4_cuda, monkeypatch, capsys,
                                                         parity, expected_code, expected_status):
    monkeypatch.setattr(nf4, "cuda_triton_parity", lambda: parity)
    assert _nf4_synthetic_main(nf4) == expected_code
    assert _json_from(capsys)["status"] == expected_status


@pytest.mark.parametrize("args", [["--cuda-shapes", ""], ["--cuda-shapes", "0x4"], ["--cuda-shapes", "-1"],
                                  ["--cuda-shapes", "2x2,"], ["--iters", "0"], ["--warmup", "-1"]])
def test_nf4_invalid_measurement_configuration_fails_before_work(nf4, monkeypatch, args):
    def unexpected():
        raise AssertionError("invalid inputs reached environment/workload probe")

    monkeypatch.setattr(nf4, "environment", unexpected)
    with pytest.raises(SystemExit) as exc:
        nf4.main(args)
    assert exc.value.code == 2


def test_nf4_non_cuda_backend_cannot_claim_cuda_acceptance(nf4, synthetic_nf4_cuda, monkeypatch, capsys):
    monkeypatch.setattr(torch.version, "hip", "synthetic HIP")
    assert _nf4_synthetic_main(nf4) == 2
    report = _json_from(capsys)
    assert report["status"] == "BLOCKED_NON_CUDA_BACKEND" and report["cuda"]["measurements"] is None


@pytest.mark.parametrize("mode", ["success", "mismatch", "launch_error", "async_error"])
def test_triton_probe_calls_kernel_directly_and_never_silent_fallback(nf4, monkeypatch, mode):
    from types import SimpleNamespace
    from hivemind.swarm import swarm_fabric as sf

    calls = []

    class SyntheticKernel:
        def __getitem__(self, grid):
            def launch(x, quantized, scales, n_cols, **kwargs):
                calls.append(grid)
                if mode == "launch_error":
                    raise RuntimeError("injected Triton launch failure")
                scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
                scales.copy_(scale)
                quantized.copy_((x / scale).clamp(-448.0, 448.0))
                if mode == "mismatch":
                    quantized.zero_()
            return launch

    def no_fallback(*args):
        raise AssertionError("must not call production wrapper with silent fallback")

    def sync(*args):
        if mode == "async_error":
            raise RuntimeError("injected asynchronous kernel failure")

    real_randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: real_randn(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(sf, "_HAS_TRITON", True)
    monkeypatch.setattr(sf, "triton", SimpleNamespace(next_power_of_2=lambda n: n), raising=False)
    monkeypatch.setattr(sf, "_triton_fp8_quant_kernel", SyntheticKernel(), raising=False)
    monkeypatch.setattr(sf, "compress_hidden_states_fp8", no_fallback)
    result = nf4.cuda_triton_parity()
    assert calls == [(64,)]
    if mode in ("launch_error", "async_error"):
        assert result["status"] == "ERROR" and result["ran"] is False
        assert "passed" not in result
    else:
        assert result["ran"] is True and result["backend"] == "triton_direct_no_fallback"
        assert result["passed"] is (mode == "success")


def test_nf4_later_int4_error_preserves_prior_failures(nf4, monkeypatch):
    count = 0

    def fail_then_raise(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected later int4 failure")
        return {"checks": {"injected_failure": False}, "metrics": {}}

    monkeypatch.setattr(nf4, "check_int4_row_case", fail_then_raise)
    result = nf4.run_correctness(0, quick=True)
    assert not result["passed"] and not result["int4_row_path"]["available"]
    assert result["summary"]["int4_row_failed"] == 1
    assert result["int4_row_path"]["cases"][0]["passed"] is False


def test_cpu_dry_run_forces_cpu_even_when_cuda_is_visible(pre, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU dry run reached a CUDA workload or hardware probe")

    for name in ("synchronize", "empty_cache", "mem_get_info", "memory_allocated", "memory_reserved",
                 "get_device_properties", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    real_build = pre.build_model
    seen = []

    def cpu_only(profile, dev, args, **kwargs):
        assert dev.type == "cpu"
        assert profile == "tiny" and args.tiny_mib == 1.0
        seen.append(str(dev))
        return real_build(profile, dev, args, **kwargs)

    monkeypatch.setattr(pre, "build_model", cpu_only)
    monkeypatch.setattr(pre, "nvidia_smi_process_memory", forbidden)
    assert pre.main(["--functional-dry-run-cpu"]) == pre.EXIT_DRY_RUN
    report = _json_from(capsys)
    assert seen == ["cpu", "cpu", "cpu"]
    assert report["environment"]["cuda_available"] is True
    assert report["environment"]["device"] == "cpu"
    assert report["status"] == "DRY_RUN_CPU_FUNCTIONAL_ONLY_NOT_ACCEPTANCE"
    assert report["reference_bare_model_to_cpu"] is None
    assert all(s["measurements"] is None for s in report["scenarios"])


@pytest.mark.parametrize("name", ["main", "unrelated_live_cuda_tensor", "model_rejects_cpu_offload"])
def test_unexpected_checkpoint_failure_is_not_functional_success(pre, no_cuda, monkeypatch, tmp_path, name):
    import hivemind.core.diloco_inner as inner

    real_run = inner.run_diloco_inner_loop

    def missing_checkpoint(*args, **kwargs):
        result = real_run(*args, **kwargs)
        result.emergency_dump_path = None
        result.emergency_checkpoint_error = "injected unexpected disk failure"
        return result

    monkeypatch.setattr(inner, "run_diloco_inner_loop", missing_checkpoint)
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    result = pre.run_scenario(name, torch.device("cpu"), args, checkpoint_dir=tmp_path / "ckpt")
    assert not result["functional_passed"]
    assert not result["functional_checks"]["checkpoint_saved_as_expected"]
    assert not result["functional_checks"]["checkpoint_has_no_error"]


def test_injected_checkpoint_failure_scenario_rejects_unexpected_success(pre, no_cuda, tmp_path):
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    result = pre.run_scenario("unwritable_but_injection_broken", torch.device("cpu"), args,
                              checkpoint_dir=tmp_path / "writable", expect_checkpoint_failure=True)
    assert not result["functional_passed"]
    assert not result["functional_checks"]["checkpoint_failed_as_expected"]
    assert not result["functional_checks"]["checkpoint_failure_is_visible"]


@pytest.mark.parametrize("functional_ok,code", [(True, 0), (False, 1)])
def test_preemption_synthetic_cuda_main_gates_functional_outcome(pre, monkeypatch, capsys, functional_ok, code):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(pre, "environment", lambda *a: {"synthetic_test_only": True})
    monkeypatch.setattr(pre, "reference_offload_time", lambda *a: {"synthetic_test_only": True})

    def synthetic_scenario(name, dev, args, **kwargs):
        assert dev.type == "cuda"
        assert kwargs.get("expect_checkpoint_failure", False) == (name == "checkpoint_dir_unwritable")
        return {"scenario": name, "functional_passed": functional_ok, "synthetic_test_only": True}

    monkeypatch.setattr(pre, "run_scenario", synthetic_scenario)
    assert pre.main([]) == code
    report = _json_from(capsys)
    assert len(report["scenarios"]) == 4
    assert report["status"] == ("MEASURED_ALL_FUNCTIONAL_PASSED" if functional_ok else "MEASURED_FUNCTIONAL_FAILURES")


def test_preemption_execution_failure_emits_failed_json(pre, no_cuda, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("injected scenario failure")

    monkeypatch.setattr(pre, "run_scenario", fail)
    assert pre.main(["--functional-dry-run-cpu"]) == 1
    report = _json_from(capsys)
    assert report["status"] == "FAIL_EXECUTION" and report["error"].endswith("injected scenario failure")


def test_preemption_hip_blocks_cuda_acceptance_before_work(pre, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "synthetic HIP")
    assert pre.main([]) == 2
    assert _json_from(capsys)["status"] == "BLOCKED_NON_CUDA_BACKEND"


@pytest.mark.parametrize("field,invalid", [("torch_rng_state", None), ("torch_rng_state", torch.ones(4)),
                                          ("cuda_rng_states", None), ("cuda_rng_states", [torch.ones(4)])])
def test_checkpoint_missing_or_invalid_rng_state_cannot_pass(pre, no_cuda, monkeypatch, tmp_path, field, invalid):
    real_load = torch.load

    def corrupt(*args, **kwargs):
        payload = real_load(*args, **kwargs)
        if invalid is None:
            payload.pop(field)
        else:
            payload[field] = invalid
        return payload

    monkeypatch.setattr(torch, "load", corrupt)
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    result = pre.run_scenario("corrupt_rng", torch.device("cpu"), args, checkpoint_dir=tmp_path / "ckpt")
    assert not result["functional_passed"]


def test_cuda_checkpoint_requires_nonempty_rng_state_list(pre):
    assert not pre._valid_cuda_rng_states([], required=True)
    assert pre._valid_cuda_rng_states([], required=False)
    assert not pre._valid_cuda_rng_states(None, required=False)
    assert pre._valid_cuda_rng_states([torch.zeros(16, dtype=torch.uint8)], required=True)


@pytest.mark.parametrize("row", [None, {"shape": None}, {"shape": [2, 2]},
                                 {"shape": [2, 2], "dtype": "float16", "status": "MEASURED"}])
def test_nf4_malformed_or_unmeasured_success_rows_fail_closed(nf4, synthetic_nf4_cuda, monkeypatch, capsys, row):
    monkeypatch.setattr(nf4, "measure_cuda", lambda *a: [row, _synthetic_nf4_row((2, 2), torch.float32)])
    assert _nf4_synthetic_main(nf4) == 1
    report = _json_from(capsys)
    assert report["status"] == "FAIL_CUDA_MEASUREMENTS"
    assert report["cuda"]["measurement_summary"]["invalid_rows"] == [0]


def test_nf4_nonfinite_timing_cannot_be_a_successful_measurement(nf4, synthetic_nf4_cuda, monkeypatch, capsys):
    row = _synthetic_nf4_row((2, 2), torch.float16)
    row["latency"]["dense_linear_reference"]["median"] = float("nan")
    monkeypatch.setattr(nf4, "measure_cuda", lambda *a: [row])
    assert _nf4_synthetic_main(nf4) == 1
    assert _json_from(capsys)["status"] == "FAIL_CUDA_MEASUREMENTS"


@pytest.mark.parametrize("profile", ["tiny", "qwen2-local"])
@pytest.mark.parametrize("available,status", [(None, "BLOCKED_HOST_RAM_UNKNOWN"),
                                             ("unknown", "BLOCKED_HOST_RAM_UNKNOWN"),
                                             (-1, "BLOCKED_HOST_RAM_UNKNOWN"),
                                             (True, "BLOCKED_HOST_RAM_UNKNOWN"),
                                             (0, "BLOCKED_INSUFFICIENT_HOST_RAM"),
                                             (1, "BLOCKED_INSUFFICIENT_HOST_RAM")])
def test_host_ram_block_precedes_all_model_and_scenario_allocations(pre, no_cuda, monkeypatch, capsys,
                                                                  profile, available, status):
    monkeypatch.setattr(pre, "host_ram", lambda: {"available_bytes": available, "total_bytes": None})

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked host-RAM preflight reached allocation or measurement")

    monkeypatch.setattr(pre, "build_model", forbidden)
    monkeypatch.setattr(pre, "cuda_mem", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "randn", forbidden)
    monkeypatch.setattr(torch, "randint", forbidden)
    assert pre.main(["--functional-dry-run-cpu", "--profile", profile]) == 2
    report = _json_from(capsys)
    assert report["status"] == status
    assert report["scenarios"] == []
    assert report["measurements"] is None and report["vram_release"] is None
    assert report["host_ram_preflight"]["required_host_ram_bytes"] > 0
    assert "model allocation was not started" in report["reason"]


@pytest.mark.parametrize("snapshot", [None, {}, {"available_bytes": float("nan")}])
def test_missing_or_malformed_host_ram_snapshot_blocks(pre, monkeypatch, snapshot):
    monkeypatch.setattr(pre, "host_ram", lambda: snapshot)
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    with pytest.raises(pre.HostRamPreflightBlocked) as exc:
        pre.require_host_ram("tiny", args)
    assert exc.value.report["status"] == "BLOCKED_HOST_RAM_UNKNOWN"


def test_host_ram_probe_exception_blocks(pre, monkeypatch):
    def unavailable():
        raise RuntimeError("synthetic telemetry failure")

    monkeypatch.setattr(pre, "host_ram", unavailable)
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    with pytest.raises(pre.HostRamPreflightBlocked) as exc:
        pre.require_host_ram("tiny", args)
    assert exc.value.report["status"] == "BLOCKED_HOST_RAM_UNKNOWN"


@pytest.mark.parametrize("available", [None, 1])
def test_reference_offload_preflight_blocks_before_gpu_or_model_work(pre, monkeypatch, available):
    monkeypatch.setattr(pre, "host_ram", lambda: {"available_bytes": available, "total_bytes": None})

    def forbidden(*args, **kwargs):
        raise AssertionError("blocked reference probe reached model/GPU work")

    monkeypatch.setattr(pre, "build_model", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    with pytest.raises(pre.HostRamPreflightBlocked):
        pre.reference_offload_time(torch.device("cuda:0"), args)


def test_preflight_requires_the_full_budget_and_bystander_headroom(pre, monkeypatch):
    args = pre.parse_args([])
    args.profile, args.tiny_mib = "tiny", 1.0
    budget = pre.HOST_RAM_MODEL_MULTIPLIER * pre.model_storage_upper_bound("tiny", args) + pre.HOST_RAM_HEADROOM_BYTES
    monkeypatch.setattr(pre, "host_ram", lambda: {"available_bytes": budget - 1})
    with pytest.raises(pre.HostRamPreflightBlocked):
        pre.require_host_ram("tiny", args)
    monkeypatch.setattr(pre, "host_ram", lambda: {"available_bytes": budget})
    assert pre.require_host_ram("tiny", args)["status"] == "ADMITTED"
    with pytest.raises(pre.HostRamPreflightBlocked):
        pre.require_host_ram("tiny", args, bystander_mib=1)


@pytest.mark.parametrize("profile", ["tiny", "qwen2-local"])
def test_preallocation_storage_bound_covers_actual_small_cpu_profiles(pre, no_cuda, profile):
    args = pre.parse_args([])
    args.profile, args.tiny_mib, args.hidden_size, args.layers = profile, 0.25, 16, 1
    bound = pre.model_storage_upper_bound(profile, args)
    model = pre.build_model(profile, torch.device("cpu"), args)
    actual = sum(t.nbytes for t in list(model.parameters()) + list(model.buffers()))
    assert 0 < actual <= bound


def test_admitted_cpu_scenarios_report_preflight_without_gpu_evidence(pre, no_cuda, capsys):
    assert pre.main(["--functional-dry-run-cpu"]) == 3
    report = _json_from(capsys)
    for scenario in report["scenarios"]:
        guard = scenario["host_ram_preflight"]
        assert guard["status"] == "ADMITTED"
        assert guard["host_ram_before"]["available_bytes"] >= guard["required_host_ram_bytes"]
        assert scenario["model_bytes"] <= guard["estimated_model_storage_upper_bound_bytes"]
        assert scenario["functional_checks"]["host_ram_sufficient_for_cpu_copy"] is True
        assert scenario["measurements"] is None


def test_unknown_ram_blocks_cuda_main_before_environment_probe(pre, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(pre, "host_ram", lambda: {"available_bytes": None})

    def forbidden(*args, **kwargs):
        raise AssertionError("RAM admission failure reached a CUDA context/environment probe")

    monkeypatch.setattr(pre, "environment", forbidden)
    monkeypatch.setattr(pre, "build_model", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    assert pre.main([]) == 2
    report = _json_from(capsys)
    assert report["status"] == "BLOCKED_HOST_RAM_UNKNOWN"
    assert report["scenarios"] == [] and "environment" not in report
