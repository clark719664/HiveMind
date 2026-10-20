"""CPU-only tests for tools/gpu_acceptance: correctness checks and the no-CUDA refusal behaviour.

Nothing here measures a GPU. The one test that executes the CUDA reporting branch feeds it
SYNTHETIC counters purely to prove the branch runs; its numbers are never asserted as results.
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
