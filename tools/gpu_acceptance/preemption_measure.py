#!/usr/bin/env python
"""Measure training-preemption behaviour as specified in docs/GPU_PREEMPTION_VALIDATION.md.

This tool REFUSES to emit preemption-latency or VRAM-release numbers unless torch.cuda is
available. Without CUDA it prints a BLOCKED status JSON and exits 2.

    python tools/gpu_acceptance/preemption_measure.py                  # needs CUDA, else BLOCKED (exit 2)
    python tools/gpu_acceptance/preemption_measure.py --out run.json
    python tools/gpu_acceptance/preemption_measure.py --functional-dry-run-cpu
        # CPU control-flow check only: NO latency and NO VRAM numbers, always exits non-zero (3 or 1)

Exit codes: 0 all functional checks passed on CUDA with measurements; 1 a functional check failed;
2 BLOCKED (no CUDA or non-CUDA backend); 3 CPU dry-run finished (not acceptance evidence).

Measurements come from torch.cuda allocator counters, torch.cuda.mem_get_info and (if present)
nvidia-smi. Allocator bytes exclude the CUDA context and driver overhead; compare with the
device-free numbers and record the difference rather than calling it "released".
The owning model object stays referenced in this process for the whole scenario.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "src").is_dir() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

SCHEMA_VERSION = 2
EXIT_OK, EXIT_FAIL, EXIT_BLOCKED, EXIT_DRY_RUN = 0, 1, 2, 3
BLOCKED_MESSAGE = (
    "BLOCKED: torch.cuda is unavailable. Preemption latency and VRAM release cannot be measured "
    "or inferred on this machine; no numbers are emitted."
)
MANUAL_CHECKS = [
    "Windows idle telemetry: GetLastInputInfo readings change after real keyboard/mouse input (needs an interactive session and owner permission).",
    "Recent input or a watched game process requests preemption through client/daemon.py on the real desktop.",
    "Missing telemetry (idle reading unavailable) pauses background work.",
    "Explicit benchmark opt-outs stay explicit and do not silently become production defaults.",
    "Controller recovery in client/worker_node.py: any cached device field is reconciled with the model's real device before the next forward pass.",
    "Trajectory-equivalent resume is NOT supported; do not claim it without a deterministic interrupted-vs-uninterrupted test.",
]
CLAIMS_NOT_SUPPORTED = [
    "100% VRAM release",
    "a fixed 1.5 second preemption deadline",
    "release of memory held by unrelated tensors or the CUDA context",
]


def blocked_report(reason: str = BLOCKED_MESSAGE) -> dict:
    return {
        "tool": "preemption_measure",
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED_NO_CUDA",
        "reason": reason,
        "preemption_latency_seconds": None,
        "vram_release": None,
        "measurements": None,
        "run_instead_on": "a Windows/CUDA machine, with the owner's permission, following docs/GPU_PREEMPTION_VALIDATION.md",
        "command": "python tools/gpu_acceptance/preemption_measure.py --out preemption_run.json",
    }


# ------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------


class TinyLoraModel(nn.Module):
    """Frozen dense base of ~`base_mib` MiB plus LoRA-named trainable parameters."""

    def __init__(self, base_mib: float = 1.0, rank: int = 8, fail_offload: bool = False):
        super().__init__()
        width = max(16, int(((base_mib * 2**20) / 4) ** 0.5))
        self.base = nn.Linear(width, width, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, width) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(width, rank) * 0.01)
        self.register_buffer("offset", torch.ones(1))
        self.fail_offload = fail_offload
        self.forward_calls = 0

    def forward(self, input_ids, labels=None):
        self.forward_calls += 1
        h = (input_ids.float().mean(dim=-1, keepdim=True) / 100.0).expand(-1, self.base.in_features).contiguous()
        out = self.base(h) + (h @ self.lora_a.T) @ self.lora_b.T + self.offset
        return SimpleNamespace(loss=(out**2).mean())

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if self.fail_offload and target is not None and torch.device(target).type == "cpu":
            raise RuntimeError("simulated model that rejects CPU offload")
        return super().to(*args, **kwargs)


def build_model(profile: str, dev: torch.device, args, *, fail_offload: bool = False) -> nn.Module:
    if profile == "tiny":
        model = TinyLoraModel(args.tiny_mib).to(dev)
        model.fail_offload = fail_offload  # armed only after the initial move to the device
        return model
    if fail_offload:
        raise ValueError("offload-rejection is only injected into the tiny profile")
    from hivemind.core.quantization import attach_qlora_adapters, build_local_qwen2_backbone

    base, _, _ = build_local_qwen2_backbone(
        vocab_size=4096, hidden_size=args.hidden_size, intermediate_size=int(args.hidden_size * 2.75),
        num_hidden_layers=args.layers, device=dev, use_4bit_nf4_if_cuda=args.nf4_reconstruct,
    )
    return attach_qlora_adapters(base, rank_r=8, lora_alpha=16, gradient_checkpointing=False).model


# ------------------------------------------------------------------------------------------
# Probes
# ------------------------------------------------------------------------------------------


def _is_cuda(dev: torch.device) -> bool:
    return dev.type == "cuda"


def cuda_mem(dev: torch.device) -> dict | None:
    if not _is_cuda(dev):
        return None
    torch.cuda.synchronize(dev)
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(dev)),
        "reserved_bytes": int(torch.cuda.memory_reserved(dev)),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _run(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        return subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def nvidia_smi_process_memory() -> dict:
    """Per-process memory as nvidia-smi reports it (WDDM often reports N/A). Read-only."""
    out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if out is None:
        return {"available": False}
    mine = [line for line in out.splitlines() if line.split(",")[0].strip() == str(os.getpid())]
    return {"available": True, "this_process_rows": mine, "all_rows_count": len(out.splitlines())}


def host_ram() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"available_bytes": int(vm.available), "total_bytes": int(vm.total)}
    except Exception:  # noqa: BLE001
        return {"available_bytes": None, "total_bytes": None}


def environment(dev: torch.device, args) -> dict:
    env = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_version_cuda": getattr(torch.version, "cuda", None),
        "torch_version_hip": getattr(torch.version, "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(dev),
        "model_profile": args.profile,
        "model_profile_args": {"tiny_mib": args.tiny_mib, "hidden_size": args.hidden_size, "layers": args.layers,
                               "nf4_reconstruct": args.nf4_reconstruct},
    }
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                           text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        env.update({"gpu_name": props.name, "gpu_capability": f"{props.major}.{props.minor}",
                    "gpu_total_memory_bytes": int(props.total_memory),
                    "gpu_backend": "hip" if env["torch_version_hip"] else "cuda",
                    "driver_and_gpu_nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])})
        if env["gpu_backend"] == "hip":
            env["warning"] = "ROCm/HIP build: results are not CUDA acceptance evidence."
    return env


# ------------------------------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------------------------------


def _all_on_cpu(model: nn.Module) -> bool:
    return all(t.device.type == "cpu" for t in list(model.parameters()) + list(model.buffers()))


def _valid_cpu_rng_state(state) -> bool:
    if not isinstance(state, torch.Tensor) or state.device.type != "cpu" or state.dtype != torch.uint8:
        return False
    try:
        torch.Generator(device="cpu").set_state(state)
        return True
    except (TypeError, RuntimeError):
        return False


def _valid_cuda_rng_states(states, required: bool) -> bool:
    return isinstance(states, list) and (bool(states) or not required) and all(
        isinstance(state, torch.Tensor) and state.device.type == "cpu"
        and state.dtype == torch.uint8 and state.ndim == 1 and state.numel() > 0
        for state in states
    )


def run_scenario(name: str, dev: torch.device, args, *, bystander_mib: int = 0, checkpoint_dir: Path | None = None,
                 fail_offload: bool = False, profile: str | None = None, restart: bool = False,
                 expect_checkpoint_failure: bool = False) -> dict:
    import hivemind.core.diloco_inner as inner
    from hivemind.core.quantization import extract_lora_state_dict

    profile = profile or args.profile
    k = args.steps_before_preempt
    functional: dict[str, bool] = {}
    notes: list[str] = []
    cuda = _is_cuda(dev)

    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    pre_model = cuda_mem(dev)
    ram_before = host_ram()
    model = build_model(profile, dev, args, fail_offload=fail_offload)  # owner stays alive below
    bystander = torch.empty(bystander_mib * 2**20, dtype=torch.uint8, device=dev) if (cuda and bystander_mib) else None
    model_bytes = sum(t.nbytes for t in list(model.parameters()) + list(model.buffers()))
    param_dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    after_model = cuda_mem(dev)
    functional["host_ram_sufficient_for_cpu_copy"] = (
        ram_before["available_bytes"] is None or ram_before["available_bytes"] > 2 * model_bytes
    )

    steps_counter = {"optimizer_steps": 0}
    real_builder = inner.build_inner_optimizer

    def tracked_builder(*a, **kw):
        opt = real_builder(*a, **kw)
        original_step = opt.step

        def counted_step(*sa, **skw):
            steps_counter["optimizer_steps"] += 1
            return original_step(*sa, **skw)

        opt.step = counted_step
        return opt

    trigger: dict = {}
    calls = {"n": 0}

    def should_preempt() -> bool:
        calls["n"] += 1
        if calls["n"] == k + 1:
            trigger["mem_at_trigger"] = cuda_mem(dev)  # optimizer state still alive here
            trigger["t"] = time.perf_counter()
            return True
        return False

    tokens = torch.randint(0, 4000, (8, 16))
    inner.build_inner_optimizer = tracked_builder
    try:
        result = inner.run_diloco_inner_loop(
            model, tokens, f"acceptance-{name}", "preemption-measure", inner_steps_H=k + 3, checkpoint_interval_s=2,
            should_preempt_fn=should_preempt, emergency_checkpoint_dir=checkpoint_dir,
        )
    finally:
        inner.build_inner_optimizer = real_builder
    t_return = time.perf_counter()
    mem_after = cuda_mem(dev)

    functional["preempted_flag_set"] = bool(result.preempted_by_daemon)
    functional["steps_completed_equals_trigger_boundary"] = result.steps_completed == k
    functional["no_optimizer_steps_after_preemption"] = steps_counter["optimizer_steps"] == k
    functional["theta_final_matches_stopped_weights"] = all(
        torch.equal(result.theta_final[key].cpu(), value.cpu()) for key, value in extract_lora_state_dict(model).items()
    ) and set(result.theta_final) == set(extract_lora_state_dict(model))
    if fail_offload:
        functional["offload_failure_is_reported"] = (not result.model_offloaded_to_cpu) and bool(result.model_offload_error)
        functional["offload_failure_not_claimed_as_success"] = not result.model_offloaded_to_cpu
    else:
        functional["registered_tensors_on_cpu"] = _all_on_cpu(model)
        functional["offload_flag_matches_reality"] = bool(result.model_offloaded_to_cpu) == _all_on_cpu(model)
        functional["no_offload_error"] = result.model_offload_error is None
    functional["original_device_recorded"] = result.preemption_original_device == str(dev)

    checkpoint_report: dict = {"requested": checkpoint_dir is not None,
                               "failure_expected": expect_checkpoint_failure}
    if checkpoint_dir is not None:
        if expect_checkpoint_failure:
            functional["checkpoint_failed_as_expected"] = result.emergency_dump_path is None
            functional["checkpoint_failure_is_visible"] = bool(result.emergency_checkpoint_error)
        else:
            functional["checkpoint_saved_as_expected"] = result.emergency_dump_path is not None
            functional["checkpoint_has_no_error"] = result.emergency_checkpoint_error is None
        leftovers = sorted(p.name for p in checkpoint_dir.glob(".emergency-*")) if checkpoint_dir.is_dir() else []
        checkpoint_report["leftover_temp_files"] = leftovers
        functional["no_partial_checkpoint_files"] = not leftovers
        if result.emergency_dump_path is not None:
            payload = torch.load(result.emergency_dump_path, map_location="cpu", weights_only=True)
            checkpoint_report.update({
                "loaded_with_cpu_map_location": True,
                "keys": sorted(payload),
                "saved_step": payload.get("step"),
                "optimizer_state_entries": len(payload["optimizer_state"]["state"]),
                "has_torch_rng_state": "torch_rng_state" in payload,
                "cuda_rng_state_devices": len(payload.get("cuda_rng_states", [])),
                "automatic_resume_supported": payload.get("automatic_resume_supported"),
            })
            functional["checkpoint_step_matches"] = payload.get("step") == k
            functional["checkpoint_has_optimizer_state"] = checkpoint_report["optimizer_state_entries"] > 0
            functional["checkpoint_has_valid_cpu_rng_state"] = _valid_cpu_rng_state(payload.get("torch_rng_state"))
            functional["checkpoint_has_valid_cuda_rng_states"] = _valid_cuda_rng_states(
                payload.get("cuda_rng_states"), required=dev.type == "cuda"
            )
            functional["checkpoint_marks_resume_unsupported"] = payload.get("automatic_resume_supported") is False
            functional["checkpoint_matches_stopped_weights"] = all(
                torch.equal(payload["lora_state"][key], result.theta_final[key]) for key in result.theta_final
            ) and set(payload["lora_state"]) == set(result.theta_final)
        else:
            checkpoint_report["emergency_checkpoint_error"] = result.emergency_checkpoint_error

    report: dict = {
        "scenario": name,
        "model_profile": profile,
        "model_bytes": int(model_bytes),
        "parameter_dtypes": param_dtypes,
        "result_fields": {
            "steps_completed": result.steps_completed,
            "model_offloaded_to_cpu": result.model_offloaded_to_cpu,
            "model_offload_error": result.model_offload_error,
            "emergency_checkpoint_error": result.emergency_checkpoint_error,
            "preemption_original_device": result.preemption_original_device,
        },
        "optimizer_steps_counted": steps_counter["optimizer_steps"],
        "host_ram_before": ram_before,
        "checkpoint": checkpoint_report,
        "notes": notes,
    }

    if cuda:
        at_trigger = trigger["mem_at_trigger"]
        measurements = {
            "owner_model_kept_alive": True,
            "pre_model": pre_model,
            "after_model_loaded": after_model,
            "at_trigger_with_optimizer_state": at_trigger,
            "after_preemption_return": mem_after,
            "result_cuda_allocated_before_bytes": result.cuda_allocated_before_bytes,
            "result_cuda_allocated_after_bytes": result.cuda_allocated_after_bytes,
            "optimizer_and_activation_bytes_over_loaded_model": at_trigger["allocated_bytes"] - after_model["allocated_bytes"],
            "allocated_bytes_released_by_preemption": at_trigger["allocated_bytes"] - mem_after["allocated_bytes"],
            "allocated_bytes_remaining_over_pre_model": mem_after["allocated_bytes"] - pre_model["allocated_bytes"],
            "reserved_bytes_remaining_over_pre_model": mem_after["reserved_bytes"] - pre_model["reserved_bytes"],
            "device_free_bytes_change_after_vs_trigger": mem_after["device_free_bytes"] - at_trigger["device_free_bytes"],
            "trigger_to_return_wall_seconds": round(t_return - trigger["t"], 4),
            "trigger_to_return_note": "includes emergency checkpoint write and post-loop delta compression; an upper bound on offload latency, not an offload-only time",
            "internal_emergency_dump_latency_sec": result.emergency_dump_latency_sec,
            "nvidia_smi_process_memory_after": nvidia_smi_process_memory(),
        }
        if bystander is not None:
            bystander_bytes = int(bystander.nbytes)
            measurements["bystander_bytes"] = bystander_bytes
            functional["bystander_tensor_still_allocated"] = mem_after["allocated_bytes"] >= bystander_bytes and bystander.device.type == "cuda"
            functional["result_does_not_claim_bystander_released"] = (
                result.cuda_allocated_after_bytes is None or result.cuda_allocated_after_bytes >= bystander_bytes
            )
        if not fail_offload:
            functional["cuda_allocated_dropped_after_offload"] = mem_after["allocated_bytes"] < at_trigger["allocated_bytes"]
        report["measurements"] = measurements
    else:
        report["measurements"] = None
        notes.append("CPU dry-run: no latency or VRAM numbers are produced or implied.")

    if restart and not fail_offload:
        # Controller-style recovery: the model stays on CPU until the caller explicitly moves it back.
        functional["model_still_on_cpu_before_explicit_restore"] = _all_on_cpu(model)
        model.to(dev)
        restored = cuda_mem(dev)
        again = inner.run_diloco_inner_loop(model, tokens, f"acceptance-{name}-restart", "preemption-measure",
                                            inner_steps_H=2, checkpoint_interval_s=2)
        functional["restart_runs_requested_steps"] = again.steps_completed == 2 and not again.preempted_by_daemon
        after_restart = cuda_mem(dev)
        report["restart"] = {
            "steps_completed": again.steps_completed,
            "continuation_is_fresh_optimizer_not_exact_resume": True,
            "after_model_restored_to_device": restored,
            "after_restart_run": after_restart,
        }

    report["functional_checks"] = functional
    report["functional_passed"] = all(functional.values())

    del model, bystander, result
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        report["after_cleanup"] = cuda_mem(dev)
    return report


def reference_offload_time(dev: torch.device, args) -> dict | None:
    """Wall time of a bare model.to('cpu') with CUDA sync, no optimizer, for comparison."""
    if dev.type != "cuda":
        return None
    model = build_model(args.profile, dev, args)
    torch.cuda.synchronize(dev)
    start = time.perf_counter()
    model.to("cpu")
    torch.cuda.synchronize(dev)
    seconds = round(time.perf_counter() - start, 4)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"model_to_cpu_wall_seconds": seconds, "note": "bare module move, synchronized; excludes optimizer, checkpoint and post-processing"}


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the JSON report here")
    ap.add_argument("--profile", choices=["tiny", "qwen2-local"], default=None,
                    help="default: qwen2-local on CUDA, tiny for the CPU dry-run")
    ap.add_argument("--tiny-mib", type=float, default=None, help="frozen base size for the tiny profile (default 256 on CUDA, 1 on CPU)")
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--nf4-reconstruct", action="store_true",
                    help="apply the repo's NF4 dense-reconstruction to the qwen2-local profile (still dense in VRAM)")
    ap.add_argument("--steps-before-preempt", type=int, default=3)
    ap.add_argument("--bystander-mib", type=int, default=64)
    ap.add_argument("--functional-dry-run-cpu", action="store_true",
                    help="exercise control flow on CPU; never produces latency/VRAM numbers; exits 3 (or 1)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cuda_available = torch.cuda.is_available()
    # An explicit CPU dry run must remain on CPU even on a CUDA-capable host.
    cuda = cuda_available and not args.functional_dry_run_cpu
    if not cuda_available and not args.functional_dry_run_cpu:
        report = blocked_report()
        _emit(report, args.out)
        print(BLOCKED_MESSAGE, file=sys.stderr)
        return EXIT_BLOCKED

    if cuda and getattr(torch.version, "hip", None):
        report = blocked_report("ROCm/HIP uses torch.cuda APIs but is not CUDA acceptance evidence.")
        report["status"] = "BLOCKED_NON_CUDA_BACKEND"
        _emit(report, args.out)
        return EXIT_BLOCKED

    dev = torch.device("cuda:0" if cuda else "cpu")
    args.profile = args.profile or ("qwen2-local" if cuda else "tiny")
    args.tiny_mib = args.tiny_mib if args.tiny_mib is not None else (256.0 if cuda else 1.0)

    report: dict = {"tool": "preemption_measure", "schema_version": SCHEMA_VERSION,
                    "environment": environment(dev, args), "manual_checks_not_run": MANUAL_CHECKS,
                    "claims_not_supported_by_this_tool_alone": CLAIMS_NOT_SUPPORTED, "scenarios": []}
    scratch = Path(tempfile.mkdtemp(prefix="hivemind-preempt-"))
    try:
        report["scenarios"].append(run_scenario("main", dev, args, checkpoint_dir=scratch / "main", restart=True))
        if cuda:
            report["scenarios"].append(run_scenario("unrelated_live_cuda_tensor", dev, args, bystander_mib=args.bystander_mib,
                                                    checkpoint_dir=scratch / "bystander"))
        blocker = scratch / "not_a_directory"
        blocker.write_text("regular file where a directory is required", encoding="utf-8")
        report["scenarios"].append(run_scenario("checkpoint_dir_unwritable", dev, args, checkpoint_dir=blocker / "sub", profile="tiny", expect_checkpoint_failure=True))
        report["scenarios"].append(run_scenario("model_rejects_cpu_offload", dev, args, checkpoint_dir=scratch / "reject",
                                                fail_offload=True, profile="tiny"))
        report["reference_bare_model_to_cpu"] = reference_offload_time(dev, args)
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if "error" in report:
        report["status"] = "FAIL_EXECUTION"
        _emit(report, args.out)
        return EXIT_FAIL
    functional_ok = all(s["functional_passed"] for s in report["scenarios"])
    if cuda:
        report["status"] = "MEASURED_ALL_FUNCTIONAL_PASSED" if functional_ok else "MEASURED_FUNCTIONAL_FAILURES"
        code = EXIT_OK if functional_ok else EXIT_FAIL
    else:
        report["status"] = "DRY_RUN_CPU_FUNCTIONAL_ONLY_NOT_ACCEPTANCE" if functional_ok else "DRY_RUN_CPU_FUNCTIONAL_FAILURES"
        report["preemption_latency_seconds"] = None
        report["vram_release"] = None
        code = EXIT_DRY_RUN if functional_ok else EXIT_FAIL
    _emit(report, args.out)
    return code


def _emit(report: dict, out: str | None) -> None:
    text = json.dumps(report, indent=2, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
