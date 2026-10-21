"""Phase B (Worker): DiLoCo-QLoRA Inner Loop with Deterministic Micro-Checkpointing & Hardware Safety Circuit-Breaker.

Executes H local AdamW optimization steps on frozen 4-bit NF4 base weights + trainable LoRA
adapters with zero network communication, recording cryptographic SHA-256 micro-checkpoints
every s steps for BOINC spot-check verification while enforcing Phase S-4 GPU hardware safety
invariants (thermal circuit-breaker, WDDM TDR step watchdog, gradient norm clipping, and
NaN/diverged loss early abort).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from hivemind.config import configure_deterministic_runtime
from hivemind.core.compression import (
    CompressedDeltaPackage,
    compress_lora_delta,
    compute_pseudo_gradient_delta,
)
from hivemind.core.quantization import (
    compute_state_dict_sha256,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from hivemind.security.gpu_execution_circuit_breaker import (
    DEFAULT_MAX_GRAD_NORM,
    GPUExecutionCircuitBreaker,
)


@dataclass
class MicroCheckpointRecord:
    """Cryptographic record of intermediate LoRA weights at step s."""

    step: int
    lora_sha256: str
    loss_value: float
    state_snapshot: dict[str, torch.Tensor] = field(repr=False)


@dataclass
class InnerLoopResult:
    """Output produced by a worker node after completing (or preempting) H inner steps."""

    worker_id: str
    branch_name: str
    outer_step: int
    steps_completed: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    theta_init: dict[str, torch.Tensor] = field(repr=False)
    theta_final: dict[str, torch.Tensor] = field(repr=False)
    raw_delta: dict[str, torch.Tensor] = field(repr=False)
    compressed_delta: CompressedDeltaPackage
    micro_checkpoints: dict[int, MicroCheckpointRecord]
    preempted_by_daemon: bool = False
    emergency_dump_path: Path | None = None
    emergency_dump_latency_sec: float = 0.0
    circuit_breaker_reason: str | None = None
    thermal_throttled_steps: int = 0
    max_step_latency_sec: float = 0.0


def build_inner_optimizer(
    trainable_params: list[nn.Parameter],
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    use_paged_8bit_adamw: bool = False,
) -> torch.optim.Optimizer:
    """Instantiate deterministic AdamW optimizer for DiLoCo inner loop."""
    _ = use_paged_8bit_adamw
    return torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def get_deterministic_batch(
    token_sequences: torch.Tensor,
    step: int,
    rng_seed: int,
    micro_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically index a micro-batch for step `step` given `rng_seed`."""
    num_samples = token_sequences.size(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(rng_seed) + int(step) * 10007)
    idx = torch.randint(0, num_samples, (micro_batch_size,), generator=gen)
    batch = token_sequences[idx].to(device, dtype=torch.long)
    return batch, batch.clone()


def run_diloco_inner_loop(
    model: nn.Module,
    token_sequences: torch.Tensor,
    worker_id: str,
    branch_name: str,
    outer_step: int = 1,
    inner_steps_H: int = 500,
    checkpoint_interval_s: int = 50,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    rng_seed: int = 42,
    micro_batch_size: int = 1,
    topk_ratio: float = 0.20,
    use_paged_8bit_adamw: bool = False,
    should_preempt_fn: Callable[[], bool] | None = None,
    emergency_checkpoint_dir: Path | None = None,
    max_grad_norm: float | None = DEFAULT_MAX_GRAD_NORM,
    circuit_breaker: GPUExecutionCircuitBreaker | None = None,
    enforce_battery_guard: bool = False,
    enable_thermal_sleep: bool = False,
) -> InnerLoopResult:
    """Execute H local AdamW steps on LoRA parameters with micro-checkpointing every s steps."""
    configure_deterministic_runtime(rng_seed)
    device = next(model.parameters()).device
    model.train()

    cb = circuit_breaker or GPUExecutionCircuitBreaker(
        allow_battery_execution=not enforce_battery_guard
    )
    seq_len = int(token_sequences.size(1)) if token_sequences.ndim >= 2 else int(token_sequences.numel())
    cb.verify_microbatch_tdr_safety(
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        raise_on_violation=True,
    )

    theta_init = extract_lora_state_dict(model)
    init_sha256 = compute_state_dict_sha256(theta_init)

    trainable_params = [p for name, p in model.named_parameters() if "lora_" in name or p.requires_grad]
    optimizer = build_inner_optimizer(
        trainable_params=trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        use_paged_8bit_adamw=use_paged_8bit_adamw,
    )

    micro_checkpoints: dict[int, MicroCheckpointRecord] = {
        0: MicroCheckpointRecord(
            step=0,
            lora_sha256=init_sha256,
            loss_value=0.0,
            state_snapshot=theta_init,
        )
    }

    start_time = time.perf_counter()
    initial_loss = 0.0
    final_loss = 0.0
    steps_completed = 0
    preempted = False
    dump_path: Path | None = None
    dump_latency = 0.0
    cb_reason: str | None = None
    thermal_throttled_steps = 0
    max_step_latency_sec = 0.0

    for step in range(1, inner_steps_H + 1):
        # 1. Hardware Circuit-Breaker & Zero-Intrusion Daemon preemption check
        should_poll_hw = (
            step == 1
            or step % max(1, checkpoint_interval_s) == 0
            or cb.simulated_telemetry is not None
            or cb.is_tripped
        )
        hw_preempt = False
        if should_poll_hw:
            cb_status = cb.evaluate(last_step_latency_sec=max_step_latency_sec)
            if cb_status.should_preempt:
                hw_preempt = True
                cb_reason = f"[{cb_status.rule_id}] {cb_status.reason}"
            elif cb_status.should_throttle:
                thermal_throttled_steps += 1
                if enable_thermal_sleep and cb_status.recommended_sleep_sec > 0.0:
                    time.sleep(cb_status.recommended_sleep_sec)

        daemon_preempt = bool(should_preempt_fn is not None and should_preempt_fn())
        if daemon_preempt or hw_preempt:
            preempted = True
            t_dump = time.perf_counter()
            if emergency_checkpoint_dir is not None:
                emergency_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                dump_path = emergency_checkpoint_dir / f"emergency_{worker_id}_step{step - 1}.pt"
                torch.save(
                    {
                        "worker_id": worker_id,
                        "branch_name": branch_name,
                        "outer_step": outer_step,
                        "step": step - 1,
                        "lora_state": extract_lora_state_dict(model),
                        "optimizer_state": optimizer.state_dict(),
                        "circuit_breaker_reason": cb_reason,
                    },
                    dump_path,
                )
            # Yield VRAM immediately
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            dump_latency = round(time.perf_counter() - t_dump, 4)
            break

        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels = get_deterministic_batch(
            token_sequences=token_sequences,
            step=step,
            rng_seed=rng_seed,
            micro_batch_size=micro_batch_size,
            device=device,
        )
        outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss_val = float(loss.detach().cpu().item())

        # Verify loss finiteness and workunit wall-clock deadline before backward/step
        elapsed_wall_clock = time.perf_counter() - start_time
        cb.verify_loss_and_wall_clock(
            loss_value=loss_val,
            step=step,
            elapsed_wall_clock_sec=elapsed_wall_clock,
            raise_on_violation=True,
        )

        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)

        del outputs, loss, input_ids, labels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_latency = time.perf_counter() - step_t0
        max_step_latency_sec = max(max_step_latency_sec, step_latency)

        if step == 1:
            initial_loss = loss_val
        final_loss = loss_val
        steps_completed = step

        if step % checkpoint_interval_s == 0 or step == inner_steps_H:
            snap = extract_lora_state_dict(model)
            sha = compute_state_dict_sha256(snap)
            micro_checkpoints[step] = MicroCheckpointRecord(
                step=step,
                lora_sha256=sha,
                loss_value=loss_val,
                state_snapshot=snap,
            )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - start_time, 4)
    if steps_completed in micro_checkpoints:
        theta_final = micro_checkpoints[steps_completed].state_snapshot
    else:
        theta_final = extract_lora_state_dict(model)
    raw_delta = compute_pseudo_gradient_delta(theta_init, theta_final)
    compressed_delta = compress_lora_delta(
        delta_dict=raw_delta,
        worker_id=worker_id,
        branch_name=branch_name,
        outer_step=outer_step,
        inner_steps_completed=steps_completed,
        topk_ratio=topk_ratio,
    )

    return InnerLoopResult(
        worker_id=worker_id,
        branch_name=branch_name,
        outer_step=outer_step,
        steps_completed=steps_completed,
        initial_loss=round(initial_loss, 6),
        final_loss=round(final_loss, 6),
        elapsed_seconds=elapsed,
        theta_init=theta_init,
        theta_final=theta_final,
        raw_delta=raw_delta,
        compressed_delta=compressed_delta,
        micro_checkpoints=micro_checkpoints,
        preempted_by_daemon=preempted,
        emergency_dump_path=dump_path,
        emergency_dump_latency_sec=dump_latency,
        circuit_breaker_reason=cb_reason,
        thermal_throttled_steps=thermal_throttled_steps,
        max_step_latency_sec=round(max_step_latency_sec, 6),
    )


def replay_inner_segment(
    model: nn.Module,
    start_lora_state: dict[str, torch.Tensor],
    token_sequences: torch.Tensor,
    start_step: int,
    end_step: int,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    rng_seed: int = 42,
    micro_batch_size: int = 1,
    initial_step_state: dict[str, torch.Tensor] | None = None,
    max_grad_norm: float | None = DEFAULT_MAX_GRAD_NORM,
) -> dict[str, torch.Tensor]:
    """Deterministically replay a segment of inner steps for BOINC Validator spot-checking."""
    configure_deterministic_runtime(rng_seed)
    device = next(model.parameters()).device
    model.train()

    # If verifying from step 0 or replaying from step 0 through end_step for exact AdamW moments
    replay_from = 0 if initial_step_state is not None else start_step
    base_state = initial_step_state if initial_step_state is not None else start_lora_state
    load_lora_state_dict(model, base_state)

    trainable_params = [p for name, p in model.named_parameters() if "lora_" in name or p.requires_grad]
    optimizer = build_inner_optimizer(
        trainable_params=trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        use_paged_8bit_adamw=False,
    )

    for step in range(replay_from + 1, end_step + 1):
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels = get_deterministic_batch(
            token_sequences=token_sequences,
            step=step,
            rng_seed=rng_seed,
            micro_batch_size=micro_batch_size,
            device=device,
        )
        outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        del outputs, loss, input_ids, labels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return extract_lora_state_dict(model)
