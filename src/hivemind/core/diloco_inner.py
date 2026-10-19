"""Phase B (Worker): DiLoCo-QLoRA Inner Loop with Deterministic Micro-Checkpointing.

Executes H local AdamW optimization steps on frozen 4-bit NF4 base weights + trainable LoRA
adapters with zero network communication, recording cryptographic SHA-256 micro-checkpoints
every s steps for BOINC spot-check verification.
"""

from __future__ import annotations

import gc
import hashlib
import os
import tempfile
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
    emergency_checkpoint_error: str | None = None
    preemption_original_device: str | None = None
    model_offloaded_to_cpu: bool = False
    model_offload_error: str | None = None
    cuda_allocated_before_bytes: int | None = None
    cuda_allocated_after_bytes: int | None = None


def _checkpoint_to_cpu(value):
    """Detach checkpoint tensors from live optimizer/model GPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _checkpoint_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_to_cpu(item) for item in value)
    return value


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
) -> InnerLoopResult:
    """Execute local AdamW steps, stopping and offloading the model on preemption.

    A preempted model stays on CPU; callers must explicitly move it back only
    after their idle policy permits. Re-entering this function starts a fresh
    optimizer and RNG sequence, not an exact continuation of an emergency dump.
    """
    if inner_steps_H <= 0 or checkpoint_interval_s <= 0 or micro_batch_size <= 0:
        raise ValueError("Inner steps, checkpoint interval and batch size must be positive.")
    configure_deterministic_runtime(rng_seed)
    device = next(model.parameters()).device
    model.train()

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
    checkpoint_error = None
    offload_error = None
    offloaded = False
    cuda_before = None
    cuda_after = None

    for step in range(1, inner_steps_H + 1):
        # Zero-Intrusion Daemon preemption check
        if should_preempt_fn is not None and should_preempt_fn():
            preempted = True
            t_dump = time.perf_counter()
            checkpoint_payload = None
            if device.type == "cuda":
                try:
                    cuda_before = int(torch.cuda.memory_allocated(device))
                except (RuntimeError, AssertionError):
                    pass  # None means unavailable, never zero/free VRAM.
            if emergency_checkpoint_dir is not None:
                try:
                    checkpoint_payload = {
                        "worker_id": worker_id,
                        "branch_name": branch_name,
                        "outer_step": outer_step,
                        "step": step - 1,
                        "lora_state": extract_lora_state_dict(model),
                        "initial_lora_state": theta_init,
                        "optimizer_state": _checkpoint_to_cpu(optimizer.state_dict()),
                        "torch_rng_state": torch.get_rng_state().cpu(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "original_device": str(device),
                        "automatic_resume_supported": False,
                    }
                except Exception as exc:
                    checkpoint_error = f"Could not prepare emergency checkpoint: {exc}"
            # empty_cache cannot release live parameter/optimizer tensors. Drop
            # optimizer references and move registered model storage to CPU first.
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            optimizer.param_groups.clear()
            optimizer = None
            trainable_params.clear()
            try:
                model.to(torch.device("cpu"))
                offloaded = all(t.device.type == "cpu" for t in model.parameters()) and all(
                    t.device.type == "cpu" for t in model.buffers()
                )
                if not offloaded:
                    offload_error = "Model still has registered non-CPU tensors after offload."
            except Exception as exc:
                offload_error = f"Model CPU offload failed: {exc}"
            gc.collect()
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    cuda_after = int(torch.cuda.memory_allocated(device))
                except (RuntimeError, AssertionError):
                    pass
            if checkpoint_payload is not None:
                temporary_path = None
                try:
                    emergency_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    safe_worker = hashlib.sha256(worker_id.encode("utf-8")).hexdigest()[:16]
                    target = emergency_checkpoint_dir / f"emergency_{safe_worker}_step{step - 1}.pt"
                    fd, temporary_path = tempfile.mkstemp(prefix=".emergency-", suffix=".pt", dir=emergency_checkpoint_dir)
                    os.close(fd)
                    torch.save(checkpoint_payload, temporary_path)
                    os.replace(temporary_path, target)
                    dump_path = target
                except Exception as exc:
                    checkpoint_error = f"Could not save emergency checkpoint: {exc}"
                finally:
                    if temporary_path is not None:
                        try:
                            os.unlink(temporary_path)
                        except OSError:
                            pass
                del checkpoint_payload
            dump_latency = round(time.perf_counter() - t_dump, 4)
            break

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
        loss_val = float(loss.detach().cpu().item())
        del outputs, loss, input_ids, labels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    del optimizer
    if device.type == "cuda" and not preempted:
        torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - start_time, 4)
    if steps_completed in micro_checkpoints:
        theta_final = micro_checkpoints[steps_completed].state_snapshot
    else:
        theta_final = extract_lora_state_dict(model)
    # A preemption between periodic checkpoints must still commit its final state.
    if steps_completed not in micro_checkpoints:
        micro_checkpoints[steps_completed] = MicroCheckpointRecord(
            step=steps_completed,
            lora_sha256=compute_state_dict_sha256(theta_final),
            loss_value=final_loss,
            state_snapshot=theta_final,
        )
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
        emergency_checkpoint_error=checkpoint_error,
        preemption_original_device=str(device) if preempted else None,
        model_offloaded_to_cpu=offloaded,
        model_offload_error=offload_error,
        cuda_allocated_before_bytes=cuda_before,
        cuda_allocated_after_bytes=cuda_after,
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
    step_callback: Callable[[int], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Replay from step zero so AdamW moments and random-number state are reproducible."""
    if not 0 <= start_step <= end_step:
        raise ValueError("Replay step range is invalid.")
    if start_step and initial_step_state is None:
        raise ValueError("Nonzero replay requires the initial state to reconstruct AdamW moments.")
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
        del outputs, loss, input_ids, labels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step_callback is not None:
            step_callback(step)

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return extract_lora_state_dict(model)
