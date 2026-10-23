"""Standalone copy of the E1 toy LoRA causal LM and dataset builder, so that a learner host
needs only this file, ``run_experiment_E8_pilot.py`` and PyTorch. Kept byte-identical to the
definitions in ``run_paper_experiments_and_figures.py`` (checked by ``tests/test_e8_pilot_toy_model.py``).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SyntheticLoRACausalLM(nn.Module):
    """Compact 2-layer causal sequence model with frozen NF4-like base weights and trainable LoRA adapters."""

    def __init__(self, vocab_size: int = 64, d_model: int = 64, rank_r: int = 8, seed: int = 42) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rank_r = rank_r
        self.scale = 2.0

        # Frozen base weights
        self.register_buffer("embed_w", torch.randn((vocab_size, d_model), generator=gen) * 0.15)
        self.register_buffer("w1_base", torch.randn((d_model, d_model), generator=gen) / math.sqrt(d_model))
        self.register_buffer("w2_base", torch.randn((d_model, d_model), generator=gen) / math.sqrt(d_model))
        self.register_buffer("head_w", torch.randn((vocab_size, d_model), generator=gen) / math.sqrt(d_model))

        # Trainable LoRA adapters (64x8 and 8x64 -> reshapeable into 8x8 DCT blocks)
        self.l1_A = nn.Parameter(torch.randn((rank_r, d_model), generator=gen) * 0.04)
        self.l1_B = nn.Parameter(torch.zeros((d_model, rank_r)))
        self.l2_A = nn.Parameter(torch.randn((rank_r, d_model), generator=gen) * 0.04)
        self.l2_B = nn.Parameter(torch.zeros((d_model, rank_r)))

    def forward(self, x_ids: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        h0 = F.embedding(x_ids, self.embed_w)
        # Causal Bigram Mixing: combine current token and shifted previous token
        h_prev = torch.roll(h0, shifts=1, dims=1)
        h_prev[:, 0, :] = 0.0
        h_in = h0 + 0.6 * h_prev

        delta1 = (h_in @ self.l1_A.t()) @ self.l1_B.t() * self.scale
        h1 = F.silu(h_in @ self.w1_base.t() + delta1) + h_in

        delta2 = (h1 @ self.l2_A.t()) @ self.l2_B.t() * self.scale
        h2 = F.silu(h1 @ self.w2_base.t() + delta2) + h1

        logits = h2 @ self.head_w.t()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


def _build_synthetic_sequence_dataset(
    vocab_size: int = 64,
    seq_len: int = 16,
    n_train_per_worker: int = 64,
    n_workers: int = 4,
    n_val: int = 128,
    seed: int = 1001,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a structured teacher-generated sequence dataset with a strictly disjoint held-out validation split."""
    teacher = SyntheticLoRACausalLM(vocab_size=vocab_size, d_model=64, rank_r=8, seed=seed)
    gen = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        teacher.l1_B.copy_(torch.randn(teacher.l1_B.shape, generator=gen) * 0.35)
        teacher.l2_B.copy_(torch.randn(teacher.l2_B.shape, generator=gen) * 0.35)

    total_train = n_train_per_worker * n_workers
    all_x = torch.randint(0, vocab_size, (total_train + n_val, seq_len), generator=gen)
    with torch.no_grad():
        logits, _ = teacher(all_x)
        all_y = logits.argmax(dim=-1)

    worker_splits: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k in range(n_workers):
        lo = k * n_train_per_worker
        hi = (k + 1) * n_train_per_worker
        worker_splits.append((all_x[lo:hi].clone(), all_y[lo:hi].clone()))

    val_split = (all_x[total_train:].clone(), all_y[total_train:].clone())
    return worker_splits, val_split
