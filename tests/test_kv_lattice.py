"""GPU/torch lattice codecs of E11 against the numpy reference decoders and exhaustive search."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from hivemind.compress.kv_lattice import (
    SQRT3,
    factorised_rate_t,
    from_rows,
    is_a2_point,
    lattice_quantize,
    nearest_a2_t,
    nearest_e8_t,
    nearest_zn_t,
    to_rows,
)
from hivemind.compress.lattice_quantizer import factorised_rate_bits_per_dim, nearest_e8, nearest_zn

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _queries(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=2.0, size=(n, 8))
    # exact ties: half-integers and integer-plus-half on several coordinates
    t = rng.integers(-3, 4, size=(n // 4, 8)).astype(float)
    t[:, ::2] += 0.5
    return np.concatenate([q, t, np.full((1, 8), 0.5), np.full((1, 8), -0.5), np.zeros((1, 8))])


@pytest.mark.parametrize("device", DEVICES)
def test_torch_decoders_bit_identical_to_numpy(device: str) -> None:
    q = _queries(2026, 20000)
    x = torch.from_numpy(q).to(device)
    assert np.array_equal(nearest_zn_t(x).cpu().numpy(), nearest_zn(q))
    assert np.array_equal(nearest_e8_t(x).cpu().numpy(), nearest_e8(q))


def _a2_points(radius: float) -> np.ndarray:
    r = int(math.ceil(2 * radius)) + 2
    pts = np.array([(a + 0.5 * b, 0.5 * SQRT3 * b) for a, b in itertools.product(range(-r, r + 1), repeat=2)])
    return pts[np.einsum("ij,ij->i", pts, pts) <= radius * radius + 1e-9]


@pytest.mark.parametrize("device", DEVICES)
def test_a2_two_sided_brute_force(device: str) -> None:
    rng = np.random.default_rng(7)
    q = rng.uniform(-3.0, 3.0, size=(5000, 2))
    c = nearest_a2_t(torch.from_numpy(q).to(device)).cpu()
    assert bool(is_a2_point(c).all())
    pts = _a2_points(6.0)
    d_min = ((q[:, None, :] - pts[None]) ** 2).sum(-1).min(axis=1)
    d_dec = ((q - c.numpy()) ** 2).sum(-1)
    assert np.max(np.abs(d_dec - d_min)) < 1e-9


def test_a2_second_moment_matches_theory() -> None:
    # uniform dither over a big cube: mean squared error / (2 * V^(2/2)) -> NSM(A2), V = sqrt(3)/2
    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.uniform(-50, 50, size=(400000, 2)))
    e = ((x - nearest_a2_t(x)) ** 2).sum(-1).mean().item()
    assert abs(e / (2 * (SQRT3 / 2)) - 5 / (36 * SQRT3)) < 1e-3


def test_rate_matches_numpy_reference_with_constant_context() -> None:
    rng = np.random.default_rng(11)
    q = rng.normal(size=(4000, 8)) * 1.7
    for kind, dec_np, dec_t in (("Z", nearest_zn, nearest_zn_t), ("E8", nearest_e8, nearest_e8_t)):
        c_np = dec_np(q)
        c_t = dec_t(torch.from_numpy(q))
        ref = factorised_rate_bits_per_dim(kind, c_np)
        got = factorised_rate_t(kind, c_t, torch.zeros(len(q), dtype=torch.int64))
        assert abs(ref - got) < 1e-9, (kind, ref, got)


@pytest.mark.parametrize("kind", ["Z", "A2", "E8"])
def test_row_layout_round_trip_and_fine_scale(kind: str) -> None:
    x = torch.randn(2, 3, 5, 64, dtype=torch.float64)
    rows, ctx = to_rows(kind, x)
    assert torch.equal(from_rows(kind, rows, x.shape), x)
    assert int(ctx.max()) == (31 if kind == "A2" else 7)
    xh, rate = lattice_quantize(kind, x, 1e-6, measure_rate=True)
    # the plug-in entropy saturates near log2(rows per context) when every codeword is distinct
    assert torch.allclose(xh, x, atol=1e-6) and rate > 3.5
    xh, rate = lattice_quantize(kind, x, 0.5, measure_rate=True)
    assert 0.5 < rate < 4.0
