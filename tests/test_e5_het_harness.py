"""Tests for the E5-het heterogeneity-controlled testbed (``scripts/run_experiment_E5_het.py``)
and the RDA norm-ratio identity it measures (Proposition ``prop:rda_multiplier`` in the paper)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment_E5_het import build_heterogeneous_dataset  # noqa: E402
from toy_model import _build_synthetic_sequence_dataset  # noqa: E402
from hivemind.swarm.decoupled_diloco import direct_weighted_average, radial_directional_average  # noqa: E402


def test_rho_zero_reproduces_the_e5_toy_dataset_exactly() -> None:
    het_shards, (het_vx, het_vy) = build_heterogeneous_dataset(0.0, n_workers=4, seed=2027)
    toy_shards, (toy_vx, toy_vy) = _build_synthetic_sequence_dataset(n_workers=4, seed=2027)
    assert len(het_shards) == len(toy_shards) == 4
    for (hx, hy), (tx, ty) in zip(het_shards, toy_shards):
        assert torch.equal(hx, tx)
        assert torch.equal(hy, ty)
    assert torch.equal(het_vx, toy_vx)
    assert torch.equal(het_vy, toy_vy)


def test_rho_positive_changes_labels_but_not_inputs() -> None:
    base_shards, (base_vx, base_vy) = build_heterogeneous_dataset(0.0, n_workers=4, seed=2027)
    het_shards, (het_vx, het_vy) = build_heterogeneous_dataset(4.0, n_workers=4, seed=2027)
    changed = 0
    for (bx, by), (hx, hy) in zip(base_shards, het_shards):
        assert torch.equal(bx, hx)
        changed += int(not torch.equal(by, hy))
    assert changed == 4, "every learner must get its own teacher at rho = 4"
    assert torch.equal(base_vx, het_vx)
    frac = float((base_vy != het_vy).float().mean())
    assert 0.2 < frac < 1.0, f"validation labels should differ in a substantial fraction of positions, got {frac}"


def test_validation_rows_are_labelled_by_the_rotating_teacher() -> None:
    # With two workers and rho > 0, validation row j must carry teacher (j mod 2)'s label; a wrong
    # mixture would show up as rows matching the other teacher more often than their own.
    _, (vx, vy) = build_heterogeneous_dataset(4.0, n_workers=2, n_val=64, seed=2027)
    shards0, _ = build_heterogeneous_dataset(4.0, n_workers=2, n_val=64, seed=2027)
    # Rebuild the two teachers' labels on the validation inputs through the public builder by
    # requesting a dataset whose "training" split is the validation inputs: simpler, recompute directly.
    import copy
    from toy_model import SyntheticLoRACausalLM

    teacher = SyntheticLoRACausalLM(vocab_size=64, d_model=64, rank_r=8, seed=2027)
    gen = torch.Generator().manual_seed(2027 + 7)
    with torch.no_grad():
        teacher.l1_B.copy_(torch.randn(teacher.l1_B.shape, generator=gen) * 0.35)
        teacher.l2_B.copy_(torch.randn(teacher.l2_B.shape, generator=gen) * 0.35)
    het_gen = torch.Generator().manual_seed(2027 + 7919)
    labels = []
    for _ in range(2):
        t = copy.deepcopy(teacher)
        with torch.no_grad():
            t.l1_B.add_(4.0 * torch.randn(t.l1_B.shape, generator=het_gen) * 0.35)
            t.l2_B.add_(4.0 * torch.randn(t.l2_B.shape, generator=het_gen) * 0.35)
            logits, _ = t(vx)
        labels.append(logits.argmax(dim=-1))
    own = torch.stack([labels[j % 2][j] for j in range(vx.shape[0])])
    other = torch.stack([labels[(j + 1) % 2][j] for j in range(vx.shape[0])])
    assert torch.equal(vy, own)
    assert not torch.equal(vy, other)
    assert len(shards0) == 2


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.lists(st.floats(-3, 3, allow_nan=False), min_size=4, max_size=4), min_size=2, max_size=6),
    st.lists(st.floats(0.05, 1.0), min_size=6, max_size=6),
)
def test_rda_norm_is_weighted_mean_norm_and_ratio_at_least_one(rows: list, raw_w: list) -> None:
    vecs = [torch.tensor(r, dtype=torch.float64) for r in rows if any(abs(x) > 1e-3 for x in r)]
    if len(vecs) < 2:
        return
    w = raw_w[: len(vecs)]
    wn = [x / sum(w) for x in w]
    direct = direct_weighted_average(vecs, wn)
    rda = radial_directional_average(vecs, wn)
    if float(direct.norm()) < 1e-6 or float(rda.norm()) < 1e-6:
        return
    expected_norm = sum(wi * float(v.norm()) for wi, v in zip(wn, vecs))
    assert math.isclose(float(rda.norm()), expected_norm, rel_tol=1e-5, abs_tol=1e-6)  # implementation works in float32
    assert float(rda.norm()) / float(direct.norm()) >= 1.0 - 1e-5


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 8), st.integers(0, 10_000))
def test_equal_norm_ratio_matches_quadratic_form_of_cosines(m: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    vecs = [torch.randn(16, generator=g, dtype=torch.float64) for _ in range(m)]
    vecs = [v / v.norm() * 3.0 for v in vecs]  # common norm R = 3
    w = [1.0 / m] * m
    direct = direct_weighted_average(vecs, w)
    rda = radial_directional_average(vecs, w)
    c = torch.tensor([[float(torch.dot(a, b)) / 9.0 for b in vecs] for a in vecs], dtype=torch.float64)
    wt = torch.full((m,), 1.0 / m, dtype=torch.float64)
    predicted = 1.0 / math.sqrt(float(wt @ c @ wt))
    assert math.isclose(float(rda.norm()) / float(direct.norm()), predicted, rel_tol=1e-4)
    # same direction in the common-norm case
    cos_dir = float(torch.dot(rda, direct)) / (float(rda.norm()) * float(direct.norm()))
    assert cos_dir > 1.0 - 1e-6
