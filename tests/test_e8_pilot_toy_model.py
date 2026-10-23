"""The standalone ``scripts/toy_model.py`` shipped to remote learner hosts must stay byte-identical
to the E1 definitions in ``scripts/run_paper_experiments_and_figures.py``, and the E8 pilot
protocol must reproduce the single-process baseline when all learners are admitted."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _source_of(path: Path, names: set[str]) -> dict[str, str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names}


def test_toy_model_copy_is_byte_identical_to_e1_definitions() -> None:
    names = {"SyntheticLoRACausalLM", "_build_synthetic_sequence_dataset"}
    a = _source_of(SCRIPTS / "run_paper_experiments_and_figures.py", names)
    b = _source_of(SCRIPTS / "toy_model.py", names)
    assert set(a) == names and set(b) == names
    for n in names:
        assert a[n] == b[n], f"{n} differs between the E1 harness and toy_model.py"


def test_e8_pilot_inner_train_and_flat_roundtrip_match_in_process_baseline() -> None:
    import run_experiment_E8_pilot as pilot

    model_cls, build = pilot._load_toy_model_cls()
    shards, (vx, vy) = build(vocab_size=64, n_train_per_worker=64, n_workers=2, n_val=128, seed=2027)
    init = model_cls(vocab_size=64, d_model=64, rank_r=8, seed=2027)
    params = {k: v.detach().clone() for k, v in init.named_parameters()}
    # the serialisation used on the wire must be lossless
    rt = pilot.b64_to_tensors(pilot.tensors_to_b64(params))
    assert all(torch.equal(rt[k], params[k]) for k in params)
    d0 = pilot.inner_train(model_cls, params, shards[0], 0, 2027)
    d1 = pilot.inner_train(model_cls, params, shards[1], 0, 2027)
    assert all(torch.isfinite(v).all() for v in d0.values())
    assert pilot.flat(d0).norm() > 0 and not torch.equal(pilot.flat(d0), pilot.flat(d1))
    # flat/unflat is an exact inverse
    back = pilot.unflat(pilot.flat(d0), d0)
    assert all(torch.equal(back[k], d0[k]) for k in d0)
    assert pilot.percentile([3.0, 1.0, 2.0], 0.5) == 2.0 and pilot.percentile([1.0], 0.95) == 1.0
