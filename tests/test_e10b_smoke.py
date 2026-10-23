"""E10b (HM-S10-01) runner smoke test: tiny random Qwen2 on CPU, 2 rounds x 2 learners, report written outside artifacts/.

Also checks that the script constants agree with the prereg (draft or committed), that the torch lattice codec equals
the numpy ``LatticeDeltaQuantizer`` at its calibrated scale, and the optional lattice value codec of
``ComputeVisibleSparseLoCoCompressor`` (loaded by file path because ``hivemind.swarm`` needs the optional aioquic).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e10b():
    return _load("run_experiment_E10b_lattice_05b", ROOT / "scripts" / "run_experiment_E10b_lattice_05b.py")


def test_smoke_cpu_run(e10b, tmp_path):
    out = tmp_path / "e10b_smoke.json"
    e10b.main(["--smoke", "--out", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["smoke"] is True and "SMOKE" in r["warning"]
    assert r["num_runs"] == 24 and len(r["cells"]) == 8
    assert set(r["analysis"]) == {"P10a", "P10b", "P10c"}
    assert r["controls"]["negative_decoders"]["passed"] and r["controls"]["negative_matched_rate"]["passed"]
    assert len(r["receipt_sha256"]) == 64
    for run in r["runs"]:
        if run["arm"] in ("Z8@1+EF", "E8@1+EF", "Z8@2+EF", "E8@2+EF"):
            assert abs(run["mean_rate_bits_per_dim"] - float(run["arm"].split("@")[1][0])) < 0.1  # per-round calibration hits the target
    assert e10b.REPORT_PATH != out  # smoke never targets the real report


def test_smoke_refuses_artifacts_and_missing_out(e10b):
    with pytest.raises(SystemExit):
        e10b.main(["--smoke"])
    with pytest.raises(SystemExit):
        e10b.main(["--smoke", "--out", str(ROOT / "artifacts" / "experiments" / "x.json")])


def test_estimate_within_budget(e10b):
    est = e10b.estimate()
    assert est["runs"] == 24 and est["total_hours"] <= 2.5


def test_prereg_matches_script_constants(e10b):
    p = e10b.PREREG_PATH if e10b.PREREG_PATH.exists() else e10b.PREREG_DRAFT_PATH
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["acceptance_rules"]["t_interval"].find(str(e10b.T_CRIT_DF2)) >= 0
    est = e10b.estimate()
    assert d["sizing"]["runs"] == est["runs"] and d["sizing"]["inner_steps_total"] == est["inner_steps_total"]
    assert d["sizing"]["lattice_codec_calls"] == est["lattice_codec_calls"] and d["sizing"]["eval_passes"] == est["eval_passes"]
    assert abs(d["sizing"]["estimated_hours"] - est["total_hours"]) < 0.01
    assert d["testbed"]["seeds"] == list(e10b.SEEDS)
    assert e10b.FULL.rounds == 12 and e10b.FULL.learners == 4 and e10b.FULL.inner_steps == 40


@pytest.mark.parametrize("kind", ["Z", "E8"])
def test_torch_codec_equals_numpy_quantizer(e10b, kind):
    from hivemind.compress import LatticeDeltaQuantizer

    g = torch.Generator().manual_seed(3)
    v = torch.randn(4096, generator=g) * 1e-3
    codec = e10b.TorchLatticeCodec(kind, 2.0, error_feedback=True)
    q = codec.quantize(v)
    assert abs(q.rate_bits_per_dim - 2.0) <= 0.05
    ref = LatticeDeltaQuantizer(kind, codec.last_scale, error_feedback=True).quantize(v)
    assert torch.equal(q.dequantized, ref.dequantized)
    assert abs(q.relative_mse - ref.relative_mse) < 1e-6
    assert abs(q.rate_bits_per_dim - ref.rate_bits_per_dim) < 1e-9
    # error feedback: the carry is added to the next vector
    q2 = codec.quantize(v)
    assert torch.allclose(q2.dequantized + q2.residual, v + q.residual, atol=1e-7)


def test_sparseloco_lattice_option_and_default_unchanged():
    nx = _load("diloco_nextgen_by_path", ROOT / "src" / "hivemind" / "swarm" / "diloco_nextgen.py")
    g = torch.Generator().manual_seed(5)
    delta = torch.randn(64, 64, generator=g) * 1e-3
    base = nx.ComputeVisibleSparseLoCoCompressor()
    assert base.value_codec == "lut"
    rec0, info0 = base.compress_and_reconstruct(delta)
    assert info0["strategy_id"] == "SI-09B-PULSE-SPARSELOCO-2BIT"
    for kind in ("Z", "E8"):
        c = nx.ComputeVisibleSparseLoCoCompressor(topk_ratio=0.25, value_codec=kind, lattice_rate_bits=2.0)
        rec, info = c.compress_and_reconstruct(delta)
        assert rec.shape == delta.shape and info["value_codec"] == kind and abs(info["measured_rate_bits_per_value"] - 2.0) < 0.1
        assert int((rec != 0).sum()) <= info["k_transmitted"]
    with pytest.raises(ValueError):
        nx.ComputeVisibleSparseLoCoCompressor(value_codec="bogus")
