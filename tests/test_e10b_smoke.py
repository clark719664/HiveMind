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
    assert r["num_runs"] == 15 and len(r["cells"]) == 5
    assert set(r["analysis"]) == {"P10a", "P10b", "P10c"}
    assert r["controls"]["negative_decoders"]["passed"] and r["controls"]["negative_matched_rate"]["passed"]
    assert len(r["receipt_sha256"]) == 64
    for run in r["runs"]:
        if run["arm"] in ("Z8@1+EF", "E8@1+EF"):
            assert abs(run["mean_rate_bits_per_dim"] - float(run["arm"].split("@")[1][0])) < 0.1  # per-round calibration hits the target
    assert e10b.REPORT_PATH != out  # smoke never targets the real report


def test_smoke_refuses_artifacts_and_missing_out(e10b):
    with pytest.raises(SystemExit):
        e10b.main(["--smoke"])
    with pytest.raises(SystemExit):
        e10b.main(["--smoke", "--out", str(ROOT / "artifacts" / "experiments" / "x.json")])


def test_estimate_within_budget(e10b):
    est = e10b.estimate()
    assert est["runs"] == 15 and est["total_hours"] <= 2.5 and est["assumed_sec_per_inner_step"] == 0.72


def test_amendment_matches_script_constants(e10b):
    d = json.loads(e10b.AMENDMENT_PATH.read_text(encoding="utf-8"))
    est = e10b.estimate()
    for k in ("runs", "inner_steps_total", "lattice_codec_calls", "eval_passes"):
        assert d["sizing"][k] == est[k]
    assert abs(d["sizing"]["estimated_hours"] - est["total_hours"]) < 0.01
    ch = d["changes"]["3_sizing"]
    assert (ch["R"], ch["M"], ch["H"]) == (e10b.FULL.rounds, e10b.FULL.learners, e10b.FULL.inner_steps) == (6, 4, 20)
    assert e10b.FULL.eval_every == 3 and ch["seeds"] == list(e10b.SEEDS)
    assert [a.name for a in e10b.ARMS] == ["none", "Z8@1+EF", "E8@1+EF", "sign", "E8@6"]
    assert e10b.RATES == (1.0,)
    assert "2026-10-24" in d["made_utc_date"] and "NOT a rule registered" in d["first_execution"]["how_it_ended"] and "INCONCLUSIVE" in d["expected_power"]


def test_lora_init_is_peft_uniform_and_always_reset(e10b):
    model = e10b.tiny_model("cpu")  # attach_lora draws the E12 N(0, 1/r) init
    std_e12 = statistics_std(model)
    e10b.reset_lora(model, 1)
    for n, p in model.named_parameters():
        if "lora_A" in n:
            assert float(p.abs().max()) <= 1.0 / (p.shape[1] ** 0.5) + 1e-9
        if "lora_B" in n:
            assert float(p.abs().max()) == 0.0
    assert statistics_std(model) < 0.25 * std_e12
    a1 = e10b.get_flat(e10b.lora_params(model))
    e10b.reset_lora(model, 1)
    assert torch.equal(a1, e10b.get_flat(e10b.lora_params(model)))  # deterministic in the seed


def statistics_std(model) -> float:
    return float(torch.cat([p.reshape(-1) for n, p in model.named_parameters() if "lora_A" in n]).std())


def test_failfast_aborts_with_exit_code_3(e10b, tmp_path, monkeypatch):
    monkeypatch.setattr(e10b, "FAILFAST_STEPS", 2)
    monkeypatch.setattr(e10b, "FAILFAST_SEC_FACTOR", 1e-9)  # any real step time exceeds it
    out = tmp_path / "aborted.json"
    assert e10b.main(["--smoke", "--out", str(out)]) == 3
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["status"].startswith("aborted by registered fail-fast rule: ")
    assert all(v["status"] == "invalid" for v in r["analysis"].values())


def test_failfast_no_learning_rule(e10b, monkeypatch):
    monkeypatch.setattr(e10b, "FAILFAST_ROUND", 1)  # smoke has no round 3; check after round 1
    shards, held, _ = e10b.smoke_data(e10b.SMOKE)
    model = e10b.tiny_model("cpu")
    monkeypatch.setattr(e10b, "INNER_LR", 0.0)  # frozen LoRA: held-out loss cannot fall
    with pytest.raises(e10b.FailFast, match="does not learn"):
        e10b.run_arm(model, e10b.lora_params(model), e10b.ARMS[0], 1, shards, held, e10b.SMOKE, "cpu")


def test_partial_ignored_under_other_design(e10b, tmp_path):
    p = tmp_path / "x.partial.json"
    p.write_text(json.dumps({"design_sha256": "aaa", "runs": [{"arm": "none", "seed": 1, "final_loss": 1.0}]}), encoding="utf-8")
    assert len(e10b.load_partial(p, "aaa")) == 1
    assert e10b.load_partial(p, "bbb") == []
    p.write_text(json.dumps([{"arm": "none", "seed": 1, "final_loss": 1.0}]), encoding="utf-8")  # old list format
    assert e10b.load_partial(p, "aaa") == []


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
