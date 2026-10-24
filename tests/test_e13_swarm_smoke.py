"""E13 (swarm simulation on one host) tests: CPU only, real subprocesses and real localhost HTTP, report written outside artifacts/.

The smoke path runs through the script's ``main()`` (tiny random Qwen2, M=2, 2 rounds x 3 inner steps, synthetic tokens, cells S0-S5 including a straggler and a
churn event). Unit tests cover the straggle schedule, quorum selection, RDA, the codec round trip with byte accounting, and late-delta discard.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

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
def e13():
    return _load("run_experiment_E13_swarm_local", ROOT / "scripts" / "run_experiment_E13_swarm_local.py")


@pytest.fixture(scope="module")
def lib(e13):
    return e13.lib


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_smoke_runs_all_cells_through_main(e13, tmp_path):
    out = tmp_path / "e13_smoke.json"
    t0 = time.perf_counter()
    assert e13.main(["--smoke", "--out", str(out), "--scratch", str(tmp_path / "scr")]) == 0
    assert time.perf_counter() - t0 < 90.0
    raw = json.loads(out.read_text(encoding="utf-8"))
    by = {r["cell"]: r for r in raw["runs"]}
    assert set(by) == set(e13.ALL_CELLS) | set(e13.EXTRA_CELLS)
    # S1 == S0: the HTTP protocol with K=M, direct, no codec reproduces the sequential reference exactly
    assert by["S1"]["identity_vs_s0"]["passed"] and by["S1"]["identity_vs_s0"]["max_abs_diff"] <= 1e-6
    assert by["S1"]["final_theta_sha256"] == by["S0"]["final_theta_sha256"]
    # thread hosting (one model, M peer threads, same HTTP protocol) is also identical to the sequential reference
    assert by["S1t"]["hosting"] == "thread" and by["S1"]["hosting"] == "process"
    assert by["S1t"]["identity_vs_s0"]["passed"] and by["S1t"]["final_theta_sha256"] == by["S0"]["final_theta_sha256"]
    for c in ("S1", "S1t", "S2", "S3", "S4", "S5"):
        r = by[c]
        assert r["init_identical"] and len(r["loss_curve"]) == 3 and len(r["round_wall_s"]) == 2 and len(r["admitted_per_round"]) == 2
        assert len(r["bytes_up_per_round"]) == 2 and r["wire"]["n_posts"] >= 2 and r["wire"]["up_bytes"] > 0 and len(r["final_theta_sha256"]) == 64
        assert Path(r["final_theta_path"]).exists() and set(r["per_domain_final"]) == {"code", "wiki-fr"}
    # real real-HTTP byte counts: raw bodies are exactly 4 x n_coords
    for c in ("S1", "S1t", "S3"):
        assert by[c]["wire"]["up_bytes"] == by[c]["wire"]["n_posts"] * 4 * by[c]["n_coords"]
    # codec cells: far fewer bytes than raw, measured rate on target, ideal entropy-coded size below the zlib size
    for c in ("S2", "S4", "S5"):
        w = by[c]["wire"]
        assert w["up_bytes"] * 4 < w["raw_delta_bytes"] and abs(w["mean_measured_rate_bits_per_dim"] - 2.0) < 0.1 and w["ideal_bytes"] < w["up_bytes"]
    # churn happened in the chaos cells (a real SIGKILL and a re-spawn), not in S1
    assert [e["event"] for e in by["S2"]["churn_events"]] == ["killed", "respawned"] and by["S1"]["churn_events"] == []
    assert by["S5"]["effective_k_per_round"] == [2, 2] and by["S2"]["K"] == 1
    rep = json.loads((tmp_path / "e13_smoke.report.json").read_text(encoding="utf-8"))
    assert rep["smoke"] is True and "SMOKE" in rep["warning"] and rep["controls_all_passed"] is True and len(rep["receipt_sha256"]) == 64
    assert "tiny" in rep["models"] and rep["models"]["tiny"]["not_included"] == e13.NOT_INCLUDED


def test_smoke_refuses_artifacts_and_missing_out(e13, tmp_path):
    with pytest.raises(SystemExit):
        e13.main(["--smoke"])
    with pytest.raises(SystemExit):
        e13.main(["--smoke", "--out", str(ROOT / "artifacts" / "experiments" / "x.json")])
    with pytest.raises(SystemExit):
        e13.main(["--smoke", "--out", str(tmp_path / "x.json"), "--scratch", str(ROOT / "artifacts" / "scr")])


def test_full_run_needs_prereg(e13, monkeypatch, tmp_path):
    monkeypatch.setattr(e13, "PREREG_PATH", tmp_path / "missing_prereg.json")
    with pytest.raises(SystemExit, match="prereg"):
        e13.main(["--model", "qwen0.5b"])


def test_failfast_learning_gate_exit_code_3(e13, monkeypatch, tmp_path):
    monkeypatch.setattr(e13, "SMOKE_LEARNING_GATE_NATS", 1e9)  # nothing can drop by a billion nats
    report = tmp_path / "aborted.json"
    rc = e13.main(["--smoke", "--out", str(tmp_path / "raw.json"), "--scratch", str(tmp_path / "scr"), "--cells", "S0", "--report", str(report)])
    assert rc == 3
    r = json.loads(report.read_text(encoding="utf-8"))
    assert r["status"].startswith("aborted by registered fail-fast rule: ") and r["controls_all_passed"] is False


def test_failfast_rules_units(e13):
    assert e13.check_learning_gate({"0": 3.0, "3": 2.9}, 3, 0.05) is None
    assert "does not learn" in e13.check_learning_gate({"0": 3.0, "3": 2.98}, 3, 0.05)
    assert e13.check_step_time(0.50, "qwen0.5b") is None  # 0.45 * 1.3 = 0.585
    assert "exceeds" in e13.check_step_time(0.60, "qwen0.5b")
    assert e13.check_step_time(99.0, "tiny") is None  # no assumption for the smoke model


def test_estimate_qwen05b(e13):
    est = e13.estimate("qwen0.5b", e13.RunCfg(rounds=6, workers=4, inner=20))
    assert est["assumed_sec_per_inner_step"] == 0.45 and est["inner_steps_per_run"] == 480 and est["runs"] == 18 and est["total_hours"] < 3


def test_registry_keys_and_meta(lib):
    assert {"tiny", "qwen0.5b", "granite1b", "granite3b"} <= set(lib.LOADERS)
    with pytest.raises(NotImplementedError, match="identical at init"):
        lib.LOADERS["qwen0.5b-moe"]("cpu")
    m = lib.MODEL_META
    assert (m["qwen0.5b"]["workers"], m["granite1b"]["workers"], m["granite3b"]["workers"]) == (4, 3, 4)
    assert m["granite3b"]["hosting"] == "thread" and m["granite1b"]["grad_ckpt"] is True and m["qwen0.5b"]["hosting"] == "process"
    assert m["granite1b"]["repo"] == "ibm-granite/granite-3.1-1b-a400m-base" and m["granite3b"]["repo"] == "ibm-granite/granite-3.1-3b-a800m-base"
    model, tok, targets = lib.LOADERS["tiny"]("cpu")
    assert tok is None and targets == ("q_proj", "v_proj")


def test_grad_ckpt_build_matches_eval_mode_forward(lib):
    m1, _, p1 = lib.build_model("tiny", "cpu", 3)
    m2, _, p2 = lib.build_model("tiny", "cpu", 3, grad_ckpt=True)
    assert m2.training and not m1.training
    ids = torch.randint(0, 256, (2, 16))
    lib.e10b.set_flat(p1, torch.randn_like(lib.e10b.get_flat(p1)) * 0.01)
    lib.e10b.set_flat(p2, lib.e10b.get_flat(p1))
    l1, l2 = m1(input_ids=ids, labels=ids).loss, m2(input_ids=ids, labels=ids).loss
    assert abs(float(l1) - float(l2)) < 1e-6
    l2.backward()
    assert all(p.grad is not None for p in p2)


def test_incarnation_mismatch_is_rejected(lib):
    c, _ = _coord(lib, workers=2, k=1)
    c.set_alive(1, False)
    c.set_alive(1, True)  # re-spawn: incarnation 1
    assert c.alive_map() == {"0": 0, "1": 1}
    old = dict(_delta_meta(1, 8), incarnation=0)
    r = c.post_delta(1, old, torch.ones(8).numpy().tobytes())
    assert r["accepted"] is False and "incarnation" in r["reason"]
    assert c.post_delta(1, dict(_delta_meta(1, 8), incarnation=1), torch.ones(8).numpy().tobytes())["accepted"] is True


def test_lora_peft_scale_init_is_identical_and_deterministic(lib):
    m1, _, p1 = lib.build_model("tiny", "cpu", 5)
    m2, _, p2 = lib.build_model("tiny", "cpu", 5)
    f1, f2 = lib.e10b.get_flat(p1), lib.e10b.get_flat(p2)
    assert torch.equal(f1, f2) and lib.tensor_sha(f1) == lib.tensor_sha(f2)
    for n, p in m1.named_parameters():
        if "lora_A" in n:
            assert float(p.abs().max()) <= 1.0 / (p.shape[1] ** 0.5) + 1e-9 and p.dtype == torch.float32
        if "lora_B" in n:
            assert float(p.abs().max()) == 0.0
        if "lora_" not in n:
            assert not p.requires_grad


def test_straggle_and_churn_schedule(lib):
    s = lib.parse_straggle("-1:1,3,5:2.0", 4)
    assert s == {(3, 1): 2.0, (3, 3): 2.0, (3, 5): 2.0}
    assert lib.parse_straggle("0:2:1.5;-1:1:3", 4) == {(0, 2): 1.5, (3, 1): 3.0}
    assert lib.parse_straggle("off", 4) == {} and lib.parse_churn("off", 4) is None
    assert lib.parse_churn("1:2:3", 4) == (1, 2, 3)
    with pytest.raises(ValueError):
        lib.parse_churn("1:3:2", 4)


def test_default_chaos_schedule_in_ctx(e13):
    ctx = e13.Ctx(cfg=e13.RunCfg(rounds=6, workers=4, inner=20))
    straggle, churn = ctx.chaos_specs()
    assert straggle == {(3, 1): 2.0, (3, 3): 2.0, (3, 5): 2.0} and churn == (1, 2, 3)
    off = e13.Ctx(cfg=e13.RunCfg(rounds=6, workers=4, inner=20), straggle="off", churn="off")
    assert off.chaos_specs() == ({}, None)


def _coord(lib, workers=3, k=2, merge="direct", grace=False, n=8):
    c = lib.Coordinator(workers, rounds=3, inner=3, k=k, merge=merge, grace=grace)
    theta0 = torch.arange(n, dtype=torch.float32) * 0.1
    for w in range(workers):
        c.hello(w, "sha", theta0.numpy().tobytes() if w == 0 else b"")
    assert c.wait_ready(1.0)
    c.open_first()
    return c, theta0


def _delta_meta(rnd, n, tokens=96, steps=3):
    return {"codec": "none", "n": n, "raw_bytes": 4 * n, "ideal_bytes": 4.0 * n, "round": rnd, "compute_s": 0.1, "tokens": tokens, "steps": steps}


def test_quorum_selection_is_used_and_first_k_admitted(lib, monkeypatch):
    c, theta0 = _coord(lib, workers=3, k=2)
    calls = []
    real = c.dd.select_quorum
    monkeypatch.setattr(c.dd, "select_quorum", lambda arr, min_quorum_k, grace_window_s: calls.append((len(arr), min_quorum_k, grace_window_s)) or real(arr, min_quorum_k, grace_window_s))
    d = torch.ones(8)
    c.post_delta(2, _delta_meta(1, 8), d.numpy().tobytes())
    assert c.try_close_round() is None  # K = 2 not reached
    c.post_delta(0, _delta_meta(1, 8), d.numpy().tobytes())
    rec = c.try_close_round()  # window 0, third worker has not arrived: closes with the first two arrivals
    assert calls == [(2, 2, 0.0)] and rec["admitted"] == [0, 2] and c.version == 1
    assert torch.allclose(c.theta, lib.outer_nesterov(theta0, torch.zeros(8), d)[0])


def test_rda_merge_is_called_and_differs_from_direct(lib, monkeypatch):
    seen = []
    real = lib.load_decoupled_diloco().radial_directional_average
    monkeypatch.setattr(lib.load_decoupled_diloco(), "radial_directional_average", lambda v, w=None, eps=1e-12: seen.append(len(v)) or real(v, w, eps))
    ds = [torch.tensor([3.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 3.0, 0.0, 0.0])]
    rda = lib.merge_deltas(ds, [10, 10], [1, 1], "rda")
    direct = lib.merge_deltas(ds, [10, 10], [1, 1], "direct")
    assert seen == [2]
    assert abs(float(rda.norm()) - 3.0) < 1e-5 and abs(float(direct.norm()) - 3.0 / 2 ** 0.5) < 1e-5  # RDA keeps the radius, direct shrinks it


@pytest.mark.parametrize("spec", ["none", "E8@2+EF", "Z@2+EF"])
def test_codec_round_trip_and_byte_accounting(lib, spec):
    g = torch.Generator().manual_seed(11)
    v = torch.randn(3001, generator=g) * 1e-3  # not a multiple of 8: exercises padding
    enc = lib.WireCodec(spec)
    body, meta = enc.encode(v)
    assert meta["raw_bytes"] == 4 * 3001
    dec = lib.decode_body(body, meta)
    if spec == "none":
        assert len(body) == 4 * 3001 and torch.equal(dec, v)
        return
    ref = lib._CapturingCodec(enc.kind, enc.rate, enc.ef).quantize(v)  # same arithmetic, fresh state
    assert torch.equal(dec, ref.dequantized) and dec.shape == v.shape
    assert abs(meta["rate"] - 2.0) <= 0.05
    npad = 3008
    assert abs(meta["ideal_bytes"] - meta["rate"] * npad / 8) < 1e-9
    assert len(body) < meta["raw_bytes"] / 4 and meta["ideal_bytes"] < len(body)
    # error feedback: the second call adds the carry of the first
    body2, meta2 = enc.encode(v)
    dec2 = lib.decode_body(body2, meta2)
    assert torch.allclose(dec2 + enc.codec._carry, v + (v - dec), atol=1e-7)


def test_late_delta_is_discarded_counted_and_bytes_come_from_http_bodies(lib):
    import threading

    c, theta0 = _coord(lib, workers=2, k=1)
    srv = lib.make_server(c)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        hdr, body = lib.http_request(f"{base}/global?worker=0&have=-1")
        assert hdr["X-Version"] == "0" and body == theta0.numpy().tobytes() and c.bytes_down[(0, 1)] == len(body) == 32
        d1 = torch.full((8,), 0.5).numpy().tobytes()
        post = lambda w, r, b: json.loads(lib.http_request(f"{base}/delta?worker={w}", b, {"X-Meta": json.dumps(_delta_meta(r, 8))})[1])  # noqa: E731
        assert post(0, 1, d1)["accepted"] is True
        rec = c.try_close_round()
        assert rec["admitted"] == [0] and c.version == 1
        theta_after = c.theta.clone()
        late = post(1, 1, d1)  # arrives after round 1 closed
        assert late["accepted"] is False and "late" in late["reason"]
        assert c.try_close_round() is None and torch.equal(c.theta, theta_after)  # not merged into anything
        assert len(c.late) == 1 and c.late[0]["worker"] == 1 and c.late[0]["delta_round"] == 1
        assert c.bytes_up[(0, 1)] == 32 and c.bytes_up[(1, 1)] == 32  # the discarded body still crossed the wire
        # the late worker re-syncs to the newest global
        hdr2, body2 = lib.http_request(f"{base}/global?worker=1&have=0")
        assert hdr2["X-Version"] == "1" and body2 == theta_after.numpy().tobytes()
        assert c.bytes_down[(1, 2)] == len(body2)
        # no new global yet: the long-poll returns an empty wait response
        hdr3, body3 = lib.http_request(f"{base}/global?worker=1&have=1")
        assert hdr3.get("X-Wait") == "1" and body3 == b""
    finally:
        srv.shutdown()
        srv.server_close()


def test_adaptive_grace_window_admits_arrivals_inside_it(lib):
    c, _ = _coord(lib, workers=3, k=1, grace=True)
    d = torch.ones(8).numpy().tobytes()
    meta = _delta_meta(1, 8)
    meta["compute_s"] = 10.0  # long step: window = 0.5 * (2*10 - t_q) ~ 10 s
    c.post_delta(0, meta, d)
    assert c.try_close_round() is None  # quorum reached but the grace window is open and 2 workers are missing
    assert c._window > 5.0
    c.post_delta(1, meta, d)
    c.post_delta(2, meta, d)
    assert c.try_close_round()["admitted"] == [0, 1, 2]


def test_routing_stats_from_counts(lib):
    flat = lib.routing_stats_from_counts([torch.full((8,), 10)], 2)
    assert abs(flat["mean_norm_entropy"] - 1.0) < 1e-9 and abs(flat["mean_load_imbalance"] - 1.0) < 1e-9 and abs(flat["mean_top25pct_expert_share"] - 0.25) < 1e-9
    peaked = lib.routing_stats_from_counts([torch.tensor([100, 0, 0, 0, 0, 0, 0, 0])], 2)
    assert peaked["mean_norm_entropy"] == 0.0 and peaked["mean_top25pct_expert_share"] == 1.0


def test_dense_model_has_no_routing_stats(lib):
    m, _, p = lib.build_model("tiny", "cpu", 1)
    r = lib.routing_stats(m, p, lib.e10b.get_flat(p), {"code": torch.zeros(2, 4, dtype=torch.int32)}, 2, "cpu", False)
    assert r["available"] is False


def test_routing_stats_via_router_forward_hooks(lib):
    import torch.nn as nn

    E, TOPK = 4, 2

    class Router(nn.Module):
        def forward(self, x):  # (top_k_index, top_k_weights, router_logits), as GraniteMoe's router
            logits = x[..., :E].reshape(-1, E).float()
            w, idx = logits.softmax(-1).topk(TOPK, dim=-1)
            return idx, w, logits

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_sparse_moe = nn.Module()
            self.block_sparse_moe.router = Router()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, input_ids, use_cache=False):  # no output_router_logits argument at all
            x = nn.functional.one_hot(input_ids % E, E).float()
            for ly in self.model.layers:
                ly.block_sparse_moe.router(x)
            return None

    r = lib.routing_stats(Fake(), [], torch.zeros(0), {"code": torch.ones(3, 5, dtype=torch.int32)}, 2, "cpu", False)
    st = r["per_domain"]["code"]
    assert r["available"] is True and st["layers"] == 2 and st["top_k"] == TOPK and st["mean_top25pct_expert_share"] == 0.5 and st["mean_norm_entropy"] < 0.6


def _fake_run(model, cell, seed, final, codec="none", n=100):
    posts = 4 if cell != "S0" else 0
    return {"model": model, "cell": cell, "seed": seed, "K": 2, "codec": codec, "n_coords": n, "initial_loss": 3.0, "final_loss": final, "per_domain_final": {"code": final, "wiki-fr": final},
            "round_wall_s": [1.0, 2.0, 3.0], "admitted_per_round": [[0, 1]] * 3, "late_discard_count": 1,
            "wire": {"up_bytes": posts * (4 * n if codec == "none" else 30), "down_bytes": 10, "n_posts": posts, "raw_delta_bytes": posts * 4 * n, "ideal_bytes": posts * (4.0 * n if codec == "none" else 25.0),
                     "mean_measured_rate_bits_per_dim": None if codec == "none" else 2.0},
            "init_identical": True, "identity_vs_s0": {"available": True, "max_abs_diff": 0.0, "passed": True}}


def test_aggregate_paired_ci_and_controls(e13):
    runs = []
    for s, (f0, f2) in zip((1, 2, 3), ((2.0, 2.1), (2.1, 2.3), (2.2, 2.2))):
        runs += [_fake_run("m", "S0", s, f0), _fake_run("m", "S1", s, f0), _fake_run("m", "S2", s, f2, codec="E8@2+EF")]
    res = e13.aggregate({"design_sha256": "x", "runs": runs})
    m = res["models"]["m"]
    d = m["paired_vs_S0"]["S2"]
    diffs = [0.1, 0.2, 0.0]
    assert d["final_loss_minus_S0_per_seed"] == pytest.approx(diffs)
    mean, sd = sum(diffs) / 3, (sum((x - sum(diffs) / 3) ** 2 for x in diffs) / 2) ** 0.5
    assert d["mean"] == pytest.approx(mean) and d["ci95"] == pytest.approx(4.302652729911275 * sd / 3 ** 0.5)
    assert m["paired_vs_S0"]["S1"]["mean"] == 0.0
    assert m["controls"]["s1_equals_s0"]["passed"] and m["controls"]["raw_wire_bytes_exact"]["passed"] and m["controls"]["codec_rate_on_target"]["passed"]
    assert m["controls_all_passed"] is True and res["controls_all_passed"] is True
    assert m["cells"]["S2"]["wire"]["compression_ratio_actual"] == pytest.approx(1600 / 120)
    assert m["cells"]["S2"]["wire"]["compression_ratio_ideal"] == pytest.approx(1600 / 100)
    assert m["cells"]["S1"]["round_latency_s"]["p50"] == 2.0 and m["cells"]["S2"]["late_discards_total"] == 3 and m["cells"]["S2"]["admitted_fraction_mean"] == 1.0
    assert len(res["receipt_sha256"]) == 64
    # a failing identity makes the controls fail; fewer than 3 seeds gives no interval
    runs[1]["identity_vs_s0"] = {"available": True, "max_abs_diff": 1e-3, "passed": False}
    assert e13.aggregate({"runs": runs})["controls_all_passed"] is False
    assert "needs exactly 3" in e13.aggregate({"runs": runs[:6]})["models"]["m"]["paired_vs_S0"]["S2"]["status"]


def test_raw_file_is_design_guarded(e13, tmp_path):
    p = tmp_path / "raw.json"
    e13.write_raw(p, "aaa", [{"model": "m", "cell": "S0", "seed": 1, "final_loss": 1.0}])
    assert len(e13.load_raw(p, "aaa")) == 1 and e13.load_raw(p, "bbb") == []
