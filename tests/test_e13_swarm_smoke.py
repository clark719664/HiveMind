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


@pytest.fixture(autouse=True)
def _restore_torch_flags():
    """main() calls configure_determinism() (global torch flags): put them back after every test."""
    det = torch.are_deterministic_algorithms_enabled()
    flags = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
    yield
    torch.use_deterministic_algorithms(det)
    torch.backends.cuda.enable_flash_sdp(flags[0])
    torch.backends.cuda.enable_mem_efficient_sdp(flags[1])
    torch.backends.cuda.enable_math_sdp(flags[2])


def test_smoke_runs_all_cells_through_main(e13, tmp_path, capsys):
    out = tmp_path / "e13_smoke.json"
    t0 = time.perf_counter()
    assert e13.main(["--smoke", "--out", str(out), "--scratch", str(tmp_path / "scr")]) == 0
    assert time.perf_counter() - t0 < 90.0
    raw = json.loads(out.read_text(encoding="utf-8"))
    by = {r["cell"]: r for r in raw["runs"]}
    assert set(by) == set(e13.ALL_CELLS) | set(e13.EXTRA_CELLS)
    assert set(by["S0"]["loss_curve"]) == {"0", "2"} and by["S0"]["eval_versions"] == [0, 2]  # registered evaluation versions only (smoke R = 2: no round 3)
    nf = by["S0"]["identity_noise_floor"]  # diagnostic S0b twin of the first seed
    assert nf["available"] and nf["max_abs_diff"] <= 1e-6 and by["S1"]["identity_vs_s0"]["max_abs_diff"] <= 1e-6
    assert all(r["status"] == "ok" and r["eval_by"] == {"0": 0, "2": 0} for r in raw["runs"] if r["cell"] in e13.CELL_SPECS)  # v mod M (M = 2), worker 0 alive
    # S1 == S0: the HTTP protocol with K=M, direct, no codec reproduces the sequential reference exactly
    assert by["S1"]["identity_vs_s0"]["passed"] and by["S1"]["identity_vs_s0"]["max_abs_diff"] <= 1e-6
    assert by["S1"]["final_theta_sha256"] == by["S0"]["final_theta_sha256"]
    # thread hosting (one model, M peer threads, same HTTP protocol) is also identical to the sequential reference
    assert by["S1t"]["hosting"] == "thread" and by["S1"]["hosting"] == "process"
    assert by["S1t"]["identity_vs_s0"]["passed"] and by["S1t"]["final_theta_sha256"] == by["S0"]["final_theta_sha256"]
    for c in ("S1", "S1t", "S2", "S3", "S4", "S5"):
        r = by[c]
        assert r["init_identical"] and set(r["loss_curve"]) == {"0", "2"} and len(r["round_wall_s"]) == 2 and len(r["admitted_per_round"]) == 2
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
    assert "identity_noise_floor" in rep["models"]["tiny"]["controls"]["s1_equals_s0"]["rows"][0]
    assert capsys.readouterr().out.strip().splitlines()[-1] == f"E13 DONE model=tiny cells_ok=7 cells_failed=0 report={tmp_path / 'e13_smoke.report.json'}"


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
    assert lib.parse_straggle("2:2:1.5;-1:1:3", 4) == {(2, 2): 1.5, (3, 1): 3.0}
    assert lib.parse_straggle("off", 4) == {} and lib.parse_churn("off", 4) is None
    assert lib.parse_churn("1:2:3", 4) == (1, 2, 3)
    with pytest.raises(ValueError):
        lib.parse_churn("1:3:2", 4)


def test_chaos_specs_reject_worker_zero_and_nonexistent_workers(lib, e13):
    for bad in ("0:1:2.0", "4:1:2.0", "-5:1:2.0", "-4:1:2.0", "1:1:2.0;0:2:1.0"):
        with pytest.raises(ValueError, match="worker"):
            lib.parse_straggle(bad, 4)
    for bad in ("0:1:2", "4:1:2", "-5:1:2", "-4:1:2"):
        with pytest.raises(ValueError, match="worker"):
            lib.parse_churn(bad, 4)
    with pytest.raises(ValueError, match="bad churn"):
        lib.parse_churn("x", 4)
    assert lib.parse_churn("-1:1:2", 4) == (3, 1, 2)
    with pytest.raises(SystemExit):  # clear error at parse time in main(), before anything runs
        e13.main(["--smoke", "--out", "/tmp/never.json", "--straggle", "0:1:2.0"])
    with pytest.raises(SystemExit):
        e13.main(["--estimate", "--churn", "9:1:2"])


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


# ---------------------------------------------------------------- review-fix tests
def _smoke_ctx(e13, tmp_path, **kw):
    return e13.Ctx(cfg=e13.SMOKE, hosting="process", scratch=tmp_path / "scr", device="cpu", **kw)


def _ok_rec(model, cell, seed, final=2.0, codec="none"):
    r = _fake_run(model, cell, seed, final, codec=codec)
    r.update(status="ok", cell_name=cell, wall_seconds=1.0, final_theta_path="/nonexistent")
    return r


def test_outer_nesterov_equals_torch_sgd(lib):
    g = torch.Generator().manual_seed(5)
    p = torch.nn.Parameter(torch.randn(50, generator=g))
    opt = torch.optim.SGD([p], lr=0.7, momentum=0.9, nesterov=True)
    theta, buf = p.detach().clone(), torch.zeros(50)
    for _ in range(4):
        grad = torch.randn(50, generator=g)
        p.grad = grad.clone()
        opt.step()
        theta, buf = lib.outer_nesterov(theta, buf, grad)
        assert float((theta - p.detach()).abs().max()) < 1e-6


def test_configure_determinism_sets_flags(lib):
    import os

    lib.configure_determinism()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and torch.backends.cuda.math_sdp_enabled()


def test_identity_noise_floor_twin_and_aggregate(e13, tmp_path):
    out = tmp_path / "raw.json"
    assert e13.main(["--smoke", "--out", str(out), "--scratch", str(tmp_path / "scr"), "--cells", "S0"]) == 0
    s0 = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
    assert s0["identity_noise_floor"]["available"] and s0["identity_noise_floor"]["max_abs_diff"] <= 1e-6
    assert (tmp_path / "scr" / "tiny" / "S0b_seed1_final.pt").exists()
    runs = [_ok_rec("m", "S0", 1, 2.0), _ok_rec("m", "S1", 1, 2.0)]
    runs[0]["identity_noise_floor"] = {"available": True, "max_abs_diff": 3e-8}
    row = e13.aggregate({"runs": runs})["models"]["m"]["controls"]["s1_equals_s0"]["rows"][0]
    assert row["identity_noise_floor"]["max_abs_diff"] == 3e-8 and row["passed"]


# ---- 1: memory
def test_plan_hosting_and_preflight_with_mocked_mem_get_info(e13, monkeypatch, tmp_path):
    G = e13.GIB
    peak = int(2.0 * G)
    # process needs 4 x (2.0 + 0.45) = 9.8 GiB; thread needs 2.5 GiB
    assert e13.plan_hosting("process", 4, peak, int(10 * G))["hosting"] == "process"
    fb = e13.plan_hosting("process", 4, peak, int(9 * G))
    assert fb["hosting"] == "thread" and fb["hosting_fallback"] is True and fb["need_process_bytes"] == 4 * (peak + e13.WORKER_OVERHEAD_B) and fb["need_thread_bytes"] == peak + e13.HOST_OVERHEAD_B
    with pytest.raises(e13.FailFast, match="2.50 GiB") as ei:
        e13.plan_hosting("process", 4, peak, int(2.4 * G))
    assert ei.value.details["free_bytes"] == int(2.4 * G)
    with pytest.raises(e13.FailFast):
        e13.plan_hosting("thread", 4, peak, int(2.4 * G))
    assert e13.plan_hosting("thread", 4, peak, int(3 * G))["hosting"] == "thread"
    # memory_preflight: reads torch.cuda.mem_get_info and the S0 peaks of this model; skipped on CPU / smoke
    ctx = e13.Ctx(cfg=e13.RunCfg(rounds=6, workers=4, inner=20), device="cuda", scratch=tmp_path)
    assert e13.memory_preflight("m", "process", ctx) is None  # no S0 peak known
    ctx.results[("m", "S0", 1)] = {"peak_reserved_bytes": peak}
    ctx.results[("m", "S0", 2)] = {"peak_reserved_bytes": int(1.5 * G)}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (int(9 * G), int(15 * G)))
    info = e13.memory_preflight("m", "process", ctx)
    assert info["hosting"] == "thread" and info["hosting_fallback"] and info["peak_reserved_bytes"] == peak
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (int(1 * G), int(15 * G)))
    with pytest.raises(e13.FailFast):
        e13.memory_preflight("m", "process", ctx)
    assert e13.memory_preflight("m", "process", e13.Ctx(cfg=e13.SMOKE, device="cpu", scratch=tmp_path)) is None


def test_hosting_fallback_spawns_one_thread_host_not_process_workers(e13, monkeypatch, tmp_path):
    info = {"requested": "process", "hosting": "thread", "hosting_fallback": True, "free_bytes": 1, "peak_reserved_bytes": 2, "need_process_bytes": 3, "need_thread_bytes": 1, "workers": 2}
    monkeypatch.setattr(e13, "memory_preflight", lambda *a, **k: info)
    spawned = []
    real = e13.lib.spawn_worker
    monkeypatch.setattr(e13.lib, "spawn_worker", lambda script, cfg_path, log_path, env: spawned.append(json.loads(Path(cfg_path).read_text())) or real(script, cfg_path, log_path, env))
    ctx = _smoke_ctx(e13, tmp_path)
    rec = e13.run_cell("tiny", "S1", 1, ctx)
    assert rec["hosting"] == "thread" and rec["hosting_fallback"] is True and rec["memory_preflight"] == info and rec["hosting_requested"] == "process"
    assert len(spawned) == 1 and spawned[0]["hosting"] == "thread" and spawned[0]["worker_id"] == -1 and spawned[0]["parent_pid"] > 0
    assert rec["status"] == "ok" and rec["identity_vs_s0"] == {"available": False}


def test_worker_env_sets_expandable_segments_and_peak_reserved_field(e13, monkeypatch, tmp_path):
    envs = []
    real = e13.lib.spawn_worker
    monkeypatch.setattr(e13.lib, "spawn_worker", lambda script, cfg_path, log_path, env: envs.append(env) or real(script, cfg_path, log_path, env))
    ctx = _smoke_ctx(e13, tmp_path)
    e13.run_cell("tiny", "S1t", 1, ctx)
    assert envs and envs[0]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    s0 = e13.run_cell("tiny", "S0", 1, ctx)
    assert "peak_reserved_bytes" in s0 and s0["peak_reserved_bytes"] is None  # CPU: no CUDA allocator


# ---- 2: orphans
def test_worker_exits_when_parent_dies(tmp_path):
    marker = tmp_path / "exit_code.txt"
    child = (f"import sys, time, os\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT / 'scripts')!r})\nfrom e13_watchdog import start_parent_watchdog\n"
             f"def ex(code):\n    Path({str(marker)!r}).write_text(str(code)); os._exit(code)\n"
             "start_parent_watchdog(0.1, int(sys.argv[1]), exit_fn=ex)\ntime.sleep(60)\n")
    parent = f"import os, subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}, str(os.getpid())])\ntime.sleep(1.5)\n"
    import subprocess

    t0 = time.time()
    subprocess.run([sys.executable, "-c", parent], check=True, timeout=30)  # the parent exits after 1.5 s without killing its child
    while not marker.exists() and time.time() - t0 < 30:
        time.sleep(0.1)
    assert marker.exists() and marker.read_text() == "3"


def test_watchdog_does_not_fire_while_parent_lives_and_fires_on_change():
    import os

    from e13_watchdog import start_parent_watchdog

    fired = []
    start_parent_watchdog(0.05, os.getppid(), exit_fn=fired.append)
    time.sleep(0.3)
    assert fired == []
    fired2 = []
    start_parent_watchdog(0.05, 10, exit_fn=fired2.append, getppid=iter([10, 10, 11] + [11] * 50).__next__)  # parent pid changes after two polls
    time.sleep(0.5)
    assert fired2[:1] == [3]


def test_spawn_worker_registers_closes_log_and_kill_spawned_kills(lib, tmp_path):
    import os

    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(60)\n")
    p = lib.spawn_worker(str(script), tmp_path / "cfg.json", tmp_path / "w.log", dict(os.environ))
    assert p in lib._SPAWNED and p.poll() is None
    open_logs = [f for f in os.listdir("/proc/self/fd") if os.path.islink(f"/proc/self/fd/{f}") and os.readlink(f"/proc/self/fd/{f}").endswith("w.log")]
    assert open_logs == []  # the parent's handle was closed
    assert lib.kill_spawned() >= 1 and p.poll() is not None and lib._SPAWNED == []


def test_main_installs_sigterm_handler_and_restores(e13, tmp_path, monkeypatch):
    import signal

    prev = signal.getsignal(signal.SIGTERM)
    seen = []
    monkeypatch.setattr(e13, "_coordinator_main", lambda a, ap: seen.append(signal.getsignal(signal.SIGTERM)) or 0)
    assert e13.main(["--smoke", "--out", str(tmp_path / "r.json")]) == 0
    assert seen == [e13._sigterm] and signal.getsignal(signal.SIGTERM) == prev
    with pytest.raises(SystemExit) as ei:
        e13._sigterm(15, None)
    assert ei.value.code == 143


# ---- 3: hang protection
def test_startup_deadline_aborts_with_clear_error(e13, monkeypatch, tmp_path):
    import subprocess

    spawned = []

    def fake_spawn(script, cfg_path, log_path, env):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        spawned.append(p)
        return p

    monkeypatch.setattr(e13.lib, "spawn_worker", fake_spawn)
    ctx = _smoke_ctx(e13, tmp_path, cell_timeout_s=2.0)  # startup deadline = min(cell_timeout, 900) = 2 s
    t0 = time.time()
    with pytest.raises(TimeoutError, match=r"did not register"):
        e13.run_cell("tiny", "S1", 1, ctx)
    assert time.time() - t0 < 30 and spawned and all(p.poll() is not None for p in spawned)  # the finally block killed them by pid


def test_stall_watchdog_limits_heartbeat_and_reset(lib):
    now = [0.0]
    beats = []
    wd = lib.StallWatchdog("lbl", clock=lambda: now[0], printer=beats.append)
    assert wd.limit_s([]) == 1800.0 and wd.limit_s([10.0, 20.0, 30.0]) == 600.0 and wd.limit_s([100.0, 100.0]) == 2000.0  # max(600, 20 x median)
    st = lambda: "state-x"  # noqa: E731
    wd.poll(("a",), [], st)
    now[0] = 61.0
    wd.poll(("a",), [], st)
    assert len(beats) == 1 and "state-x" in beats[0] and "lbl" in beats[0]
    now[0] = 1700.0
    wd.poll(("b",), [], st)  # progress resets the stall clock
    now[0] = 1700.0 + 1799.0
    wd.poll(("b",), [], st)
    now[0] = 1700.0 + 1801.0
    with pytest.raises(TimeoutError, match="no progress.*state-x"):
        wd.poll(("b",), [], st)
    now[0] = 0.0
    wd2 = lib.StallWatchdog("x", clock=lambda: now[0], printer=beats.append)
    wd2.poll(1, [20.0], st)
    now[0] = 599.0
    wd2.poll(1, [20.0], st)
    now[0] = 601.0
    with pytest.raises(TimeoutError):
        wd2.poll(1, [20.0], st)


def test_coordinator_progress_token_and_summary(lib):
    c, _ = _coord(lib, workers=3, k=2)
    t0 = c.progress_token()
    c.post_delta(0, _delta_meta(1, 8), torch.ones(8).numpy().tobytes())
    assert c.progress_token() != t0 and "round 1/3" in c.state_summary() and c.missing_workers() == []


def test_cell_timeout_default_is_3600(e13):
    assert e13.DEFAULT_CELL_TIMEOUT_S == 3600.0 and e13.Ctx().cell_timeout_s == 3600.0 and e13.lib.STARTUP_TIMEOUT_S == 900.0


def test_chaos_specs_reject_worker_zero_and_nonexistent_workers(lib, e13):
    for bad in ("0:1:2.0", "4:1:2.0", "-5:1:2.0", "-4:1:2.0", "1:1:2.0;0:2:1.0"):
        with pytest.raises(ValueError, match="worker"):
            lib.parse_straggle(bad, 4)
    for bad in ("0:1:2", "4:1:2", "-5:1:2", "-4:1:2"):
        with pytest.raises(ValueError, match="worker"):
            lib.parse_churn(bad, 4)
    with pytest.raises(ValueError, match="bad churn"):
        lib.parse_churn("x", 4)
    assert lib.parse_churn("-1:1:2", 4) == (3, 1, 2)
    with pytest.raises(SystemExit):  # clear error at parse time in main(), before anything runs
        e13.main(["--smoke", "--out", "/tmp/never.json", "--straggle", "0:1:2.0"])
    with pytest.raises(SystemExit):
        e13.main(["--estimate", "--churn", "9:1:2"])


def test_failed_cell_is_recorded_continues_and_is_rerun_on_resume(e13, monkeypatch, tmp_path, capsys):
    calls = []
    state = {"s2_ok": False}

    def fake_run_cell(model, cell, seed, ctx=None):
        calls.append(cell)
        if cell == "S0b":
            raise RuntimeError("no twin in this test")
        if cell == "S2" and not state["s2_ok"]:
            raise RuntimeError("boom")
        return _ok_rec(model, cell, seed, 2.0, codec="E8@2+EF" if cell == "S2" else "none")

    monkeypatch.setattr(e13, "run_cell", fake_run_cell)
    out, rep = tmp_path / "raw.json", tmp_path / "rep.json"
    argv = ["--smoke", "--out", str(out), "--scratch", str(tmp_path / "scr"), "--cells", "S0,S2,S3", "--report", str(rep), "--parallel-cells", "1"]
    assert e13.main(argv) == 0
    runs = json.loads(out.read_text())["runs"]
    st = {r["cell"]: r for r in runs}
    assert st["S2"]["status"] == "failed" and "boom" in st["S2"]["error"] and st["S2"]["traceback_tail"] and st["S0"]["status"] == "ok" and st["S3"]["status"] == "ok"  # continued after the failure
    assert st["S0"]["identity_noise_floor"]["available"] is False
    report = json.loads(rep.read_text())
    m = report["models"]["tiny"]
    assert m["cells"]["S2"]["status"] == "failed" and m["controls_all_passed"] is False and report["controls_all_passed"] is False and m["failed_cells"][0]["cell"] == "S2"
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == f"E13 DONE model=tiny cells_ok=2 cells_failed=1 report={rep.resolve()}"
    # resume: only status-ok records are skipped, the failed cell is retried
    calls.clear()
    state["s2_ok"] = True
    assert e13.main(argv + ["--resume"]) == 0
    assert calls == ["S2"]
    runs2 = json.loads(out.read_text())["runs"]
    assert sorted((r["cell"], r["status"]) for r in runs2) == [("S0", "ok"), ("S2", "ok"), ("S3", "ok")]
    assert json.loads(rep.read_text())["models"]["tiny"]["failed_cells"] == []


def test_load_raw_returns_only_ok_runs(e13, tmp_path):
    p = tmp_path / "raw.json"
    e13.write_raw(p, "d", [{"model": "m", "cell": "S0", "seed": 1, "status": "ok"}, {"model": "m", "cell": "S2", "seed": 1, "status": "failed", "error": "x"}, {"model": "m", "cell": "S3", "seed": 1}])
    assert [r["cell"] for r in e13.load_raw(p, "d")] == ["S0", "S3"]


def test_aggregate_tolerates_failed_and_missing_cells(e13):
    runs = [_ok_rec("m", "S0", 1, 2.0), {"model": "m", "cell": "S1", "seed": 1, "status": "failed", "error": "RuntimeError('x')"}, {"model": "m", "cell": "S2", "seed": 1, "status": "failed", "error": "e"}]
    m = e13.aggregate({"runs": runs})["models"]["m"]
    assert m["cells"]["S1"]["status"] == "failed" and m["controls_all_passed"] is False and len(m["failed_cells"]) == 2
    assert m["controls"]["no_failed_cells"]["passed"] is False and m["controls"]["s1_equals_s0"]["passed"] is None
    only_failed = e13.aggregate({"runs": [{"model": "z", "cell": "S0", "seed": 1, "status": "failed", "error": "e"}]})
    assert only_failed["models"]["z"]["controls_all_passed"] is False and only_failed["controls_all_passed"] is False
    assert e13.aggregate({"runs": []})["controls_all_passed"] is False
    ok_later = [{"model": "m", "cell": "S2", "seed": 1, "status": "failed", "error": "e"}, _ok_rec("m", "S2", 1, 2.0, codec="E8@2+EF"), _ok_rec("m", "S0", 1, 2.0)]
    assert e13.aggregate({"runs": ok_later})["models"]["m"]["failed_cells"] == []


def test_report_keeps_other_models_sections(e13, tmp_path):
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps(e13.aggregate({"runs": [_ok_rec("qwen", "S0", 1, 2.0)]})))
    merged = e13.merge_with_existing_report(rep, e13.aggregate({"runs": [_ok_rec("granite", "S0", 1, 2.0)]}))
    assert set(merged["models"]) == {"qwen", "granite"} and len(merged["receipt_sha256"]) == 64


# ---- 5: Granite / time budget / finite checks
def test_time_budget_failfast_after_first_s0_round(e13, tmp_path):
    ctx = _smoke_ctx(e13, tmp_path, max_hours=1e-12, planned_runs=1000)
    with pytest.raises(e13.FailFast, match=r"projected \d+\.\d+ h > --max-hours") as ei:
        e13.run_s0("tiny", 1, ctx)
    assert ei.value.details["planned_runs"] == 1000 and ei.value.details["measured_sec_per_step"] > 0
    ok = _smoke_ctx(e13, tmp_path / "b", max_hours=3.0, planned_runs=8)
    assert e13.run_s0("tiny", 1, ok)["status"] == "ok" and ok.s0_done
    assert e13.lib.MODEL_META["tiny"]["sec_per_step"] is None  # the projection uses the measured step time, so it also covers models without an assumption


def test_time_budget_exit_code_3_and_status(e13, tmp_path, capsys):
    rc = e13.main(["--smoke", "--out", str(tmp_path / "raw.json"), "--scratch", str(tmp_path / "scr"), "--cells", "S0", "--report", str(tmp_path / "a.json"), "--max-hours", "1e-12"])
    assert rc == 3 and json.loads((tmp_path / "a.json").read_text())["status"].startswith("aborted by registered fail-fast rule: projected ")
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("E13 DONE model=tiny cells_ok=0 cells_failed=0")


def test_nonfinite_round0_is_failfast_and_nan_fails_the_learning_gate(e13, monkeypatch, tmp_path):
    assert "does not learn" in e13.check_learning_gate({"0": float("nan"), "3": 2.0}, 3, 0.05)
    assert "does not learn" in e13.check_learning_gate({"0": 3.0, "3": float("nan")}, 3, 0.05)
    assert e13.check_learning_gate({"0": 3.0, "3": 2.0}, 3, 0.05) is None
    monkeypatch.setattr(e13.lib, "eval_domains", lambda model, params, theta, held, bs, device, amp: {d: float("nan") for d in held})
    with pytest.raises(e13.FailFast, match=r"non-finite round-0 loss \(fp16 overflow\?\)"):
        e13.run_s0("tiny", 1, _smoke_ctx(e13, tmp_path))


def test_granite_meta_experts_impl_and_loader_fallback(lib, monkeypatch):
    assert lib.MODEL_META["granite1b"]["experts_implementation"] == "eager" and lib.MODEL_META["granite3b"]["experts_implementation"] == "eager"
    assert "experts_implementation" not in lib.MODEL_META["qwen0.5b"]
    assert lib.resolve_experts_impl("granite1b", None) == "eager" and lib.resolve_experts_impl("granite1b", "grouped_mm") == "grouped_mm"
    assert lib.resolve_experts_impl("granite1b", "default") is None and lib.resolve_experts_impl("qwen0.5b", None) is None
    import transformers

    seen = []

    class FakeLM:
        reject = False

        @staticmethod
        def from_pretrained(path, **kw):
            seen.append(sorted(kw))
            if "experts_implementation" in kw and FakeLM.reject:
                raise TypeError("unexpected keyword experts_implementation")
            return lib.load_tiny("cpu")[0]

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeLM)
    monkeypatch.setattr(transformers, "AutoTokenizer", type("T", (), {"from_pretrained": staticmethod(lambda p: None)}))
    monkeypatch.setattr(lib, "_snapshot", lambda repo: "/x")
    lib.LOADERS["granite1b"]("cpu", "eager")
    assert seen[-1] == ["dtype", "experts_implementation"] and lib.LOAD_INFO["experts_impl_used"] == "eager"
    FakeLM.reject = True
    lib.LOADERS["granite1b"]("cpu", "eager")
    assert seen[-2:] == [["dtype", "experts_implementation"], ["dtype"]] and lib.LOAD_INFO["experts_impl_used"] is None and "TypeError" in lib.LOAD_INFO["experts_impl_error"]
    lib.LOADERS["granite1b"]("cpu", None)
    assert seen[-1] == ["dtype"]


def test_experts_impl_cli_flag(e13):
    assert e13.main(["--experts-impl", "grouped_mm", "--estimate"]) == 0
    with pytest.raises(SystemExit):
        e13.main(["--experts-impl", "bogus", "--estimate"])


# ---- 6: codec
def test_codec_int8_path_byte_identical_and_int16_when_overflow(lib, monkeypatch):
    import zlib

    v = torch.randn(2048, generator=torch.Generator().manual_seed(2)) * 1e-3
    enc = lib.WireCodec("E8@2+EF")
    body, meta = enc.encode(v)
    ref = lib._CapturingCodec(enc.kind, enc.rate, enc.ef)
    ref.quantize(v)
    expect = zlib.compress((ref.last_codewords * lib.E8_CODEWORD_SCALE).round().to(torch.int8).numpy().tobytes(), 6)
    assert body == expect and meta["cw_dtype"] == "int8" and enc.n_int16 == 0
    # force |2c| > 127 (scale 200): int16 on the wire, tagged, decodes to the encoder's dequantized vector
    monkeypatch.setattr(lib, "E8_CODEWORD_SCALE", 200.0)
    enc2 = lib.WireCodec("E8@2+EF")
    body2, meta2 = enc2.encode(v)
    assert meta2["cw_dtype"] == "int16" and enc2.n_int16 == 1
    assert len(zlib.decompress(body2)) == 2 * 2048
    ref2 = lib._CapturingCodec(enc2.kind, enc2.rate, enc2.ef).quantize(v)
    assert torch.equal(lib.decode_body(body2, meta2), ref2.dequantized)
    c, _ = _coord(lib, workers=1, k=1, n=2048)  # the coordinator accepts the int16 body and counts it; nothing crashes
    meta2.update(round=1, compute_s=0.1, tokens=1, steps=1)
    assert c.post_delta(0, meta2, body2)["accepted"] and c.posts[0]["cw_dtype"] == "int16"


# ---- 7: evaluation duty
def test_eval_versions_and_assignee_rule(lib):
    assert lib.eval_versions(6) == (0, 3, 6) and lib.eval_versions(2) == (0, 2) and lib.eval_versions(3) == (0, 3)
    c = lib.Coordinator(4, rounds=6, inner=3, k=2, merge="direct", grace=False)
    assert c.eval_versions == (0, 3, 6)
    c.hello(0, "s", torch.zeros(8).numpy().tobytes())
    for w in (1, 2, 3):
        c.hello(w, "s", b"")
    c.open_first()
    assert [c.eval_assignee(v) for v in (0, 3, 6)] == [0, 3, 2]  # v mod M
    assert c.eval_todo(1) == [] and c.eval_todo(3) == [] and c.eval_todo(2) == [] and c.eval_todo(0) == [0]  # only version 0 exists; it is assigned to worker 0
    c.history.update({3: b"x", 6: b"y"})  # pretend they were published
    assert c.eval_todo(3) == [3] and c.eval_todo(2) == [6] and c.eval_todo(0) == [0]  # worker 0 re-polling keeps its own claim on version 0
    c.post_eval(0, {"mixture": 1.0}, 10)
    c.post_eval(3, {"mixture": 1.0}, 10)
    assert c.eval_todo(3) == [] and c.eval_todo(0) == []
    c.set_alive(2, False)  # assignee of version 6 is dead: the lowest alive id takes over
    assert c.eval_assignee(6) == 0 and c.eval_todo(0) == [6] and c.eval_todo(1) == []
    c.set_alive(0, False)  # worker 0 dies mid-evaluation: its claim is void and the next lowest alive takes over
    assert c.eval_assignee(6) == 1 and c.eval_todo(1) == [6]
    assert not c.wait_evals(0.01)
    c.post_eval(6, {"mixture": 1.0}, 1)
    assert c.wait_evals(0.01)


def test_eval_todo_endpoint_over_http(lib):
    import threading

    c, _ = _coord(lib, workers=2, k=1)
    srv = lib.make_server(c)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.loads(lib.http_request(f"{base}/evaltodo?worker=1&incarn=0")[1]) == []
        assert json.loads(lib.http_request(f"{base}/evaltodo?worker=0&incarn=0")[1]) == [0]
    finally:
        srv.shutdown()
        srv.server_close()


# ---- 8: held-out sampling
def test_heldout_is_stride_sampled_over_the_whole_region(lib):
    assert lib.stride_indices(10, 4) == [0, 2, 5, 7] and lib.stride_indices(4, 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        lib.stride_indices(3, 4)
    seq, E, n = 8, 4, 8 * 100
    ids = list(range(n))
    _, ev, prov = lib._split_ids(ids, seq, E)
    held_start = int(0.8 * n)  # 640 -> 160 tokens = 20 sequences
    n_held_seq = (n - held_start) // seq
    assert n_held_seq == 20 and prov["eval_sequence_indices"] == lib.stride_indices(20, E) == [0, 5, 10, 15]
    assert ev.shape == (E, seq) and [int(x) for x in ev[:, 0]] == [held_start + i * seq for i in prov["eval_sequence_indices"]]
    _, ev2, prov2 = lib._split_ids(ids, seq, E)
    assert torch.equal(ev, ev2) and prov2["eval_slice_sha256"] == prov["eval_slice_sha256"] and "stride" in prov["eval_sampling"]
    contiguous = torch.tensor(ids[held_start : held_start + E * seq], dtype=torch.int32).reshape(E, seq)
    assert prov["eval_slice_sha256"] != lib.tensor_sha(contiguous)


def test_default_scratch_is_not_under_artifacts(e13):
    import tempfile

    assert str(e13.Ctx().scratch).startswith(tempfile.gettempdir()) and (e13.PROJECT_ROOT / "artifacts") not in e13.Ctx().scratch.parents


def test_estimate_counts_three_evaluations_and_twin(e13):
    est = e13.estimate("qwen0.5b", e13.RunCfg(rounds=6, workers=4, inner=20))
    assert est["breakdown_s"]["eval_s"] == 18 * 3 * e13.SEC_PER_EVAL and est["breakdown_s"]["diagnostic_twin_S0b_s"] > 0
