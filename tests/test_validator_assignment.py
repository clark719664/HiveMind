"""Tests for HM-S10-08 architecture-matched validator assignment (CPU only, synthetic data)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hivemind.boinc.validator_assignment import (
    AcceleratorFingerprint,
    EpsilonTable,
    FallbackPolicy,
    ValidatorRegistry,
)


def fp(cls="cuda", cc="7.5", tv="2.4", dtype="float32", name="dev"):
    return AcceleratorFingerprint(cls, name, cc, tv, "12.1", dtype)


def test_class_key_ignores_name_and_runtime():
    a = fp(name="Tesla T4")
    b = AcceleratorFingerprint("cuda", "other", "7.5", "2.4", "11.8", "float32")
    assert a.class_key() == b.class_key() == ("cuda", "7.5", "2.4", "float32")
    assert fp(cc="8.9").class_key() != a.class_key()
    assert fp(dtype="bfloat16").class_key() != a.class_key()


def test_invalid_device_class_and_frozen():
    with pytest.raises(ValueError):
        fp(cls="fpga")
    with pytest.raises(Exception):
        fp().dtype = "x"  # type: ignore[misc]


def test_from_local_cpu_only():
    local = AcceleratorFingerprint.from_local("float32")
    assert local.device_class in {"cuda", "rocm", "mps", "cpu"}
    assert local.dtype == "float32"
    assert local.class_key()[2].count(".") <= 1


def test_assignment_deterministic_and_matched():
    reg = ValidatorRegistry()
    for i in range(5):
        reg.register(f"t4-{i}", fp())
    reg.register("ada", fp(cc="8.9"))
    first = reg.assign(fp(), rng_seed=7)
    assert first == reg.assign(fp(), rng_seed=7)
    vid, policy, _ = first
    assert vid.startswith("t4-") and policy is None
    assert {reg.assign(fp(), rng_seed=s)[0] for s in range(40)} > {vid}
    assert reg.assign(fp(cc="8.9"), 1)[0] == "ada"
    assert reg.assign(fp(), 7, exclude=[vid])[0] != vid


def test_fallback_when_no_match():
    reg = ValidatorRegistry()
    reg.register("cpu-1", fp("cpu", None))
    vid, policy, reason = reg.assign(fp(), rng_seed=1)
    assert vid is None and policy is FallbackPolicy.REJECT_UNMATCHED and "no validator" in reason
    reg2 = ValidatorRegistry(default_fallback=FallbackPolicy.LOSS_TRAJECTORY_CHECK)
    assert reg2.assign(fp(), 1)[1] is FallbackPolicy.LOSS_TRAJECTORY_CHECK
    assert {p.name for p in FallbackPolicy} == {"REJECT_UNMATCHED", "FIXED_POINT_REPLAY", "LOSS_TRAJECTORY_CHECK"}


def test_epsilon_defaults():
    t = EpsilonTable()
    a, b = fp().class_key(), fp("cpu", None).class_key()
    assert t.eps_for(a, a) == 1e-4
    assert t.eps_for(a, b) is None and t.eps_for(b, a) is None
    t.set_eps(a, b, 5e-3)
    assert t.eps_for(b, a) == 5e-3


def _report(tmp_path, b_linf):
    cell = lambda m, n=3: {"max_linf": m, "min_linf": 0.0, "n_above_eps": 0, "n": n}
    p = tmp_path / "E12.json"
    p.write_text(json.dumps({
        "eps": 1e-4,
        "summary": {
            "A_same_process_T4": cell(0.0),
            "B_separate_process_T4": cell(b_linf),
            "C_cross_T4_vs_cpu": cell(2e-3),
            "D_proxy_fp64_snap_T4_vs_cpu": cell(1e-6),
            "E_tamper": cell(1.0),
        },
        "analysis": {},
    }))
    return p


def test_from_e12_report_missing_file(tmp_path):
    t = EpsilonTable.from_e12_report(tmp_path / "nope.json")
    assert t.measured == {} and t.flags == [] and t.same_class_eps == 1e-4


def test_from_e12_report_records_without_changing_eps(tmp_path):
    t = EpsilonTable.from_e12_report(_report(tmp_path, 1e-6))
    assert t.measured["C_cross_T4_vs_cpu"]["max_linf"] == 2e-3
    assert t.measured["C_cross_T4_vs_cpu"]["kind"] == "cross_class"
    assert t.flags == []
    a, b = fp().class_key(), fp("cpu", None).class_key()
    assert t.eps_for(a, a) == 1e-4 and t.eps_for(a, b) is None


def test_from_e12_report_flags_same_class_exceedance(tmp_path):
    t = EpsilonTable.from_e12_report(_report(tmp_path, 3e-4))
    assert len(t.flags) == 1 and "B_separate_process_T4" in t.flags[0]
    assert t.same_class_eps == 1e-4  # not changed silently


def _validator_cls():
    from hivemind.boinc.validator import DeterministicSpotCheckValidator

    return DeterministicSpotCheckValidator


def _stubs():
    wu = SimpleNamespace(wu_id="wu-1", rng_seed=0, inner_steps_H=500)
    sub = SimpleNamespace(worker_id="w1", micro_checkpoints={})  # would flag the account if replay path ran
    return wu, sub


def test_validator_cross_architecture_short_circuit():
    v = _validator_cls()()
    wu, sub = _stubs()
    verdict = v.verify_submission(
        None, wu, sub, None, "steam-1",  # type: ignore[arg-type]
        worker_fingerprint=fp(cc="7.5"), validator_fingerprint=fp(cc="8.9"),
    )
    assert verdict.passed is False and verdict.steam_account_flagged is False
    assert verdict.reason == "not comparable: cross-architecture replay (HM-S10-08)"
    assert v.flagged_steam_accounts == {}


def test_validator_default_behaviour_unchanged_and_same_class_proceeds():
    wu, sub = _stubs()
    for kwargs in ({}, {"worker_fingerprint": fp(), "validator_fingerprint": fp()}, {"worker_fingerprint": fp()}):
        v = _validator_cls()()
        verdict = v.verify_submission(None, wu, sub, None, "steam-1", **kwargs)  # type: ignore[arg-type]
        assert verdict.steam_account_flagged is True  # reached the existing zero-checkpoint path
        assert "zero micro-checkpoints" in verdict.reason
