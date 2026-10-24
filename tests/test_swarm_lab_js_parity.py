"""Parity of the JavaScript Swarm Lab engine (docs/site/swarm_lab.js) with the Python contract (src/hivemind/lab/swarm_lab.py).

Runs node on docs/site/parity_driver.js with a grid of scenarios and compares loss curves, per-round admitted/late/required/carried lists, round times,
total simulated time, retention and the digest with Python's ``simulate`` to 1e-9. Skipped (with a clear message) when node is not installed.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from hivemind.lab.swarm_lab import POLICIES, PRESETS, STRAGGLES, Rng, simulate

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "docs" / "site" / "parity_driver.js"
TOL = 1e-9
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed: the JavaScript parity test needs `node` (>= 18) on PATH")


def _grid() -> list[dict[str, Any]]:
    """presets x seeds 1..5 x policies x straggle modes x K, plus hand-picked edge scenarios (rda, odd straggle pattern, churn, grace 0, M=2, M=6)."""
    grid: list[dict[str, Any]] = []
    for preset in PRESETS.values():
        for seed in range(1, 6):
            for policy in POLICIES:
                for straggle in STRAGGLES:
                    for k in (1, 2, 4):
                        grid.append({**preset, "seed": seed, "policy": policy, "straggle": straggle, "K": k})
    grid += [
        {"merge": "rda", "straggle": "persistent", "policy": "deferred", "seed": 3},
        {"merge": "rda", "straggle": "rotating", "policy": "fair", "seed": 4, "K": 2},
        {"straggle": "persistent", "straggle_every": 2, "policy": "discard", "seed": 2},
        {"straggle": "rotating", "straggle_every": 2, "policy": "deferred", "seed": 5, "K": 2},
        {"M": 6, "K": 4, "straggle": "rotating", "policy": "deferred", "seed": 7, "rounds": 15},
        {"M": 6, "K": 3, "straggle": "persistent", "policy": "fair", "seed": 8, "churn_start": 2, "churn_end": 9},
        {"M": 2, "K": 1, "straggle": "persistent", "policy": "fair", "seed": 9},
        {"M": 3, "K": 1, "grace": 0.0, "policy": "fair", "seed": 11, "churn_start": 3, "churn_end": 4},
        {"M": 1, "K": 1, "seed": 6},
        {"spread": 0.0, "offset": 0.0, "sigma": 0.0},
        {"seed": 2**32 + 7, "straggle": "persistent", "policy": "deferred"},
        {"dim": 5, "H": 3, "eta": 0.05, "sigma": 0.5, "outer_lr": 0.3, "outer_momentum": 0.5, "penalty": 0.5, "straggle": "persistent", "policy": "fair", "seed": 12},
    ]
    return grid


INVALID: list[dict[str, Any]] = [
    {"M": 0}, {"M": 7}, {"K": 0}, {"K": 5}, {"rounds": 0}, {"dim": 1}, {"eta": 0.0}, {"sigma": -1.0}, {"outer_lr": 0.0}, {"outer_momentum": 1.0},
    {"merge": "x"}, {"policy": "x"}, {"straggle": "x"}, {"M": 1, "K": 1, "straggle": "persistent"}, {"churn_start": 5, "churn_end": 2}, {"churn_start": 1, "churn_end": 99},
    {"M": 1, "K": 1, "churn_start": 1, "churn_end": 2}, {"offset": -1.0},
]


@pytest.fixture(scope="module")
def node_output(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    spec = tmp_path_factory.mktemp("parity") / "spec.json"
    scenarios = _grid()
    spec.write_text(json.dumps({"scenarios": scenarios, "invalid": INVALID}), encoding="utf-8")
    proc = subprocess.run([str(NODE), str(DRIVER), str(spec)], capture_output=True, text=True, timeout=50, check=False,
                          env={"PATH": os.environ.get("PATH", ""), "CUDA_VISIBLE_DEVICES": ""})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    out["scenarios"] = scenarios
    out["python"] = [simulate(sc) for sc in scenarios]  # computed once, shared by the tests below
    return out


def _close(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def test_grid_size_is_meaningful(node_output: dict[str, Any]) -> None:
    assert len(node_output["scenarios"]) >= 500
    assert len(node_output["results"]) == len(node_output["scenarios"])


def test_pinned_prng_values_match_python(node_output: dict[str, Any]) -> None:
    prng = node_output["prng"]
    assert prng["uniform_12345"] == [0.9797282677609473, 0.3067522644996643, 0.484205421525985, 0.817934412509203, 0.5094283693470061]  # exact
    want = [-0.07064807449259204, 0.49860719149527216, -0.6676634049638013]
    assert _close(prng["gauss_12345"], want) < 1e-12
    r = Rng(12345)
    assert prng["uniform_12345"] == [r.uniform() for _ in range(5)]
    assert prng["state_wrap"][0] == prng["state_wrap"][1] == Rng(12345).state
    assert prng["sha256_abc"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_simulate_parity_over_grid(node_output: dict[str, Any]) -> None:
    worst = 0.0
    for sc, js, py in zip(node_output["scenarios"], node_output["results"], node_output["python"]):
        rows = py["rounds"]
        worst = max(worst, _close(js["loss_curve"], py["loss_curve"]), abs(js["total_time"] - py["total_time"]), _close(js["round_time"], [r["round_time"] for r in rows]),
                    _close(js["domain_losses_final"], py["final_domain_losses"]), abs(js["sync_final_loss"] - py["sync_final_loss"]))
        assert js["admitted"] == [r["admitted"] for r in rows], sc
        assert js["late"] == [r["late"] for r in rows], sc
        assert js["required"] == [r["required"] for r in rows], sc
        assert js["carried"] == [r["carried"] for r in rows], sc
        assert js["dropped_stale"] == [r["dropped_stale"] for r in rows], sc
        if py["retention"] is None:
            assert js["retention"] is None, sc
        else:
            assert js["retention"] is not None and abs(js["retention"] - py["retention"]) <= TOL * max(1.0, abs(py["retention"])), sc
        assert math.isfinite(worst)
    print(f"parity grid: {len(node_output['scenarios'])} scenarios, max deviation {worst:.3e}")
    assert worst <= TOL, f"max deviation {worst:.3e} exceeds {TOL}"


def test_digest_matches_python_for_nearly_all_scenarios(node_output: dict[str, Any]) -> None:
    # The digest hashes the curve rounded to 9 decimals, so a sub-1e-15 libm difference can flip a rounding tie at most very rarely.
    mismatches = [sc for sc, js, py in zip(node_output["scenarios"], node_output["results"], node_output["python"]) if js["digest"] != py["digest"]]
    assert len(mismatches) <= max(1, len(node_output["scenarios"]) // 500), mismatches[:3]


def test_invalid_scenarios_are_rejected_like_python(node_output: dict[str, Any]) -> None:
    assert node_output["invalid_rejected"] == [True] * len(INVALID)
    for bad in INVALID:
        with pytest.raises(ValueError):
            simulate(bad)
