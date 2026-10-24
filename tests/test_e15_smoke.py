"""E15 --smoke end to end (tiny random model, CPU, synthetic problems; plumbing only) and the registered fail-fast rule."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_experiment_E15_gwaya_rft.py"
sys.path.insert(0, str(ROOT / "scripts"))
import e15_gate as gate_mod  # noqa: E402

pytestmark = pytest.mark.skipif(not gate_mod.gate_available(), reason="sandbox unavailable")


def run(args, timeout=170):
    return subprocess.run([sys.executable, "-W", "ignore", str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd="/")


def test_smoke_end_to_end_and_resume(tmp_path):
    before = sorted(p.name for p in (ROOT / "artifacts" / "experiments").glob("E15*")) if (ROOT / "artifacts" / "experiments").exists() else []
    out = tmp_path / "o"
    r = run(["--smoke", "--out", str(out)])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    done = [l for l in r.stdout.splitlines() if l.startswith("E15 DONE")]
    assert len(done) == 1 and "stages_failed=0" in done[0]
    rep = json.loads((out / "E15_gwaya_rft_report.json").read_text())
    assert rep["smoke"] and rep["controls_all_passed"] and set(rep["conditions"]) == {"C0", "C1", "C2", "C3", "C0h", "C2h"}
    for k in ("C1-C0", "C2-C0", "C3-C1", "C3-C0"):
        assert k in rep["paired_comparisons"] and 0 <= rep["paired_comparisons"][k]["mcnemar_p"] <= 1
    c = rep["controls"]
    assert c["ii_planted_stubs_rejected"]["pass"] and c["iii_containment_probes"]["pass"] and c["iv_no_leakage"]["pass"] and c["v_gate_available"]["pass"]
    assert rep["stages"]["train"]["lora_nonzero_B"] and rep["stages"]["rft"]["problems_with_accepted"] >= 1
    assert (out / "scratch" / "E15_lora.pt").exists()
    # nothing outside --out and the system temp dir was written under artifacts/
    after = sorted(p.name for p in (ROOT / "artifacts" / "experiments").glob("E15*")) if (ROOT / "artifacts" / "experiments").exists() else []
    assert before == after
    # resume: every stage is skipped
    r2 = run(["--smoke", "--out", str(out), "--resume"])
    assert r2.returncode == 0 and r2.stdout.count("resumed from") == 6, r2.stdout[-1500:]


def test_fail_fast_exit_3(tmp_path, monkeypatch):
    """With an impossible yield threshold the rft stage must abort with exit 3 and status 'aborted by registered fail-fast rule'."""
    # smoke injects the reference into sample 0, so every problem yields; force the abort through the threshold override module
    code = ("import sys, runpy\nsys.argv = ['x', '--smoke', '--out', %r]\nsys.path.insert(0, %r)\n"
            "import run_experiment_E15_gwaya_rft as m\n"
            "orig = m.Ctx.__init__\n"
            "def init(self, a):\n    orig(self, a)\n    self.fail_min = 10**6\n"
            "m.Ctx.__init__ = init\nsys.exit(m.main(['--smoke', '--out', %r]))\n") % (str(tmp_path / "o"), str(ROOT / "scripts"), str(tmp_path / "o"))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=170, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd="/")
    assert r.returncode == 3, r.stdout[-1500:] + r.stderr[-1500:]
    rec = json.loads((tmp_path / "o" / "E15_gwaya_rft_rft.json").read_text())
    assert rec["status"] == "aborted by registered fail-fast rule"
    assert not (tmp_path / "o" / "E15_gwaya_rft_train.json").exists()
    assert "E15 ABORTED" in r.stdout
