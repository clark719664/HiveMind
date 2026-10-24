"""Tests for scripts/aggregate_E11b_sweep.py on raw files built in tmp_path from the first-execution fixture (CPU only)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_E11b_sweep as agg  # noqa: E402

FIXTURE = ROOT / "artifacts" / "experiments" / "E11_first_execution_qwen0.5b.json"


def _perturbed(raw: Dict[str, Any]) -> Dict[str, Any]:
    """E11b-like copy: nll_sum of every non-baseline cell scaled up a little; kltc cells cloned from klt."""
    out = copy.deepcopy(raw)
    extra = []
    for r in out["runs"]:
        if r["cell"] != "none":
            r["nll_sum"] *= 1.0005
        if r["cell"].endswith("|klt"):
            c = copy.deepcopy(r)
            c["cell"] = r["cell"][: -len("klt")] + "kltc"
            c["nll_sum"] = r["nll_sum"] * 0.999
            extra.append(c)
    out["runs"] += extra
    out["dataset"], out["registered_configuration"] = "code", False
    return out


@pytest.fixture()
def run(tmp_path: Path):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (tmp_path / "E11_raw_qwen0.5b.json").write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "E11b_raw_qwen0.5b_code_w1024.json").write_text(json.dumps(_perturbed(raw)), encoding="utf-8")
    (tmp_path / "E11b_raw_qwen7b_gsm8k_w1024.json").write_text(json.dumps(_perturbed(raw) | {"model": "qwen7b"}), encoding="utf-8")
    out, figs = tmp_path / "out" / "report.json", tmp_path / "figs"
    rc = agg.main(["--in-dir", str(tmp_path), "--out", str(out), "--fig-dir", str(figs)])
    return rc, json.loads(out.read_text(encoding="utf-8")), figs


def test_report_structure(run) -> None:
    rc, rep, _ = run
    assert rc == 0
    assert rep["status"] == "exploratory, decides nothing"
    assert set(rep["entries"]) == {"qwen0.5b|code|w1024", "qwen7b|gsm8k|w1024"}
    e = rep["entries"]["qwen0.5b|code|w1024"]
    assert e["transforms_present"] == ["none", "klt", "kltc"]
    assert "P11a" in e["decision"] and "P11b" in e["decision"]
    pd = e["paired_differences"]
    for key in ("Z-A2@R=3|none", "Z-E8@R=3|klt", "Z-A2@R=2|kltc", "Z-E8@R=4|kltc", "kltc-klt:A2@R=3", "kltc-klt:E8@R=2"):
        assert key in pd
    assert pd["kltc-klt:Z@R=3"]["mean"] < 0  # kltc cloned 0.1% lower than klt


def test_agreement_table(run) -> None:
    _, rep, _ = run
    ag = rep["entries"]["qwen0.5b|code|w1024"]["agreement_vs_E11"]
    assert ag["n_compared"] > 0 and "Z-A2@R=3|none" in ag["rows"]
    assert "kltc-klt:Z@R=3" not in ag["rows"]  # E11 has no kltc
    row = ag["rows"]["Z-E8@R=3|klt"]
    assert isinstance(row["same_sign"], bool) and isinstance(row["both_exclude_zero"], bool)
    # model without a registered E11 run
    assert "no registered" in rep["entries"]["qwen7b|gsm8k|w1024"]["agreement_vs_E11"]["status"]


def test_receipt(run) -> None:
    _, rep, _ = run
    body = {k: v for k, v in rep.items() if k != "receipt_sha256"}
    assert rep["receipt_sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def test_figure_written_in_tmp(run) -> None:
    _, _, figs = run
    assert (figs / "fig9_e11b_sweep.png").stat().st_size > 0
    assert (figs / "fig9_e11b_sweep.pdf").stat().st_size > 0


def test_empty_dir(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert agg.main(["--in-dir", str(tmp_path), "--out", str(out), "--fig-dir", str(tmp_path / "f")]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["entries"] == {}
    assert not (tmp_path / "f").exists()
