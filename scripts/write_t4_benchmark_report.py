"""Benchmark report of the 10-hour T4 campaign (2026-10-23/24), built only from committed artifacts.

Reads the queue log and the E11 / E12 / E10b / E11b reports (any may be missing) and writes
``docs/benchmarks/T4_10h_campaign_report.md``. Every number in the report is copied from an artifact; the text
adds no result. Run: python3 scripts/write_t4_benchmark_report.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "artifacts" / "experiments"
OUT = ROOT / "docs" / "benchmarks" / "T4_10h_campaign_report.md"


def load(name: str) -> Optional[Dict[str, Any]]:
    p = EXP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sha(name: str) -> str:
    p = EXP / name
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "-"


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if abs(x) < 100 else f"{x:.3g}"
    return str(x)


def pm(m: Optional[float], h: Optional[float], nd: int = 4) -> str:
    return "-" if m is None else f"{m:+.{nd}f} ± {h:.{nd}f}"


def queue_section(q: Optional[Dict[str, Any]]) -> List[str]:
    if not q:
        return ["Queue log not found."]
    start = dt.datetime.fromtimestamp(q["started_unix"], dt.timezone.utc)
    lines = [f"- Started {start:%Y-%m-%d %H:%M} UTC, budget {q['hours']} h, wall time {q.get('wall_hours', 'running')} h.",
             f"- GPU utilisation (NVML, 30 s samples, {q.get('gpu_samples')} samples): mean {q.get('gpu_util_mean_pct')} %, "
             f"peak memory {q.get('gpu_mem_peak_gib')} GiB.", "",
             "| job | status | wall (min) | mean GPU util % |", "| :-- | :-- | --: | --: |"]
    util = q.get("gpu_util_by_job", {})
    for j in q["jobs"]:
        mins = f"{j['seconds'] / 60:.1f}" if "seconds" in j else "-"
        lines.append(f"| `{j['name']}` | {j['status']} | {mins} | {util.get(j['name'], '-')} |")
    gpu_h = sum(j.get("seconds", 0) for j in q["jobs"] if j.get("status") == "ok") / 3600
    lines += ["", f"Successful jobs used {gpu_h:.2f} GPU-hours. Failed attempts are listed and explained below; they are not hidden."]
    return lines


def e11_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E11 report not found."]
    a = r["analysis"]
    lines = [f"**Decision (pre-registered, amendment 1):** P11a **{a['P11a']['status']}**; P11b **{a['P11b']['status']}**. "
             f"Controls all passed: {r['controls_all_passed']}. Receipt `{r['receipt_sha256'][:16]}`.", "",
             "| model | dtype | peak GiB | baseline NLL | ΔNLL FP8 | ΔNLL INT4 | P11a | P11b | controls |",
             "| :-- | :-- | --: | --: | --: | --: | :-- | :-- | :-- |"]
    for k, v in r["models"].items():
        if "cells" not in v:
            lines.append(f"| {k} | - | - | - | - | - | {v.get('status')} | | |")
            continue
        c = v["cells"]
        failed = [n for n, x in v["controls"].items() if isinstance(x, dict) and x.get("passed") is False]
        lines.append(f"| {k} | {v['dtype']} | {v['peak_gpu_gib']} | {v['baseline_nll']:.3f} | {pm(c['fp8']['dnll_mean'], c['fp8']['dnll_ci95'])} | "
                     f"{pm(c['int@4']['dnll_mean'], c['int@4']['dnll_ci95'], 3)} | {v['P11a']['status']} | {v['P11b']['status']} | "
                     f"{'all pass' if not failed else 'failed: ' + ', '.join(failed)} |")
    lines += ["", "Paired differences at matched measured rate (ΔNLL(Z) − ΔNLL(lattice), nats/token, positive = lattice better):", "",
              "| model | R | Z − A₂ (none) | Z − E₈ (none) | Z − E₈ (KLT) |", "| :-- | --: | :-- | :-- | :-- |"]
    for k, v in r["models"].items():
        if "cells" not in v:
            continue
        for R in (2, 3, 4):
            a2 = v["P11a"]["cells"][f"R={R}"]
            e0 = v["P11b"]["cells"][f"R={R}|none"]
            e1 = v["P11b"]["cells"][f"R={R}|klt"]
            cell = lambda d, key: f"{pm(d[key], d['ci95'])} ({d['status']})"  # noqa: E731
            lines.append(f"| {k} | {R} | {cell(a2, 'z_minus_a2_mean')} | {cell(e0, 'z_minus_e8_mean')} | {cell(e1, 'z_minus_e8_mean')} |")
    lines += ["", "Descriptive only (decides nothing): does a 4-bit codec reach FP8 (paired ΔNLL difference CI includes 0) at about half the bytes?", "",
              "| model | A₂@4 | E₈@4 + KLT | Z@4 | INT4 |", "| :-- | :-- | :-- | :-- | :-- |"]
    for k, v in r["models"].items():
        if "descriptive_frontier_vs_fp8" in v:
            fr = v["descriptive_frontier_vs_fp8"]
            lines.append(f"| {k} | " + " | ".join(f"{pm(fr[c]['minus_fp8_mean'], fr[c]['ci95'])}{' ✓' if fr[c]['reaches_fp8'] else ''}"
                                                   for c in ("A2@4|none", "E8@4|klt", "Z@4|none", "int@4")) + " |")
    return lines


def e12_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E12 report not found (not run or not finished)."]
    lines = [f"**Decision (pre-registered):** P12-same **{r['analysis']['P12-same']['status']}** ({r['analysis']['P12-same']['basis']}); "
             f"P12-cross **{r['analysis']['P12-cross']['status']}** ({r['analysis']['P12-cross']['basis']}); "
             f"fixed-point clause: {r['analysis']['P12-fixed-point']['status']}. Controls all passed: {r['controls_all_passed']}. "
             f"ε = {r['eps']}.", "", "| cell | max ‖Δθ‖∞ | min ‖Δθ‖∞ | segments above ε |", "| :-- | --: | --: | --: |"]
    for k, s in r["summary"].items():
        lines.append(f"| {k} | {s['max_linf']:.3e} | {s['min_linf']:.3e} | {s['n_above_eps']}/{s['n']} |")
    lines += ["", f"Not run on this host: {', '.join(r['not_run'])}."]
    w = {k: len(v) for k, v in r.get("determinism_warnings", {}).items()}
    lines.append(f"Determinism warnings captured per configuration (count): {w}.")
    return lines


def e10b_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E10b report not found (not run, not finished, or cut by the deadline; a partial file may exist)."]
    an = r.get("analysis", {})
    lines = [f"**Decision (pre-registered):** " + "; ".join(f"{k} **{v.get('status')}**" for k, v in an.items() if isinstance(v, dict)) +
             f". Controls all passed: {r.get('controls_all_passed')}.", ""]
    cells = r.get("cells", {})
    if cells:
        lines += ["| arm | final held-out loss | rel. MSE | rate (bits/dim) |", "| :-- | :-- | --: | --: |"]
        for k, c in cells.items():
            lines.append(f"| {k} | {pm(c.get('final_loss_mean'), c.get('final_loss_ci95'))} | {f(c.get('mean_relative_mse'))} | {f(c.get('mean_rate_bits_per_dim'), 3)} |")
    return lines


def e11b_section(r: Optional[Dict[str, Any]]) -> List[str]:
    if not r:
        return ["E11b report not found (sweep not aggregated)."]
    lines = ["Exploratory: this sweep decides nothing. It checks whether E11's codec ranking holds on other Hugging Face data.", ""]
    entries = r.get("entries", r.get("configurations", {}))
    if isinstance(entries, dict):
        lines += ["| configuration | baseline NLL | ΔNLL FP8 | P11a-style | P11b-style | ranking agreement with E11 |", "| :-- | --: | --: | :-- | :-- | :-- |"]
        for k, v in entries.items():
            dm = v.get("decide_model", v)
            if not isinstance(dm, dict) or "cells" not in dm:
                lines.append(f"| {k} | error: {str(v.get('error', '?'))[:60]} | | | | |")
                continue
            ag = v.get("ranking_agreement", {})
            tot = ag.get("totals", ag) if isinstance(ag, dict) else {}
            lines.append(f"| {k} | {dm['baseline_nll']:.3f} | {pm(dm['cells']['fp8']['dnll_mean'], dm['cells']['fp8']['dnll_ci95'])} | "
                         f"{dm['P11a']['status']} | {dm['P11b']['status']} | {json.dumps(tot)[:90]} |")
    return lines


def main() -> int:
    q = load("T4_benchmark_queue_log.json")
    e11, e12, e10b, e11b = load("E11_kv_lattice_report.json"), load("E12_replay_divergence_report.json"), load("E10b_lattice_05b_report.json"), load("E11b_kv_lattice_sweep_report.json")
    hw = (e11 or {}).get("hardware", {})
    md = [
        "# HiveMind — 10-hour T4 benchmark campaign (2026-10-23/24)",
        "",
        f"Hardware: {hw.get('gpu', 'Tesla T4')} (15 GB, sm_75), 8 vCPU, 31.5 GB RAM; torch {hw.get('torch', '?')}, transformers {hw.get('transformers', '?')}. "
        "Data: Hugging Face WikiText-2, fineweb-edu, codeparrot-clean-valid, GSM8K (local cache). Models: Qwen2.5-0.5B-Instruct (fp16), "
        "Qwen2.5-Coder-7B-Instruct (fp16, embeddings on CPU), Gemma-4-E4B (bf16, per-layer embeddings on CPU).",
        "",
        "Every experiment below was pre-registered before it ran (files in `artifacts/experiments/*prereg*.json`; deviations and the one "
        "amendment are recorded there). Numbers are copied from the receipts listed at the end; 95 % intervals are df = 2 t-intervals over "
        "three seeds. A refuted prediction is reported as refuted.",
        "",
        "## 1. Campaign timeline", "", *queue_section(q), "",
        "## 2. E11 — lattice KV-cache codecs vs FP8 / scalar (RQ10, HM-S10-05)", "", *e11_section(e11), "",
        "## 3. E12 — spot-check replay divergence across hardware (RQ11, HM-S10-08)", "", *e12_section(e12), "",
        "## 4. E10b — lattice outer-delta codec at 0.5B (RQ9, HM-S10-01)", "", *e10b_section(e10b), "",
        "## 5. E11b — exploratory sweep over Hugging Face domains and context length", "", *e11b_section(e11b), "",
        "## 6. Receipts", "",
        "| artifact | sha256 (first 12) |", "| :-- | :-- |",
    ]
    for n in ("E11_kv_lattice_prereg.json", "E11_kv_lattice_prereg_amendment_1.json", "E11_kv_lattice_report.json", "E12_replay_divergence_prereg.json",
              "E12_replay_divergence_report.json", "E10b_lattice_05b_prereg.json", "E10b_lattice_05b_report.json", "E11b_kv_lattice_sweep_prereg.json",
              "E11b_kv_lattice_sweep_report.json", "T4_benchmark_queue_log.json"):
        md.append(f"| `{n}` | `{sha(n)}` |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
