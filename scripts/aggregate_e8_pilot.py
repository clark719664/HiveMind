"""Aggregate the two-host WAN pilot runs into ``artifacts/experiments/E8_pilot_report.json``.

Campaigns (each: seeds 1-3, syncer on this box, learner 0 on this box, learner 1 on the GCP TPU VM
``gwenlaya-tpu-1`` in ``us-west4-a`` through an SSH reverse tunnel, plus a single-host baseline with
identical seeds, shards and hyper-parameters):

  e8_pilot_runs             H = 8 inner steps,   K = 1, adaptive grace, merge rda and direct
  e8_pilot_runs_h400_k1     H = 400 inner steps, K = 1, adaptive grace, merge rda
  e8_pilot_runs_h400_k2     H = 400 inner steps, K = 2 (wait for both), merge rda

For every campaign: paired (WAN - baseline) final-loss difference with a 95 % t-interval (df = 2),
pooled per-round latency percentiles, how often the remote learner was admitted, and its HTTP
round-trip time. Nothing is interpreted here; the paper does that.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP = PROJECT_ROOT / "artifacts" / "experiments"
OUT = EXP / "E8_pilot_report.json"
T_CRIT_DF2 = 4.302652729911275


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def mean_ci(vals: List[float]) -> Dict[str, float]:
    m = statistics.fmean(vals)
    hw = T_CRIT_DF2 * statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) >= 2 else 0.0
    return {"mean": m, "ci95_halfwidth_df2": hw, "n": len(vals)}


def summarise(campaign_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"dir": campaign_dir.name, "merges": {}}
    for merge in ("rda", "direct"):
        wan_files = sorted(campaign_dir.glob(f"wan_seed*_{merge}.json"))
        if not wan_files:
            continue
        diffs, wan_losses, base_losses, latencies, remote_admit, rounds_total, remote_rtt_ms = [], [], [], [], 0, 0, []
        remote_label = None
        for wf in wan_files:
            seed = wf.name.split("_seed")[1].split("_")[0]
            bf = campaign_dir / f"baseline_seed{seed}_{merge}.json"
            wan = json.loads(wf.read_text())
            base = json.loads(bf.read_text()) if bf.exists() else None
            wan_losses.append(wan["final_loss"])
            if base is not None:
                base_losses.append(base["final_loss"])
                diffs.append(wan["final_loss"] - base["final_loss"])
            for r in wan["rounds"]:
                latencies.append(r["latency_s"])
                rounds_total += 1
                if "1" in r["admitted"] or "1" in r.get("stale_applied", []):
                    remote_admit += 1
            remote_label = wan.get("hosts", {}).get("1", remote_label)
            log = campaign_dir / f"learner1_remote_seed{seed}_{merge}.log"
            if log.exists():
                for line in log.read_text().splitlines():
                    if line.startswith('{"learner_id": 1'):
                        remote_rtt_ms.append(json.loads(line)["http_rtt_ms"]["p50"])
        cfg = json.loads(wan_files[0].read_text())["config"]
        out["merges"][merge] = {
            "config": cfg, "seeds": len(wan_files), "remote_host": remote_label,
            "final_loss_wan": mean_ci(wan_losses), "final_loss_baseline": mean_ci(base_losses) if base_losses else None,
            "paired_diff_wan_minus_baseline": mean_ci(diffs) if diffs else None,
            "round_latency_s": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95), "p99": percentile(latencies, 0.99), "n_rounds": len(latencies)},
            "remote_learner_admitted_fraction": remote_admit / rounds_total if rounds_total else None,
            "remote_http_rtt_ms_p50_per_run": remote_rtt_ms,
        }
    return out


def main() -> int:
    campaigns = [d for d in sorted(EXP.glob("e8_pilot_runs*")) if d.is_dir()]
    report: Dict[str, Any] = {
        "schema_version": "hivemind-experiment-E8-pilot-aggregate/v1.0",
        "experiment_id": "E8-pilot",
        "hosts": {"syncer_and_learner_0": "this workstation (NVIDIA T4 box)", "learner_1": "GCP TPU VM gwenlaya-tpu-1, us-west4-a, CPU-only PyTorch 2.14.1, reached via SSH reverse tunnel"},
        "icmp_rtt_ms_to_remote": {"min": 150.770, "avg": 151.346, "max": 151.838, "n": 5, "note": "ping -c 5 measured 2026-10-23 before the runs"},
        "campaigns": {c.name: summarise(c) for c in campaigns},
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope_note": "Two real hosts, toy synthetic LM, 12 outer rounds, 3 seeds per cell; the SSH tunnel adds its own latency on top of the ICMP round trip.",
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode()
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, c in report["campaigns"].items():
        for merge, m in c["merges"].items():
            d = m["paired_diff_wan_minus_baseline"]
            diff_txt = f"{d['mean']:+.5f} ± {d['ci95_halfwidth_df2']:.5f} (n={d['n']})" if d else "baseline pending"
            print(f"{name:26s} {merge:6s} H={m['config']['inner_steps']} K={m['config']['k']} grace={m['config']['grace']} | WAN-base diff {diff_txt} | latency p50 {m['round_latency_s']['p50']:.2f}s p95 {m['round_latency_s']['p95']:.2f}s | remote admitted {m['remote_learner_admitted_fraction']:.2f} | remote rtt p50 {[round(x) for x in m['remote_http_rtt_ms_p50_per_run']]}")
    print("receipt", report["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
