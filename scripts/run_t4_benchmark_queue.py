"""Sequential GPU job queue for a bounded benchmark campaign on one GPU (here: the 10-hour T4 campaign of 2026-10-23).

Reads the job list from ``artifacts/experiments/T4_benchmark_queue.json`` before every job, so jobs can be appended
while the queue runs. A job is skipped when its output file already exists (resume), never started after the
deadline, and killed at the deadline. GPU utilisation and memory are sampled every 30 s through NVML so the report
states how busy the GPU actually was. The log is ``artifacts/experiments/T4_benchmark_queue_log.json``.

    python3 scripts/run_t4_benchmark_queue.py --hours 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
QUEUE_PATH = EXP_DIR / "T4_benchmark_queue.json"
LOG_PATH = EXP_DIR / "T4_benchmark_queue_log.json"


class GPUSampler(threading.Thread):
    def __init__(self, period: float = 30.0) -> None:
        super().__init__(daemon=True)
        self.period, self.samples, self.stop = period, [], threading.Event()
        self.current_job = None

    def run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(h)
        except Exception as exc:  # NVML missing: record why and stop sampling
            self.samples.append({"error": repr(exc)})
            return
        while not self.stop.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.samples.append({"t": round(time.time(), 1), "job": self.current_job, "util_pct": int(u.gpu), "mem_used_gib": round(m.used / 2**30, 2),
                                 "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)})
            self.stop.wait(self.period)


def load_jobs() -> List[Dict[str, Any]]:
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))["jobs"]


def _set_paths(queue: str | None, log: str | None) -> None:
    """Optional second campaign: another job file and log, same runner."""
    global QUEUE_PATH, LOG_PATH
    if queue:
        QUEUE_PATH = PROJECT_ROOT / queue
    if log:
        LOG_PATH = PROJECT_ROOT / log


def write_log(log: Dict[str, Any], sampler: GPUSampler) -> None:
    s = [x for x in sampler.samples if "util_pct" in x]
    log["gpu_samples"] = len(s)
    log["gpu_util_mean_pct"] = round(statistics.fmean(x["util_pct"] for x in s), 1) if s else None
    log["gpu_util_by_job"] = {j: round(statistics.fmean(x["util_pct"] for x in s if x["job"] == j), 1) for j in {x["job"] for x in s if x["job"]}}
    log["gpu_mem_peak_gib"] = max((x["mem_used_gib"] for x in s), default=None)
    log["gpu_trace"] = s[::4]  # every 2 minutes
    LOG_PATH.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--logs", default=str(Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "queue_logs"))
    ap.add_argument("--queue", help="job file relative to the repo root (default artifacts/experiments/T4_benchmark_queue.json)")
    ap.add_argument("--log", help="log file relative to the repo root (default artifacts/experiments/T4_benchmark_queue_log.json)")
    a = ap.parse_args()
    _set_paths(a.queue, a.log)
    logs = Path(a.logs)
    logs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = start + a.hours * 3600.0
    sampler = GPUSampler()
    sampler.start()
    log: Dict[str, Any] = {"started_unix": start, "deadline_unix": deadline, "hours": a.hours, "jobs": []}
    done: set = set()
    while True:
        pending = [j for j in load_jobs() if j["name"] not in done]
        if not pending:
            break
        job = pending[0]
        done.add(job["name"])
        entry: Dict[str, Any] = {"name": job["name"], "argv": job["argv"]}
        out = job.get("output")
        if out and (PROJECT_ROOT / out).exists():
            entry["status"] = "skipped (output exists)"
            log["jobs"].append(entry)
            write_log(log, sampler)
            continue
        if time.time() >= deadline:
            entry["status"] = "not started (deadline reached)"
            log["jobs"].append(entry)
            write_log(log, sampler)
            continue
        entry["start_unix"] = time.time()
        sampler.current_job = job["name"]
        print(f"[queue] start {job['name']} ({(deadline - time.time()) / 3600:.2f} h left)", flush=True)
        log_file = logs / f"{job['name']}.log"
        with open(log_file, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable] + job["argv"], cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)  # own session: killpg reaches child workers
            while proc.poll() is None and time.time() < deadline:
                time.sleep(5)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # the job and every worker it spawned
                except ProcessLookupError:
                    proc.kill()
                proc.wait()
                entry["status"] = "killed at deadline"
            else:
                entry["status"] = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        sampler.current_job = None
        entry["returncode"] = proc.returncode
        entry["seconds"] = round(time.time() - entry["start_unix"], 1)
        entry["log_tail"] = log_file.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()[-15:]
        print(f"[queue] end {job['name']}: {entry['status']} in {entry['seconds']} s", flush=True)
        log["jobs"].append(entry)
        write_log(log, sampler)
    sampler.stop.set()
    log["finished_unix"] = time.time()
    log["wall_hours"] = round((log["finished_unix"] - start) / 3600.0, 3)
    write_log(log, sampler)
    print(f"[queue] finished after {log['wall_hours']} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
