#!/usr/bin/env python3
"""Idle-time governor for the shared T4: QLoRA training only when nobody is asking for inference.

Inference always has priority. The governor polls nginx's loopback ``stub_status`` once a second (live in-flight
requests, including ones that are still generating, which an access log would only show afterwards). While requests
are in flight it keeps the BOINC client's GPU mode at ``never`` (the client stops the GPU task; the QLoRA app resumes
from its segment checkpoint later). After ``--idle-start-s`` seconds without requests, and only if the card has room
and is cool, it sets the GPU mode to ``always`` so the training workunits run again.

It measures the yield latency (request seen -> BOINC GPU process gone) and writes state + events as JSON lines.
Standard library only. Run with --dry-run to log decisions without calling boinccmd.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def stub_status(url: str) -> dict[str, int]:
    txt = urllib.request.urlopen(url, timeout=2).read().decode()
    out = {"active": 0, "writing": 0, "reading": 0, "waiting": 0, "requests": 0}
    for ln in txt.splitlines():
        if ln.startswith("Active connections:"):
            out["active"] = int(ln.split(":")[1])
        elif ln.startswith("Reading:"):
            p = ln.replace(":", " ").split()
            out.update(reading=int(p[1]), writing=int(p[3]), waiting=int(p[5]))
        elif ln.strip() and ln.split()[0].isdigit() and len(ln.split()) == 3:
            out["requests"] = int(ln.split()[2])
    return out


def gpu_state() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(",")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    procs = [tuple(x.strip() for x in ln.split(",")) for ln in apps.splitlines() if ln.strip()]
    return {"used_mib": int(q[0]), "total_mib": int(q[1]), "temp_c": int(q[2]), "util_pct": int(q[3]), "procs": procs}


def boinc_gpu_pids(procs: list[tuple[str, ...]]) -> list[str]:
    """GPU processes that belong to the BOINC QLoRA app (python from the project venv running moe_app)."""
    out = []
    for pid, *_ in procs:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "real_client.moe_app" in cmd:
            out.append(pid)
    return out


def in_night(now: float, tz: str, start_h: int, end_h: int) -> bool:
    """True inside the owner's night window (wraps past midnight when start_h > end_h)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    h = datetime.fromtimestamp(now, ZoneInfo(tz)).hour
    return (start_h <= h < end_h) if start_h < end_h else (h >= start_h or h < end_h)


class Boinc:
    def __init__(self, boinccmd: str, lib: str, port: int, datadir: Path, dry: bool) -> None:
        self.cmd, self.lib, self.port, self.dir, self.dry = boinccmd, lib, port, datadir, dry
        self.mode = "unknown"

    def set_gpu(self, mode: str) -> bool:
        if self.dry:
            self.mode = mode
            return True
        pw = (self.dir / "gui_rpc_auth.cfg").read_text().strip()
        r = subprocess.run([self.cmd, "--host", f"127.0.0.1:{self.port}", "--passwd", pw, "--set_gpu_mode", mode, "0"],
                           capture_output=True, text=True, timeout=15, env=dict(os.environ, LD_LIBRARY_PATH=self.lib))
        ok = r.returncode == 0
        if ok:
            self.mode = mode
        return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-url", default="http://127.0.0.1:8091/nginx_status")
    ap.add_argument("--boinccmd", default="/mnt/disks/disk-socrateai-local-1/boinc/root/usr/bin/boinccmd")
    ap.add_argument("--lib", default="/mnt/disks/disk-socrateai-local-1/boinc/root/usr/lib/x86_64-linux-gnu")
    ap.add_argument("--boinc-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/node/client"))
    ap.add_argument("--rpc-port", type=int, default=31420)
    ap.add_argument("--telemetry", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor.jsonl"))
    ap.add_argument("--state-file", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor_state.json"))
    ap.add_argument("--idle-start-s", type=float, default=1800.0, help="quiet seconds before training may run INSIDE the night window")
    ap.add_argument("--day-idle-s", type=float, default=7200.0, help="quiet seconds required OUTSIDE the night window (training is meant to be mostly nocturnal)")
    ap.add_argument("--night-tz", default="Europe/Paris", help="timezone of the owner's night (an assumption: change it if wrong)")
    ap.add_argument("--night-start-h", type=int, default=22)
    ap.add_argument("--night-end-h", type=int, default=7)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--min-free-mib", type=int, default=4300, help="free VRAM needed (beyond what BOINC itself holds) to start training")
    ap.add_argument("--max-temp-c", type=int, default=82)
    ap.add_argument("--hold-file", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD"),
                    help="operator switch: while this file exists the governor keeps training OFF however quiet the node is "
                         "(`touch` to pause training, `rm` to allow it again)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.telemetry.parent.mkdir(parents=True, exist_ok=True)
    boinc = Boinc(a.boinccmd, a.lib, a.rpc_port, a.boinc_dir, a.dry_run)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    def log(ev: dict) -> None:
        ev["t"] = round(time.time(), 2)
        with open(a.telemetry, "a") as f:
            f.write(json.dumps(ev) + "\n")

    # a fresh start counts as "just busy": training waits out a full quiet period first
    state, last_busy, yields, last_yield_ms, last_snapshot = "init", time.time(), 0, None, 0.0
    # every (re)start begins with training OFF; it can only be turned on by a full quiet period
    boinc.set_gpu("never")
    state = "yielded"
    log({"event": "start", "dry_run": a.dry_run, "idle_start_s": a.idle_start_s, "day_idle_s": a.day_idle_s,
         "night": f"{a.night_start_h:02d}:00-{a.night_end_h:02d}:00 {a.night_tz}"})
    while not stop["now"]:
        t0 = time.time()
        try:
            st = stub_status(a.status_url)
            in_flight = max(st["writing"] - 1, 0)  # minus this poll itself
            http_ok = True
        except Exception as exc:
            in_flight, http_ok = 0, False
            st = {"error": str(exc)[:80]}
        busy = in_flight >= 1
        if busy:
            last_busy = t0
        if a.hold_file.exists():
            if state != "held":
                boinc.set_gpu("never")
                state = "held"
                log({"event": "hold", "reason": f"{a.hold_file} exists: training is paused by the operator"})
            last_busy = t0  # leaving the hold starts a fresh quiet period
        elif state == "held":
            state = "yielded"
            log({"event": "hold_released"})
        elif busy and state != "yielded":
            had_task = bool(boinc_gpu_pids(gpu_state()["procs"])) if not a.dry_run else False
            ok = boinc.set_gpu("never") if boinc.mode != "never" or state != "yielded" else True
            t_req = time.time()
            freed_ms = None
            if ok and not a.dry_run:
                deadline = t_req + 30
                while time.time() < deadline:
                    if not boinc_gpu_pids(gpu_state()["procs"]):
                        freed_ms = int((time.time() - t_req) * 1000)
                        break
                    time.sleep(0.25)
            elif ok:
                freed_ms = 0
            yields += 1
            last_yield_ms = freed_ms
            state = "yielded"
            # gpu_freed_ms is a real measurement only when a BOINC GPU task was running (had_boinc_gpu_task); otherwise there was nothing to free
            log({"event": "yield", "in_flight": in_flight, "set_never_ok": ok, "had_boinc_gpu_task": had_task, "gpu_freed_ms": freed_ms, "yields": yields})
        elif not busy and state in ("init", "yielded") and (t0 - last_busy) >= (a.idle_start_s if in_night(t0, a.night_tz, a.night_start_h, a.night_end_h) else a.day_idle_s):
            g = gpu_state()
            mine = sum(int(p[2]) for p in g["procs"] if p[0] in boinc_gpu_pids(g["procs"]))
            free = g["total_mib"] - g["used_mib"] + mine
            if free >= a.min_free_mib and g["temp_c"] < a.max_temp_c and http_ok:
                ok = boinc.set_gpu("always")
                if ok:
                    state = "training"
                    log({"event": "resume", "idle_s": round(t0 - last_busy, 1), "free_mib": free, "temp_c": g["temp_c"],
                         "night": in_night(t0, a.night_tz, a.night_start_h, a.night_end_h)})
            elif state == "init":
                boinc.set_gpu("never")
                state = "yielded"
        if t0 - last_snapshot >= 10:
            g = gpu_state()
            snap = {"state": state, "boinc_gpu_mode": boinc.mode, "in_flight": in_flight, "http_ok": http_ok, "yields": yields,
                    "last_yield_ms": last_yield_ms, "quiet_s": round(t0 - last_busy, 1) if last_busy else None,
                    "gpu_used_mib": g["used_mib"], "gpu_temp_c": g["temp_c"], "gpu_util_pct": g["util_pct"], "boinc_procs": boinc_gpu_pids(g["procs"]), "t": round(t0, 1)}
            a.state_file.write_text(json.dumps(snap))
            log({"event": "snapshot", **{k: snap[k] for k in ("state", "in_flight", "gpu_used_mib", "gpu_util_pct", "yields", "last_yield_ms")}})
            last_snapshot = t0
        time.sleep(max(a.poll_s - (time.time() - t0), 0.05))
    boinc.set_gpu("never")
    log({"event": "stop"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
