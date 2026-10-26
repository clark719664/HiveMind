#!/usr/bin/env python3
"""End-to-end check of the STANDARD BOINC distribution channel: a fresh stock BOINC client (no app_info.xml, its own data dir and host record) attaches to
the project, downloads the signed self-contained app version through the project's download URL, unpacks it, runs a real training workunit on the GPU,
uploads the result, and the server validates it by replay.

  <venv>/bin/python scripts/boinc_distribution_e2e.py --version 102 [--abort-held]

The volunteer account opts in to beta work (``allow_beta_work``), so it receives the beta app version while the production node stays on the stable one.
``--abort-held`` aborts the (idle) task the production node client is holding so BOINC reissues it to this volunteer; it counts as one error for that workunit.
Writes artifacts/boinc_distribution/report.json. Runs a GPU task: start it only when the node is idle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.client_setup import Client, prepare_data_dir, project_dir_name  # noqa: E402
from hivemind.boinc.real_client.server_stack import Stack  # noqa: E402


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-base", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack"))
    ap.add_argument("--boinc-src", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/src"))
    ap.add_argument("--boinc-root", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/root"))
    ap.add_argument("--work", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/dist-test"))
    ap.add_argument("--node-client-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/node/client"))
    ap.add_argument("--node-rpc-port", type=int, default=31420)
    ap.add_argument("--rpc-port", type=int, default=31431)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--abort-held", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=3600)
    a = ap.parse_args()
    st = Stack(a.stack_base, a.boinc_src, sys.executable, ROOT / "src")
    if not st.wait_http(15):
        print("project web server is not up")
        return 2
    report: dict = {"version": a.version, "events": []}

    def note(msg: str) -> None:
        report["events"].append({"t": round(time.time()), "msg": msg})
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    auth = st.create_account(f"volunteer-{int(time.time())}@test.invalid", "dist-test")
    st.sql(f"use {st.dbname}; update user set project_prefs='<project_preferences><allow_beta_work>1</allow_beta_work></project_preferences>' where authenticator='{auth}';")
    uid = int(st.sql(f"use {st.dbname}; select id from user where authenticator='{auth}';").strip().splitlines()[-1])
    note(f"volunteer account created (user {uid}), beta work allowed")

    if a.abort_held:
        node = Client(a.boinc_root / "usr/bin/boinc", a.boinc_root / "usr/lib/x86_64-linux-gnu", a.node_client_dir, a.node_rpc_port)
        tasks = node.cmd("--get_tasks")
        for name in re.findall(r"^\s+name: (hmq_\S+)", tasks, re.M):
            note(f"aborting the node's held task {name}: {node.cmd('--task', st.master_url, name, 'abort').strip() or 'ok'}")

    a.work.mkdir(parents=True, exist_ok=True)
    prepare_data_dir(a.work / "client", st.master_url, ROOT / "src", install_apps=False, authenticator=auth)
    client = Client(a.boinc_root / "usr/bin/boinc", a.boinc_root / "usr/lib/x86_64-linux-gnu", a.work / "client", a.rpc_port)
    client.start()
    note(f"stock client {client.cmd('--get_state')[:0]}started on port {a.rpc_port}" if client.wait_rpc(60) else "client RPC did not come up")
    pdir = a.work / "client" / "projects" / project_dir_name(st.master_url)
    t0, done = time.time(), False
    seen: set[str] = set()
    try:
        while time.time() - t0 < a.timeout_s and not done:
            time.sleep(20)
            files = {p.name for p in pdir.glob("hivemind_*")} if pdir.exists() else set()
            for n in sorted(files - seen):
                note(f"project dir now has {n}")
            seen |= files
            hid = st.sql(f"use {st.dbname}; select id from host where userid={uid} order by id desc limit 1;").strip().splitlines()
            if hid and hid[-1].isdigit():
                rows = st.sql(f"use {st.dbname}; select name,server_state,outcome,validate_state,exit_status,round(elapsed_time) from result where hostid={hid[-1]} order by id desc limit 3;").strip()
                report["host_id"], report["results"] = int(hid[-1]), rows
                if re.search(r"\t5\t\d\t[12]\t", rows):  # over, validated (1) or invalid (2)
                    done = True
        report["client_state"] = client.cmd("--get_tasks")[-1500:]
    finally:
        client.quit()
    dl = st.root / "download"
    report["files"] = {}
    for p in sorted(pdir.glob("hivemind_*")) if pdir.exists() else []:
        if p.is_file() and not p.is_symlink():
            report["files"][p.name] = {"bytes": p.stat().st_size, "md5_matches_server": (dl / p.name).exists() and md5(p) == md5(dl / p.name)}
    report["unpacked"] = sorted(x.name for x in (pdir / "hivemind_env").glob("*")) if (pdir / "hivemind_env").exists() else []
    report["validated"] = bool(re.search(r"\t5\t\d\t1\t", report.get("results", "")))
    out = ROOT / "artifacts" / "boinc_distribution"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    note(f"validated={report['validated']} files={len(report['files'])} unpacked={report['unpacked']}")
    return 0 if report["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
