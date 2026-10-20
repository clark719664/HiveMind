"""Out-of-checkout smoke test for the frozen HiveMind HUD.

    py -3.11 packaging\\windows\\smoke_test_standalone.py --zip build\\standalone\\HiveMindHUD-<sha>-win64-cpu.zip --out results.json

Extracts the release archive into a directory whose path contains spaces and
non-ASCII characters, runs the executable with PATH stripped of Python and git
and with outbound HTTP(S) pointed at a dead proxy, and records each check.
This is a smoke test on the build machine, not clean-machine acceptance.
Standard library only, so it can run from any Python.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

APP = "HiveMindHUD"
STARTUP_TIMEOUT = 240.0


class Recorder:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.observations: list[dict] = []

    def check(self, name: str, ok: bool, **evidence) -> bool:
        self.checks.append({"name": name, "passed": bool(ok), **evidence})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(evidence, default=str)[:300]}", flush=True)
        return ok

    def observe(self, name: str, **evidence) -> None:
        """Record behaviour that has no agreed expected result yet; never counted as pass or fail."""
        self.observations.append({"name": name, **evidence})
        print(f"[OBSERVED] {name} {json.dumps(evidence, default=str)[:300]}", flush=True)


class ProxyRecorder:
    """Stand-in HTTP(S) proxy that records each outbound attempt and then drops it."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.attempts: list[str] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(2)
                try:
                    first = conn.recv(1024).split(b'\r\n', 1)[0].decode("latin-1")
                except OSError:
                    first = "<no request line>"
                self.attempts.append(first)

    def close(self) -> None:
        self.sock.close()


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def clean_env(local_appdata: Path, proxy_port: int) -> dict[str, str]:
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    keep = {"SystemRoot", "SystemDrive", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERNAME",
            "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS"}
    # os.environ keys are upper-cased on Windows; dropping SYSTEMROOT breaks Winsock (WinError 10106).
    keep = {k.upper() for k in keep}
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    env["PATH"] = os.pathsep.join([rf"{sysroot}\System32", sysroot, rf"{sysroot}\System32\Wbem"])
    env["LOCALAPPDATA"] = str(local_appdata)
    env["APPDATA"] = str(local_appdata.parent / "Roaming")
    env["HTTP_PROXY"] = env["HTTPS_PROXY"] = env["http_proxy"] = env["https_proxy"] = f"http://127.0.0.1:{proxy_port}"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost,::1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def http(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def ws_first_frame(port: int, path: str = "/ws/swarm-telemetry", timeout: float = 15.0) -> dict:
    key = base64.b64encode(os.urandom(16)).decode()
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += s.recv(4096)
        head, buf = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError(head.decode(errors="replace"))

        def need(n: int) -> bytes:
            nonlocal buf
            while len(buf) < n:
                buf += s.recv(65536)
            out, buf = buf[:n], buf[n:]
            return out

        b0, b1 = need(2)
        length = b1 & 0x7F
        if length == 126:
            length = int.from_bytes(need(2), "big")
        elif length == 127:
            length = int.from_bytes(need(8), "big")
        payload = need(length)
        return {"opcode": b0 & 0x0F, "frame": json.loads(payload.decode("utf-8"))}


def listening_addresses(pid: int) -> list[str]:
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, encoding="utf-8", errors="replace").stdout
    out6 = subprocess.run(["netstat", "-ano", "-p", "TCPv6"], capture_output=True, encoding="utf-8", errors="replace").stdout
    addrs = []
    for line in (out + out6).splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING" and parts[4] == str(pid):
            addrs.append(parts[1])
    return addrs


def processes_under(root: Path) -> list[dict]:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith("
          f"'{str(root)}', [StringComparison]::OrdinalIgnoreCase) }} | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, encoding="utf-8", errors="replace").stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


class Instance:
    def __init__(self, exe: Path, args: list[str], env: dict, logs: Path, tag: str) -> None:
        self.stdout = (logs / f"{tag}.stdout.txt").open("w", encoding="utf-8")
        self.stderr_path = logs / f"{tag}.stderr.txt"
        self.stderr = self.stderr_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen([str(exe), *args], env=env, cwd=str(logs), stdin=subprocess.DEVNULL,
                                     stdout=self.stdout, stderr=self.stderr,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        self.out_path = logs / f"{tag}.stdout.txt"

    def wait_url(self, timeout: float = STARTUP_TIMEOUT) -> str | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.out_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "HiveMind HUD running at " in line:
                    return line.split("running at ", 1)[1].split()[0]
            if self.proc.poll() is not None:
                return None
            time.sleep(0.5)
        return None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        self.stdout.close()
        self.stderr.close()



def wait_exit(instance: Instance, timeout: float = STARTUP_TIMEOUT) -> int | None:
    try:
        return instance.proc.wait(timeout)
    except subprocess.TimeoutExpired:
        return None


def read_owner(data_dir: Path) -> dict:
    try:
        status = json.loads((data_dir / "run" / "hud-server.json").read_text(encoding="utf-8"))
        return status if isinstance(status, dict) else {}
    except (OSError, ValueError):
        return {}


def stop_instance(exe: Path, env: dict, logs: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), "--stop"], env=env, cwd=str(logs), capture_output=True,
                          encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL)


def check_explicit_port_conflict(exe: Path, env: dict, logs: Path, base: Path, port: int, rec: Recorder) -> None:
    # Ownership is checked before binding. Use another directory so this check
    # actually reaches the occupied-port path instead of the owner lock (exit 8).
    conflict_env = dict(env, HIVEMIND_DATA_DIR=str(base / "port conflict data"))
    conflict = Instance(exe, ["--no-browser", "--port", str(port)], conflict_env, logs, "conflict")
    try:
        code = wait_exit(conflict)
        rec.check("second instance on same explicit port with distinct data dir exits 3", code == 3,
                  exit_code=code, data_dir=conflict_env["HIVEMIND_DATA_DIR"],
                  stderr_tail=conflict.stderr_path.read_text(encoding="utf-8", errors="replace")[-300:])
    finally:
        conflict.close()


def check_shared_directory_ownership(exe: Path, env: dict, logs: Path, base: Path, rec: Recorder) -> None:
    shared_dir = base / "shared data"
    shared_env = dict(env, HIVEMIND_DATA_DIR=str(shared_dir))
    other_dir = base / "independent data"
    other_env = dict(env, HIVEMIND_DATA_DIR=str(other_dir))
    instances = []
    try:
        # Port 0 avoids racing between reserving and releasing two ephemeral ports.
        owner = Instance(exe, ["--no-browser", "--port", "0"], shared_env, logs, "shared_owner")
        instances.append(owner)
        owner_url = owner.wait_url()
        before = read_owner(shared_dir)
        rec.check("shared data dir owner starts with generation identity",
                  bool(owner_url) and bool(before.get("instance_id")) and before.get("pid") == owner.proc.pid,
                  url=owner_url, status=before)
        contender = Instance(exe, ["--no-browser", "--port", "0"], shared_env, logs, "shared_contender")
        instances.append(contender)
        code = wait_exit(contender)
        rec.check("second instance sharing data dir is rejected with exit 8", code == 8, exit_code=code,
                  stderr_tail=contender.stderr_path.read_text(encoding="utf-8", errors="replace")[-300:])
        after = read_owner(shared_dir)
        rec.check("rejected contender leaves original owner running and status unchanged",
                  owner.proc.poll() is None and bool(before.get("instance_id")) and after == before,
                  owner_exit=owner.proc.poll(), before=before, after=after)

        other = Instance(exe, ["--no-browser", "--port", "0"], other_env, logs, "independent_owner")
        instances.append(other)
        other_url = other.wait_url()
        other_before = read_owner(other_dir)
        rec.check("independent data dir starts a different owner",
                  bool(other_url) and bool(other_before.get("instance_id"))
                  and other_before.get("instance_id") != before.get("instance_id"),
                  url=other_url, status=other_before)
        stop = stop_instance(exe, shared_env, logs)
        owner_code = wait_exit(owner, 60)
        rec.check("shared data dir stop shuts down the original owner", stop.returncode == 0 and owner_code == 0,
                  stop_exit=stop.returncode, owner_exit=owner_code, stopped_instance_id=before.get("instance_id"),
                  stop_stdout=stop.stdout.strip()[-300:], stop_stderr=stop.stderr.strip()[-300:])
        rec.check("shared owner status removed after stop", not (shared_dir / "run" / "hud-server.json").exists())
        rec.check("shared data dir stop leaves independent owner running",
                  other.proc.poll() is None and bool(other_before.get("instance_id"))
                  and read_owner(other_dir) == other_before,
                  independent_exit=other.proc.poll(), independent_instance_id=other_before.get("instance_id"))
        other_stop = stop_instance(exe, other_env, logs)
        other_code = wait_exit(other, 60)
        rec.check("independent owner stops with its own data dir", other_stop.returncode == 0 and other_code == 0,
                  stop_exit=other_stop.returncode, owner_exit=other_code)
    finally:
        for inst in reversed(instances):
            inst.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=None, help="Parent directory for the extraction (default: system temp).")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-real-training", action="store_true")
    args = ap.parse_args()

    rec = Recorder()
    base = Path(tempfile.mkdtemp(prefix="hm-smoke-", dir=args.workdir))
    install_parent = base / "Hive Mind Smoke Ünïcødé 测试"
    local_appdata = base / "User Ü" / "AppData" / "Local"
    logs = base / "logs"
    for d in (install_parent, local_appdata, logs):
        d.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as zf:
        zf.extractall(install_parent)
    root = install_parent / APP
    exe = root / f"{APP}.exe"
    manifest = json.loads((root / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    proxy = ProxyRecorder()
    env = clean_env(local_appdata, proxy.port)
    results: dict = {
        "zip": str(args.zip), "zip_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        "source_sha": manifest["source_sha"], "exe_sha256": manifest["exe_sha256"],
        "install_dir": str(root), "scope": "smoke test on build machine; NOT clean-machine acceptance",
    }

    rec.check("no python or git on PATH", shutil.which("python", path=env["PATH"]) is None
              and shutil.which("git", path=env["PATH"]) is None and shutil.which("py", path=env["PATH"]) is None,
              path=env["PATH"])
    rec.check("install path has spaces and non-ASCII", " " in str(root) and not str(root).isascii(), path=str(root))
    before = tree_state(root)

    chk = subprocess.run([str(exe), "--check"], env=env, cwd=str(logs), capture_output=True, encoding="utf-8",
                         errors="replace", timeout=STARTUP_TIMEOUT, stdin=subprocess.DEVNULL)
    report = {}
    try:
        report = json.loads(chk.stdout)
    except json.JSONDecodeError:
        pass
    data_dir = local_appdata / "HiveMind"
    rec.check("--check exits 0 with JSON", chk.returncode == 0 and bool(report), exit_code=chk.returncode, stderr_tail=chk.stderr[-400:])
    rec.check("--check reports frozen CPU build", report.get("frozen") is True and report.get("compute_backend") == "cpu",
              compute_backend=report.get("compute_backend"), gpu_features=report.get("gpu_features"))
    rec.check("data dir resolves to per-user LOCALAPPDATA", Path(report.get("data_dir", "")) == data_dir.resolve(), data_dir=report.get("data_dir"))
    roots = report.get("writable_roots", {})
    rec.check("writable roots outside install dir", bool(roots) and all(not Path(p).resolve().is_relative_to(root.resolve()) for p in roots.values()), writable_roots=roots)
    results["capabilities"] = report

    bad_host = subprocess.run([str(exe), "--host", "0.0.0.0", "--no-browser"], env=env, cwd=str(logs), capture_output=True,
                              encoding="utf-8", errors="replace", timeout=STARTUP_TIMEOUT, stdin=subprocess.DEVNULL)
    rec.check("non-loopback bind refused without --allow-network", bad_host.returncode == 2, exit_code=bad_host.returncode, stderr_tail=bad_host.stderr[-300:])

    port = free_port()
    main_inst = Instance(exe, ["--no-browser", "--port", str(port)], env, logs, "main")
    instances = [main_inst]
    try:
        t0 = time.monotonic()
        url = main_inst.wait_url()
        startup_s = round(time.monotonic() - t0, 1)
        if not rec.check("server starts and announces URL", url == f"http://127.0.0.1:{port}/", url=url, startup_seconds=startup_s,
                         stderr_tail=main_inst.stderr_path.read_text(encoding="utf-8", errors="replace")[-600:]):
            raise SystemExit("startup failed")
        pid = main_inst.proc.pid
        addrs = listening_addresses(pid)
        rec.check("only loopback listeners", bool(addrs) and all(a.startswith(("127.0.0.1:", "[::1]:")) for a in addrs), listeners=addrs)

        status, html = http("GET", url)
        rec.check("GET / serves HUD HTML from bundle", status == 200 and 'id="webgl-georing-canvas"' in html, status=status, bytes=len(html))
        status, body = http("GET", url + "api/georing-topology")
        topo = json.loads(body) if status == 200 else {}
        rec.check("topology labelled demo fixture", topo.get("source") == "demo_fixture" and topo.get("live_peer_discovery") is False,
                  status=status, source=topo.get("source"), nodes=len(topo.get("nodes", [])))
        status, body = http("GET", url + "api/rtx-tool-status")
        tool = json.loads(body) if status == 200 else {}
        hw = tool.get("hardware", {})
        rec.check("rtx-tool-status reports no CUDA", status == 200 and hw.get("cuda_available") is False and str(hw.get("torch_version", "")).endswith("+cpu"),
                  status=status, cuda_available=hw.get("cuda_available"), gpu_backend=hw.get("gpu_backend"), torch_version=hw.get("torch_version"))
        try:
            ws = ws_first_frame(port)
            rec.check("WebSocket telemetry frame labelled demo fixture", ws["opcode"] == 1 and ws["frame"].get("source") == "demo_fixture",
                      source=ws["frame"].get("source"), keys=sorted(ws["frame"])[:12])
        except Exception as exc:
            rec.check("WebSocket telemetry frame labelled demo fixture", False, error=repr(exc))

        status, body = http("POST", url + "api/scan-hf-shard", {"gamer_tag": "Smoke", "custom_rows": ["def add(a, b): return a + b", "fn main() { println!(\"hi\"); }"]})
        rec.check("offline: shard scan on supplied rows", status == 200, status=status, body_head=body[:200])
        status, body = http("POST", url + "api/scan-hf-shard", {"gamer_tag": "Smoke", "dataset_id": "sahil2801/CodeAlpaca-20k"}, timeout=120)
        rec.check("offline: dataset scan returns a response without hanging", status in (200, 500, 503), status=status, body_head=body[:300],
                  note="records behaviour only; whether it fell back to bundled/synthetic rows is visible in body")
        rec.check("offline: no outbound HTTP(S) attempt reached the recording proxy", not proxy.attempts, attempts=proxy.attempts[:10],
                  note="proxy env vars point at a local recorder; a direct connection that ignores proxy settings would not be seen")
        status, body = http("POST", url + "api/forge-lora-mission", {"gamer_tag": "Smoke", "real_training": False}, timeout=180)
        rec.check("forge (synthetic micro-forge) responds", status == 200, status=status, body_head=body[:200])
        if not args.skip_real_training:
            t1 = time.monotonic()
            status, body = http("POST", url + "api/forge-lora-mission", {"gamer_tag": "Smoke", "real_training": True}, timeout=900)
            try:
                detail = json.loads(body)
            except json.JSONDecodeError:
                detail = None
            outcome = "trained" if status == 200 else "refused_by_validation" if status == 422 else "server_error"
            rec.check("forge real_training returns a structured result (200 trained or 422 validation refusal), no server error",
                      status in (200, 422) and isinstance(detail, dict), status=status, outcome=outcome,
                      seconds=round(time.monotonic() - t1, 1), body_head=body[:600],
                      note="byte-level micro model trained on bundled fixture rows; not evidence of model quality or accepted work")
            results["real_training_response"] = detail if isinstance(detail, dict) else body[:2000]

        check_explicit_port_conflict(exe, env, logs, base, port, rec)

        held = None
        try:
            held = socket.socket()
            held.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            held.bind(("127.0.0.1", 8090))
            held.listen(1)
        except OSError:
            held = None
        fallback_env = dict(env, HIVEMIND_DATA_DIR=str(base / "fallback data"))
        fb = Instance(exe, ["--no-browser"], fallback_env, logs, "fallback")
        instances.append(fb)
        fb_url = fb.wait_url()
        rec.check("default port busy -> falls back to a free loopback port", bool(fb_url) and fb_url != "http://127.0.0.1:8090/" and fb_url.startswith("http://127.0.0.1:"),
                  url=fb_url, port_8090_held_by_test=held is not None)
        stop_fb = subprocess.run([str(exe), "--stop"], env=fallback_env, cwd=str(logs), capture_output=True, encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL)
        try:
            fb_code = fb.proc.wait(60)
        except subprocess.TimeoutExpired:
            fb_code = None
        rec.check("fallback instance stops via --stop", stop_fb.returncode == 0 and fb_code == 0, stop_exit=stop_fb.returncode, server_exit=fb_code)
        if held:
            held.close()

        t2 = time.monotonic()
        stop = subprocess.run([str(exe), "--stop"], env=env, cwd=str(logs), capture_output=True, encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL)
        try:
            main_code = main_inst.proc.wait(60)
        except subprocess.TimeoutExpired:
            main_code = None
        rec.check("graceful shutdown via --stop", stop.returncode == 0 and main_code == 0, stop_exit=stop.returncode, server_exit=main_code,
                  seconds=round(time.monotonic() - t2, 1), stop_stdout=stop.stdout.strip())
        rec.check("status file removed after shutdown", not (data_dir / "run" / "hud-server.json").exists())
        time.sleep(1.0)
        leftovers = processes_under(root)
        rec.check("no processes left running from install dir", leftovers == [], leftovers=leftovers)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                still_open = True
        except OSError:
            still_open = False
        rec.check("port released after shutdown", not still_open, port=port)
    finally:
        for inst in instances:
            inst.close()

    check_shared_directory_ownership(exe, env, logs, base, rec)
    results["proxy_attempts_total"] = list(proxy.attempts)
    proxy.close()

    after = tree_state(root)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    rec.check("install directory unchanged by running the app", not changed, changed=changed[:20])
    written = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*") if p.is_file())[:40] if data_dir.exists() else []
    rec.check("app wrote its files under per-user data dir", bool(written), sample=written[:15])

    results["checks"] = rec.checks
    results["observations"] = rec.observations
    results["passed"] = sum(c["passed"] for c in rec.checks)
    results["failed"] = sum(not c["passed"] for c in rec.checks)
    results["logs_dir"] = str(logs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{results['passed']} passed, {results['failed']} failed -> {args.out}")
    if not args.keep:
        shutil.rmtree(install_parent, ignore_errors=True)
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
