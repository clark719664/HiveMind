"""Experiment S-1 (HiveMind issue #266) on this CPU class: build the pinned solver offline, run every problem 10 times.

Runs after pytest inside the offline container. The Rust dependencies were vendored at image build time from the
Cargo.toml/Cargo.lock copies in this harness, so the project's solver is built ONLY when its manifest and lockfile are
byte-identical to those copies; otherwise the stage is skipped with a reason (a stale vendor directory cannot build it).

Part (a) of S-1: 10 repeated runs of the same binary on the same host must give bitwise-identical stdout per problem.
Part (b): the report carries this host's CPU model, rustc, glibc and every result's sha256 and final state, so another
host's run can be compared against it. Nothing here edits the project. Exit status: 0 skipped or all identical,
3 build/test failure, 4 a problem whose 10 outputs differ.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

WORK = Path("/work")
SOLVER_DIR = WORK / "solvers" / "hivemind-solver"
PROBLEM_DIRS = (WORK / "solvers" / "problems", WORK / "solvers" / "problems" / "heldout")
VENDORED = Path("/opt/solver-manifest")
VENDOR_CONFIG = Path("/opt/cargo-vendor.toml")
TARGET = WORK / "target-s1"
RUNS = 10
BEGIN, END = "===S1-REPORT-BEGIN===", "===S1-REPORT-END==="


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit(report: dict) -> None:
    print(BEGIN, flush=True)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)
    print(END, flush=True)


def host_facts() -> dict:
    facts = {"machine": platform.machine(), "kernel": platform.release()}
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name") and "cpu_model" not in facts:
                facts["cpu_model"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and "cpu_flag_count" not in facts:
                facts["cpu_flag_count"] = len(line.split(":", 1)[1].split())
    except OSError as exc:
        facts["cpuinfo_error"] = str(exc)
    for key, cmd in (("rustc", ["rustc", "-vV"]), ("cargo", ["cargo", "--version"]), ("glibc", ["ldd", "--version"])):
        try:
            facts[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0:4]
        except (OSError, subprocess.TimeoutExpired, IndexError) as exc:
            facts[key] = f"unavailable: {exc}"
    return facts


def cargo(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CARGO_HOME="/tmp/hivemind/cargo", CARGO_TARGET_DIR=str(TARGET), CARGO_NET_OFFLINE="true")
    cmd = ["cargo", "--config", str(VENDOR_CONFIG), *args, "--locked", "--offline"]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=SOLVER_DIR, env=env, text=True, capture_output=True, timeout=1500)


def main() -> int:
    report: dict = {"experiment": "S-1 (a) same binary, same host, 10 runs", "host": host_facts(), "runs_per_problem": RUNS}
    if not SOLVER_DIR.is_dir():
        report["skipped"] = "no solvers/hivemind-solver in this revision"
        emit(report)
        return 0
    for name in ("Cargo.toml", "Cargo.lock"):
        ours, theirs = (VENDORED / name).read_bytes(), (SOLVER_DIR / name).read_bytes()
        if ours != theirs:
            report["skipped"] = f"{name} differs from the harness copy the dependencies were vendored from (harness {sha256(ours)[:12]}, revision {sha256(theirs)[:12]}); update sandbox/solver/ and rebuild"
            emit(report)
            return 0
    for step in (("build", "--release"), ("test", "--release")):
        done = cargo(*step)
        tail = (done.stdout + done.stderr)[-4000:]
        report[f"cargo_{step[0]}"] = {"returncode": done.returncode, "tail": tail}
        if done.returncode != 0:
            report["failed"] = f"cargo {step[0]} exited {done.returncode}"
            emit(report)
            return 3
    binary = TARGET / "release" / "hivemind-solver"
    version = subprocess.run([str(binary), "version"], capture_output=True, text=True, timeout=60)
    report["solver_version"] = version.stdout.strip()

    frozen: dict = {}
    frozen_path = PROBLEM_DIRS[1] / "FROZEN.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8")).get("files", {})
    problems: list[dict] = []
    differing = 0
    for directory in PROBLEM_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name == "FROZEN.json":
                continue
            raw = path.read_bytes()
            document = json.loads(raw)
            spec = document.get("spec", document)
            stdin = json.dumps(spec, separators=(",", ":"), sort_keys=True).encode("utf-8")
            outputs: list[bytes] = []
            exit_codes: set[int] = set()
            for _ in range(RUNS):
                run = subprocess.run([str(binary), "run"], input=stdin, capture_output=True, timeout=300)
                outputs.append(run.stdout)
                exit_codes.add(run.returncode)
            hashes = sorted({sha256(out) for out in outputs})
            identical = len(hashes) == 1 and len(exit_codes) == 1
            differing += not identical
            entry: dict = {
                "problem": f"{directory.relative_to(WORK)}/{path.name}",
                "problem_sha256": sha256(raw),
                "input_sha256": sha256(stdin),
                "identical_10_runs": identical,
                "output_sha256": hashes if not identical else hashes[0],
                "exit_codes": sorted(exit_codes),
            }
            if path.name in frozen:
                entry["matches_frozen_manifest"] = frozen[path.name] == entry["problem_sha256"]
            try:
                result = json.loads(outputs[0])
                entry["status"] = result.get("status")
                entry["stats"] = result.get("stats")
                entry["error"] = result.get("error")
                if isinstance(result.get("y"), list) and result["y"]:
                    entry["final_t"] = result.get("t", [None])[-1]
                    entry["final_y"] = result["y"][-1]
            except (ValueError, IndexError, TypeError) as exc:
                entry["parse_error"] = str(exc)
            problems.append(entry)
    report["problems"] = problems
    report["problems_total"] = len(problems)
    report["problems_with_differing_runs"] = differing
    report["part_a_result"] = "PASS: bitwise-identical across 10 runs for every problem" if differing == 0 else "FAIL"
    emit(report)
    return 4 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
