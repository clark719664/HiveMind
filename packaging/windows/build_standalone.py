"""Reproducible CPU-only Windows build of the standalone HiveMind HUD.

    py -3.11 packaging\\windows\\build_standalone.py [--reuse-venv] [--allow-dirty]

Creates ``build/standalone/venv`` from the invoking Python 3.11 interpreter,
installs the project wheel with every dependency pinned by
``constraints-windows-cpu.txt``, freezes it with PyInstaller, gathers
third-party license files, and writes a manifest with the source SHA,
dependency versions, build command and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = REPO / "build" / "standalone"
CONSTRAINTS = HERE / "constraints-windows-cpu.txt"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
APP_NAME = "HiveMindHUD"
BUILD_TOOLING = {"pip", "setuptools", "wheel", "pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes"}
LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "authors")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, encoding="utf-8", errors="replace", **kw)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, encoding="utf-8").stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_DIST_SCRIPT = r"""
import json, importlib.metadata as md
out = []
for d in md.distributions():
    name = d.metadata["Name"]
    files = [str(f) for f in (d.files or []) if f.name.lower().startswith(%r) and ".dist-info" in str(f).replace("\\", "/")]
    files += [str(f) for f in (d.files or []) if "/licenses/" in str(f).replace("\\", "/").lower() and ".dist-info" in str(f)]
    out.append({
        "name": name, "version": d.version,
        "license": d.metadata.get("License-Expression") or d.metadata.get("License") or "",
        "classifiers": [c for c in d.metadata.get_all("Classifier") or [] if c.startswith("License")],
        "root": str(d.locate_file("")), "license_files": sorted(set(files)),
    })
print(json.dumps(out))
""" % (LICENSE_PREFIXES,)


def collect_licenses(venv_python: Path, dest: Path) -> list[dict]:
    raw = json.loads(sh([venv_python, "-c", _DIST_SCRIPT], capture_output=True).stdout)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for dist in sorted(raw, key=lambda d: d["name"].lower()):
        if dist["name"].lower() in BUILD_TOOLING:
            continue
        target = dest / f"{dist['name']}-{dist['version']}"
        copied = []
        for rel in dist["license_files"]:
            src = Path(dist["root"]) / rel
            if src.is_file():
                target.mkdir(parents=True, exist_ok=True)
                out = target / src.name
                n = 1
                while out.exists():
                    out = target / f"{src.stem}.{n}{src.suffix}"
                    n += 1
                shutil.copy2(src, out)
                copied.append(out.name)
        records.append({k: dist[k] for k in ("name", "version", "license", "classifiers")} | {"license_files_copied": copied})
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        (dest / f"CPython-{platform.python_version()}").mkdir(exist_ok=True)
        shutil.copy2(python_license, dest / f"CPython-{platform.python_version()}" / "LICENSE.txt")
        records.append({"name": "CPython", "version": platform.python_version(), "license": "PSF-2.0", "classifiers": [], "license_files_copied": ["LICENSE.txt"]})
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-venv", action="store_true", help="Reuse build/standalone/venv if present.")
    ap.add_argument("--allow-dirty", action="store_true", help="Build from a working tree with uncommitted changes (recorded in the manifest).")
    args = ap.parse_args()

    if sys.platform != "win32":
        sys.exit("This build targets Windows only.")
    if sys.version_info[:2] != (3, 11):
        sys.exit(f"Use Python 3.11 (validated baseline); got {platform.python_version()}.")

    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not args.allow_dirty:
        sys.exit("Working tree has uncommitted changes; commit first or pass --allow-dirty.")

    started = time.time()
    venv = BUILD / "venv"
    py = venv / "Scripts" / "python.exe"
    if not (args.reuse_venv and py.exists()):
        shutil.rmtree(venv, ignore_errors=True)
        sh([sys.executable, "-m", "venv", venv])
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "PIP_REQUIRE_VIRTUALENV"}}
    pip = [py, "-m", "pip", "--disable-pip-version-check"]
    sh([*pip, "install", "--upgrade", "pip==25.2"], env=env)
    sh([*pip, "install", "--extra-index-url", TORCH_CPU_INDEX, "-c", CONSTRAINTS,
        REPO, "pyinstaller", "pyinstaller-hooks-contrib"], env=env)
    # A reused venv already holds hivemind 1.1.0, so pip would skip the rebuilt source without this.
    sh([*pip, "install", "--no-deps", "--force-reinstall", REPO], env=env)
    sh([*pip, "check"], env=env)
    freeze = sh([*pip, "freeze", "--all"], env=env, capture_output=True).stdout.splitlines()

    dist_root = BUILD / "dist"
    shutil.rmtree(dist_root, ignore_errors=True)
    sh([py, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", dist_root,
        "--workpath", BUILD / "work", HERE / f"{APP_NAME}.spec"], env=env, cwd=HERE)
    app_dir = dist_root / APP_NAME
    exe = app_dir / f"{APP_NAME}.exe"
    if not exe.is_file():
        sys.exit(f"PyInstaller did not produce {exe}")

    licenses = collect_licenses(py, app_dir / "licenses")
    shutil.copy2(HERE / "STANDALONE_README.txt", app_dir / "README.txt")
    project_license = next((p.name for p in REPO.iterdir() if p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))), None)

    sums = []
    for f in sorted(p for p in app_dir.rglob("*") if p.is_file()):
        sums.append(f"{sha256(f)}  {f.relative_to(app_dir).as_posix()}")
    (app_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    manifest = {
        "artifact": APP_NAME,
        "kind": "PyInstaller one-folder, console, CPU-only",
        "source_repo": git("remote", "get-url", "fork") if "fork" in git("remote").split() else git("remote", "get-url", "origin"),
        "source_sha": sha,
        "source_dirty": dirty,
        "build_command": f"py -3.11 packaging\\windows\\build_standalone.py{' --reuse-venv' if args.reuse_venv else ''}{' --allow-dirty' if args.allow_dirty else ''}",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - started, 1),
        "build_host": {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine()},
        "dependencies": freeze,
        "excluded_from_bundle": ["bitsandbytes (GPU-only)", "triton (GPU-only)", "tkinter", "matplotlib", "IPython", "pytest", "tensorboard"],
        "supported": "Windows 10/11 x64, CPU inference/training paths, local HUD on loopback",
        "unsupported": "CUDA/ROCm acceleration, NF4 fused kernels, Steamworks (no steam_api64.dll bundled), Lean proof checking (no compiler bundled)",
        "project_license_file": project_license or "NOT FOUND in source tree: redistribution terms for HiveMind itself are unresolved",
        "third_party_licenses": licenses,
        "exe_sha256": sha256(exe),
        "file_count": len(sums),
        "total_bytes": sum(p.stat().st_size for p in app_dir.rglob("*") if p.is_file()),
    }
    (app_dir / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = BUILD / f"{APP_NAME}-{sha[:12]}-win64-cpu.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(p for p in app_dir.rglob("*") if p.is_file()):
            zf.write(f, f"{APP_NAME}/{f.relative_to(app_dir).as_posix()}")
    manifest["zip"] = {"file": zip_path.name, "sha256": sha256(zip_path), "bytes": zip_path.stat().st_size}
    (BUILD / f"{zip_path.stem}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_sha", "source_dirty", "exe_sha256", "file_count", "total_bytes", "zip")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
