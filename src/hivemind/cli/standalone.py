"""Standalone (frozen) Windows launcher for the HiveMind Web HUD.

This is the entry point bundled by ``packaging/windows/build_standalone.py``. It
keeps every generated file in a per-user data directory, binds the HUD to
loopback with an exclusive socket, reports which optional capabilities are
actually present in this build, and shuts the server down cleanly.

Environment configuration must happen before ``hivemind.config`` is imported,
because that module resolves its cache and artifact roots at import time.
"""

from __future__ import annotations

import argparse
import errno
import ipaddress
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import webbrowser
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

APP_DIR_NAME = "HiveMind"
DEFAULT_PORT = 8090
REQUIRED_MODULES = ("fastapi", "uvicorn", "starlette", "torch", "transformers", "peft", "yaml", "numpy")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PORT_IN_USE = 3
EXIT_MISSING_DEPENDENCY = 4
EXIT_UNSAFE_PATHS = 5
EXIT_STARTUP_FAILED = 6
EXIT_STOP_TIMEOUT = 7
EXIT_ALREADY_RUNNING = 8

log = logging.getLogger("hivemind.standalone")


class LauncherError(RuntimeError):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def default_data_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    explicit = env.get("HIVEMIND_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = env.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local") if os.name == "nt" else str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def configure_environment(
    data_dir: Path, offline: bool, env: MutableMapping[str, str] | None = None
) -> dict[str, str]:
    """Point every HiveMind/Hugging Face cache at ``data_dir``; explicit user overrides win."""
    env = os.environ if env is None else env
    defaults = {
        "HIVEMIND_CACHE_DIR": str(data_dir / "cache"),
        "HIVEMIND_ARTIFACTS_DIR": str(data_dir / "artifacts"),
        "HF_HOME": str(data_dir / "cache" / "huggingface"),
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    return {key: env[key] for key in (*defaults, "HF_HUB_OFFLINE") if key in env}


def assert_writable_roots_outside_install(roots: Mapping[str, Path], install_root: Path) -> None:
    """A frozen build must never write next to its own executable."""
    offenders = {name: str(path) for name, path in roots.items() if _is_within(path, install_root)}
    if offenders:
        raise LauncherError(
            "Refusing to start: these writable locations resolve inside the installation "
            f"directory {install_root}: {offenders}. Set HIVEMIND_DATA_DIR to a user-owned folder.",
            EXIT_UNSAFE_PATHS,
        )


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def bind_exclusive(host: str, port: int) -> socket.socket:
    """Bind a listening socket nobody else can share.

    uvicorn binds with SO_REUSEADDR, which on Windows lets a second process bind
    the same port and silently split traffic. SO_EXCLUSIVEADDRUSE makes a
    conflicting bind fail instead.
    """
    family = socket.AF_INET6 if ":" in host.strip("[]") else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host.strip("[]"), port))
        sock.listen(128)
        sock.set_inheritable(False)
        return sock
    except OSError:
        sock.close()
        raise


def choose_socket(host: str, port: int | None) -> socket.socket:
    """Explicit ports fail loudly when taken; the default port falls back to a free one."""
    if port is not None:
        try:
            return bind_exclusive(host, port)
        except OSError as exc:
            raise LauncherError(
                f"Port {port} on {host} is already in use ({exc.strerror or exc}). "
                "Close the other program or HiveMind window, or pass --port 0 to pick a free port.",
                EXIT_PORT_IN_USE,
            ) from exc
    try:
        return bind_exclusive(host, DEFAULT_PORT)
    except OSError:
        log.warning("Default port %s is busy; asking the OS for a free loopback port.", DEFAULT_PORT)
        return bind_exclusive(host, 0)


def find_missing_modules(
    modules: tuple[str, ...] = REQUIRED_MODULES,
    importer: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    import importlib

    importer = importer or importlib.import_module
    missing: dict[str, str] = {}
    for name in modules:
        try:
            importer(name)
        except Exception as exc:  # broken native DLLs raise OSError, not ImportError
            missing[name] = f"{type(exc).__name__}: {exc}"
    return missing


def probe_capabilities(install_root: Path) -> dict[str, Any]:
    """Describe what this build can really do; never upgrade a simulation to a claim."""
    import torch

    cuda = bool(torch.cuda.is_available())
    steam_dll = next(
        (str(p) for p in (install_root / "steam_api64.dll", install_root / "steam_api.dll") if p.exists()),
        None,
    )
    lean = shutil.which("lean")
    return {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "compute_backend": "cuda" if cuda else "cpu",
        "gpu_features": "available" if cuda else "unsupported in this build (CPU-only PyTorch; CUDA/NF4 kernels, VRAM preemption timing and DMA paths are not exercised)",
        "steamworks": (
            f"native DLL present at {steam_dll} (not validated against Valve services)"
            if steam_dll
            else "steam_api64.dll not bundled; Steam features are local HMAC simulation only"
        ),
        "lean_compiler": lean or "not found on PATH; formal proof checking unavailable",
        "telemetry_source": "demo_fixture (no live peer discovery)",
    }


def _status_file(data_dir: Path) -> Path:
    return data_dir / "run" / "hud-server.json"


def _stop_request_file(data_dir: Path, instance_id: str) -> Path:
    # Never let status contents choose an arbitrary filesystem path.
    if uuid.UUID(instance_id).hex != instance_id:
        raise ValueError("Invalid HUD instance ID")
    return data_dir / "run" / f"stop-request-{instance_id}"


def _read_status(data_dir: Path) -> dict[str, Any]:
    try:
        status = json.loads(_status_file(data_dir).read_text(encoding="utf-8"))
        instance_id = status["instance_id"]
        _stop_request_file(data_dir, instance_id)
        return status
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _write_status(data_dir: Path, status: Mapping[str, Any]) -> None:
    """Readers must see one complete generation, never a partial JSON write."""
    path = _status_file(data_dir)
    fd, name = tempfile.mkstemp(prefix="hud-status-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(status, stream)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class _DataDirectoryLock:
    """Kernel-held ownership; crash recovery never guesses from a reusable PID.

    Keep the lock file in place forever. Unlinking it would let another process
    lock a new inode while an existing contender still holds the old one.
    The OS releases the lock if its owner exits, including an unclean exit.
    """

    def __init__(self, data_dir: Path, filename: str = "hud-server.lock") -> None:
        self.path = data_dir / "run" / filename
        self.stream: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                # Windows byte-range locks may extend beyond EOF. Do not write
                # to the locked byte while another process owns this file.
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise LauncherError(
                f"A HiveMind HUD already owns data directory {self.path.parent.parent}. "
                "Use --stop with that data directory, or use a different --data-dir.",
                EXIT_ALREADY_RUNNING,
            ) from exc
        except BaseException:
            stream.close()
            raise
        self.stream = stream

    def acquire_until(self, deadline: float) -> bool:
        while True:
            try:
                self.acquire()
                return True
            except LauncherError as exc:
                if exc.exit_code != EXIT_ALREADY_RUNNING:
                    raise
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def close(self) -> None:
        if self.stream is not None:
            # Closing the handle releases both Windows and POSIX locks.
            self.stream.close()
            self.stream = None


class _InstanceOwner:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.instance_id = uuid.uuid4().hex
        self.stop_path = _stop_request_file(data_dir, self.instance_id)
        self.lock = _DataDirectoryLock(data_dir)
        self._mutex = threading.Lock()
        self._closed = True

    def __enter__(self) -> "_InstanceOwner":
        # Coordinate initial status handover with --stop. A stopper must not
        # select the previous crashed generation before this ID is published.
        control = _DataDirectoryLock(self.data_dir, "hud-control.lock")
        if not control.acquire_until(time.monotonic() + 30.0):
            raise LauncherError("Timed out waiting for HUD launch/stop coordination.", EXIT_STARTUP_FAILED)
        try:
            self.lock.acquire()
            with self._mutex:
                self._closed = False
                try:
                    self._publish_locked(state="starting")
                except BaseException:
                    self._close_locked()
                    raise
        finally:
            control.close()
        return self

    def _publish_locked(self, **fields: Any) -> None:
        _write_status(self.data_dir, {
            **fields, "instance_id": self.instance_id, "pid": os.getpid(),
        })

    def publish(self, **fields: Any) -> bool:
        with self._mutex:
            if self._closed:
                return False
            # Serialize all status reads/writes, including Windows where an
            # open reader may prevent an atomic replace or unlink.
            control = _DataDirectoryLock(self.data_dir, "hud-control.lock")
            if not control.acquire_until(time.monotonic() + 30.0):
                raise LauncherError("Timed out publishing HUD status.", EXIT_STARTUP_FAILED)
            try:
                self._publish_locked(**fields)
            finally:
                control.close()
            return True

    def _close_locked(self) -> None:
        self._closed = True
        try:
            if _read_status(self.data_dir).get("instance_id") == self.instance_id:
                _status_file(self.data_dir).unlink(missing_ok=True)
            self.stop_path.unlink(missing_ok=True)
        finally:
            self.lock.close()

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            control = _DataDirectoryLock(self.data_dir, "hud-control.lock")
            if not control.acquire_until(time.monotonic() + 30.0):
                # Leave stale metadata rather than racing a reader or retaining
                # lifetime ownership forever; the next launch can recover it.
                self._closed = True
                self.lock.close()
                log.warning("Timed out cleaning HUD status; stale files retained.")
                return
            try:
                self._close_locked()
            finally:
                control.close()

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _instance_is_locked(data_dir: Path) -> bool:
    probe = _DataDirectoryLock(data_dir)
    try:
        probe.acquire()
    except LauncherError as exc:
        if exc.exit_code != EXIT_ALREADY_RUNNING:
            raise
        return True
    finally:
        probe.close()
    return False


def request_stop(data_dir: Path, timeout: float = 30.0) -> int:
    """Target one launch generation, without signalling any PID or newer launch."""
    deadline = time.monotonic() + timeout
    target: str | None = None
    while True:
        control = _DataDirectoryLock(data_dir, "hud-control.lock")
        if not control.acquire_until(deadline):
            break
        try:
            locked = _instance_is_locked(data_dir)
            status = _read_status(data_dir)
            if target is None:
                if not locked:
                    print("No running HiveMind HUD found for this data directory.", flush=True)
                    return EXIT_OK
                target = status.get("instance_id")
                if target is not None:
                    _stop_request_file(data_dir, target).write_text("stop\n", encoding="utf-8")
            elif not locked or (status and status["instance_id"] != target):
                print("HiveMind HUD stopped.", flush=True)
                return EXIT_OK
        finally:
            control.close()
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    print(f"HiveMind HUD did not stop within {timeout:.0f}s.", file=sys.stderr, flush=True)
    return EXIT_STOP_TIMEOUT


def _terminate_owned_children(timeout: float = 5.0) -> list[int]:
    try:
        import psutil
    except Exception:
        return []
    children = psutil.Process().children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    return [c.pid for c in children]


def _setup_logging(data_dir: Path) -> Path:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "hud-launcher.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="HiveMindHUD",
        description="Run the HiveMind Web HUD locally (CPU build; demo-fixture telemetry).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only).")
    parser.add_argument("--port", type=int, default=None, help=f"Port (default {DEFAULT_PORT}, falls back to a free port if busy; 0 = any free port).")
    parser.add_argument("--allow-network", action="store_true", help="Permit binding a non-loopback address. The HUD has no authentication.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the default browser.")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face Hub downloads (offline by default).")
    parser.add_argument("--data-dir", type=Path, default=None, help="Per-user data/cache directory (default %%LOCALAPPDATA%%\\HiveMind).")
    parser.add_argument("--check", action="store_true", help="Verify bundled dependencies and paths, print capabilities as JSON, then exit.")
    parser.add_argument("--stop", action="store_true", help="Stop the HUD running for this data directory, then exit.")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.port is not None and not 0 <= args.port <= 65535:
        raise LauncherError(f"--port must be between 0 and 65535, got {args.port}", EXIT_USAGE)
    if not is_loopback_host(args.host) and not args.allow_network:
        raise LauncherError(
            f"Refusing to bind {args.host}: the HUD is unauthenticated. Use a loopback address, "
            "or pass --allow-network if you understand it will be reachable from your network.",
            EXIT_USAGE,
        )

    data_dir = (args.data_dir or default_data_dir()).expanduser().resolve()
    root = install_dir()
    if is_frozen() and _is_within(data_dir, root):
        raise LauncherError(
            f"Data directory {data_dir} is inside the installation directory {root}.",
            EXIT_UNSAFE_PATHS,
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.stop:
        return request_stop(data_dir)
    if args.check:
        return _run_configured(args, data_dir, root)
    # Acquire before importing the app, binding a port, or touching status/logs.
    # A busy default port may still fall back for *different* data directories.
    with _InstanceOwner(data_dir) as owner:
        return _run_configured(args, data_dir, root, owner)


def _run_configured(
    args: argparse.Namespace, data_dir: Path, root: Path, owner: _InstanceOwner | None = None,
) -> int:
    log_path = _setup_logging(data_dir)
    # Never resolve relative paths (including the Steam DLL probe) against an
    # arbitrary launch directory such as Downloads.
    os.chdir(data_dir)
    env_used = configure_environment(data_dir, offline=not args.online)

    missing = find_missing_modules()
    if missing:
        details = "; ".join(f"{k} ({v})" for k, v in missing.items())
        raise LauncherError(
            f"This HiveMind build is missing required components: {details}. "
            "Re-extract the complete release archive; do not move files out of the _internal folder.",
            EXIT_MISSING_DEPENDENCY,
        )

    from hivemind import config as hm_config

    writable = {
        "PROJECT_ROOT": Path(hm_config.PROJECT_ROOT),
        "CACHE_ROOT": Path(hm_config.CACHE_ROOT),
        "ARTIFACTS_ROOT": Path(hm_config.ARTIFACTS_ROOT),
    }
    if is_frozen():
        assert_writable_roots_outside_install(writable, root)

    capabilities = probe_capabilities(root)
    capabilities.update(
        {
            "frozen": is_frozen(),
            "install_dir": str(root),
            "data_dir": str(data_dir),
            "log_file": str(log_path),
            "writable_roots": {k: str(v) for k, v in writable.items()},
            "environment": env_used,
        }
    )
    if args.check:
        print(json.dumps(capabilities, indent=2))
        return EXIT_OK
    for key in ("compute_backend", "gpu_features", "steamworks", "lean_compiler", "telemetry_source"):
        log.info("%s: %s", key, capabilities[key])

    if not is_loopback_host(args.host):
        log.warning("Binding %s: the unauthenticated HUD is reachable from your network.", args.host)
    sock = choose_socket(args.host, args.port)
    bound_port = sock.getsockname()[1]
    host_for_url = f"[{args.host.strip('[]')}]" if ":" in args.host else args.host
    url = f"http://{host_for_url}:{bound_port}/"

    try:
        import uvicorn

        from hivemind.game.steam_edition import create_hack_miner_fastapi_app

        app = create_hack_miner_fastapi_app()
        server = uvicorn.Server(uvicorn.Config(app, log_level="info", lifespan="on"))
    except Exception as exc:
        sock.close()
        log.exception("HUD failed to initialise")
        raise LauncherError(f"HUD failed to initialise: {exc}. See {log_path}.", EXIT_STARTUP_FAILED) from exc

    assert owner is not None
    return _serve_hud(server, sock, owner, url, bound_port, no_browser=args.no_browser)


def _serve_hud(
    server: Any, sock: socket.socket, owner: _InstanceOwner, url: str, bound_port: int,
    *, no_browser: bool,
) -> int:
    shutdown = threading.Event()

    def _open_browser() -> None:
        try:
            webbrowser.open(url)
        except Exception:
            log.exception("Could not open the default browser; use %s manually", url)

    def _announce() -> None:
        try:
            while not server.started and not server.should_exit and not shutdown.is_set():
                if owner.stop_path.exists():
                    server.should_exit = True
                    return
                shutdown.wait(0.05)
            if not server.started or shutdown.is_set():
                return
            if not owner.publish(state="running", url=url, port=bound_port, started_at=time.time()):
                return
            print(f"HiveMind HUD running at {url}  (Ctrl+C to stop)", flush=True)
            if not no_browser:
                # Some browser handlers wait until their browser exits. Keep
                # that wait independent from the stop/control watcher.
                threading.Thread(target=_open_browser, name="hud-browser", daemon=True).start()
            while not server.should_exit and not shutdown.is_set():
                if owner.stop_path.exists():
                    log.info("Stop requested for instance %s", owner.instance_id)
                    server.should_exit = True
                    break
                shutdown.wait(0.05)
        except Exception:
            log.exception("HUD status/control watcher failed; stopping the server")
            server.should_exit = True

    announce = threading.Thread(target=_announce, name="hud-announce", daemon=True)
    announce.start()
    try:
        server.run(sockets=[sock])
    finally:
        shutdown.set()
        server.should_exit = True
        announce.join(timeout=2.0)
        sock.close()
        reaped = _terminate_owned_children()
        if reaped:
            log.info("Terminated leftover child processes: %s", reaped)
        # The outer owner context removes only this generation's files and then
        # releases the kernel lock, including on startup or runtime failures.
        log.info("HUD stopped.")
    return EXIT_OK if server.started else EXIT_STARTUP_FAILED


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        if is_frozen() and sys.stdin is not None and sys.stdin.isatty():
            try:
                input("Press Enter to close this window...")
            except EOFError:
                pass
        return exc.exit_code
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
