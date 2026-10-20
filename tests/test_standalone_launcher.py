"""Regression tests for the standalone Windows HUD launcher (hivemind.cli.standalone)."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest

from hivemind.cli import standalone


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", True), ("127.5.0.9", True), ("::1", True), ("[::1]", True), ("localhost", True),
     ("0.0.0.0", False), ("::", False), ("192.168.1.10", False), ("example.com", False)],
)
def test_loopback_classification(host: str, expected: bool) -> None:
    assert standalone.is_loopback_host(host) is expected


def test_default_data_dir_prefers_explicit_then_localappdata(tmp_path: Path) -> None:
    assert standalone.default_data_dir({"HIVEMIND_DATA_DIR": str(tmp_path / "x")}) == tmp_path / "x"
    assert standalone.default_data_dir({"LOCALAPPDATA": str(tmp_path)}) == tmp_path / "HiveMind"


def test_configure_environment_keeps_user_overrides_and_forces_offline(tmp_path: Path) -> None:
    env = {"HF_HOME": "C:/custom/hf"}
    used = standalone.configure_environment(tmp_path, offline=True, env=env)
    assert env["HF_HOME"] == "C:/custom/hf"
    assert env["HIVEMIND_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["HIVEMIND_ARTIFACTS_DIR"] == str(tmp_path / "artifacts")
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert used["HF_HUB_OFFLINE"] == "1"
    online = {}
    standalone.configure_environment(tmp_path, offline=False, env=online)
    assert "HF_HUB_OFFLINE" not in online


def test_writable_roots_inside_install_dir_are_rejected(tmp_path: Path) -> None:
    install = tmp_path / "Program Files" / "HiveMind"
    install.mkdir(parents=True)
    standalone.assert_writable_roots_outside_install({"cache": tmp_path / "data"}, install)
    with pytest.raises(standalone.LauncherError) as err:
        standalone.assert_writable_roots_outside_install({"cache": install / "cache"}, install)
    assert err.value.exit_code == standalone.EXIT_UNSAFE_PATHS


def test_explicit_port_conflict_fails_with_clear_exit_code() -> None:
    held = standalone.bind_exclusive("127.0.0.1", 0)
    try:
        port = held.getsockname()[1]
        with pytest.raises(standalone.LauncherError) as err:
            standalone.choose_socket("127.0.0.1", port)
        assert err.value.exit_code == standalone.EXIT_PORT_IN_USE
        assert str(port) in str(err.value)
    finally:
        held.close()


def test_busy_default_port_falls_back_to_a_free_port(monkeypatch: pytest.MonkeyPatch) -> None:
    held = standalone.bind_exclusive("127.0.0.1", 0)
    try:
        monkeypatch.setattr(standalone, "DEFAULT_PORT", held.getsockname()[1])
        sock = standalone.choose_socket("127.0.0.1", None)
        try:
            assert sock.getsockname()[1] not in (0, held.getsockname()[1])
            assert sock.getsockname()[0] == "127.0.0.1"
        finally:
            sock.close()
    finally:
        held.close()


@pytest.mark.skipif(os.name != "nt", reason="SO_REUSEADDR port sharing is Windows-specific")
def test_exclusive_bind_blocks_reuseaddr_port_sharing_on_windows() -> None:
    held = standalone.bind_exclusive("127.0.0.1", 0)
    try:
        intruder = socket.socket()
        intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            intruder.bind(("127.0.0.1", held.getsockname()[1]))
        intruder.close()
    finally:
        held.close()


def test_missing_modules_are_reported_with_reason() -> None:
    def importer(name: str):
        if name == "peft":
            raise ImportError("No module named 'peft'")
        if name == "torch":
            raise OSError("[WinError 126] torch_cpu.dll not found")

    missing = standalone.find_missing_modules(("peft", "torch", "yaml"), importer=importer)
    assert set(missing) == {"peft", "torch"}
    assert "torch_cpu.dll" in missing["torch"]


def test_non_loopback_bind_requires_explicit_opt_in() -> None:
    with pytest.raises(standalone.LauncherError) as err:
        standalone.run(["--host", "0.0.0.0", "--no-browser"])
    assert err.value.exit_code == standalone.EXIT_USAGE
    assert standalone.main(["--port", "70000"]) == standalone.EXIT_USAGE


def test_stop_without_running_instance_is_a_noop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert standalone.run(["--stop", "--data-dir", str(tmp_path)]) == standalone.EXIT_OK
    assert "No running" in capsys.readouterr().out


def test_check_reports_cpu_capabilities_and_user_data_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in ("HIVEMIND_CACHE_DIR", "HIVEMIND_ARTIFACTS_DIR", "HF_HOME", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(standalone, "_setup_logging", lambda d: d / "logs" / "hud-launcher.log")
    data = tmp_path / "Data Ü 测试"
    assert standalone.run(["--check", "--data-dir", str(data)]) == standalone.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["frozen"] is False
    assert report["data_dir"] == str(data.resolve())
    assert report["environment"]["HF_HUB_OFFLINE"] == "1"
    assert report["telemetry_source"].startswith("demo_fixture")
    if report["compute_backend"] == "cpu":
        assert "unsupported" in report["gpu_features"]
    assert "simulation" in report["steamworks"] or "not validated" in report["steamworks"]
    assert Path.cwd() == data.resolve()


def test_same_data_directory_is_rejected_before_startup(tmp_path, monkeypatch):
    def forbidden_startup(*args, **kwargs):
        pytest.fail("second launch must fail before startup or port fallback")

    monkeypatch.setattr(standalone, "_run_configured", forbidden_startup)
    with standalone._InstanceOwner(tmp_path) as first:
        status = standalone._status_file(tmp_path).read_bytes()
        first.stop_path.write_text("stop")
        assert standalone.main(["--data-dir", str(tmp_path), "--port", "0"]) == standalone.EXIT_ALREADY_RUNNING
        assert standalone._status_file(tmp_path).read_bytes() == status
        assert first.stop_path.read_text() == "stop"


def test_different_data_directories_have_independent_ownership(tmp_path):
    with standalone._InstanceOwner(tmp_path / "one") as first:
        with standalone._InstanceOwner(tmp_path / "two") as second:
            assert first.instance_id != second.instance_id
            assert standalone._instance_is_locked(first.data_dir)
            assert standalone._instance_is_locked(second.data_dir)
    assert not standalone._instance_is_locked(first.data_dir)
    assert not standalone._instance_is_locked(second.data_dir)


def test_failed_startup_releases_ownership_and_can_restart(tmp_path, monkeypatch):
    def fail_startup(*args, **kwargs):
        raise standalone.LauncherError("broken dependency", standalone.EXIT_MISSING_DEPENDENCY)

    monkeypatch.setattr(standalone, "_run_configured", fail_startup)
    assert standalone.main(["--data-dir", str(tmp_path)]) == standalone.EXIT_MISSING_DEPENDENCY
    assert not standalone._status_file(tmp_path).exists()
    with standalone._InstanceOwner(tmp_path):
        assert standalone._instance_is_locked(tmp_path)


def test_stale_status_and_reused_pid_never_signal_a_process(tmp_path, monkeypatch):
    import uuid

    standalone._status_file(tmp_path).parent.mkdir()
    old_id = uuid.uuid4().hex
    standalone._write_status(tmp_path, {"instance_id": old_id, "pid": os.getpid(), "state": "running"})
    old_stop = standalone._stop_request_file(tmp_path, old_id)
    old_stop.write_text("stale request")
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("must never signal a PID from status"))
    assert standalone.request_stop(tmp_path, timeout=0) == standalone.EXIT_OK
    with standalone._InstanceOwner(tmp_path) as current:
        assert current.instance_id != old_id
        assert not current.stop_path.exists()
        assert standalone._read_status(tmp_path)["instance_id"] == current.instance_id
        assert old_stop.read_text() == "stale request"
    # The stable lock file must never be unlinked, even during cleanup.
    assert (tmp_path / "run" / "hud-server.lock").exists()


def test_stop_targets_starting_owner_and_waits_for_release(tmp_path, monkeypatch):
    with standalone._InstanceOwner(tmp_path) as owner:
        assert standalone._read_status(tmp_path)["state"] == "starting"
        def release_on_request(_):
            assert owner.stop_path.exists()
            owner.close()
        monkeypatch.setattr(standalone.time, "sleep", release_on_request)
        assert standalone.request_stop(tmp_path, timeout=1) == standalone.EXIT_OK
    assert not standalone._status_file(tmp_path).exists()
    assert not owner.stop_path.exists()


def test_stop_timeout_keeps_request_for_only_that_generation(tmp_path):
    with standalone._InstanceOwner(tmp_path) as owner:
        assert standalone.request_stop(tmp_path, timeout=0) == standalone.EXIT_STOP_TIMEOUT
        assert owner.stop_path.exists()
        assert standalone._instance_is_locked(tmp_path)
    with standalone._InstanceOwner(tmp_path) as restarted:
        assert not restarted.stop_path.exists()


def test_stop_racing_with_restart_does_not_stop_replacement(tmp_path, monkeypatch):
    first = standalone._InstanceOwner(tmp_path).__enter__()
    replacement = None
    try:
        def restart_after_request(_):
            nonlocal replacement
            assert first.stop_path.exists()
            first.close()
            replacement = standalone._InstanceOwner(tmp_path).__enter__()
        monkeypatch.setattr(standalone.time, "sleep", restart_after_request)
        assert standalone.request_stop(tmp_path, timeout=1) == standalone.EXIT_OK
        assert replacement is not None
        assert not replacement.stop_path.exists()
        assert standalone._instance_is_locked(tmp_path)
        assert standalone._read_status(tmp_path)["instance_id"] == replacement.instance_id
        # A late watcher/cleanup from the previous owner cannot affect this run.
        assert first.publish(state="running", url="http://stale/") is False
        first.close()
        assert standalone._read_status(tmp_path)["instance_id"] == replacement.instance_id
    finally:
        first.close()
        if replacement is not None:
            replacement.close()


def test_cleanup_does_not_remove_unowned_status(tmp_path):
    import uuid

    with standalone._InstanceOwner(tmp_path) as owner:
        replacement_id = uuid.uuid4().hex
        standalone._write_status(tmp_path, {"instance_id": replacement_id})
        owner.close()
        assert standalone._read_status(tmp_path)["instance_id"] == replacement_id


@pytest.mark.parametrize("bad_id", ["../escape", "x" * 32, "", None, 42])
def test_malformed_status_never_selects_a_stop_path(tmp_path, bad_id):
    with standalone._InstanceOwner(tmp_path):
        standalone._write_status(tmp_path, {"instance_id": bad_id, "pid": os.getpid()})
        assert standalone.request_stop(tmp_path, timeout=0) == standalone.EXIT_STOP_TIMEOUT
        assert not list((tmp_path / "run").glob("stop-request-*"))


def _ownership_process(data_dir, ready, release, result, start=None):
    """Spawn-compatible child: exercise native locking rather than mocked PIDs."""
    if start is not None:
        start.wait(10)
    try:
        with standalone._InstanceOwner(Path(data_dir)) as owner:
            result.put(("owner", owner.instance_id))
            ready.set()
            release.wait(20)
    except standalone.LauncherError as exc:
        result.put(("rejected", exc.exit_code))


def test_process_crash_releases_native_lock_and_stale_status_is_recoverable(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    ready, release, result = ctx.Event(), ctx.Event(), ctx.Queue()
    child = ctx.Process(target=_ownership_process, args=(str(tmp_path), ready, release, result))
    child.start()
    try:
        assert ready.wait(15)
        kind, original_id = result.get(timeout=5)
        assert kind == "owner"
        with pytest.raises(standalone.LauncherError):
            standalone._InstanceOwner(tmp_path).__enter__()
        child.terminate()  # Only the specific test child we just created.
        child.join(10)
        assert not child.is_alive()
        assert standalone._read_status(tmp_path)["instance_id"] == original_id
        with standalone._InstanceOwner(tmp_path) as restarted:
            assert restarted.instance_id != original_id
            assert standalone._read_status(tmp_path)["instance_id"] == restarted.instance_id
    finally:
        # Never reuse multiprocessing synchronization objects after abruptly
        # killing a waiter: the child can die while holding their internal lock.
        if child.is_alive():
            child.terminate()
        child.join(10)
        result.close()
        result.join_thread()


def test_simultaneous_process_launches_have_exactly_one_owner(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    ready, release, start, result = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=_ownership_process, args=(str(tmp_path), ready, release, result, start))
                for _ in range(4)]
    for child in children:
        child.start()
    try:
        start.set()
        outcomes = [result.get(timeout=15) for _ in children]
        assert [kind for kind, _ in outcomes].count("owner") == 1
        assert [kind for kind, _ in outcomes].count("rejected") == 3
        assert all(value == standalone.EXIT_ALREADY_RUNNING for kind, value in outcomes if kind == "rejected")
        winner_id = next(value for kind, value in outcomes if kind == "owner")
        assert standalone._read_status(tmp_path)["instance_id"] == winner_id
    finally:
        release.set()
        for child in children:
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join(10)
        result.close()
        result.join_thread()
    assert all(child.exitcode == 0 for child in children)
    with standalone._InstanceOwner(tmp_path):
        pass


def test_missing_status_while_lock_is_held_does_not_report_stopped(tmp_path, monkeypatch):
    with standalone._InstanceOwner(tmp_path) as owner:
        original_read = standalone._read_status
        reads = 0
        def disappear_after_targeting(data_dir):
            nonlocal reads
            reads += 1
            return original_read(data_dir) if reads == 1 else {}
        monkeypatch.setattr(standalone, "_read_status", disappear_after_targeting)
        assert standalone.request_stop(tmp_path, timeout=0.05) == standalone.EXIT_STOP_TIMEOUT
        assert owner.stop_path.exists()
        assert standalone._instance_is_locked(tmp_path)


def test_stop_waits_for_initial_status_handover_after_crash(tmp_path, monkeypatch):
    import concurrent.futures
    import threading
    import uuid

    standalone._status_file(tmp_path).parent.mkdir()
    stale_id = uuid.uuid4().hex
    standalone._write_status(tmp_path, {"instance_id": stale_id, "pid": os.getpid()})
    owner = standalone._InstanceOwner(tmp_path)
    acquired, publish_allowed, stop_seen = threading.Event(), threading.Event(), threading.Event()
    real_acquire = owner.lock.acquire

    def paused_acquire():
        real_acquire()
        acquired.set()
        assert publish_allowed.wait(5)
    monkeypatch.setattr(owner.lock, "acquire", paused_acquire)

    def start_and_stop():
        with owner:
            for _ in range(500):
                if owner.stop_path.exists():
                    stop_seen.set()
                    return
                threading.Event().wait(0.01)
            pytest.fail("new owner never received the stop request")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        launch = pool.submit(start_and_stop)
        try:
            assert acquired.wait(5)
            stop = pool.submit(standalone.request_stop, tmp_path, 5)
            # While publication is paused, the short control lock must protect
            # stale metadata from selection, including immediate-timeout stops.
            assert standalone.request_stop(tmp_path, timeout=0) == standalone.EXIT_STOP_TIMEOUT
            assert not standalone._stop_request_file(tmp_path, stale_id).exists()
        finally:
            publish_allowed.set()
        assert stop.result(timeout=10) == standalone.EXIT_OK
        launch.result(timeout=10)
    assert stop_seen.is_set()
    assert not standalone._instance_is_locked(tmp_path)


def test_all_owner_status_reads_writes_and_cleanup_hold_control_lock(tmp_path, monkeypatch):
    real_read, real_write = standalone._read_status, standalone._write_status
    observed = []

    def assert_control_held():
        probe = standalone._DataDirectoryLock(tmp_path, "hud-control.lock")
        try:
            with pytest.raises(standalone.LauncherError) as error:
                probe.acquire()
            assert error.value.exit_code == standalone.EXIT_ALREADY_RUNNING
        finally:
            probe.close()

    def checked_read(data_dir):
        assert_control_held()
        observed.append("read")
        return real_read(data_dir)

    def checked_write(data_dir, value):
        assert_control_held()
        observed.append("write")
        return real_write(data_dir, value)

    monkeypatch.setattr(standalone, "_read_status", checked_read)
    monkeypatch.setattr(standalone, "_write_status", checked_write)
    with standalone._InstanceOwner(tmp_path) as owner:
        owner.publish(state="running")
        assert standalone.request_stop(tmp_path, timeout=0) == standalone.EXIT_STOP_TIMEOUT
    assert observed.count("write") == 2
    assert observed.count("read") == 2


def test_blocked_browser_does_not_block_stop_or_ownership_release(tmp_path, monkeypatch):
    import concurrent.futures
    import threading

    browser_entered, browser_release = threading.Event(), threading.Event()
    def blocking_browser(url):
        browser_entered.set()
        assert browser_release.wait(10)
    monkeypatch.setattr(standalone.webbrowser, "open", blocking_browser)
    monkeypatch.setattr(standalone, "_terminate_owned_children", lambda: [])

    class Server:
        started = False
        should_exit = False
        def run(self, sockets):
            self.started = True
            for _ in range(1000):
                if self.should_exit:
                    return
                threading.Event().wait(0.01)
            raise AssertionError("stop watcher did not exit")

    sock = standalone.bind_exclusive("127.0.0.1", 0)
    def launch():
        with standalone._InstanceOwner(tmp_path) as owner:
            return standalone._serve_hud(Server(), sock, owner, "http://127.0.0.1/", 0, no_browser=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(launch)
        try:
            assert browser_entered.wait(5)
            assert standalone.request_stop(tmp_path, timeout=5) == standalone.EXIT_OK
            assert running.result(timeout=5) == standalone.EXIT_OK
            assert not browser_release.is_set()
            assert not standalone._instance_is_locked(tmp_path)
            assert not standalone._status_file(tmp_path).exists()
        finally:
            browser_release.set()
            sock.close()


@pytest.mark.parametrize("runtime_error", [False, True])
def test_server_failure_cleans_up_without_late_status_publication(tmp_path, monkeypatch, runtime_error):
    monkeypatch.setattr(standalone, "_terminate_owned_children", lambda: [])
    class Server:
        started = False
        should_exit = False
        def run(self, sockets):
            if runtime_error:
                raise RuntimeError("startup failed")
    sock = standalone.bind_exclusive("127.0.0.1", 0)
    def launch():
        with standalone._InstanceOwner(tmp_path) as owner:
            return standalone._serve_hud(Server(), sock, owner, "http://127.0.0.1/", 0, no_browser=True)
    if runtime_error:
        with pytest.raises(RuntimeError, match="startup failed"):
            launch()
    else:
        assert launch() == standalone.EXIT_STARTUP_FAILED
    assert sock.fileno() == -1
    assert not standalone._status_file(tmp_path).exists()
    with standalone._InstanceOwner(tmp_path) as replacement:
        standalone.time.sleep(0.1)
        assert standalone._read_status(tmp_path)["instance_id"] == replacement.instance_id
