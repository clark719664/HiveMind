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
