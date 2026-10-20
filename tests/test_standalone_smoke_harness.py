"""Portable tests of the frozen Windows smoke-test assertions (no EXE claims)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).resolve().parents[1] / "packaging/windows/smoke_test_standalone.py"
    spec = importlib.util.spec_from_file_location("standalone_smoke_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self, pid, code=None):
        self.pid = pid
        self.code = code

    def poll(self):
        return self.code

    def wait(self, timeout):
        if self.code is None:
            raise subprocess.TimeoutExpired("fake EXE", timeout)
        return self.code


def test_explicit_port_conflict_uses_distinct_data_directory(harness, tmp_path, monkeypatch):
    created = []

    class Instance:
        def __init__(self, exe, args, env, logs, tag):
            self.env = env
            self.args = args
            self.closed = False
            self.proc = FakeProcess(100, 3)
            self.stderr_path = logs / "conflict.stderr.txt"
            self.stderr_path.write_text("Port is already in use")
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness, "Instance", Instance)
    env = {"HIVEMIND_DATA_DIR": str(tmp_path / "owner")}
    rec = harness.Recorder()
    harness.check_explicit_port_conflict(Path("HiveMindHUD.exe"), env, tmp_path, tmp_path, 8123, rec)
    assert all(check["passed"] for check in rec.checks)
    assert created[0].env["HIVEMIND_DATA_DIR"] != env["HIVEMIND_DATA_DIR"]
    assert created[0].args == ["--no-browser", "--port", "8123"]
    assert created[0].closed
    assert env == {"HIVEMIND_DATA_DIR": str(tmp_path / "owner")}


@pytest.mark.parametrize("fault", [None, "wrong_rejection", "replaced_status", "owner_dies", "stops_other"])
def test_shared_directory_checks_enforce_owner_isolation(harness, tmp_path, monkeypatch, fault):
    created = []
    owners = {}
    stops = []

    class Instance:
        def __init__(self, exe, args, env, logs, tag):
            self.data_dir = Path(env["HIVEMIND_DATA_DIR"])
            self.status_path = self.data_dir / "run/hud-server.json"
            self.proc = FakeProcess(100 + len(created))
            self.tag = tag
            self.closed = False
            self.stderr_path = logs / f"{tag}.stderr.txt"
            self.stderr_path.write_text("")
            created.append(self)
            if tag == "shared_contender":
                self.proc.code = 3 if fault == "wrong_rejection" else 8
                if fault == "replaced_status":
                    self.status_path.write_text(json.dumps({"pid": self.proc.pid, "instance_id": tag}))
                if fault == "owner_dies":
                    owners[self.data_dir].proc.code = 1
            else:
                owners[self.data_dir] = self
                self.status_path.parent.mkdir(parents=True)
                self.status_path.write_text(json.dumps({"pid": self.proc.pid, "instance_id": tag}))

        def wait_url(self):
            return f"http://127.0.0.1:{8000 + self.proc.pid}/"

        def close(self):
            self.closed = True

    def stop(exe, env, logs):
        data_dir = Path(env["HIVEMIND_DATA_DIR"])
        stops.append(data_dir)
        owner = owners[data_dir]
        owner.proc.code = 0
        owner.status_path.unlink(missing_ok=True)
        if fault == "stops_other" and owner.tag == "shared_owner":
            for other in owners.values():
                other.proc.code = 0
        return SimpleNamespace(returncode=0, stdout="HiveMind HUD stopped.", stderr="")

    monkeypatch.setattr(harness, "Instance", Instance)
    monkeypatch.setattr(harness, "stop_instance", stop)
    rec = harness.Recorder()
    harness.check_shared_directory_ownership(Path("HiveMindHUD.exe"), {}, tmp_path, tmp_path, rec)
    assert len(rec.checks) == 8
    assert all(check["passed"] for check in rec.checks) is (fault is None)
    assert rec.observations == []
    assert stops == [tmp_path / "shared data", tmp_path / "independent data"]
    assert all(instance.closed for instance in created)


def test_shared_directory_instances_closed_if_assertion_setup_raises(harness, tmp_path, monkeypatch):
    created = []

    class Instance:
        def __init__(self, *args):
            self.closed = False
            created.append(self)

        def wait_url(self):
            raise OSError("failed to inspect process logs")

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness, "Instance", Instance)
    with pytest.raises(OSError, match="process logs"):
        harness.check_shared_directory_ownership(Path("HiveMindHUD.exe"), {}, tmp_path, tmp_path, harness.Recorder())
    assert created and all(instance.closed for instance in created)


@pytest.mark.parametrize("content", ["not-json", "[]", "null"])
def test_invalid_owner_status_is_not_accepted(harness, tmp_path, content):
    (tmp_path / "run").mkdir()
    (tmp_path / "run/hud-server.json").write_text(content)
    assert harness.read_owner(tmp_path) == {}
