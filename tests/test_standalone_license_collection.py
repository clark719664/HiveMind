"""Offline license collection checks; no Windows build or ML packages required."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "standalone_license_builder", ROOT / "packaging/windows/build_standalone.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def distribution(root, name="tokenizers", version="0.22.2", files=()):
    return {
        "name": name, "version": version, "license": "", "classifiers": [],
        "root": str(root), "license_files": list(files),
    }


@pytest.fixture
def collect(tmp_path, monkeypatch):
    # Never copy the test runner's actual CPython license or installed packages.
    python_root = tmp_path / "python"
    python_root.mkdir()
    monkeypatch.setattr(builder.sys, "base_prefix", str(python_root))

    def run(distributions):
        calls = []

        def metadata_only(cmd, **kwargs):
            calls.append(cmd)
            assert cmd == [Path("test-python"), "-c", builder._DIST_SCRIPT]
            assert kwargs == {"capture_output": True}
            return SimpleNamespace(stdout=json.dumps(distributions))

        monkeypatch.setattr(builder, "sh", metadata_only)
        dest = tmp_path / "licenses"
        records = builder.collect_licenses(Path("test-python"), dest)
        assert len(calls) == 1
        return records, dest

    return run


def test_exact_version_fallback_copies_verified_upstream_bytes(collect, tmp_path):
    records, dest = collect([distribution(tmp_path, name="Tokenizers")])
    record = records[0]
    assert record["name"] == "Tokenizers"
    assert record["license"] == ""  # Do not rewrite installed license metadata.
    assert record["license_files_copied"] == ["LICENSE"]
    assert record["license_collection_status"] == "files_collected"
    copied = dest / "Tokenizers-0.22.2/LICENSE"
    assert copied.read_bytes() == (builder.VENDORED_LICENSES / "tokenizers-0.22.2/LICENSE").read_bytes()
    provenance = record["license_file_provenance"][0]
    assert provenance["kind"] == "vendored_upstream"
    assert provenance["sha256"] == builder.sha256(copied)
    assert provenance["source_tag"] == "v0.22.2"
    assert provenance["source_commit"] == "f383101a26663708484cac0727792aad74f78234"
    assert provenance["source_url"].endswith(provenance["source_commit"] + "/LICENSE")
    assert provenance["license_expression"] == "Apache-2.0"


def test_installed_files_take_precedence_and_keep_collision_provenance(collect, tmp_path):
    paths = ["tokenizers.dist-info/LICENSE", "tokenizers.dist-info/licenses/nested/LICENSE"]
    for i, relative in enumerate(paths):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"installed license {i}\n", encoding="utf-8")
    records, dest = collect([distribution(tmp_path, files=paths)])
    record = records[0]
    assert record["license_files_copied"] == ["LICENSE", "LICENSE.1"]
    assert [item["source_path"] for item in record["license_file_provenance"]] == paths
    for item in record["license_file_provenance"]:
        assert item["kind"] == "installed_distribution"
        assert item["sha256"] == builder.sha256(dest / "tokenizers-0.22.2" / item["file"])


def test_missing_installed_file_uses_fallback(collect, tmp_path):
    records, _ = collect([distribution(tmp_path, files=["tokenizers.dist-info/LICENSE"])])
    assert records[0]["license_file_provenance"][0]["kind"] == "vendored_upstream"


def test_no_fallback_for_another_version_or_project_license(collect, tmp_path, capsys):
    records, dest = collect([
        distribution(tmp_path, version="0.22.3"),
        distribution(tmp_path, name="project-hivemind", version="1.1.0"),
        distribution(tmp_path, name="pip", version="25.2"),
    ])
    assert {item["name"] for item in records} == {"tokenizers", "project-hivemind", "CPython"}
    for item in records:
        assert item["license_files_copied"] == []
        assert item["license_file_provenance"] == []
        assert item["license_collection_status"] == "no_license_files_found"
    assert list(dest.iterdir()) == []
    warning = capsys.readouterr().err
    assert "tokenizers==0.22.3" in warning
    assert "project-hivemind==1.1.0" in warning
    assert "CPython==" in warning


@pytest.mark.parametrize("damage", ["changed", "missing", "outside"])
def test_bad_vendored_source_fails_closed(collect, tmp_path, monkeypatch, damage):
    vendor = tmp_path / "vendor"
    shutil.copytree(builder.VENDORED_LICENSES, vendor)
    monkeypatch.setattr(builder, "VENDORED_LICENSES", vendor)
    source = vendor / "tokenizers-0.22.2/LICENSE"
    if damage == "changed":
        source.write_bytes(b"unverified substitute\n")
    elif damage == "missing":
        source.unlink()
    else:
        manifest = json.loads((vendor / "manifest.json").read_text(encoding="utf-8"))
        manifest["licenses"][0]["path"] = "../outside/LICENSE"
        (vendor / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Vendored license"):
        collect([distribution(tmp_path)])
    assert not (tmp_path / "licenses/tokenizers-0.22.2/LICENSE").exists()


def test_cpython_license_has_copy_and_hash_provenance(collect, tmp_path):
    (tmp_path / "python/LICENSE.txt").write_bytes(b"CPython test license\n")
    records, dest = collect([])
    record, = records
    assert record["name"] == "CPython"
    assert record["license_collection_status"] == "files_collected"
    copied = dest / f"CPython-{builder.platform.python_version()}" / "LICENSE.txt"
    assert copied.read_bytes() == b"CPython test license\n"
    assert record["license_file_provenance"] == [{
        "file": "LICENSE.txt", "kind": "build_python", "source_path": "LICENSE.txt",
        "sha256": builder.sha256(copied),
    }]


def test_vendored_index_matches_pin_and_official_git_blob():
    manifest = json.loads((builder.VENDORED_LICENSES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    pins = builder.CONSTRAINTS.read_text(encoding="utf-8").splitlines()
    for item in manifest["licenses"]:
        assert f"{item['name']}=={item['version']}" in pins
        content = (builder.VENDORED_LICENSES / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        git_blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
        assert hashlib.sha1(git_blob).hexdigest() == item["source_blob_sha"]
        assert b"\r" not in content


def test_metadata_script_collects_dist_info_notices_and_nested_licenses(tmp_path):
    site = tmp_path / "site"
    dist_info = site / "sample-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Name: sample\nVersion: 1.0\nLicense-Expression: MIT\n", encoding="utf-8")
    relative_files = [
        "sample-1.0.dist-info/LICENSE", "sample-1.0.dist-info/NOTICE.txt",
        "sample-1.0.dist-info/licenses/vendor/Apache-2.0.txt", "sample-1.0.dist-info/README.txt",
    ]
    for relative in relative_files:
        file = site / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("\n".join(f"{p},," for p in relative_files), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(site))
    result = subprocess.run(
        [sys.executable, "-S", "-c", builder._DIST_SCRIPT], env=env,
        check=True, capture_output=True, text=True,
    )
    record, = json.loads(result.stdout)
    assert record["name"] == "sample"
    assert record["license"] == "MIT"
    assert record["license_files"] == sorted(relative_files[:3])
