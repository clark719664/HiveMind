"""Build and install the actual wheel without duplicating runtime dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
CACHE_VARIABLES = (
    "HIVEMIND_CACHE_DIR", "HIVEMIND_ARTIFACTS_DIR", "HF_HOME", "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "UV_CACHE_DIR", "TMP", "TEMP",
)


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    root = tmp_path_factory.mktemp("installed-hivemind")
    wheels = root / "wheels"
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(wheels)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    wheel, = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as package:
        assert "hivemind/resources/rtx2070_dual_profile.yaml" in package.namelist()
    target = root / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile",
         "--target", str(target), str(wheel)],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return target


def _read_installed_config(target, tmp_path, overrides=None, config_path=None):
    env = {key: value for key, value in os.environ.items() if key not in CACHE_VARIABLES}
    env.update({
        "PYTHONPATH": str(target), "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_CACHE_HOME": str(tmp_path / "user-cache"),
        "HOME": str(tmp_path / "home"),
    })
    env.update(overrides or {})
    config_argument = repr(str(config_path)) if config_path is not None else "None"
    script = f"""
import json, os
import hivemind.config as c
cfg = c.HiveMindConfig.from_yaml({config_argument})
print(json.dumps({{
    'module': c.__file__, 'profiles': list(cfg.profiles),
    'cache': str(c.CACHE_ROOT), 'artifacts': str(c.ARTIFACTS_ROOT),
    'environment': {{key: os.environ.get(key) for key in {CACHE_VARIABLES!r}}},
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def test_installed_wheel_loads_profiles_and_never_writes_beside_code(installed_wheel, tmp_path):
    before = set(installed_wheel.rglob("*"))
    result = _read_installed_config(installed_wheel, tmp_path)
    assert Path(result["module"]).is_relative_to(installed_wheel)
    assert {"rtx2070_moe_e2e", "rtx2070_7b_worker", "rtx4070_14b_spec"} <= set(result["profiles"])
    assert Path(result["cache"]) == tmp_path / "user-cache" / "hivemind"
    assert Path(result["artifacts"]) == Path(result["cache"]) / "artifacts"
    assert set(installed_wheel.rglob("*")) == before


def test_installed_cache_environment_overrides_are_preserved(installed_wheel, tmp_path):
    overrides = {key: str(tmp_path / key.lower()) for key in CACHE_VARIABLES}
    result = _read_installed_config(installed_wheel, tmp_path, overrides)
    assert result["environment"] == overrides
    assert result["cache"] == overrides["HIVEMIND_CACHE_DIR"]
    assert result["artifacts"] == overrides["HIVEMIND_ARTIFACTS_DIR"]


def test_hf_home_override_controls_default_hub_paths(installed_wheel, tmp_path):
    hf_home = str(tmp_path / "selected-hf-cache")
    result = _read_installed_config(installed_wheel, tmp_path, {"HF_HOME": hf_home})
    assert result["environment"]["HF_HOME"] == hf_home
    assert result["environment"]["HF_HUB_CACHE"] == str(Path(hf_home) / "hub")
    assert result["environment"]["HUGGINGFACE_HUB_CACHE"] == str(Path(hf_home) / "hub")


def test_invalid_default_cache_location_falls_back_to_temp(installed_wheel, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("not a cache directory")
    fallback = tmp_path / "temporary"
    fallback.mkdir()
    result = _read_installed_config(installed_wheel, tmp_path, {
        "XDG_CACHE_HOME": str(blocked), "TMPDIR": str(fallback),
    })
    assert Path(result["cache"]).is_relative_to(fallback)
    assert result["profiles"]


def test_installed_explicit_config_is_used(installed_wheel, tmp_path):
    configuration = tmp_path / "custom.yaml"
    configuration.write_text("profiles:\n  custom_worker:\n    hf_model_id: test/model\n")
    result = _read_installed_config(installed_wheel, tmp_path, config_path=configuration)
    assert result["profiles"] == ["custom_worker"]


def test_bundled_default_profiles_match_checkout():
    assert (ROOT / "src/hivemind/resources/rtx2070_dual_profile.yaml").read_bytes() == (
        ROOT / "configs/rtx2070_dual_profile.yaml"
    ).read_bytes()
