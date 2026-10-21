"""Import-order regressions for hivemind.security, checked in fresh interpreters."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hivemind.security as security

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize(
    "first_import",
    ["hivemind.core.diloco_outer", "hivemind.boinc.coordinator_api", "hivemind.security.federated_learning_hardening"],
)
def test_no_circular_import_from_any_entry_point(first_import: str) -> None:
    result = _run(f"import {first_import}")
    assert result.returncode == 0, result.stderr[-2000:]


def test_pr_scanner_imports_without_torch() -> None:
    result = _run(
        "import sys; sys.modules['torch'] = None\n"
        "from hivemind.security.pr_security_scanner import PRSecurityScanner\n"
        "assert 'hivemind.security.swarm_runtime_hardening' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_lazy_exports_resolve_every_public_name() -> None:
    assert sorted(security._EXPORTS) == sorted(security.__all__)
    for name in security.__all__:
        assert getattr(security, name) is not None
    with pytest.raises(AttributeError):
        security.not_a_real_export  # noqa: B018
