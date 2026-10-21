"""Shared test fixtures."""

from __future__ import annotations

import pytest

from hivemind.security import gpu_execution_circuit_breaker as _breaker

_NOMINAL_HOST = {"system_ram_used_pct": 50.0, "on_battery_power": False, "battery_percent": 100.0}


@pytest.fixture(autouse=True)
def _nominal_host_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin host RAM and battery readings so the HW circuit breaker does not abort training
    tests on a busy or unplugged machine. Explicit simulated overrides still take precedence."""
    live_query = _breaker.query_live_gpu_hardware_telemetry

    def query(simulated_overrides=None):
        return live_query(simulated_overrides={**_NOMINAL_HOST, **(simulated_overrides or {})})

    monkeypatch.setattr(_breaker, "query_live_gpu_hardware_telemetry", query)
