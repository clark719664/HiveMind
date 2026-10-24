"""Tests for scripts/hivemind_producer_agent.py. HTTP is external I/O, so it is replaced by httpx.MockTransport."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_AGENT_PATH: Path = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_producer_agent.py"


def _load_agent() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hivemind_producer_agent", _AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent: ModuleType = _load_agent()

GATEWAY: str = "https://gateway.example.org"
API_KEY: str = "hm-test-key-for-producer"
ENDPOINT: str = "https://producer.example.org"
ACK: dict[str, object] = {"account_id": "acc123", "status": "live", "models": ["qwen3-8b"]}


def _recording_transport(
    status: int, body: dict[str, object], seen: list[httpx.Request]
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def test_once_sends_expected_heartbeat_and_prints_response(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(agent.API_KEY_ENV, API_KEY)
    seen: list[httpx.Request] = []
    transport = _recording_transport(200, ACK, seen)

    code = agent.main(
        [
            "--gateway",
            GATEWAY + "/",
            "--endpoint",
            ENDPOINT,
            "--model",
            "qwen3-8b=qwen3:8b",
            "--model",
            "coder-7b=qwen2.5-coder:7b",
            "--cost-per-1k",
            "0.4",
            "--capacity-1k",
            "50",
            "--execution-class",
            "t4",
            "--once",
        ],
        transport=transport,
    )

    assert code == 0
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == GATEWAY + agent.HEARTBEAT_PATH
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "endpoint_url": ENDPOINT,
        "models": [
            {"id": "qwen3-8b", "local_name": "qwen3:8b"},
            {"id": "coder-7b", "local_name": "qwen2.5-coder:7b"},
        ],
        "cost_per_1k": 0.4,
        "capacity_1k": 50.0,
        "execution_class": "t4",
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed == ACK
    assert printed["account_id"] == "acc123"


def test_once_reports_gateway_rejection_with_exit_code_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(agent.API_KEY_ENV, API_KEY)
    seen: list[httpx.Request] = []
    transport = _recording_transport(403, {"error": {"message": "forbidden"}}, seen)

    code = agent.main(
        [
            "--gateway",
            GATEWAY,
            "--endpoint",
            ENDPOINT,
            "--model",
            "qwen3-8b=qwen3:8b",
            "--cost-per-1k",
            "0.4",
            "--capacity-1k",
            "50",
            "--once",
        ],
        transport=transport,
    )

    assert code == 1
    assert len(seen) == 1
    err = capsys.readouterr().err
    assert "heartbeat failed" in err
    assert API_KEY not in err


def test_missing_api_key_environment_exits_with_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(agent.API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        agent.main(
            [
                "--gateway",
                GATEWAY,
                "--endpoint",
                ENDPOINT,
                "--model",
                "qwen3-8b=qwen3:8b",
                "--cost-per-1k",
                "0.4",
                "--capacity-1k",
                "50",
                "--once",
            ]
        )
    assert excinfo.value.code == 2


def test_model_spec_rejects_missing_separator_and_blank_parts() -> None:
    for bad in ("qwen3-8b", "=qwen3:8b", "qwen3-8b=", "  =  "):
        with pytest.raises(argparse.ArgumentTypeError):
            agent.parse_model_spec(bad)
    assert agent.parse_model_spec(" qwen3-8b = qwen3:8b ") == {
        "id": "qwen3-8b",
        "local_name": "qwen3:8b",
    }


_NAME_TEXT: st.SearchStrategy[str] = st.text(
    alphabet=st.characters(blacklist_characters="=\x00", blacklist_categories=("Cs", "Zs", "Cc")),
    min_size=1,
    max_size=40,
)


@settings(max_examples=200, deadline=None)
@given(model_id=_NAME_TEXT, local_name=_NAME_TEXT)
def test_model_spec_round_trips_any_id_and_local_name(model_id: str, local_name: str) -> None:
    parsed = agent.parse_model_spec(f"{model_id}={local_name}")
    assert parsed == {"id": model_id, "local_name": local_name}
    assert set(parsed) == {"id", "local_name"}


@settings(max_examples=100, deadline=None)
@given(
    models=st.lists(st.tuples(_NAME_TEXT, _NAME_TEXT), min_size=1, max_size=8),
    cost=st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
    capacity=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False),
)
def test_heartbeat_payload_preserves_models_in_order(
    models: list[tuple[str, str]], cost: float, capacity: float
) -> None:
    entries = [{"id": a, "local_name": b} for a, b in models]
    payload = agent.build_heartbeat_payload(
        endpoint=ENDPOINT,
        models=entries,
        cost_per_1k=cost,
        capacity_1k=capacity,
        execution_class=agent.DEFAULT_EXECUTION_CLASS,
    )
    assert payload["models"] == entries
    assert len(payload["models"]) == len(models)
    entries[0]["id"] = "mutated"
    assert payload["models"][0]["id"] == models[0][0]


def test_send_heartbeat_returns_object_and_rejects_non_object() -> None:
    def make_client(body: object) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        return httpx.Client(transport=httpx.MockTransport(handler))

    payload: dict[str, object] = {"endpoint_url": ENDPOINT, "models": []}
    with make_client(ACK) as client:
        assert agent.send_heartbeat(client, GATEWAY, API_KEY, payload) == ACK
    with make_client(["not", "an", "object"]) as client:
        with pytest.raises(ValueError):
            agent.send_heartbeat(client, GATEWAY, API_KEY, payload)


@pytest.mark.parametrize("gateway", ["http://gateway.example.com", "ftp://gateway.example.com", "gateway.example.com"])
def test_agent_refuses_a_non_https_gateway_before_sending_the_key(monkeypatch: pytest.MonkeyPatch, gateway: str) -> None:
    agent = _load_agent()
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-key")
    sent: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(200, json={}))

    with pytest.raises(SystemExit) as exit_info:
        agent.main(["--gateway", gateway, "--endpoint", "https://gpu.example.com", "--model", "m=local", "--once"], transport=transport)

    assert exit_info.value.code == 2
    assert sent == []
