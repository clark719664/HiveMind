"""Response-format regression tests without model downloads or GPU execution."""

import json

import pytest
import torch
from fastapi.testclient import TestClient

from hivemind.boinc import coordinator_api
from hivemind.config import HiveMindConfig
from hivemind.game import hf_hack_miner


@pytest.mark.parametrize(
    "solution",
    [
        "    return 42\n",
        "\n\n```python\ndef answer():\n    return 42\n```\n",
        "\t\treturn 42\n",
        "\r\n    return 42\r\n",
        " \t\r\n ",
        "",
        "\u2003\u00a0return 42\n",
        "  one two three four five six seven eight nine ten eleven  \n",
    ],
    ids=[
        "indentation", "leading-newlines", "tabs", "crlf",
        "whitespace-only", "empty", "unicode", "multiple-batches",
    ],
)
def test_stream_preserves_completion_whitespace(solution, monkeypatch, tmp_path):
    monkeypatch.setattr(coordinator_api, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(
        hf_hack_miner,
        "execute_interactive_bounty_probe",
        lambda **kwargs: {
            "generated_solution": solution,
            "top2_active_experts": ["code", "math"],
            "expert_routing_weights": {"code": 0.5, "math": 0.5},
        },
    )
    coordinator = coordinator_api.BOINCCoordinatorService(
        config=HiveMindConfig.from_yaml(),
        initial_lora_state={"lora_A": torch.zeros(4, 8)},
        base_model_sha256="test",
        shards_dir=tmp_path / "shards",
    )
    with TestClient(coordinator_api.create_coordinator_fastapi_app(coordinator)) as client:
        payload = {"messages": [{"role": "user", "content": "Return code."}]}
        plain = client.post("/v1/chat/completions", json=payload)
        streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})

    assert plain.status_code == streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert content == plain.json()["choices"][0]["message"]["content"] == solution
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
