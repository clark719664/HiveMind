#!/usr/bin/env python3
"""HiveMind producer agent: heartbeats a local OpenAI-compatible server into the OpenRouter gateway.

The agent does not run inference. It advertises an endpoint you already run (Ollama, vLLM or
llama.cpp server), the models behind it, a price per 1k tokens and a capacity per epoch. The
gateway routes consumer requests to live producers and credits HMT for the tokens they serve.

Usage:
    export HIVEMIND_API_KEY=hm-...
    python scripts/hivemind_producer_agent.py \
        --gateway https://GATEWAY_HOST \
        --endpoint https://producer.example.org \
        --model qwen3-8b=qwen3:8b \
        --cost-per-1k 0.4 --capacity-1k 50 --once

The API key is read from the HIVEMIND_API_KEY environment variable only, so it never appears in
shell history or process listings. The endpoint must be reachable over https at a public address;
the gateway's SSRF guard rejects anything else unless it was started with allow_private_producers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

HEARTBEAT_PATH: str = "/v1/hivemind/producer/heartbeat"
API_KEY_ENV: str = "HIVEMIND_API_KEY"
DEFAULT_EXECUTION_CLASS: str = "gpu"
DEFAULT_INTERVAL_S: float = 60.0
HTTP_TIMEOUT_S: float = 30.0


def parse_model_spec(raw: str) -> dict[str, str]:
    """Parse one ``--model id=local_name`` argument into a heartbeat model entry."""
    model_id, separator, local_name = raw.partition("=")
    model_id = model_id.strip()
    local_name = local_name.strip()
    if not separator or not model_id or not local_name:
        raise argparse.ArgumentTypeError(f"--model expects id=local_name, got {raw!r}")
    return {"id": model_id, "local_name": local_name}


def build_heartbeat_payload(
    *,
    endpoint: str,
    models: list[dict[str, str]],
    cost_per_1k: float,
    capacity_1k: float,
    execution_class: str,
) -> dict[str, Any]:
    """Build the JSON body for POST /v1/hivemind/producer/heartbeat."""
    return {
        "endpoint_url": endpoint,
        "models": [dict(model) for model in models],
        "cost_per_1k": cost_per_1k,
        "capacity_1k": capacity_1k,
        "execution_class": execution_class,
    }


def send_heartbeat(
    client: httpx.Client, gateway: str, api_key: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST one heartbeat and return the gateway's JSON object. Raises on HTTP or JSON errors."""
    response = client.post(
        gateway.rstrip("/") + HEARTBEAT_PATH,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("heartbeat response is not a JSON object")
    return body


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Advertise a local OpenAI-compatible server to the HiveMind OpenRouter gateway."
    )
    parser.add_argument("--gateway", required=True, help="Gateway base URL, e.g. https://host")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Public https base URL of your OpenAI-compatible server (gateway appends /v1/...)",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model_spec,
        metavar="ID=LOCAL_NAME",
        help="Public model id = the name your server uses. Repeat for several models.",
    )
    parser.add_argument(
        "--cost-per-1k",
        type=float,
        required=True,
        help="Your cost in HMT per 1,000 tokens (must be >= 0)",
    )
    parser.add_argument(
        "--capacity-1k",
        type=float,
        required=True,
        help="Capacity per epoch in 1,000-token units (must be > 0)",
    )
    parser.add_argument(
        "--execution-class",
        default=DEFAULT_EXECUTION_CLASS,
        help=f"Hardware class label, default {DEFAULT_EXECUTION_CLASS!r}",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"Seconds between heartbeats (default {DEFAULT_INTERVAL_S:g})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send a single heartbeat, print the gateway's JSON response and exit",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None
) -> int:
    """Run the producer agent. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        parser.error(f"set the {API_KEY_ENV} environment variable to your producer API key")
    gateway_url = urlsplit(args.gateway)
    if gateway_url.scheme != "https" and not (gateway_url.scheme == "http" and gateway_url.hostname in ("localhost", "127.0.0.1", "::1")):
        parser.error("--gateway must be https (plain http is allowed only for localhost): your API key is sent in every heartbeat")
    if args.cost_per_1k < 0:
        parser.error("--cost-per-1k must be >= 0")
    if args.capacity_1k <= 0:
        parser.error("--capacity-1k must be > 0")
    if args.interval_s <= 0:
        parser.error("--interval-s must be > 0")

    payload = build_heartbeat_payload(
        endpoint=args.endpoint,
        models=args.model,
        cost_per_1k=args.cost_per_1k,
        capacity_1k=args.capacity_1k,
        execution_class=args.execution_class,
    )
    with httpx.Client(timeout=HTTP_TIMEOUT_S, transport=transport) as client:
        if args.once:
            try:
                body = send_heartbeat(client, args.gateway, api_key, payload)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"heartbeat failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            print(json.dumps(body, indent=2, sort_keys=True))
            return 0

        try:
            while True:
                try:
                    body = send_heartbeat(client, args.gateway, api_key, payload)
                    print(f"heartbeat ok: {json.dumps(body, sort_keys=True)}", flush=True)
                except (httpx.HTTPError, ValueError) as exc:
                    print(
                        f"heartbeat failed, retrying: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(args.interval_s)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
