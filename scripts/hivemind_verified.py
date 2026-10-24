#!/usr/bin/env python3
"""Verified mode for HiveMind coding requests: generate, run YOUR tests in a sandbox on YOUR machine, repair, and only call it
verified when the tests pass.

Why this runs on your side: the router and the GPU producers must never execute code that a stranger sends. Here the only code
that runs is the answer to your own request, against your own tests, inside the local fail-closed sandbox of ``e15_gate``
(Linux user namespaces, no network, empty environment). If that sandbox is unavailable the result is BLOCKED, never "verified".

Measured (MBPP-sanitized, 257 held-out tasks, the 7B ``coder`` tier through the live router, 2026-10-24): first attempt 196/257
(76.3%), with this loop 213/257 (82.9%); 17 tasks rescued, none lost, exact McNemar p = 1.5e-5. Sampling was not controlled, so
expect noise. The loop is policy: the first candidate that passes your tests wins; if none does, the FIRST attempt is returned
and flagged UNVERIFIED_FALLBACK.

    export HIVEMIND_API_KEY=hm-...
    python hivemind_verified.py --task "Write clamp(x, lo, hi)" --test "assert clamp(5,0,3)==3" --test "assert clamp(-1,0,3)==0"

Exit status: 0 verified, 1 unverified fallback (code printed only with --allow-unverified), 2 blocked / no code / request failed.
Dependency: ``pip install httpx`` (the sandbox ships in this repository's scripts/e15_gate.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GATEWAY: str = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
GATEWAY_ENV: str = "HIVEMIND_GATEWAY"
API_KEY_ENV: str = "HIVEMIND_API_KEY"
DEFAULT_MODEL: str = "hivemind/coder"
DEFAULT_MAX_TOKENS: int = 700
DEFAULT_MAX_REPAIRS: int = 3
HTTP_TIMEOUT_S: float = 300.0
SYSTEM_PROMPT: str = "You are a careful Python programming assistant. Reply with exactly one ```python code block containing the complete solution."
STATUS_VERIFIED, STATUS_FALLBACK, STATUS_BLOCKED, STATUS_NO_CODE = "VERIFIED", "UNVERIFIED_FALLBACK", "BLOCKED", "NO_CODE"

Messages = list[dict[str, str]]
Ask = Callable[[Messages], str]


@dataclass(frozen=True)
class Judgement:
    """What the sandbox said about one candidate. ``blocked`` means it could not judge at all (fail closed)."""

    accepted: bool
    verdict: str
    feedback: str = ""
    blocked: bool = False


Judge = Callable[[str], Judgement]


@dataclass(frozen=True)
class Attempt:
    round: int
    verdict: str
    feedback: str


@dataclass(frozen=True)
class VerifiedResult:
    status: str
    code: str
    rounds: int
    attempts: tuple[Attempt, ...]

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "verified": self.verified, "code": self.code, "rounds": self.rounds,
                "attempts": [{"round": a.round, "verdict": a.verdict, "feedback": a.feedback} for a in self.attempts]}


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """The first fenced code block of a reply, or ``None`` when there is none (prose is never run)."""
    match = _FENCE.search(text)
    return match.group(1) if match and match.group(1).strip() else None


def verified_solve(ask: Ask, messages: Messages, judge: Judge, *, max_repairs: int = DEFAULT_MAX_REPAIRS) -> VerifiedResult:
    """Ask, judge, and repair up to ``max_repairs`` times. Pure orchestration: ``ask`` and ``judge`` are injected.

    The first accepted candidate wins. If none is accepted the first candidate is returned as UNVERIFIED_FALLBACK. A judge that
    cannot run (``blocked``) stops the loop at once: more rounds cannot help and the answer must not look verified.
    """
    if max_repairs < 0:
        raise ValueError("max_repairs must be >= 0")
    convo = list(messages)
    attempts: list[Attempt] = []
    first: str | None = None
    for rnd in range(max_repairs + 1):
        reply = ask(convo)
        code = extract_code(reply)
        if code is None:
            attempts.append(Attempt(rnd, "NO_CODE", "no ```python block in the reply"))
            convo += [{"role": "assistant", "content": reply},
                      {"role": "user", "content": "Reply with exactly one ```python block with the complete solution."}]
            continue
        if first is None:
            first = code
        verdict = judge(code)
        attempts.append(Attempt(rnd, verdict.verdict, verdict.feedback))
        if verdict.blocked:
            return VerifiedResult(STATUS_BLOCKED, "", rnd + 1, tuple(attempts))
        if verdict.accepted:
            return VerifiedResult(STATUS_VERIFIED, code, rnd + 1, tuple(attempts))
        convo += [{"role": "assistant", "content": reply},
                  {"role": "user", "content": f"That failed: {verdict.feedback}\nFix it. Reply with exactly one ```python block with the complete solution."}]
    if first is None:
        return VerifiedResult(STATUS_NO_CODE, "", len(attempts), tuple(attempts))
    return VerifiedResult(STATUS_FALLBACK, first, len(attempts), tuple(attempts))


def sandbox_judge(tests: Sequence[str], setup: str = "") -> Judge:
    """A judge backed by the repository's fail-closed sandbox (``scripts/e15_gate.py``). Unavailable sandbox -> blocked."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import e15_gate as gate
    except ImportError as exc:  # pragma: no cover - the file ships beside this script
        raise RuntimeError("scripts/e15_gate.py must sit next to hivemind_verified.py") from exc

    def judge(code: str) -> Judgement:
        result = gate.gate(code, list(tests), setup)
        if result.verdict == gate.UNVERIFIED:
            return Judgement(False, result.verdict, result.feedback(), blocked=True)
        return Judgement(result.accepted, result.verdict, "" if result.accepted else result.feedback())

    return judge


def router_ask(client: Any, gateway: str, api_key: str, model: str, max_tokens: int) -> Ask:
    """An ``ask`` that posts to the router's OpenAI-compatible endpoint with an ``httpx.Client``."""

    def ask(messages: Messages) -> str:
        response = client.post(f"{gateway.rstrip('/')}/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                               json={"model": model, "messages": messages, "max_tokens": max_tokens}, timeout=HTTP_TIMEOUT_S)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "")

    return ask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Python through HiveMind and keep only code that passes your tests.")
    parser.add_argument("--task", required=True, help="What the code must do (state the function name and behaviour).")
    parser.add_argument("--test", action="append", default=[], help="A test statement, e.g. \"assert f(1)==2\" (repeatable).")
    parser.add_argument("--tests-file", help="A file with one test statement per blank-line-separated block.")
    parser.add_argument("--setup", default="", help="Code run before the candidate (imports your tests need).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--allow-unverified", action="store_true", help="Also print the first attempt when no candidate passed.")
    parser.add_argument("--json", action="store_true", help="Print the whole result as JSON.")
    return parser


def run(argv: Sequence[str] | None = None, *, client: Any = None, judge: Judge | None = None) -> int:
    args = build_parser().parse_args(argv)
    tests = list(args.test)
    if args.tests_file:
        tests += [block.strip() for block in Path(args.tests_file).read_text(encoding="utf-8").split("\n\n") if block.strip()]
    if not tests:
        print("give at least one --test (or --tests-file): without a test nothing can be verified", file=sys.stderr)
        return 2
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(f"set {API_KEY_ENV} to your HiveMind API key", file=sys.stderr)
        return 2
    import httpx

    owns = client is None
    http = client or httpx.Client()
    user = args.task.strip() + "\nYour code should pass these tests:\n" + "\n".join(tests)
    messages: Messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    try:
        ask = router_ask(http, os.environ.get(GATEWAY_ENV, DEFAULT_GATEWAY), api_key, args.model, args.max_tokens)
        result = verified_solve(ask, messages, judge or sandbox_judge(tests, args.setup), max_repairs=args.max_repairs)
    except httpx.HTTPError as exc:
        print(f"request failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    finally:
        if owns:
            http.close()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif result.verified or (args.allow_unverified and result.code):
        print(result.code)
    print(f"[{result.status}] after {result.rounds} model call(s)", file=sys.stderr)
    if result.verified:
        return 0
    return 1 if result.status == STATUS_FALLBACK else 2


if __name__ == "__main__":
    sys.exit(run())
