#!/usr/bin/env python3
"""GWAYA-Lite coding assistant: the deployable form of experiment E16 (stdlib + the reused E15/E16 modules, no pip).

A small-first model ladder (Qwen2.5-Coder 1.5B -> 3B -> 7B, Q4_K_M, served by the local Ollama daemon) behind the fail-closed GWAYA-style gate
(``scripts/e15_gate.py``: vendored GWAYA zero-stub auditor + namespace sandbox).  Subcommands::

    gwaya_lite.py ask PROMPT [--tests FILE | --test 'assert f(1)==2' ...] [--ladder 1.5b,3b,7b] [--ctx 4096] [--max-rounds 3] [--json] [--keep-warm]
    gwaya_lite.py serve [--host 127.0.0.1] [--port 8099] [--ladder ...] [--ctx 4096] [--allow-nonlocal --api-key-file F] [--log-prompts]
    gwaya_lite.py doctor

THE LOOP (the E16 "C1" loop, per tier): greedy candidate, then up to 2 samples (T=0.7, top_p=0.95), then up to ``--max-rounds`` greedy repair rounds that show
the model its previous code plus the gate feedback.  Every candidate goes through the gate; the first one that passes wins.  The next tier is tried only when
the whole loop of the current tier found no passing candidate.  The first test (if any) is shown to the model in the prompt, as in E16; ALL tests gate.

VERDICTS (strict; the tool never claims more than it checked)::

    VERIFIED    audit clean AND the user-supplied tests pass in the sandbox
    AUDITED     no tests given: syntax + zero-stub audit only; NOT tested
    UNVERIFIED  the gate / sandbox could not run (fail-closed)
    FAILED      nothing passed the gate after the whole ladder

Exit codes: 0 VERIFIED/AUDITED, 1 FAILED, 2 UNVERIFIED (or doctor found problems), 3 usage / environment error (Ollama down, model missing, bad flags).
VERIFIED is NOT "correct": in the E16 measurement 5-7% of VERIFIED solutions still failed the hidden tests (artifacts/experiments/E16_gwaya_lite_t4_report.json).

GPU DISCIPLINE (8 GB card): ``Residency`` keeps the sum of the resident Ollama models under ``--max-resident-mib`` (default 7500) using the measured E16 numbers,
unloads other resident models before a load that would exceed it, loads the active tier with keep_alive 5m, and unloads on exit / SIGTERM / SIGINT / ``serve``
shutdown (never leaves a model resident after ``ask`` unless ``--keep-warm``).

SECURITY: candidate code runs ONLY inside ``e15_gate``'s sandbox (namespaces, tmpfs over $HOME, no network, dropped capabilities, rlimits).  That sandbox is NOT
GWAYA's Bubblewrap oracle and has no seccomp filter.  This file contains no dynamic code execution, no shell and no subprocess; the server never runs request-supplied
commands.  ``serve`` binds loopback only by default and then has no auth: any local user / process on the machine can use it (document it, do not expose the port).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e15_data as data  # noqa: E402  (prompt format, SYSTEM_PROMPT; light imports only)
import e15_gate as gate_mod  # noqa: E402  (gate(), gate_available(), verdicts)
import e16_ollama as oll  # noqa: E402  (stdlib Ollama client, GPU readers)

PROJECT_ROOT = HERE.parent
REPORT_PATH = PROJECT_ROOT / "artifacts" / "experiments" / "E16_gwaya_lite_t4_report.json"
REPORT_REL = "artifacts/experiments/E16_gwaya_lite_t4_report.json"

TIER_MODELS = {"1.5b": "qwen2.5-coder:1.5b", "3b": "qwen2.5-coder:3b", "7b": "qwen2.5-coder:7b-instruct"}
FALLBACK_MIB = {"1.5b": {4096: 1651}, "3b": {4096: 2713}, "7b": {4096: 5483}}  # E16 measured at num_ctx 4096 (used only if the report file is absent)
KV_KIB = {"1.5b": 28, "3b": 36, "7b": 56}  # KV cache per token, KiB (E16 design)
UNKNOWN_MODEL_OVERHEAD_MIB = 450  # added to ps size_vram for a resident model that is not a ladder tier (CUDA context + buffers)
DEFAULT_LADDER, DEFAULT_CTX, DEFAULT_BUDGET_MIB, KEEP_ALIVE = "1.5b,3b,7b", 4096, 7500, "5m"
N_SAMPLES, SAMPLE_T, SAMPLE_TOP_P = 2, 0.7, 0.95
GATE_TIMEOUT_S, GATE_MEM_MB = 10, 2048
MAX_BODY_BYTES, MAX_TOKENS_CAP, DEFAULT_TOKENS = 64 * 1024, 1024, 512
MAX_PROMPT_CHARS, MAX_TESTS, MAX_TEST_CHARS = 16000, 100, 2000
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
SERVED_MODEL_ID = "gwaya-lite"

VERIFIED, AUDITED, UNVERIFIED, FAILED = "VERIFIED", "AUDITED", "UNVERIFIED", "FAILED"
EXIT_CODES = {VERIFIED: 0, AUDITED: 0, FAILED: 1, UNVERIFIED: 2}
EXIT_USAGE = 3
CAVEAT = ("In the E16 measurement 5-7% of VERIFIED solutions still failed the hidden tests (precision 92.6-94.8%; " + REPORT_REL + "): VERIFIED means 'audit clean and "
          "your tests passed in the sandbox', never 'correct'.")


class GwayaLiteError(RuntimeError):
    """Environment / usage problem (exit code 3, HTTP 400/502)."""


class ResidencyError(GwayaLiteError):
    pass


# ================================================================ planning table (E16 numbers)
def load_planning_table(report_path: Path = REPORT_PATH) -> Dict[str, Dict[int, int]]:
    """{tier: {num_ctx: measured MiB}} read from the E16 report (probes[].measured_mib); the E16 constants if the report is missing or unreadable."""
    table = {t: dict(v) for t, v in FALLBACK_MIB.items()}
    try:
        rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
        found: Dict[str, Dict[int, int]] = {}
        for p in rep.get("probes", []):
            if p.get("tier") in TIER_MODELS and isinstance(p.get("measured_mib"), (int, float)) and isinstance(p.get("ctx"), int):
                found.setdefault(p["tier"], {})[p["ctx"]] = int(p["measured_mib"])
        for t, v in found.items():
            if 4096 in v:
                table[t] = v
    except (OSError, ValueError, AttributeError):
        pass
    return table


def need_mib(table: Dict[str, Dict[int, int]], tier: str, ctx: int) -> int:
    """Planned GPU memory of ``tier`` at ``ctx``: the measured value if that context was measured, else the 4096 value plus the KV delta (28/36/56 KiB per token)."""
    row = table[tier]
    if ctx in row:
        return row[ctx]
    return max(1, int(row[4096] + KV_KIB[tier] * (ctx - 4096) / 1024))


def choose_unloads(resident: Dict[str, int], need: int, budget: int) -> List[str]:
    """Names (largest first) to unload from ``resident`` {name: MiB} so that what stays plus ``need`` fits ``budget``.  Minimal prefix of the largest-first order."""
    keep = dict(resident)
    out: List[str] = []
    for name in sorted(resident, key=lambda n: -resident[n]):
        if sum(keep.values()) + need <= budget:
            break
        out.append(name)
        del keep[name]
    return out


def plan_sequence(ladder: Sequence[str], ctx: int, budget: int, table: Dict[str, Dict[int, int]]) -> List[Dict[str, Any]]:
    """The 'max resident' plan: loading the ladder tiers in order, which model is unloaded before each load and what stays resident."""
    resident: Dict[str, int] = {}
    steps = []
    for t in ladder:
        need = need_mib(table, t, ctx)
        unload = choose_unloads(resident, need, budget) if need <= budget else []
        for n in unload:
            del resident[n]
        resident[t] = need
        steps.append({"load": t, "need_mib": need, "unload_first": unload, "resident_after": sorted(resident), "total_mib": sum(resident.values()), "fits": sum(resident.values()) <= budget})
    return steps


class Residency:
    """Keeps the resident Ollama models under ``budget_mib`` before every tier load; unloads on demand and at exit."""

    def __init__(self, client: oll.Client, budget_mib: int = DEFAULT_BUDGET_MIB, table: Optional[Dict[str, Dict[int, int]]] = None, log: Callable[[str], None] = lambda m: None, unload_wait_s: float = 20.0) -> None:
        self.client, self.budget, self.table, self.log = client, int(budget_mib), table or load_planning_table(), log
        self.unload_wait_s = unload_wait_s
        self.touched: set = set()
        self.unload_events: List[str] = []

    def _usage(self, entry: Dict[str, Any]) -> int:
        for tier, model in TIER_MODELS.items():
            if entry["name"] == model:
                return need_mib(self.table, tier, int(entry.get("context_length") or DEFAULT_CTX))
        return int(entry.get("size_vram_mib") or 0) + UNKNOWN_MODEL_OVERHEAD_MIB

    def ensure(self, tier: str, ctx: int) -> List[str]:
        """Make room for ``tier`` at ``ctx``; returns the names unloaded.  Raises ResidencyError if the tier alone exceeds the budget or an unload cannot be verified."""
        model, need = TIER_MODELS[tier], need_mib(self.table, tier, ctx)
        if need > self.budget:
            raise ResidencyError(f"tier {tier} at num_ctx {ctx} needs about {need} MiB, above --max-resident-mib {self.budget}")
        others = {e["name"]: self._usage(e) for e in self.client.ps() if e["name"] != model}
        victims = choose_unloads(others, need, self.budget)
        for name in victims:
            if not self.client.unload(name, self.unload_wait_s):
                raise ResidencyError(f"could not unload {name} (still resident); refusing to exceed the {self.budget} MiB budget")
            self.unload_events.append(name)
            self.log(f"unloaded {name} to stay under {self.budget} MiB before loading {model}")
        self.touched.add(model)
        return victims

    def unload_all(self, tiers: Sequence[str] = tuple(TIER_MODELS)) -> List[str]:
        """Unload every ladder model that is resident or was touched by this process (best effort, never raises)."""
        names = set(self.touched) | {TIER_MODELS[t] for t in tiers}
        done: List[str] = []
        try:
            resident = {e["name"] for e in self.client.ps()}
        except oll.OllamaError:
            return done
        for n in sorted(names & resident):
            try:
                if self.client.unload(n, self.unload_wait_s):
                    done.append(n)
            except oll.OllamaError:
                pass
        return done


# ================================================================ the ladder loop
@dataclass
class TierLog:
    tier: str
    candidates: int = 0
    repair_rounds: int = 0
    tokens: int = 0
    passed: bool = False
    last_error: str = ""


@dataclass
class Result:
    verdict: str
    detail: str = ""  # human wording for the verdict line
    code: str = ""
    tier: str = ""
    candidates: int = 0
    repair_rounds: int = 0
    tokens: int = 0
    prompt_tokens: int = 0
    seconds: float = 0.0
    model_seconds: float = 0.0
    gate_verdict: str = ""
    error: str = ""
    n_tests: int = 0
    tiers: List[TierLog] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "verdict_text": self.detail, "tier": self.tier, "model": TIER_MODELS.get(self.tier), "candidates": self.candidates,
                "repair_rounds": self.repair_rounds, "generated_tokens": self.tokens, "seconds": round(self.seconds, 2), "model_seconds": round(self.model_seconds, 2),
                "gate_verdict": self.gate_verdict, "tests": self.n_tests, "error": self.error,
                "tiers_tried": [{"tier": t.tier, "candidates": t.candidates, "repair_rounds": t.repair_rounds, "passed": t.passed, "last_error": t.last_error} for t in self.tiers],
                "caveat": CAVEAT, "code": self.code}


def parse_ladder(spec: Any) -> List[str]:
    items = [s.strip().lower() for s in (spec.split(",") if isinstance(spec, str) else list(spec))]
    out: List[str] = []
    for s in items:
        if s not in TIER_MODELS:
            raise GwayaLiteError(f"unknown tier {s!r}; choose from {', '.join(TIER_MODELS)}")
        if s not in out:
            out.append(s)
    if not out:
        raise GwayaLiteError("empty ladder")
    return out


def validate_tests(tests: Sequence[Any]) -> List[str]:
    """Only single-line ``assert ...`` statements are accepted as tests."""
    out: List[str] = []
    for t in tests:
        if not isinstance(t, str):
            raise GwayaLiteError("a test must be a string")
        t = t.strip()
        if not t or t.startswith("#"):
            continue
        if not re.match(r"assert\b", t) or "\n" in t:
            raise GwayaLiteError(f"only single-line 'assert ...' tests are accepted, got: {t[:60]!r}")
        if len(t) > MAX_TEST_CHARS:
            raise GwayaLiteError(f"a test is longer than {MAX_TEST_CHARS} characters")
        out.append(t)
    if len(out) > MAX_TESTS:
        raise GwayaLiteError(f"more than {MAX_TESTS} tests")
    return out


def _seed(prompt: str, idx: int) -> int:
    return int(hashlib.sha256(f"{prompt}|{idx}".encode()).hexdigest()[:8], 16) % (2 ** 31)


def repair_messages(messages: Sequence[Dict[str, str]], code: str, feedback: str) -> List[Dict[str, str]]:
    """Same text as scripts/e15_lib.py repair_messages (not imported: e15_lib pulls in torch)."""
    prev = f"```python\n{code}\n```" if code.strip() else "(no ```python code block was found in your previous reply)"
    user = (messages[1]["content"] + "\n\nYour previous attempt:\n" + prev + "\n\nVerification feedback: " + feedback +
            "\n\nFix the solution. Reply with exactly one ```python code block containing the complete corrected solution.")
    return [messages[0], {"role": "user", "content": user}]


class Assistant:
    """Runs the ladder.  Not thread-safe by design: callers serialise (the server holds a lock)."""

    def __init__(self, client: oll.Client, ladder: Sequence[str] = ("1.5b", "3b", "7b"), ctx: int = DEFAULT_CTX, max_rounds: int = 3, budget_mib: int = DEFAULT_BUDGET_MIB,
                 num_predict: int = DEFAULT_TOKENS, residency: Optional[Residency] = None, log: Callable[[str], None] = lambda m: None) -> None:
        self.client, self.ladder, self.ctx, self.max_rounds, self.num_predict, self.log = client, list(ladder), int(ctx), int(max_rounds), int(num_predict), log
        self.residency = residency or Residency(client, budget_mib, log=log)

    def _gate(self, code: str, tests: Sequence[str]) -> gate_mod.GateResult:
        return gate_mod.gate(code, list(tests), "", GATE_TIMEOUT_S, GATE_MEM_MB, True)

    def _gen(self, tier: str, messages: Sequence[Dict[str, str]], seed: int, sample: bool, acc: Dict[str, float], tl: TierLog) -> str:
        opts: Dict[str, Any] = {"num_ctx": self.ctx, "seed": seed, "num_predict": self.num_predict, "temperature": SAMPLE_T if sample else 0.0}
        if sample:
            opts["top_p"] = SAMPLE_TOP_P
        try:
            res = self.client.chat(TIER_MODELS[tier], messages, opts, KEEP_ALIVE)
        except oll.OllamaError as exc:
            if "not found" in str(exc):
                raise GwayaLiteError(f"model {TIER_MODELS[tier]} is not installed: ollama pull {TIER_MODELS[tier]}") from None
            raise
        tl.tokens += res["eval_count"]
        acc["tokens"] += res["eval_count"]
        acc["prompt_tokens"] += res["prompt_eval_count"]
        acc["model_s"] += (res["total_duration"] - res["load_duration"]) / 1e9
        return res["text"]

    def solve(self, prompt: str, tests: Sequence[str] = (), ladder: Optional[Sequence[str]] = None) -> Result:
        prompt = prompt.strip()
        if not prompt:
            raise GwayaLiteError("empty prompt")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise GwayaLiteError(f"prompt longer than {MAX_PROMPT_CHARS} characters")
        tests = validate_tests(tests)
        ladder = list(ladder) if ladder else self.ladder
        messages = data.build_messages(prompt, tests[0] if tests else None)
        t0 = time.monotonic()
        acc = {"tokens": 0, "prompt_tokens": 0, "model_s": 0.0}
        res = Result(verdict=FAILED, n_tests=len(tests))
        best_code = ""

        def finish(verdict: str, detail: str, tl: Optional[TierLog], code: str, gate_v: str, error: str) -> Result:
            res.verdict, res.detail, res.code, res.gate_verdict, res.error = verdict, detail, code, gate_v, error
            res.tier = tl.tier if tl else (ladder[-1] if ladder else "")
            res.candidates, res.repair_rounds = sum(t.candidates for t in res.tiers), sum(t.repair_rounds for t in res.tiers)
            res.tokens, res.prompt_tokens, res.model_seconds, res.seconds = int(acc["tokens"]), int(acc["prompt_tokens"]), acc["model_s"], time.monotonic() - t0
            return res

        for tier in ladder:
            self.residency.ensure(tier, self.ctx)
            tl = TierLog(tier)
            res.tiers.append(tl)
            last_text, last_gate = "", None

            def attempt(msgs: Sequence[Dict[str, str]], idx: int, sample: bool) -> Tuple[str, str, gate_mod.GateResult]:
                text = self._gen(tier, msgs, _seed(prompt, idx), sample, acc, tl)
                code = data.extract_code(text) or ""
                g = self._gate(code, tests)
                tl.candidates += 1
                tl.last_error = g.feedback() if g.verdict != gate_mod.ACCEPTED else ""
                return text, code, g

            def done_if_decided(code: str, g: gate_mod.GateResult) -> Optional[Result]:
                if g.verdict == gate_mod.ACCEPTED:
                    tl.passed = True
                    if tests:
                        return finish(VERIFIED, f"VERIFIED (zero-stub audit clean AND {len(tests)} test(s) passed in the sandbox)", tl, code, g.verdict, "")
                    return finish(AUDITED, "AUDITED (syntax + zero-stub audit only; NOT tested)", tl, code, g.verdict, "")
                if g.verdict == gate_mod.UNVERIFIED:
                    why = g.test_error or "gate unavailable"
                    return finish(UNVERIFIED, f"UNVERIFIED (the gate could not run, so nothing was checked: {why})", tl, code, g.verdict, why)
                return None

            # candidate 0 (greedy), then N_SAMPLES samples, lazily: stop at the first pass
            for idx in range(1 + N_SAMPLES):
                text, code, g = attempt(messages, idx, sample=idx > 0)
                if idx == 0:
                    last_text, last_gate, best_code = text, g, code  # best_code ends as the LAST tier's first (greedy) attempt
                out = done_if_decided(code, g)
                if out:
                    return out
            # repair rounds start from the greedy candidate and then follow the latest repair (as in E16)
            last_code = data.extract_code(last_text) or ""
            for r in range(1, self.max_rounds + 1):
                text, code, g = attempt(repair_messages(messages, last_code, last_gate.feedback()), 1 + N_SAMPLES + r, sample=False)
                tl.repair_rounds = r
                out = done_if_decided(code, g)
                if out:
                    return out
                last_code, last_gate = code, g
            self.log(f"tier {tier}: no candidate passed the gate ({tl.candidates} candidates, {tl.repair_rounds} repair rounds)")
        tl = res.tiers[-1]
        return finish(FAILED, f"FAILED (no candidate passed the gate after the ladder {','.join(ladder)}; last error: {tl.last_error or 'none'}; "
                              "the code shown is the last tier's first attempt and is NOT accepted)", tl, best_code, "REJECTED", tl.last_error)

    def close(self, keep_warm: bool = False) -> List[str]:
        return [] if keep_warm else self.residency.unload_all()


# ================================================================ rendering
def render_provenance(r: Result) -> str:
    lines = ["--- PROVENANCE (GWAYA-Lite) ---", f"verdict: {r.detail}", f"tier used: {r.tier} ({TIER_MODELS.get(r.tier, '?')})",
             f"candidates tried: {r.candidates} | repair rounds: {r.repair_rounds} | generated tokens: {r.tokens} | seconds: {r.seconds:.1f} (model {r.model_seconds:.1f})",
             f"gate: {r.gate_verdict or 'n/a'} | tests: {r.n_tests} | tiers: " + "; ".join(f"{t.tier}={t.candidates}c/{t.repair_rounds}r{'+' if t.passed else '-'}" for t in r.tiers),
             "sandbox: e15 namespace sandbox (NOT GWAYA's Bubblewrap oracle, no seccomp)", "NOTE: " + CAVEAT]
    return "\n".join(lines)


def render_reply(r: Result) -> str:
    """Chat content: the final code in a fenced python block followed by a short provenance footer."""
    code = r.code.strip("\n") if r.code.strip() else "# (no code block was produced)"
    return (f"```python\n{code}\n```\n\n---\nGWAYA-Lite: {r.detail}\ntier {r.tier} | candidates {r.candidates} | repair rounds {r.repair_rounds} | tokens {r.tokens} | {r.seconds:.1f}s\n"
            f"Note: {CAVEAT}")


def gwaya_field(r: Result) -> Dict[str, Any]:
    return {"verdict": r.verdict, "verdict_text": r.detail, "tier": r.tier, "candidates": r.candidates, "repair_rounds": r.repair_rounds, "seconds": round(r.seconds, 2),
            "generated_tokens": r.tokens, "gate_verdict": r.gate_verdict, "tests": r.n_tests, "caveat": CAVEAT}


# ================================================================ the server
TESTS_BLOCK = re.compile(r"```tests[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_tests_block(text: str) -> Tuple[str, List[str]]:
    """Strip every ```tests fenced block from ``text``; return (text without it, the assert lines inside)."""
    lines: List[str] = []
    for m in TESTS_BLOCK.finditer(text):
        lines += m.group(1).splitlines()
    return TESTS_BLOCK.sub("", text).strip(), lines


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _is_loopback_host(h: str) -> bool:
    h = h.strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _hostname_of(host_header: str) -> str:
    h = host_header.strip()
    if h.startswith("["):
        return h[1:].split("]")[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


class Gateway:
    """The OpenAI-compatible gateway state (assistant + lock + auth + logging)."""

    def __init__(self, assistant: Assistant, api_key: Optional[str] = None, log_prompts: bool = False, loopback_only: bool = True,
                 log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> None:
        self.assistant, self.api_key, self.log_prompts, self.loopback_only, self.log = assistant, api_key, log_prompts, loopback_only, log
        self.lock = threading.Lock()  # requests are serialised: one GPU job, bounded memory

    def authorised(self, header: str) -> bool:
        if not self.api_key:
            return True
        m = re.match(r"Bearer\s+(.+)$", header or "")
        return bool(m) and hmac.compare_digest(m.group(1).strip().encode(), self.api_key.encode())

    def chat(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        """(status, JSON) for a parsed /v1/chat/completions body.  Raises nothing; errors are OpenAI-style error objects."""
        def err(code: int, msg: str, typ: str = "invalid_request_error") -> Tuple[int, Dict[str, Any]]:
            return code, {"error": {"message": msg, "type": typ}}

        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return err(400, "body must be a JSON object with a non-empty 'messages' list")
        users = [m for m in body["messages"] if isinstance(m, dict) and m.get("role") == "user"]
        if not users:
            return err(400, "no user message")
        prompt, block_tests = extract_tests_block(_content_text(users[-1].get("content")))
        ext = body.get("gwaya") or {}
        if not isinstance(ext, dict):
            return err(400, "'gwaya' must be an object")
        try:
            raw_tests = list(block_tests) + list(ext.get("tests") or [])
            tests = validate_tests(raw_tests)
            ladder = parse_ladder(ext["ladder"]) if ext.get("ladder") else None
            mt = body.get("max_tokens", body.get("max_completion_tokens", DEFAULT_TOKENS))
            if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
                return err(400, "max_tokens must be a positive integer")
        except (GwayaLiteError, TypeError) as exc:
            return err(400, str(exc))
        with self.lock:
            a = self.assistant
            saved = a.num_predict
            a.num_predict = min(mt, MAX_TOKENS_CAP)
            try:
                r = a.solve(prompt, tests, ladder)
            except GwayaLiteError as exc:
                return err(400 if "prompt" in str(exc) or "empty" in str(exc) else 502, str(exc))
            except oll.OllamaError as exc:
                return err(502, f"Ollama error: {exc}", "upstream_error")
            except Exception as exc:  # noqa: BLE001  never leak internals / prompts
                self.log(f"internal error: {type(exc).__name__}")
                return err(500, "internal error", "server_error")
            finally:
                a.num_predict = saved
        self.log(f"chat verdict={r.verdict} tier={r.tier} candidates={r.candidates} rounds={r.repair_rounds} tokens={r.tokens} seconds={r.seconds:.1f} tests={r.n_tests}")
        if self.log_prompts:
            self.log("prompt=" + json.dumps(prompt) + " tests=" + json.dumps(tests) + " code=" + json.dumps(r.code))
        return 200, {"id": "chatcmpl-gwaya-" + secrets.token_hex(6), "object": "chat.completion", "created": int(time.time()), "model": SERVED_MODEL_ID,
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": render_reply(r)}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.tokens, "total_tokens": r.prompt_tokens + r.tokens}, "gwaya": gwaya_field(r)}


def sse_chunks(obj: Dict[str, Any], size: int = 40) -> List[bytes]:
    """The SSE frames (bytes) of a finished completion: role delta, content deltas of ``size`` chars, the stop chunk (carrying ``gwaya``), [DONE]."""
    content = obj["choices"][0]["message"]["content"]
    base = {"id": obj["id"], "object": "chat.completion.chunk", "created": obj["created"], "model": obj["model"]}

    def frame(delta: Dict[str, Any], finish: Optional[str], extra: Optional[Dict[str, Any]] = None) -> bytes:
        d = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}], **(extra or {}))
        return ("data: " + json.dumps(d) + "\n\n").encode()

    frames = [frame({"role": "assistant", "content": ""}, None)]
    frames += [frame({"content": content[i:i + size]}, None) for i in range(0, len(content), size)]
    frames.append(frame({}, "stop", {"gwaya": obj["gwaya"], "usage": obj["usage"]}))
    frames.append(b"data: [DONE]\n\n")
    return frames


def make_handler(gw: Gateway) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "GwayaLite/1"

        def log_message(self, *a: Any) -> None:  # request lines are logged by us, metadata only
            del a

        def _send(self, code: int, obj: Dict[str, Any]) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b)

        def _guard(self) -> bool:
            """Host / Origin / auth checks.  Returns True when the request may proceed."""
            if gw.loopback_only:  # anti DNS-rebinding and anti drive-by-from-a-web-page
                if not _is_loopback_host(_hostname_of(self.headers.get("Host", ""))):
                    self._send(403, {"error": {"message": "Host header is not loopback", "type": "forbidden"}})
                    return False
                origin = self.headers.get("Origin")
                if origin is not None and not _is_loopback_host(_hostname_of(re.sub(r"^[a-z]+://", "", origin))):
                    self._send(403, {"error": {"message": "cross-origin request refused", "type": "forbidden"}})
                    return False
            if not gw.authorised(self.headers.get("Authorization", "")):
                self._send(401, {"error": {"message": "missing or wrong Bearer token", "type": "invalid_api_key"}})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if not self._guard():
                return
            if self.path == "/health":
                self._send(200, {"status": "ok", "service": "gwaya-lite", "ladder": gw.assistant.ladder, "ctx": gw.assistant.ctx, "auth": bool(gw.api_key)})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": SERVED_MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}]})
            else:
                self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self) -> None:  # noqa: N802
            if not self._guard():
                return
            if self.path != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            if self.headers.get("Transfer-Encoding"):
                return self._send(411, {"error": {"message": "Content-Length required (chunked bodies are not supported)", "type": "invalid_request_error"}})
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._send(411, {"error": {"message": "Content-Length required", "type": "invalid_request_error"}})
            if n < 0 or n > MAX_BODY_BYTES:
                return self._send(413, {"error": {"message": f"request body limited to {MAX_BODY_BYTES} bytes", "type": "invalid_request_error"}})
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            code, obj = gw.chat(body)
            if code == 200 and isinstance(body, dict) and body.get("stream") is True:
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for f in sse_chunks(obj):
                        self.wfile.write(f)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self._send(code, obj)

    return Handler


def check_bind(host: str, allow_nonlocal: bool, api_key_file: Optional[str]) -> Optional[str]:
    """Returns the API key to enforce (or None).  Refuses a non-loopback bind unless --allow-nonlocal AND --api-key-file are both given."""
    key = None
    if api_key_file:
        try:
            key = Path(api_key_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise GwayaLiteError(f"cannot read --api-key-file: {exc.strerror}") from None
        if len(key) < 16:
            raise GwayaLiteError("the API key must be at least 16 characters")
    if not _is_loopback_host(host):
        if not (allow_nonlocal and key):
            raise GwayaLiteError(f"refusing to bind to non-loopback address {host!r}: pass BOTH --allow-nonlocal and --api-key-file (Bearer auth is then required)")
    return key


def build_server(gw: Gateway, host: str, port: int) -> ThreadingHTTPServer:
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), make_handler(gw))


# ================================================================ CLI
class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors exit 3 (2 means UNVERIFIED)
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise SystemExit(EXIT_USAGE)


class _Terminated(SystemExit):
    pass


class signal_guard:
    """SIGTERM -> SystemExit(143) and SIGINT -> KeyboardInterrupt in the main thread so that ``finally`` blocks (model unload) always run."""

    def __enter__(self) -> "signal_guard":
        self.old: Dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            def term(signum: int, frame: Any) -> None:
                raise _Terminated(128 + signum)

            for s in (signal.SIGTERM, signal.SIGINT):
                self.old[s] = signal.signal(s, term)
        return self

    def __exit__(self, *a: Any) -> None:
        for s, h in self.old.items():
            signal.signal(s, h)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ladder", default=DEFAULT_LADDER, help="comma list of tiers, small first (default %(default)s)")
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX, help="Ollama num_ctx (default %(default)s)")
    p.add_argument("--max-rounds", type=int, default=3, help="repair rounds per tier (default %(default)s)")
    p.add_argument("--max-resident-mib", type=int, default=DEFAULT_BUDGET_MIB, help="GPU budget for resident Ollama models (default %(default)s)")
    p.add_argument("--ollama-url", default=oll.DEFAULT_URL, help="local Ollama URL (default %(default)s; non-local URLs are refused)")


def build_parser() -> argparse.ArgumentParser:
    ap = Parser(prog="gwaya_lite.py", description="GWAYA-Lite: small-first Qwen2.5-Coder ladder behind a fail-closed gate (Ollama, <8 GB VRAM).")
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=Parser)
    a = sub.add_parser("ask", help="solve one coding task")
    a.add_argument("prompt")
    a.add_argument("--tests", metavar="FILE", help="file of single-line assert statements")
    a.add_argument("--test", action="append", default=[], help="one assert statement (repeatable)")
    a.add_argument("--json", action="store_true")
    a.add_argument("--keep-warm", action="store_true", help="leave the last model resident (default: unload on exit)")
    a.add_argument("--max-tokens", type=int, default=DEFAULT_TOKENS)
    _common(a)
    s = sub.add_parser("serve", help="localhost OpenAI-compatible gateway")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8099)
    s.add_argument("--allow-nonlocal", action="store_true")
    s.add_argument("--api-key-file", help="file holding the Bearer token (>= 16 chars); required with --allow-nonlocal")
    s.add_argument("--log-prompts", action="store_true", help="also log prompts, tests and code (default: metadata only)")
    _common(s)
    d = sub.add_parser("doctor", help="environment check")
    _common(d)
    return ap


def cmd_ask(args: argparse.Namespace, client: Optional[oll.Client] = None) -> int:
    tests = list(args.test)
    if args.tests:
        try:
            tests += Path(args.tests).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise GwayaLiteError(f"cannot read --tests file: {exc.strerror}") from None
    tests = validate_tests(tests)
    ladder = parse_ladder(args.ladder)
    client = client or oll.Client(args.ollama_url)
    asst = Assistant(client, ladder, args.ctx, args.max_rounds, args.max_resident_mib, min(args.max_tokens, MAX_TOKENS_CAP),
                     log=lambda m: print("[gwaya-lite] " + m, file=sys.stderr))
    try:
        with signal_guard():
            r = asst.solve(args.prompt, tests)
    finally:
        asst.close(args.keep_warm)
    if args.json:
        print(json.dumps(r.as_dict(), indent=2))
    else:
        print(r.code.strip("\n") if r.code.strip() else "# (no code)")
        print()
        print(render_provenance(r))
    return EXIT_CODES[r.verdict]


def cmd_serve(args: argparse.Namespace, client: Optional[oll.Client] = None, ready: Optional[Callable[[ThreadingHTTPServer], None]] = None) -> int:
    key = check_bind(args.host, args.allow_nonlocal, args.api_key_file)
    ladder = parse_ladder(args.ladder)
    client = client or oll.Client(args.ollama_url)
    log = lambda m: print(time.strftime("%H:%M:%S ") + m, file=sys.stderr, flush=True)  # noqa: E731
    asst = Assistant(client, ladder, args.ctx, args.max_rounds, args.max_resident_mib, log=log)
    gw = Gateway(asst, key, args.log_prompts, _is_loopback_host(args.host), log)
    httpd = build_server(gw, args.host, args.port)
    try:
        with signal_guard():
            log(f"gwaya-lite listening on http://{args.host}:{httpd.server_address[1]} ladder={','.join(ladder)} ctx={args.ctx} auth={'bearer' if key else 'none'} log_prompts={args.log_prompts}")
            if ready:
                ready(httpd)
            httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        log("shutting down")
    finally:
        httpd.server_close()
        asst.close()
    return 0


def cmd_doctor(args: argparse.Namespace, client: Optional[oll.Client] = None, out: Callable[[str], None] = print) -> int:
    ladder = parse_ladder(args.ladder)
    table = load_planning_table()
    rc = 0
    out("GWAYA-Lite doctor")
    client = client or oll.Client(args.ollama_url, timeout=15, retries=0)
    try:
        ver = client.version()
        tags = client.tags()
        out(f"  [ok]   Ollama reachable at {client.base} (version {ver or '?'})")
    except (oll.OllamaError, GwayaLiteError) as exc:
        out(f"  [FAIL] Ollama not reachable: {exc}")
        out("         start it (systemctl start ollama / ollama serve) and re-run")
        return EXIT_USAGE
    missing = [t for t in ladder if TIER_MODELS[t] not in tags]
    for t in ladder:
        out(f"  [{'ok  ' if t not in missing else 'MISS'}] model {TIER_MODELS[t]}" + ("" if t not in missing else f"  ->  ollama pull {TIER_MODELS[t]}"))
    rc = 2 if missing else rc
    ok = gate_mod.gate_available()
    out(f"  [{'ok  ' if ok else 'FAIL'}] gate / sandbox self-test: " + ("passed" if ok else f"FAILED ({gate_mod.availability_reason()}); every verdict would be UNVERIFIED"))
    rc = 2 if not ok else rc
    info = oll.gpu_info()
    try:
        used = oll.gpu_memory_used_mib()
        total = info.get("memory_total_mib")
        free = (total - used) if total else None
        biggest = max(need_mib(table, t, args.ctx) for t in ladder)
        out(f"  [{'ok  ' if (free is None or free >= biggest) else 'WARN'}] GPU {info.get('name')}: used {used} MiB, free {free if free is not None else '?'} MiB of {total}; largest tier needs about {biggest} MiB")
        if free is not None and free < biggest:
            out("         not enough free GPU memory for the largest tier right now (another job is using the card?)")
    except oll.OllamaError as exc:
        out(f"  [WARN] GPU memory not readable: {exc}")
    try:
        resident = [f"{e['name']} ({e['size_vram_mib']:.0f} MiB)" for e in client.ps()]
        out("  resident now: " + (", ".join(resident) if resident else "none"))
    except oll.OllamaError:
        pass
    out(f"  max-resident budget {args.max_resident_mib} MiB, ctx {args.ctx} (planning numbers from {REPORT_REL if REPORT_PATH.exists() else 'built-in E16 constants'}):")
    for st in plan_sequence(ladder, args.ctx, args.max_resident_mib, table):
        out(f"    load {st['load']:>4} ({st['need_mib']} MiB): unload first {st['unload_first'] or 'nothing'}; resident after {st['resident_after']} = {st['total_mib']} MiB"
            + ("" if st["fits"] else "  ** over budget **"))
    out("  verdicts: VERIFIED = audit clean + your tests pass (NOT 'correct': 5-7% failed hidden tests in E16); AUDITED = no tests, audit only; UNVERIFIED = gate cannot run; FAILED.")
    out("status: " + ("ready" if rc == 0 else "NOT ready (see above)"))
    return rc


def main(argv: Optional[Sequence[str]] = None, client: Optional[oll.Client] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.cmd == "ask":
            return cmd_ask(args, client)
        if args.cmd == "serve":
            return cmd_serve(args, client)
        return cmd_doctor(args, client)
    except (GwayaLiteError, oll.OllamaError) as exc:
        print(f"gwaya_lite: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
