"""E16 test double: an in-process fake Ollama HTTP server (random localhost port) with canned MBPP-like answers, realistic timing fields and a fake GPU-memory reader.

Used by ``run_experiment_E16_gwaya_lite_t4.py --smoke`` and by tests/test_e16_*.py.  It never touches the GPU or the real daemon.
Endpoints: GET /api/version, /api/tags, /api/ps; POST /api/chat, /api/generate (keep_alive 0 unloads).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

WEIGHTS_MIB = {"1.5b": 940, "3b": 1841, "7b": 4466}
KV_KIB = {"1.5b": 28, "3b": 36, "7b": 56}
TPS = {"1.5b": 120.0, "3b": 70.0, "7b": 35.0}
ABILITY = {"1.5b": 0.45, "3b": 0.65, "7b": 0.85}
CUDA_CTX_MIB, COMPUTE_BUF_MIB = 300, 150


def tier_of(model: str) -> str:
    for t in ("1.5b", "3b", "7b"):
        if t in model:
            return t
    return "3b"


def _u(*parts: Any) -> float:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:6], "big") / float(1 << 48)


class FakeOllama:
    def __init__(self, answers: Dict[str, str], baseline_mib: int = 400, cpu_offload: bool = False, stuck: Sequence[str] = (), leak_mib: int = 0,
                 ability: Optional[Dict[str, float]] = None, preloaded: Sequence[Tuple[str, int]] = (), missing: Sequence[str] = (), flaky_mod: int = 0,
                 repair_ability: float = 0.6, tps: Optional[Dict[str, float]] = None, fail_chat_after: Optional[int] = None, wrong: Optional[str] = None) -> None:
        self.wrong = wrong  # force the wrong-answer flavor: 'cheat' (passes the visible test only) | 'stub' | 'zero' | 'nocode'
        self.answers, self.baseline, self.cpu_offload, self.stuck, self.leak = dict(answers), baseline_mib, cpu_offload, set(stuck), leak_mib
        self.ability, self.repair_ability, self.flaky_mod = dict(ABILITY, **(ability or {})), repair_ability, flaky_mod
        self.tps = dict(TPS, **(tps or {}))
        self.missing = set(missing)
        self.fail_chat_after = fail_chat_after
        self.loaded: Dict[str, Dict[str, Any]] = {}
        self.ever_unloaded: Set[str] = set()
        self.calls: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.n_chat = 0
        self._leaked = False
        for name, ctx in preloaded:
            self.loaded[name] = {"ctx": ctx, "filled": 0}
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> str:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:  # silence
                pass

            def _send(self, code: int, obj: Dict[str, Any]) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self) -> None:  # noqa: N802
                with outer.lock:
                    if self.path == "/api/version":
                        self._send(200, {"version": "0.0.0-fake"})
                    elif self.path == "/api/tags":
                        self._send(200, {"models": [{"name": n, "model": n} for n in outer.tag_names()]})
                    elif self.path == "/api/ps":
                        self._send(200, {"models": outer.ps_models()})
                    else:
                        self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                try:
                    body = json.loads(self.rfile.read(n).decode() or "{}")
                except ValueError:
                    return self._send(400, {"error": "bad json"})
                with outer.lock:
                    if self.path == "/api/chat":
                        code, obj = outer.chat(body)
                    elif self.path == "/api/generate":
                        code, obj = outer.generate(body)
                    else:
                        code, obj = 404, {"error": "not found"}
                self._send(code, obj)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    # ------------------------------------------------------------ state
    def tag_names(self) -> List[str]:
        return [n for n in ("qwen2.5-coder:1.5b", "qwen2.5-coder:3b", "qwen2.5-coder:7b-instruct", "other:1b") if n not in self.missing]

    def _model_bytes(self, name: str, ctx: int) -> Tuple[int, int]:
        t = tier_of(name)
        size = (WEIGHTS_MIB[t] + KV_KIB[t] * ctx // 1024) * 2 ** 20
        return size, (int(size * 0.6) if self.cpu_offload else size)

    def ps_models(self) -> List[Dict[str, Any]]:
        out = []
        for name, st in self.loaded.items():
            size, vram = self._model_bytes(name, st["ctx"])
            out.append({"name": name, "model": name, "size": size, "size_vram": vram, "context_length": st["ctx"], "expires_at": "2099-01-01T00:00:00Z"})
        return out

    def gpu_read(self) -> int:
        """Fake nvidia-smi memory.used (MiB)."""
        with self.lock:
            used = self.baseline + (self.leak if self._leaked else 0)
            for name, st in self.loaded.items():
                t = tier_of(name)
                _, vram = self._model_bytes(name, st["ctx"])
                used += vram // 2 ** 20 + CUDA_CTX_MIB + COMPUTE_BUF_MIB + (32 if st["filled"] else 0)
            return used

    def procs(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [{"pid": 9000 + i, "process_name": "/usr/local/bin/ollama", "used_memory_mib": 100} for i, _ in enumerate(self.loaded)]

    # ------------------------------------------------------------ API
    def generate(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        model = body.get("model", "")
        if body.get("keep_alive") == 0:
            if model in self.stuck:
                return 200, {"model": model, "done": True}
            if model in self.loaded:
                del self.loaded[model]
                self.ever_unloaded.add(model)
                self._leaked = self.leak > 0
            return 200, {"model": model, "done": True, "done_reason": "unload"}
        return 400, {"error": "fake supports only keep_alive 0 on /api/generate"}

    def _answer(self, model: str, content: str, temperature: float, seed: int) -> str:
        key = next((k for k in self.answers if k in content), None)
        if key is None:
            return "OK"
        ref = self.answers[key]
        repair = "Verification feedback:" in content
        kind = "repair" if repair else ("greedy" if temperature == 0 else f"sample{seed}")
        extra = ""
        if kind == "greedy" and self.flaky_mod and int(_u("flaky", key) * 1e6) % self.flaky_mod == 0:
            extra = str(self.n_chat)
        u = _u(model, key, kind, hashlib.sha256(content.encode()).hexdigest()[:8] if repair else "", extra)
        ok_p = self.repair_ability if repair else self.ability[tier_of(model)]
        if u < ok_p:
            code = ref
        else:
            m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", ref)
            name, args = (m.group(1), m.group(2)) if m else ("f", "x")
            v = _u("flavor", model, key, kind, extra)
            if self.wrong:
                v = {"cheat": 0.0, "stub": 0.5, "zero": 0.8, "nocode": 0.95}[self.wrong]
            tm = re.search(r"Your code should pass this test:\s*assert\s+.*?==\s*(.+?)\s*$", content.split("Your previous attempt:")[0], flags=re.S | re.M)
            if v < 0.45 and tm:
                code = f"def {name}({args}):\n    return {tm.group(1).strip().splitlines()[0]}\n"  # passes the visible test, fails the others
            elif v < 0.75:
                code = f"def {name}({args}):\n    pass\n"
            elif v < 0.9:
                code = f"def {name}({args}):\n    return 0\n"
            else:
                return "I think the answer is simple, but here is no code."
        return f"```python\n{code.strip()}\n```"

    def chat(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        model = body.get("model", "")
        if model not in self.tag_names():
            return 404, {"error": f"model '{model}' not found"}
        self.n_chat += 1
        if self.fail_chat_after is not None and self.n_chat > self.fail_chat_after:
            return 500, {"error": "injected failure"}
        opts = body.get("options") or {}
        ctx = int(opts.get("num_ctx", 2048))
        msgs = body.get("messages") or []
        content = "\n".join(m.get("content", "") for m in msgs)
        load_ns = 0
        st = self.loaded.get(model)
        if st is None or st["ctx"] != ctx:
            self.loaded[model] = st = {"ctx": ctx, "filled": 0}
            load_ns = 1_500_000_000
        pcount = max(1, int(len(content) / 3.5))
        if pcount > ctx:
            pcount = ctx  # truncation, as the real server does
        if pcount > ctx * 0.5:
            st["filled"] = 1
        user = msgs[-1].get("content", "") if msgs else ""
        text = self._answer(model, user, float(opts.get("temperature", 0.8)), int(opts.get("seed", 0)))
        npred = int(opts.get("num_predict", 512))
        n_out = max(1, len(text) // 3)
        reason = "stop"
        if n_out > npred:
            n_out, reason = npred, "length"
            text = text[: max(1, npred * 3)] if npred > 1 else text[:1]
        tps = self.tps[tier_of(model)]
        ev_ns, pe_ns = int(n_out / tps * 1e9), int(pcount / 2000.0 * 1e9)
        self.calls.append({"model": model, "ctx": ctx, "temperature": opts.get("temperature"), "seed": opts.get("seed"), "pcount": pcount, "n_out": n_out})
        return 200, {"model": model, "message": {"role": "assistant", "content": text}, "done": True, "done_reason": reason, "total_duration": load_ns + ev_ns + pe_ns + 1_000_000,
                     "load_duration": load_ns, "prompt_eval_count": pcount, "prompt_eval_duration": pe_ns, "eval_count": n_out, "eval_duration": ev_ns}
