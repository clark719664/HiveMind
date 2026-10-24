"""E16: a small stdlib-only Ollama client (urllib) plus GPU memory readers (nvidia-smi / pynvml).

* Talks only to localhost (127.0.0.1, ::1, localhost); a non-local base URL is refused.  Environment proxies are ignored.
* Timeouts and at most 2 retries (connection errors, timeouts, HTTP 5xx); HTTP 4xx is not retried.  Error messages carry the path and the HTTP status, never headers
  or request bodies, and this module never reads or prints any token / credential.
* ``chat`` returns the reply text and the Ollama timing fields (nanoseconds, as reported); ``ps`` the loaded models with ``size_vram``; ``unload`` asks for
  ``keep_alive: 0`` and verifies through ``ps``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_URL = "http://127.0.0.1:11434"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
TIMING_FIELDS = ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration", "load_duration")


class OllamaError(RuntimeError):
    pass


def check_local(url: str) -> str:
    u = urllib.parse.urlparse(url)
    if u.scheme != "http" or (u.hostname or "") not in tuple(h.strip("[]") for h in LOCAL_HOSTS):
        raise OllamaError(f"refusing non-local Ollama URL {u.scheme}://{u.hostname}: only http://127.0.0.1 / localhost is allowed")
    return url.rstrip("/")


class Client:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600.0, retries: int = 2, sleep: Callable[[float], None] = time.sleep) -> None:
        self.base = check_local(base_url)
        self.timeout, self.retries, self._sleep = timeout, min(2, max(0, retries)), sleep
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost traffic through a proxy

    # ------------------------------------------------------------ transport
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        last: Optional[str] = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
            try:
                with self._opener.open(req, timeout=timeout or self.timeout) as r:
                    raw = r.read()
                try:
                    return json.loads(raw.decode("utf-8")) if raw.strip() else {}
                except ValueError as exc:
                    raise OllamaError(f"{method} {path}: invalid JSON in the response ({exc})") from None
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    msg = ""
                    try:
                        msg = json.loads(exc.read().decode("utf-8")).get("error", "")
                    except Exception:  # noqa: BLE001
                        pass
                    raise OllamaError(f"{method} {path}: HTTP {exc.code} {msg}".strip()) from None
                last = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
            if attempt < self.retries:
                self._sleep(1.0 + attempt)
        raise OllamaError(f"{method} {path} failed after {self.retries + 1} attempts ({last}); is the Ollama daemon running on {self.base}?")

    # ------------------------------------------------------------ API
    def version(self) -> str:
        return str(self._request("GET", "/api/version", timeout=10).get("version", ""))

    def tags(self) -> List[str]:
        return [str(m.get("name") or m.get("model")) for m in self._request("GET", "/api/tags", timeout=30).get("models", [])]

    def ps(self) -> List[Dict[str, Any]]:
        out = []
        for m in self._request("GET", "/api/ps", timeout=30).get("models", []):
            size, vram = int(m.get("size", 0) or 0), int(m.get("size_vram", 0) or 0)
            out.append({"name": str(m.get("name") or m.get("model")), "size": size, "size_vram": vram, "size_mib": size / 2 ** 20, "size_vram_mib": vram / 2 ** 20,
                        "gpu_pct": (100.0 * vram / size) if size > 0 else None, "context_length": m.get("context_length"), "expires_at": m.get("expires_at")})
        return out

    def chat(self, model: str, messages: Sequence[Dict[str, str]], options: Optional[Dict[str, Any]] = None, keep_alive: Any = "2m") -> Dict[str, Any]:
        """One non-streaming chat completion.  ``options``: num_ctx, temperature, top_p, seed, num_predict.  Returns text + timing fields (ns) + done_reason."""
        allowed = ("num_ctx", "temperature", "top_p", "seed", "num_predict")
        opts = {k: v for k, v in (options or {}).items() if k in allowed}
        body: Dict[str, Any] = {"model": model, "messages": list(messages), "stream": False, "options": opts}
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        t0 = time.monotonic()
        d = self._request("POST", "/api/chat", body)
        if "error" in d:
            raise OllamaError(f"chat {model}: {str(d['error'])[:200]}")
        msg = d.get("message") or {}
        res: Dict[str, Any] = {"text": str(msg.get("content", "")), "done_reason": d.get("done_reason"), "model": d.get("model", model), "wall_s": time.monotonic() - t0}
        for f in TIMING_FIELDS:
            res[f] = int(d.get(f, 0) or 0)
        return res

    def unload(self, model: str, wait_s: float = 20.0, poll_s: float = 0.5) -> bool:
        """keep_alive 0 for ``model``, then poll ``ps`` until it is gone.  Returns True when verified gone, False otherwise (never raises on a still-loaded model)."""
        try:
            self._request("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False}, timeout=60)
        except OllamaError:
            pass  # e.g. the model is unknown / already gone: the ps check below decides
        deadline = time.monotonic() + wait_s
        while True:
            if all(m["name"] != model for m in self.ps()):
                return True
            if time.monotonic() >= deadline:
                return False
            self._sleep(poll_s)

    def unload_all(self, wait_s: float = 20.0) -> Dict[str, Any]:
        loaded = [m["name"] for m in self.ps()]
        res = {name: self.unload(name, wait_s) for name in loaded}
        return {"was_loaded": loaded, "unloaded": res, "still_loaded": [m["name"] for m in self.ps()]}


# ---------------------------------------------------------------- GPU readers
def _smi(args: List[str], timeout: float = 20.0) -> str:
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise OllamaError(f"nvidia-smi failed (rc {r.returncode}): {r.stderr.strip()[:200]}")
    return r.stdout


def gpu_memory_used_mib(index: Optional[int] = None) -> int:
    """memory.used of GPU ``index`` (default env E16_GPU_INDEX or 0), MiB.  nvidia-smi first, pynvml as a fallback."""
    idx = int(os.environ.get("E16_GPU_INDEX", "0")) if index is None else index
    try:
        out = _smi(["-i", str(idx), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(float(out.strip().splitlines()[0]))
    except (OllamaError, OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(idx))
            return int(info.used // 2 ** 20)
        except Exception:  # noqa: BLE001
            raise OllamaError(f"cannot read GPU memory (nvidia-smi: {exc})") from None


def gpu_info(index: Optional[int] = None) -> Dict[str, Any]:
    idx = int(os.environ.get("E16_GPU_INDEX", "0")) if index is None else index
    try:
        out = _smi(["-i", str(idx), "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]).strip().splitlines()[0]
        name, total, drv = [x.strip() for x in out.split(",")]
        return {"name": name, "memory_total_mib": int(float(total)), "driver": drv}
    except Exception:  # noqa: BLE001
        return {"name": None, "memory_total_mib": None, "driver": None}


def gpu_compute_processes() -> List[Dict[str, Any]]:
    """GPU compute processes (all GPUs): pid, process_name, used_memory_mib.  Empty list when none."""
    out = _smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            mem = int(float(parts[-1]))
        except ValueError:
            mem = -1
        procs.append({"pid": int(parts[0]), "process_name": ",".join(parts[1:-1]), "used_memory_mib": mem})
    return procs


def is_ollama_process(p: Dict[str, Any]) -> bool:
    return "ollama" in str(p.get("process_name", "")).lower()
