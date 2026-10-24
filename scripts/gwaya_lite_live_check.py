"""Live deployment check of the running GWAYA-Lite gateway on the T4; writes artifacts/gwaya_lite_live_check.json (generated, not typed).

Calls ONLY the already-running localhost gateway (default http://127.0.0.1:8099) and read-only host probes (nvidia-smi, Ollama /api/ps):
  * /health and /v1/models;
  * three coding tasks with user tests through /v1/chat/completions (non-stream): verdict, tier, candidates, repair rounds, tokens, seconds;
  * one SSE request (stream=true): number of chunks and the [DONE] terminator;
  * a request without tests (must say AUDITED, never VERIFIED);
  * foreign Host and Origin headers (must be 403) and the listening address;
  * GPU memory and resident Ollama models after the requests.
Nothing else is touched; the gateway must already be running (``python3 scripts/gwaya_lite.py serve``).

    python3 scripts/gwaya_lite_live_check.py [--url http://127.0.0.1:8099]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "gwaya_lite_live_check.json"

TASKS = [
    ("is_palindrome", "Write a Python function is_palindrome(s) that returns True if the string s reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
     ["assert is_palindrome('A man, a plan, a canal: Panama') == True", "assert is_palindrome('race a car') == False", "assert is_palindrome('') == True", "assert is_palindrome('No lemon, no melon') == True"]),
    ("longest_valid_parentheses", "Write a Python function longest_valid_parentheses(s) that returns the length of the longest valid (well-formed) parentheses substring of s, which contains only '(' and ')'.",
     ["assert longest_valid_parentheses('(()') == 2", "assert longest_valid_parentheses(')()())') == 4", "assert longest_valid_parentheses('') == 0", "assert longest_valid_parentheses('()(()') == 2",
      "assert longest_valid_parentheses('()(())') == 6", "assert longest_valid_parentheses('(()())') == 6", "assert longest_valid_parentheses('))))((()((') == 2"]),
    ("fizzbuzz", "Write a Python function fizzbuzz(n) returning a list of strings for 1..n: Fizz for multiples of 3, Buzz for 5, FizzBuzz for both, else the number as a string.",
     ["assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']", "assert fizzbuzz(15)[-1] == 'FizzBuzz'", "assert fizzbuzz(0) == []"]),
]


def post(url: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def get(url: str, path: str, headers: dict | None = None) -> tuple:
    req = urllib.request.Request(url + path, headers=headers or {})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    a = ap.parse_args()
    rep: dict = {"schema_version": "hivemind-gwaya-lite-live-check/v1.0", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": a.url}
    rep["health"] = get(a.url, "/health")
    rep["models"] = get(a.url, "/v1/models")
    rep["tasks"] = []
    for name, prompt, tests in TASKS:
        t0 = time.perf_counter()
        d = post(a.url, {"model": "gwaya-lite", "messages": [{"role": "user", "content": prompt + "\n```tests\n" + "\n".join(tests) + "\n```"}]})
        g = d.get("gwaya", {})
        rep["tasks"].append({"task": name, "tests": len(tests), "wall_s": round(time.perf_counter() - t0, 2), "verdict": g.get("verdict"), "tier": g.get("tier"), "candidates": g.get("candidates"),
                             "repair_rounds": g.get("repair_rounds"), "generated_tokens": g.get("generated_tokens"), "gate_verdict": g.get("gate_verdict"),
                             "code_sha256": hashlib.sha256(d["choices"][0]["message"]["content"].encode()).hexdigest()[:16]})
    d = post(a.url, {"model": "gwaya-lite", "messages": [{"role": "user", "content": "Write a Python function add(a, b) returning the sum of two numbers."}]})
    rep["untested_request"] = {"verdict": d.get("gwaya", {}).get("verdict"), "never_verified_without_tests": d.get("gwaya", {}).get("verdict") != "VERIFIED"}
    host, port = a.url.split("//")[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=300)
    conn.request("POST", "/v1/chat/completions", json.dumps({"model": "gwaya-lite", "stream": True, "messages": [{"role": "user", "content": "Write a Python function double(x) returning 2*x.\n```tests\nassert double(2) == 4\n```"}]}),
                 {"Content-Type": "application/json"})
    raw = conn.getresponse().read().decode()
    rep["sse"] = {"data_chunks": raw.count("data: ") - raw.count("data: [DONE]"), "done_terminator": "data: [DONE]" in raw}
    rep["foreign_host_status"] = get(a.url, "/health", {"Host": "evil.example.com"})[0]
    rep["foreign_origin_status"] = get(a.url, "/health", {"Origin": "https://evil.example.com"})[0]
    rep["listening"] = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout.count(f"{host}:{port}") > 0 and "0.0.0.0:" + port not in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    rep["gpu_memory_used_mib_after"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    try:
        rep["ollama_resident_after"] = [(m["name"], round(m["size_vram"] / 2**20)) for m in json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5))["models"]]
    except Exception as exc:
        rep["ollama_resident_after"] = repr(exc)[:120]
    rep["all_ok"] = bool(rep["health"][0] == 200 and all(t["verdict"] == "VERIFIED" for t in rep["tasks"]) and rep["untested_request"]["never_verified_without_tests"] and rep["sse"]["done_terminator"]
                         and rep["foreign_host_status"] == 403 and rep["foreign_origin_status"] == 403)
    rep["receipt_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in rep.items()}, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("tasks", "untested_request", "sse", "foreign_host_status", "foreign_origin_status", "listening", "gpu_memory_used_mib_after", "ollama_resident_after", "all_ok")}, indent=1))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
