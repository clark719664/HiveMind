#!/usr/bin/env python3
"""E17 stage 'sample': rejection sampling from the SERVED 7B (Ollama native API, through an ssh tunnel) on the MBPP TRAIN problems.

Preregistered in artifacts/experiments/E17_router_7b_rft_prereg.json. 8 samples per train problem (T=0.8, top_p=0.95), each judged by
scripts/e15_gate.py with ALL tests; at most 2 distinct accepted solutions are kept. Journal is resumable. Controls run first.
Usage: python scripts/e17_sample.py --ollama http://127.0.0.1:11435 --out DIR [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15_data as D  # noqa: E402
import e15_gate as G  # noqa: E402

MODEL = "qwen2.5-coder:7b-instruct"
SAMPLES, KEEP, FAIL_FAST_MIN = 8, 2, 150


def digest(client: httpx.Client, base: str, model: str) -> str:
    tags = client.get(f"{base}/api/tags", timeout=30).json()["models"]
    return next(m["digest"] for m in tags if m["name"] == model)


def chat(client: httpx.Client, base: str, model: str, messages: list[dict[str, str]], *, temperature: float, top_p: float, seed: int) -> str:
    body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature, "top_p": top_p, "seed": seed, "num_predict": 700}}
    r = client.post(f"{base}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return str(r.json()["message"]["content"])


def norm(code: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", code.strip()).encode()).hexdigest()


def controls(train: list[D.Problem], evals: list[D.Problem]) -> dict:
    refs = [p for p in train if p.reference][:100]
    ok = sum(1 for p in refs if G.gate(p.reference, p.all_tests, p.setup).accepted)
    stub = G.gate("def f(x):\n    pass\n", refs[0].all_tests, refs[0].setup)
    leak = D.assert_no_leakage(train, evals)
    out = {"references_accepted": ok, "references_tested": len(refs), "references_ok": ok >= 0.95 * len(refs), "stub_rejected": not stub.accepted, "no_leakage": leak}
    if not (out["references_ok"] and out["stub_rejected"]):
        raise SystemExit(f"control failed: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11435")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, evals, _he, _meta = D.load_all(check_counts=True)
    problems = train[: args.limit] if args.limit else train
    client = httpx.Client()
    d0 = digest(client, args.ollama, MODEL)
    (out / "controls.json").write_text(json.dumps({"base_digest": d0, **controls(train, evals)}, indent=1))
    print("controls ok; base digest", d0[:12], flush=True)
    journal = out / "samples_journal.jsonl"
    done: dict[str, dict] = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec
    lock, sem = threading.Lock(), threading.Semaphore(args.workers)

    def work(p: D.Problem) -> None:
        with sem:
            msgs = D.build_messages(p.user_text, p.visible_tests[0] if p.visible_tests else None)
            accepted: list[str] = []
            seen: set[str] = set()
            tried = 0
            for i in range(SAMPLES):
                try:
                    text = chat(client, args.ollama, MODEL, msgs, temperature=0.8, top_p=0.95, seed=1000 + i)
                except httpx.HTTPError:
                    continue
                tried += 1
                code = G.extract_code(text)
                if code is None or norm(code) in seen:
                    continue
                seen.add(norm(code))
                if G.gate(code, p.all_tests, p.setup).accepted and len(accepted) < KEEP:
                    accepted.append(code)
                if len(accepted) >= KEEP:
                    break  # the kept set is the first KEEP accepted in sample order, so stopping here changes nothing but the cost
            rec = {"key": p.key, "task_id": p.task_id, "tried": tried, "accepted": accepted}
            with lock:
                with journal.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[p.key] = rec

    todo = [p for p in problems if p.key not in done]
    threads = [threading.Thread(target=work, args=(p,)) for p in todo]
    t0 = time.time()
    for t in threads:
        t.start()
    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        if len(done) - last >= 10:
            print(f"{len(done)}/{len(problems)} problems, {round(time.time() - t0)}s", flush=True)
            last = len(done)
    solved = sum(1 for p in problems if done.get(p.key, {}).get("accepted"))
    pairs = [{"key": p.key, "messages": D.build_messages(p.user_text, p.visible_tests[0] if p.visible_tests else None), "completion": "```python\n" + c.strip("\n") + "\n```"}
             for p in problems for c in done.get(p.key, {}).get("accepted", [])]
    (out / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pairs))
    summary = {"problems": len(problems), "problems_with_accepted": solved, "pairs": len(pairs), "wall_s": round(time.time() - t0), "fail_fast_min": FAIL_FAST_MIN}
    (out / "sample_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 3 if (not args.limit and solved < FAIL_FAST_MIN) else 0


if __name__ == "__main__":
    sys.exit(main())
