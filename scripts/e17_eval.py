#!/usr/bin/env python3
"""E17 stage 'eval': MBPP-sanitized (257 held-out) for ONE served model through Ollama's native API (temperature 0, fixed seed).

Writes <out>/<tag>_journal.jsonl with, per problem: first-attempt verdict (ALL tests) and verified-loop verdict (visible-test gate, up to 3
repairs, the policy of scripts/hivemind_verified.py), then <tag>_summary.json. Also stores the first-attempt code for the adapter-applied
control. Use --determinism N to run the first N problems twice and compare first attempts.
Usage: python scripts/e17_eval.py --ollama http://127.0.0.1:11435 --model qwen2.5-coder:7b-instruct --tag B --out DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e15_data as D  # noqa: E402
import e15_gate as G  # noqa: E402


def load_verified():
    for cand in (HERE / "hivemind_verified.py", *sorted(HERE.parent.parent.glob("wt-verified/scripts/hivemind_verified.py"))):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("hivemind_verified_e17", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("scripts/hivemind_verified.py not found (PR #153)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="1 with a one-slot llama-server keeps decoding deterministic")
    ap.add_argument("--determinism", type=int, default=0)
    ap.add_argument("--backend", choices=("ollama", "llamacpp"), default="llamacpp", help="amendment 1: llamacpp = llama-server OpenAI endpoint, base or base+--lora")
    args = ap.parse_args()
    V = load_verified()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _tr, evals, _he, _m = D.load_all(check_counts=True)
    probs = evals[: args.limit] if args.limit else evals
    client = httpx.Client()

    def ask_for(model: str):
        def ask(messages):
            if args.backend == "llamacpp":
                body = {"model": model, "messages": messages, "temperature": 0, "seed": 7, "max_tokens": 700, "cache_prompt": False}  # a cached prefix changes the numerics of the next identical request
                r = client.post(f"{args.ollama}/v1/chat/completions", json=body, timeout=600)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0, "seed": 7, "num_predict": 700}}
            r = client.post(f"{args.ollama}/api/chat", json=body, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]
        return ask

    ask = ask_for(args.model)
    ref = probs[0]
    assert G.gate(ref.reference, ref.all_tests, ref.setup).accepted and not G.gate("def f(x):\n    pass\n", ref.all_tests, ref.setup).accepted, "gate controls failed"
    if args.determinism:
        diffs = 0
        for p in probs[: args.determinism]:
            m = D.build_messages(p.user_text, p.visible_tests[0])
            diffs += ask(m) != ask(m)
        (out / f"{args.tag}_determinism.json").write_text(json.dumps({"checked": args.determinism, "nondeterministic": diffs}))
        print("determinism: nondeterministic", diffs, "of", args.determinism, flush=True)
    journal = out / f"{args.tag}_journal.jsonl"
    done = {json.loads(l)["key"] for l in journal.read_text().splitlines()} if journal.exists() else set()
    lock, sem = threading.Lock(), threading.Semaphore(args.workers)

    def work(p) -> None:
        with sem:
            msgs = D.build_messages(p.user_text, p.visible_tests[0])
            first_text = ask(msgs)
            first = G.extract_code(first_text)
            c0 = G.gate(first, p.all_tests, p.setup).verdict if first else "NO_CODE"
            res = V.verified_solve(ask, msgs, V.sandbox_judge(p.visible_tests, p.setup))
            final = res.code if res.code else first
            c1 = G.gate(final, p.all_tests, p.setup).verdict if final else "NO_CODE"
            rec = {"key": p.key, "c0": c0, "c1": c1, "loop_status": res.status, "rounds": res.rounds, "first_code": first}
            with lock:
                journal.open("a").write(json.dumps(rec) + "\n")

    th = [threading.Thread(target=work, args=(p,)) for p in probs if p.key not in done]
    t0 = time.time()
    [t.start() for t in th]
    while any(t.is_alive() for t in th):
        time.sleep(30)
        n = sum(1 for _ in journal.open()) if journal.exists() else 0
        print(f"{n}/{len(probs)} {round(time.time()-t0)}s", flush=True)
    recs = [json.loads(l) for l in journal.read_text().splitlines()]
    n = len(recs)
    summ = {"tag": args.tag, "model": args.model, "n": n, "first_passes": sum(r["c0"] == "ACCEPTED" for r in recs), "loop_passes": sum(r["c1"] == "ACCEPTED" for r in recs),
            "unverified": sum("UNVERIFIED" in (r["c0"], r["c1"]) for r in recs), "wall_s": round(time.time() - t0)}
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
