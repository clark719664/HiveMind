"""Logged model review of the foundation article (HM-S9-09, pilot).

This is the opposite of the old scripted "peer review": a real language model is called, through
AutoevolveAI's ``anse.core.api_extractor.APIExtractor`` so that the call is logged (JSONL), and the
full prompt and the full response are committed verbatim under ``paper/review/``. The script
emits **no score and no verdict**; it records what the model said, which model, which prompt,
and how long it took. A local 7B model is a weak reviewer; the transcript says so.

Run (from the HiveMind checkout):
  PYTHONPATH=src python scripts/logged_model_review.py --autoevolve-root ../../.. --model qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEX = PROJECT_ROOT / "paper" / "hivemind_foundation_article.tex"
OUT_DIR = PROJECT_ROOT / "paper" / "review"


def _section(tex: str, title_start: str) -> str:
    i = tex.find(title_start)
    if i < 0:
        return ""
    j = tex.find("\\section{", i + 10)
    return tex[i : j if j > 0 else len(tex)]


def _strip_latex(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\(texttt|textbf|textit|emph|path|url|cite|ref|label)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}$]", "", s)
    return re.sub(r"[ \t]+", " ", s)


def build_prompt(tex: str) -> str:
    abstract = _strip_latex(tex[tex.find("\\begin{abstract}") : tex.find("\\end{abstract}")])
    research = _strip_latex(_section(tex, "\\section{Research Program"))[:9000]
    limits = _strip_latex(_section(tex, "\\section{Limitations"))[:5000]
    return (
        "You are reviewing a preprint on decentralized Mixture-of-Experts training over volunteer consumer GPUs. "
        "Read the abstract, the research-program section and the limitations section below. Then answer, in plain prose, "
        "exactly these four questions, each in at most 120 words:\n"
        "1. Which single claim in the abstract is least supported by the evidence the paper itself describes?\n"
        "2. Which of the eight predictions (RQ1-RQ8) is most likely to be refuted, and why?\n"
        "3. Name one control that is missing or weaker than it looks in the pre-registered experiments.\n"
        "4. What would you need to see before believing the thesis?\n"
        "Do not give a numeric score, do not say accept or reject.\n\n"
        f"=== ABSTRACT ===\n{abstract}\n\n=== RESEARCH PROGRAM ===\n{research}\n\n=== LIMITATIONS ===\n{limits}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autoevolve-root", type=Path, default=PROJECT_ROOT.parent.parent.parent)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout-s", type=float, default=2400.0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.autoevolve_root.resolve()))
    from anse.core.api_extractor import APIExtractor  # noqa: E402  (logged LLM client)

    tex = TEX.read_text(encoding="utf-8")
    prompt = build_prompt(tex)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    call_log = OUT_DIR / "llm_calls.jsonl"
    client = APIExtractor(model_name=args.model, call_log_path=call_log, ollama_native=True, timeout_s=args.timeout_s)
    client._base = args.base_url.rstrip("/")  # local Ollama
    t0 = time.perf_counter()
    text, record = client.extract(prompt, max_new_tokens=args.max_tokens, temperature=0.2)
    elapsed = time.perf_counter() - t0
    stamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    out: Dict[str, Any] = {
        "kind": "logged_model_review_pilot",
        "is_peer_review": False,
        "disclosure": (
            "A single local language model was asked four open questions about the paper. This is not a peer review, "
            "carries no score, and a 7B coder model is a weak reviewer; it is recorded so that any future claim of model "
            "review has a transcript behind it."
        ),
        "model": args.model, "base_url": args.base_url, "elapsed_seconds": round(elapsed, 2),
        "tex_sha256": hashlib.sha256(tex.encode()).hexdigest(),
        "prompt": prompt, "response": text,
        "call_record": (dataclasses.asdict(record) if dataclasses.is_dataclass(record) else str(record)),
        "call_log_path": str(call_log.relative_to(PROJECT_ROOT)),
        "timestamp_utc": stamp,
    }
    path = OUT_DIR / f"logged_model_review_{args.model.replace(':', '_').replace('/', '_')}_{stamp}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)} ({len(text)} chars in {elapsed:.1f}s)")
    print(text[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
