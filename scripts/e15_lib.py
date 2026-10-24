"""E15 helpers: model/tokenizer loading, batched generation, the GWAYA inference loop, RFT data collection, LoRA training + merge, statistics, stage metering.

Reuses the E13 hand-rolled LoRA (``e13_swarm_lib.LoRALinear / attach_lora / reset_lora_all``; ``peft`` is not installed, no bitsandbytes).  The gate
(``e15_gate``) is the only place candidate code is executed, always in the sandbox.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import e13_swarm_lib as lib  # noqa: E402  (LoRALinear, attach_lora, reset_lora_all, configure_determinism, LOSS_SCALE_FP16)
import e15_data as data  # noqa: E402
import e15_gate as gate_mod  # noqa: E402
from e15_data import Problem  # noqa: E402

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
GIB = 1024 ** 3
VRAM_CAP_GIB, VRAM_LIMIT_GIB = 7.5, 8.0

# ---------------------------------------------------------------- registered constants (all enter the design hash)
LORA_R, LORA_ALPHA = 16, 32.0
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_INIT_SEED = 15
DESIGN: Dict[str, Any] = {
    "model": MODEL_REPO, "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": list(LORA_TARGETS), "init": "PEFT-scale (A uniform +-1/sqrt(fan_in), B=0)", "init_seed": LORA_INIT_SEED},
    "c0": {"decode": "greedy", "max_new": 512},
    "c1": {"candidates": ["greedy", "sample x2 (T=0.7, top_p=0.95)"], "visible_gate_only": True, "repair_rounds": 3, "repair_decode": "greedy", "fallback": "first greedy candidate"},
    "rft": {"n_samples": 8, "T": 0.8, "top_p": 0.95, "gate": "ALL tests", "keep_per_problem": 2, "dedup": "normalised AST, prefer shorter", "fail_fast_min_problems": 150},
    "train": {"epochs": 3, "lr": 2e-4, "schedule": "cosine", "warmup": 10, "wd": 0.0, "micro_batch": 4, "grad_accum": 4, "max_seq": 768, "grad_ckpt": True,
              "loss": "assistant tokens only", "amp": "fp16 autocast, static loss scale 256, finite checks", "adamw_betas": [0.9, 0.999]},
    "vram_cap_gib": VRAM_CAP_GIB, "vram_limit_gib": VRAM_LIMIT_GIB,
    "gate": {"timeout_s": 10, "mem_mb": 2048, "auditor_sha256": gate_mod.AUDITOR_SHA256},
    "data": data.EXPECTED, "seeds": "torch.manual_seed(sha256 of the (task_id, sample_index) keys of the batch)",
}
DESIGN_HASH = hashlib.sha256(json.dumps(DESIGN, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p (binomial, p=0.5) from the discordant counts b (A pass, B fail) and c (A fail, B pass)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, Any]:
    """Paired comparison B vs A on the same items: difference in pass rate, discordant counts, exact McNemar p."""
    assert len(a) == len(b)
    n = len(a)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {"n": n, "diff": (sum(b) - sum(a)) / n if n else float("nan"), "b_only": only_b, "a_only": only_a, "discordant": only_a + only_b, "mcnemar_p": mcnemar_exact(only_a, only_b)}


# ---------------------------------------------------------------- tokenizers
class ByteTok:
    """Smoke-only byte-level tokenizer (vocab 256): pad 0, eos 1."""

    pad_id, eos_ids = 0, [1]

    def encode_prompt(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>"
        return [min(b, 255) for b in text.encode("utf-8")]

    def encode_completion(self, text: str) -> List[int]:
        return [b for b in text.encode("utf-8")] + [1]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i > 1).decode("utf-8", "replace")


class HFTok:
    def __init__(self, tok: Any) -> None:
        self.tok = tok
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        ends = [i for i in (im_end, tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0]
        self.eos_ids = sorted(set(ends))
        self.im_end = im_end

    def encode_prompt(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        text = self.tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode_completion(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] + [self.im_end]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)


def snapshot(repo: str) -> str:
    hits = sorted(glob.glob(str(HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / "*")))
    hits = [h for h in hits if os.path.exists(os.path.join(h, "config.json"))]
    if not hits:
        raise FileNotFoundError(f"{repo} is not (completely) in the HF cache")
    return hits[0]


def load_base(kind: str, device: str) -> Tuple[nn.Module, Any]:
    """Plain (no LoRA) eval-mode model + tokenizer wrapper.  kind: 'coder1.5b' (fp16 on CUDA, fp32 on CPU) or 'tiny' (smoke)."""
    if kind == "tiny":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=4096, tie_word_embeddings=True)
        model = Qwen2ForCausalLM(cfg).to(device)
        model.train(False)
        return model, ByteTok()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = snapshot(MODEL_REPO)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device)
    model.train(False)
    return model, HFTok(AutoTokenizer.from_pretrained(path))


# ---------------------------------------------------------------- stage metering (wall, GPU seconds, peak reserved, NVML energy)
class Energy:
    """NVML power x time proxy (whole GPU, i.e. including any other process on it)."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval, self.wh, self._stop, self._t = interval, 0.0, threading.Event(), None
        self.ok = False
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml

            pynvml.nvmlInit()
            self._nv, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("E15_NVML_INDEX", "0")))
            self.ok = True
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            try:
                self.wh += self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0 * (now - last) / 3600.0
            except Exception:  # noqa: BLE001
                pass
            last = now

    def start(self) -> None:
        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def stop(self) -> Optional[float]:
        if not self.ok:
            return None
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        return self.wh


class Meter:
    """Context manager: wall seconds, peak reserved GiB (CUDA), energy Wh."""

    def __init__(self, device: str) -> None:
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.rec: Dict[str, Any] = {}
        self.energy = Energy() if self.cuda else None

    def __enter__(self) -> "Meter":
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.energy.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.cuda:
            torch.cuda.synchronize()
        self.rec = {"wall_s": time.perf_counter() - self.t0,
                    "peak_reserved_gib": (torch.cuda.max_memory_reserved() / GIB) if self.cuda else None,
                    "peak_allocated_gib": (torch.cuda.max_memory_allocated() / GIB) if self.cuda else None,
                    "energy_wh": self.energy.stop() if self.energy else None}


def cap_vram(device: str) -> Optional[float]:
    """Hard cap of the CUDA caching allocator: allocations beyond VRAM_CAP_GIB raise OutOfMemoryError instead of silently growing."""
    if device.startswith("cuda") and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GIB * GIB / total))
        return total / GIB
    return None


# ---------------------------------------------------------------- journals (jsonl, keyed; resume = skip keys present)
class Journal:
    def __init__(self, path: Path, resume: bool) -> None:
        self.path, self.recs = path, {}
        if path.exists():
            if resume:
                for line in path.read_text().splitlines():
                    if line.strip():
                        try:
                            r = json.loads(line)
                            self.recs[r["key"]] = r
                        except (ValueError, KeyError):
                            pass  # a torn last line is recomputed
            else:
                path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rec: Dict[str, Any]) -> None:
        self.recs[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- generation
def batch_seed(keys: Sequence[str]) -> int:
    return int.from_bytes(hashlib.sha256("|".join(keys).encode()).digest()[:4], "big")


class Gen:
    """Batched HF generation with left padding.  Counters: ``tokens``, ``seconds``."""

    def __init__(self, model: nn.Module, tk: Any, device: str, max_new: int = 512, bs: int = 16) -> None:
        self.model, self.tk, self.device, self.max_new, self.bs = model, tk, device, max_new, bs
        self.tokens, self.seconds, self.oom_retries, self.failed_batches = 0, 0.0, 0, 0

    @torch.no_grad()
    def _batch(self, prompts: List[List[int]], keys: List[str], sample: Optional[Tuple[float, float]]) -> List[Tuple[str, int]]:
        L = max(len(p) for p in prompts)
        pad = self.tk.pad_id
        ids = torch.full((len(prompts), L), pad, dtype=torch.long)
        mask = torch.zeros((len(prompts), L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
            mask[i, L - len(p):] = 1
        kw: Dict[str, Any] = dict(max_new_tokens=self.max_new, pad_token_id=pad, eos_token_id=self.tk.eos_ids, repetition_penalty=1.0, use_cache=True)
        if sample is None:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            kw.update(do_sample=True, temperature=sample[0], top_p=sample[1], top_k=0)
            torch.manual_seed(batch_seed(keys))
        out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **kw)[:, L:].cpu()
        res = []
        eos = set(self.tk.eos_ids)
        for row in out.tolist():
            n = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
            res.append((self.tk.decode(row[:n]), n))
        return res

    def run(self, items: Sequence[Tuple[str, List[int]]], sample: Optional[Tuple[float, float]]) -> Dict[str, Tuple[str, int]]:
        """items: (key, prompt_ids).  Returns key -> (text, generated token count).  Sorted by length for padding efficiency; halves the batch on OOM."""
        order = sorted(items, key=lambda kv: -len(kv[1]))
        out: Dict[str, Tuple[str, int]] = {}
        i, bs = 0, self.bs
        while i < len(order):
            chunk = order[i:i + bs]
            t0 = time.perf_counter()
            try:
                res = self._batch([p for _, p in chunk], [k for k, _ in chunk], sample)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs, self.oom_retries = max(1, bs // 2), self.oom_retries + 1
                continue
            except RuntimeError as exc:  # e.g. fp16 sampling produced a non-finite distribution: those items get an empty reply (counted)
                if "probability tensor" not in str(exc) and "nan" not in str(exc).lower():
                    raise
                self.failed_batches += 1
                res = [("", 0)] * len(chunk)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            self.seconds += time.perf_counter() - t0
            for (k, _), r in zip(chunk, res):
                out[k] = r
                self.tokens += r[1]
            i += len(chunk)
        return out


# ---------------------------------------------------------------- gating + judging
def gate_many(jobs: Sequence[Tuple[str, Sequence[str], str, bool]], workers: int = 6, timeout_s: float = 10.0) -> List[gate_mod.GateResult]:
    """jobs: (code, tests, setup, audit).  Order preserved."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda j: gate_mod.gate(j[0], j[1], j[2], timeout_s=timeout_s, audit=j[3]), jobs))


def code_of(text: str) -> str:
    return gate_mod.extract_code(text) or ""


def judge(problems: Sequence[Problem], texts: Sequence[str], workers: int = 6) -> List[Dict[str, Any]]:
    """Final judgement with ALL tests.  ``functional`` = tests pass (stub audit not applied); ``audit_clean`` = zero-stub audit clean; ``accepted`` = both (GWAYA ACCEPTED)."""
    codes = [code_of(t) for t in texts]
    res = gate_many([(c, p.all_tests, p.setup, False) for c, p in zip(codes, problems)], workers)
    out = []
    for p, c, r in zip(problems, codes, res):
        viol: List[str] = []
        if c.strip():
            try:
                ast.parse(c)
                viol = gate_mod.audit_stub(c)
            except SyntaxError:
                viol = ["SyntaxError"]
        out.append({"code": c, "functional": r.verdict == gate_mod.ACCEPTED, "verdict": r.verdict, "audit_clean": not viol and bool(c.strip()), "accepted": r.verdict == gate_mod.ACCEPTED and not viol,
                    "unverified": r.verdict == gate_mod.UNVERIFIED, "test_error": r.test_error[:200]})
    return out


def eval_greedy(gen: Gen, tk: Any, problems: Sequence[Problem], journal: Journal, label: str, workers: int = 6) -> List[Dict[str, Any]]:
    """C0 / C2: one greedy reply per problem, judged with ALL tests.  Per-problem records go to the journal (resume = skip done keys)."""
    todo = [p for p in problems if f"{label}:{p.key}" not in journal.recs]
    if todo:
        s0, t0 = gen.seconds, gen.tokens
        rep = gen.run([(p.key, tk.encode_prompt(p.messages)) for p in todo], None)
        j = judge(todo, [rep[p.key][0] for p in todo], workers)
        nsec = (gen.seconds - s0)
        for p, jr in zip(todo, j):
            journal.add({"key": f"{label}:{p.key}", "problem": p.key, "text": rep[p.key][0], "tokens": rep[p.key][1], "gen_seconds_share": nsec / len(todo), **jr})
    return [journal.recs[f"{label}:{p.key}"] for p in problems]


def repair_messages(p: Problem, code: str, feedback: str) -> List[Dict[str, str]]:
    prev = f"```python\n{code}\n```" if code.strip() else "(no ```python code block was found in your previous reply)"
    user = (p.messages[1]["content"] + "\n\nYour previous attempt:\n" + prev + "\n\nVerification feedback: " + feedback +
            "\n\nFix the solution. Reply with exactly one ```python code block containing the complete corrected solution.")
    return [p.messages[0], {"role": "user", "content": user}]


def gwaya_loop(gen: Gen, tk: Any, problems: Sequence[Problem], greedy: Sequence[Dict[str, Any]], journal: Journal, label: str, workers: int = 6,
               n_samples: int = 2, rounds: int = 3) -> List[Dict[str, Any]]:
    """C1 / C3.  Candidates = [greedy (reused from C0 / C2: same prompt, same decode), n_samples samples (T=0.7, top_p=0.95)]; each gated with the VISIBLE test only; the
    first gate-passing candidate wins; otherwise up to ``rounds`` greedy repair rounds from the previous candidate + gate feedback; otherwise the first greedy candidate."""
    todo = [(p, g) for p, g in zip(problems, greedy) if f"{label}:{p.key}" not in journal.recs]
    if todo:
        probs = [p for p, _ in todo]
        s0 = gen.seconds
        samp = gen.run([(f"{p.key}#c{j}", tk.encode_prompt(p.messages)) for p in probs for j in range(n_samples)], (0.7, 0.95))
        cands: Dict[str, List[Tuple[str, int]]] = {p.key: [(g["text"], g["tokens"])] + [samp[f"{p.key}#c{j}"] for j in range(n_samples)] for p, g in todo}
        vis = gate_many([(code_of(t), p.visible_tests, p.setup, True) for p in probs for t, _ in cands[p.key]], workers)
        state: Dict[str, Dict[str, Any]] = {}
        k = 0
        for p in probs:
            rs = vis[k:k + len(cands[p.key])]
            k += len(cands[p.key])
            win = next((i for i, r in enumerate(rs) if r.accepted), None)
            state[p.key] = {"win_idx": win, "rounds": 0, "chosen": cands[p.key][win][0] if win is not None else None, "tried": (win + 1) if win is not None else len(rs),
                            "last_text": cands[p.key][0][0], "last_res": rs[0], "tokens": sum(n for _, n in cands[p.key]), "vis_pass": win is not None, "source": "candidate" if win is not None else None}
        for r in range(1, rounds + 1):
            pend = [p for p in probs if state[p.key]["chosen"] is None]
            if not pend:
                break
            rep = gen.run([(f"{p.key}#fix{r}", tk.encode_prompt(repair_messages(p, code_of(state[p.key]["last_text"]), state[p.key]["last_res"].feedback()))) for p in pend], None)
            rr = gate_many([(code_of(rep[f"{p.key}#fix{r}"][0]), p.visible_tests, p.setup, True) for p in pend], workers)
            for p, res in zip(pend, rr):
                st = state[p.key]
                text, n = rep[f"{p.key}#fix{r}"]
                st["rounds"], st["tokens"], st["tried"] = r, st["tokens"] + n, st["tried"] + 1
                st["last_text"], st["last_res"] = text, res
                if res.accepted:
                    st["chosen"], st["vis_pass"], st["source"] = text, True, f"repair{r}"
        final_texts = [state[p.key]["chosen"] if state[p.key]["chosen"] is not None else cands[p.key][0][0] for p in probs]
        j = judge(probs, final_texts, workers)
        nsec = gen.seconds - s0
        for p, text, jr in zip(probs, final_texts, j):
            st = state[p.key]
            journal.add({"key": f"{label}:{p.key}", "problem": p.key, "text": text, "tokens": st["tokens"], "candidates_tried": st["tried"], "repair_rounds": st["rounds"],
                         "verified_visible": st["vis_pass"], "source": st["source"] or "fallback_first_greedy", "gen_seconds_share": nsec / len(probs), **jr})
    return [journal.recs[f"{label}:{p.key}"] for p in problems]


# ---------------------------------------------------------------- RFT data collection
def norm_ast(code: str) -> str:
    """Normalised AST dump (docstrings removed, no positions): identical up to formatting / comments / docstrings."""
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def select_kept(codes: Sequence[str], k: int = 2) -> List[str]:
    """At most ``k`` distinct (normalised AST) solutions, shorter first."""
    seen, kept = set(), []
    for c in sorted(codes, key=lambda s: (len(s), s)):
        try:
            key = norm_ast(c)
        except SyntaxError:
            continue
        if key not in seen:
            seen.add(key)
            kept.append(c)
        if len(kept) == k:
            break
    return kept


def collect_rft(gen: Gen, tk: Any, train: Sequence[Problem], journal: Journal, n_samples: int, chunk_problems: int, workers: int, smoke_inject: bool = False,
                progress: Callable[[str], None] = print, after_chunk: Optional[Callable[[int, int, float], None]] = None) -> Dict[str, Any]:
    t_start = time.time()
    todo = [p for p in train if f"rft:{p.key}" not in journal.recs]
    for ci in range(0, len(todo), chunk_problems):
        ps = todo[ci:ci + chunk_problems]
        rep = gen.run([(f"{p.key}#r{j}", tk.encode_prompt(p.messages)) for p in ps for j in range(n_samples)], (0.8, 0.95))
        flat = [(p, j, rep[f"{p.key}#r{j}"]) for p in ps for j in range(n_samples)]
        texts = [t for _, _, (t, _) in flat]
        if smoke_inject:  # smoke only: plumbing needs accepted samples from a random-init model; the reference replaces sample 0
            texts = [(f"```python\n{p.reference}\n```" if j == 0 else t) for (p, j, (t, _)), t in zip(flat, texts)]
        res = gate_many([(code_of(t), p.all_tests, p.setup, True) for (p, _, _), t in zip(flat, texts)], workers)
        k = 0
        for p in ps:
            rs, ts = res[k:k + n_samples], texts[k:k + n_samples]
            toks = [flat[k + j][2][1] for j in range(n_samples)]
            k += n_samples
            acc = [code_of(t) for t, r in zip(ts, rs) if r.accepted]
            journal.add({"key": f"rft:{p.key}", "problem": p.key, "n_samples": n_samples, "accepted": len(acc), "yield": len(acc) / n_samples, "kept": select_kept(acc, 2),
                         "verdicts": [r.verdict for r in rs], "tokens": sum(toks), "unverified": sum(r.verdict == gate_mod.UNVERIFIED for r in rs)})
        progress(f"[E15 rft] {min(ci + chunk_problems, len(todo))}/{len(todo)} problems  {time.time() - t_start:.0f}s")
        if after_chunk is not None:
            after_chunk(min(ci + chunk_problems, len(todo)), len(todo), time.time() - t_start)  # may raise (registered time fail-fast)
    recs = [journal.recs[f"rft:{p.key}"] for p in train]
    return {"problems": len(recs), "problems_with_accepted": sum(1 for r in recs if r["accepted"] >= 1), "mean_yield": sum(r["yield"] for r in recs) / max(1, len(recs)),
            "kept_pairs": sum(len(r["kept"]) for r in recs), "unverified_samples": sum(r["unverified"] for r in recs), "tokens": sum(r["tokens"] for r in recs)}


# ---------------------------------------------------------------- LoRA training / merge
def build_pairs(train: Sequence[Problem], tk: Any, journal: Journal, max_seq: int) -> Tuple[List[Tuple[List[int], List[int]]], Dict[str, int]]:
    pairs, dropped = [], 0
    for p in train:
        for code in journal.recs[f"rft:{p.key}"]["kept"]:
            pi, ci = tk.encode_prompt(p.messages), tk.encode_completion(f"```python\n{code.strip(chr(10))}\n```")
            if len(pi) + len(ci) > max_seq:
                dropped += 1
                continue
            pairs.append((pi, ci))
    return pairs, {"pairs": len(pairs), "dropped_too_long": dropped}


def _chunked_sum_ce(lm_head: nn.Module, h: torch.Tensor, tgt: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint

    def f(hh: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(lm_head(hh).float(), tt, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(f, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
    return total


def lora_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    st = {}
    for name, m in model.named_modules():
        if isinstance(m, lib.LoRALinear):
            st[f"{name}.lora_A"], st[f"{name}.lora_B"] = m.lora_A.detach().cpu().clone(), m.lora_B.detach().cpu().clone()
    return st


def lr_at(step: int, total: int, lr0: float, warmup: int) -> float:
    if step < warmup:
        return lr0 * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * lr0 * (1 + math.cos(math.pi * min(1.0, prog)))


def train_lora(model: nn.Module, tk: Any, pairs: Sequence[Tuple[List[int], List[int]]], device: str, amp: bool, cfg: Dict[str, Any], seed: int = 15,
               progress: Callable[[str], None] = print) -> Dict[str, Any]:
    """Attach r=16 LoRA to ``model`` (in place), train on (prompt, completion) pairs with loss on completion tokens only."""
    n_wrapped = lib.attach_lora(model, LORA_TARGETS, LORA_R, LORA_ALPHA)
    lib.reset_lora_all(model, LORA_INIT_SEED)
    params = lora_params(model)
    model.config.use_cache = False
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(True)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["adamw_betas"]), weight_decay=cfg["wd"])
    mb, accum = cfg["micro_batch"], cfg["grad_accum"]
    per_step = mb * accum
    steps_per_epoch = math.ceil(len(pairs) / per_step)
    total = steps_per_epoch * cfg["epochs"]
    scale = lib.LOSS_SCALE_FP16 if amp else 1.0
    pad = tk.pad_id
    step, skipped, losses, tokens_seen, t0 = 0, 0, [], 0, time.perf_counter()
    for ep in range(cfg["epochs"]):
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        order = torch.randperm(len(pairs), generator=g).tolist()
        ep_loss, ep_tok = 0.0, 0
        for s in range(steps_per_epoch):
            idx = order[s * per_step:(s + 1) * per_step]
            ntgt = sum(len(pairs[i][1]) for i in idx)
            for gparam in params:
                gparam.grad = None
            step_loss = 0.0
            for m0 in range(0, len(idx), mb):
                sub = [pairs[i] for i in idx[m0:m0 + mb]]
                L = max(len(a) + len(b) for a, b in sub)
                ids = torch.full((len(sub), L), pad, dtype=torch.long)
                att = torch.zeros((len(sub), L), dtype=torch.long)
                lab = torch.full((len(sub), L), -100, dtype=torch.long)
                for r, (a, b) in enumerate(sub):
                    seq = a + b
                    ids[r, :len(seq)], att[r, :len(seq)] = torch.tensor(seq), 1
                    lab[r, len(a):len(seq)] = torch.tensor(b)
                ids, att, lab = ids.to(device), att.to(device), lab.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
                    sel = lab[:, 1:] != -100  # position t predicts token t+1
                    loss_sum = _chunked_sum_ce(model.lm_head, h[:, :-1][sel], lab[:, 1:][sel])
                loss = loss_sum / max(1, ntgt - 0)
                (loss * scale).backward()
                step_loss += float(loss_sum)
                tokens_seen += int(att.sum())
            if scale != 1.0:
                for gparam in params:
                    if gparam.grad is not None:
                        gparam.grad.div_(scale)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step, total, cfg["lr"], cfg["warmup"])
            if bool(torch.isfinite(gn)):
                opt.step()
            else:
                skipped += 1
            ep_loss += step_loss
            ep_tok += ntgt
            step += 1
        losses.append(ep_loss / max(1, ep_tok))
        progress(f"[E15 train] epoch {ep + 1}/{cfg['epochs']} mean target-token CE {losses[-1]:.4f} steps {step}/{total} skipped {skipped}")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_disable()
    model.train(False)
    return {"wrapped_linears": n_wrapped, "lora_params": int(sum(p.numel() for p in params)), "steps": step, "steps_per_epoch": steps_per_epoch, "skipped_nonfinite_steps": skipped,
            "epoch_mean_ce": losses, "tokens_processed": tokens_seen, "seconds": time.perf_counter() - t0, "pairs": len(pairs)}


@torch.no_grad()
def merge_lora_into_base(model: nn.Module, state: Dict[str, torch.Tensor], rank: int = LORA_R, alpha: float = LORA_ALPHA) -> int:
    """W += (alpha / r) B A, in place on a PLAIN model (a freshly loaded fp16 copy of the base weights); returns the number of merged matrices.  Computed in fp32, stored in W's dtype."""
    s, n = alpha / rank, 0
    mods = dict(model.named_modules())
    for key in state:
        if not key.endswith(".lora_A"):
            continue
        name = key[: -len(".lora_A")]
        lin = mods[name]
        assert isinstance(lin, nn.Linear), name
        A, B = state[key].to(lin.weight.device, torch.float32), state[name + ".lora_B"].to(lin.weight.device, torch.float32)
        lin.weight.copy_((lin.weight.float() + s * (B @ A)).to(lin.weight.dtype))
        n += 1
    return n
