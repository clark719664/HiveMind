"""Experiment E16: GWAYA-Lite on a Tesla T4 (GWAYA draft PR #21, P1 and P2): measured VRAM, tokens/s and MBPP pass@1 (raw vs GWAYA loop) of the Qwen2.5-Coder
1.5B / 3B / 7B-instruct Q4_K_M tiers served by the local Ollama daemon.  Contract: artifacts/experiments/E16_gwaya_lite_t4_prereg.json (registered before any measurement).

Stages (each writes <out>/E16_gwaya_lite_t4_<stage>.json; journals <out>/E16_gwaya_lite_t4_journal_<name>.jsonl; --resume skips finished work):
  preflight  idle-GPU control (memory.used <= 1500 MiB, no loaded model; a loaded model is unloaded first), Ollama reachable, tiers present, gate + reference control
  probes     per tier x num_ctx: VRAM after load+1 token and after a prompt filling ~85% of the context, ollama ps, PR estimator, decode / prompt-eval tok/s (12 prompts)
  accuracy   MBPP-sanitized (n=257): raw R and GWAYA loop L per tier at num_ctx 4096, 7B raw at 8192, determinism repeat on 10 problems
  unload     every model unloaded, nvidia-smi back to within 300 MiB of the start baseline (runs right after the GPU stages, also after errors / SIGTERM)
  analysis   CPU only: pass@1 + Wilson, paired L-R (exact McNemar), H16a-H16f, P2 precision / recall, escalation ladder, determinism
Execution order: preflight, probes, accuracy, unload, analysis (the GPU is released before the CPU-only analysis).

    python3 scripts/run_experiment_E16_gwaya_lite_t4.py --estimate
    python3 scripts/run_experiment_E16_gwaya_lite_t4.py --smoke --out /tmp/e16_smoke            # fake in-process Ollama, synthetic problems, CPU only, not evidence
    python3 scripts/run_experiment_E16_gwaya_lite_t4.py --resume                                # the registered run (needs the prereg file and an idle GPU)

Exit codes: 0 done, 1 a stage failed, 2 cannot start, 3 'aborted by registered fail-fast rule' (idle-GPU control failed, 7B decode < 5 tok/s, projected time > --max-hours).
Final stdout line: 'E16 DONE ...'.  This script reports numbers; it never claims a result.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # E16 never needs CUDA in this process; Ollama owns the GPU

import argparse
import hashlib
import json
from pathlib import Path
import platform
import signal
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import e15_data as data  # noqa: E402
import e15_gate as gate_mod  # noqa: E402
import e15_lib as L  # noqa: E402
import e16_ollama as oll  # noqa: E402

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = EXP_DIR / "E16_gwaya_lite_t4_prereg.json"
PREFIX = "E16_gwaya_lite_t4"
STAGES = ("preflight", "probes", "accuracy", "unload", "analysis")
GPU_STAGES = ("preflight", "probes", "accuracy")

TIERS = {"1.5b": "qwen2.5-coder:1.5b", "3b": "qwen2.5-coder:3b", "7b": "qwen2.5-coder:7b-instruct"}
# PR #21 estimator: weights + kv_kib_per_token x ctx / 1024 + 600 MiB  (weights as assumed by the PR table, not the measured file sizes)
PR_WEIGHTS_MIB = {"1.5b": 1070, "3b": 1840, "7b": 4466}
KV_KIB_PER_TOKEN = {"1.5b": 28, "3b": 36, "7b": 56}
OVERHEAD_MIB = 600
ACC_CTX, CHECK_CTX = 4096, 8192
FILL_FRAC, FILL_BAND = 0.85, (0.80, 0.90)
MAX_NEW, PROBE_NEW, N_PROBE_PROMPTS, N_DET = 512, 256, 12, 10
IDLE_MAX_MIB, RETURN_TOL_MIB, VRAM_THRESHOLD_MIB = 1500, 300, 6656
FAILFAST_DECODE_TPS, H16F_DECODE_TPS = 5.0, 20.0
SAMPLE_T, SAMPLE_TOP_P, N_SAMPLES, N_REPAIR = 0.7, 0.95, 2, 3
REF_CONTROL_N, REF_CONTROL_MIN = 100, 0.95
KEEP_ALIVE = "2m"  # a crashed / killed run leaves nothing resident for longer than this
ESTIMATE_ASSUMED = {"decode_tps": {"1.5b": 110.0, "3b": 65.0, "7b": 32.0}, "avg_tokens_per_call": 180, "overhead_s_per_call": 0.4, "extra_calls_per_problem_in_loop": 0.91,
                    "load_s_per_phase": 10.0}

DESIGN: Dict[str, Any] = {
    "tiers": TIERS, "pr_weights_mib": PR_WEIGHTS_MIB, "kv_kib_per_token": KV_KIB_PER_TOKEN, "overhead_mib": OVERHEAD_MIB, "ctx": [ACC_CTX, CHECK_CTX], "fill_frac": FILL_FRAC,
    "probe": {"prompts": N_PROBE_PROMPTS, "num_predict": PROBE_NEW, "greedy": True}, "max_new": MAX_NEW,
    "loop": {"samples": N_SAMPLES, "T": SAMPLE_T, "top_p": SAMPLE_TOP_P, "repair_rounds": N_REPAIR, "visible_gate_only": True, "fallback": "first greedy candidate", "lazy_candidates": True},
    "thresholds": {"idle_mib": IDLE_MAX_MIB, "return_tol_mib": RETURN_TOL_MIB, "vram_mib": VRAM_THRESHOLD_MIB, "failfast_tps": FAILFAST_DECODE_TPS, "h16f_tps": H16F_DECODE_TPS},
    "gate": {"timeout_s": 10, "mem_mb": 2048, "auditor_sha256": gate_mod.AUDITOR_SHA256}, "data": data.EXPECTED, "det_problems": N_DET,
}
DESIGN_HASH = hashlib.sha256(json.dumps(DESIGN, sort_keys=True).encode()).hexdigest()


class FailFast(Exception):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


# ================================================================ pure helpers (unit-tested)
def estimator_mib(tier: str, ctx: int) -> int:
    return PR_WEIGHTS_MIB[tier] + KV_KIB_PER_TOKEN[tier] * ctx // 1024 + OVERHEAD_MIB


def seed_for(task_id: Any, idx: int) -> int:
    """Fixed seed from (task_id, sample_index); idx 0 greedy, 1..2 samples, 3..5 repair rounds."""
    return int.from_bytes(hashlib.sha256(f"E16|{task_id}|{idx}".encode()).digest()[:4], "big") & 0x7FFFFFFF


def model_seconds(call: Dict[str, Any]) -> float:
    """Model time of one call from the logged Ollama durations: total minus model load (ns -> s)."""
    return max(0, int(call.get("total_duration", 0)) - int(call.get("load_duration", 0))) / 1e9


def tok_per_s(count: int, dur_ns: int) -> Optional[float]:
    return (count / (dur_ns / 1e9)) if dur_ns > 0 and count > 0 else None


def median(xs: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return float(statistics.median(v)) if v else None


def call_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    return {k: res.get(k, 0) for k in oll.TIMING_FIELDS} | {"done_reason": res.get("done_reason")}


def build_fill_prompt(texts: Sequence[str], n_chars: int, nonce: int) -> str:
    """Deterministic long prompt from repeated MBPP problem texts.  The nonce at the very start defeats Ollama's prefix cache so prompt_eval_count is the real length."""
    head = f"[fill {nonce}] Below is a list of programming problems. Read them, then reply with the single word OK.\n"
    tail = "\nReply with the single word OK."
    body, size, i = [], len(head) + len(tail), 0
    while size < n_chars:
        s = f"Problem {i}: {texts[i % len(texts)]}\n"
        body.append(s)
        size += len(s)
        i += 1
    return (head + "".join(body))[: max(0, n_chars - len(tail))] + tail


# ---- hypotheses (exact registered rules; integer arithmetic at the boundaries)
def h16a(probes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ctx in (ACC_CTX, CHECK_CTX):
        r = next((p for p in probes if p["tier"] == "7b" and p["ctx"] == ctx), None)
        if r is None:
            out[str(ctx)] = {"status": "not_evaluated", "threshold_mib": VRAM_THRESHOLD_MIB}
        else:
            m = int(r["measured_mib"])
            out[str(ctx)] = {"measured_mib": m, "threshold_mib": VRAM_THRESHOLD_MIB, "status": "supported" if m <= VRAM_THRESHOLD_MIB else "refuted"}
    return out


def h16b(probes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for p in probes:
        m, e = int(p["measured_mib"]), int(p["estimator_mib"])
        rows.append({"tier": p["tier"], "ctx": p["ctx"], "measured_mib": m, "estimator_mib": e, "ratio": m / e if e else None, "in_band": 100 * m >= 80 * e and 100 * m <= 125 * e})
    if not rows:
        return {"status": "not_evaluated", "rows": []}
    ok = all(r["in_band"] for r in rows)
    return {"status": "adequate" if ok else "discrepancy", "band": [0.80, 1.25], "rows": rows, "complete": len(rows) == 6,
            "discrepancies": [{k: r[k] for k in ("tier", "ctx", "ratio")} for r in rows if not r["in_band"]]}


def h16c(pr: Dict[str, Any]) -> str:
    """L - R >= 0 -> supported; L - R < 0 with exact two-sided McNemar p < 0.05 -> refuted; otherwise inconclusive."""
    if pr["b_only"] >= pr["a_only"]:
        return "supported"
    return "refuted" if pr["mcnemar_p"] < 0.05 else "inconclusive"


def h16d(pr: Dict[str, Any]) -> str:
    """Improvement >= +5.0 points with p < 0.05 -> supported; <= 0 -> refuted; otherwise inconclusive."""
    n, d = pr["n"], pr["b_only"] - pr["a_only"]
    if n and d * 20 >= n and pr["mcnemar_p"] < 0.05:
        return "supported"
    if d <= 0:
        return "refuted"
    return "inconclusive"


def h16e(tp: int, verified: int) -> str:
    if verified == 0:
        return "inconclusive"
    return "supported" if 5 * tp >= 4 * verified else "refuted"


def h16f(probes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    r = next((p for p in probes if p["tier"] == "7b" and p["ctx"] == ACC_CTX), None)
    if r is None or r.get("decode_tps_median") is None:
        return {"status": "not_evaluated", "bar_tps": H16F_DECODE_TPS}
    v = float(r["decode_tps_median"])
    return {"decode_tps_median": v, "bar_tps": H16F_DECODE_TPS, "status": "met" if v >= H16F_DECODE_TPS else "not met"}


def gate_calibration(loop_recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """P2: VERIFIED verdict (audit clean + visible test pass) against the all-tests outcome, over every gated candidate of the loop runs, and over the chosen outputs."""
    tp = fp = fn = tn = 0
    for r in loop_recs:
        for c in r["candidates"]:
            v, f = bool(c["verified"]), bool(c["functional"])
            tp, fp, fn, tn = tp + (v and f), fp + (v and not f), fn + (f and not v), tn + (not v and not f)
    chosen = [r for r in loop_recs if r["verified_visible"]]
    ctp = sum(1 for r in chosen if r["functional"])
    lo, hi = L.wilson(tp, tp + fp) if tp + fp else (float("nan"), float("nan"))
    return {"candidates": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "verified": tp + fp, "precision": tp / (tp + fp) if tp + fp else None, "precision_wilson95": [lo, hi],
            "recall": tp / (tp + fn) if tp + fn else None, "chosen_verified_problems": len(chosen), "chosen_precision": ctp / len(chosen) if chosen else None,
            "status_H16e": h16e(tp, tp + fp)}


def ladder(l1: Dict[str, Dict[str, Any]], l3: Dict[str, Dict[str, Any]], l7: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Small-first ladder 1.5B-loop -> 3B-loop on FAILED (loop found no verified candidate) -> 7B-loop on FAILED, versus 7B-loop only.  Inputs: problem key -> loop record
    with ``verified_visible``, ``functional``, ``model_s``.  Cost = summed model seconds of every loop actually run (model load excluded)."""
    keys = sorted(l7)
    assert set(keys) == set(l1) == set(l3), "ladder needs the three loop runs on the same problems"
    final: List[bool] = []
    secs = esc3 = esc7 = 0
    cost = 0.0
    for k in keys:
        cost += l1[k]["model_s"]
        if l1[k]["verified_visible"]:
            final.append(bool(l1[k]["functional"]))
            continue
        esc3 += 1
        cost += l3[k]["model_s"]
        if l3[k]["verified_visible"]:
            final.append(bool(l3[k]["functional"]))
            continue
        esc7 += 1
        cost += l7[k]["model_s"]
        final.append(bool(l7[k]["functional"]))
    only7 = [bool(l7[k]["functional"]) for k in keys]
    n = len(keys)
    cost7 = sum(l7[k]["model_s"] for k in keys)
    return {"n": n, "ladder_pass_at_1": sum(final) / n if n else None, "ladder_passes": sum(final), "seven_b_only_pass_at_1": sum(only7) / n if n else None, "seven_b_only_passes": sum(only7),
            "escalated_to_3b": esc3, "escalated_to_7b": esc7, "escalation_rate_3b": esc3 / n if n else None, "escalation_rate_7b": esc7 / n if n else None,
            "ladder_model_seconds": cost, "seven_b_only_model_seconds": cost7, "ladder_over_7b_only_time": cost / cost7 if cost7 > 0 else None,
            "paired_ladder_vs_7b_only": L.paired(only7, final), "note": "descriptive, no status"}


def estimate(a: argparse.Namespace) -> Dict[str, Any]:
    A = ESTIMATE_ASSUMED
    n = 257 if not getattr(a, "limit", None) else int(a.limit)
    per_tier = {}
    total = 0.0
    for t in TIERS:
        tps = A["decode_tps"][t]
        call_s = A["avg_tokens_per_call"] / tps + A["overhead_s_per_call"]
        raw_s = n * call_s + A["load_s_per_phase"]
        loop_s = n * A["extra_calls_per_problem_in_loop"] * call_s
        det_s = N_DET * 2 * call_s
        probe_s = len(_ctx_list(a)) * (N_PROBE_PROMPTS * call_s + 40 + A["load_s_per_phase"])
        per_tier[t] = {"assumed_decode_tps": tps, "raw_s": round(raw_s), "loop_extra_s": round(loop_s), "determinism_s": round(det_s), "probes_s": round(probe_s)}
        total += raw_s + loop_s + det_s + probe_s
    extra7_8k = n * (A["avg_tokens_per_call"] / A["decode_tps"]["7b"] + A["overhead_s_per_call"]) + A["load_s_per_phase"]
    total += extra7_8k
    return {"problems": n, "tiers": list(TIERS), "assumed": A, "per_tier": per_tier, "seven_b_raw_8192_s": round(extra7_8k), "est_total_hours": round(total / 3600, 2),
            "max_hours_fail_fast": getattr(a, "max_hours", 3.0), "note": "ASSUMPTIONS (T4 decode rates and call counts), not measurements; gating on the CPU is small. The registered run re-projects with the "
            "measured decode tok/s and exits 3 if the projection exceeds --max-hours."}


def _ctx_list(a: argparse.Namespace) -> List[int]:
    return [int(x) for x in str(a.ctx).split(",") if x]


# ================================================================ context
class Ctx:
    def __init__(self, a: argparse.Namespace, client: oll.Client, mem_fn: Callable[[], int], procs_fn: Callable[[], List[Dict[str, Any]]], fast: bool) -> None:
        self.a, self.smoke, self.client, self.mem, self._procs = a, a.smoke, client, mem_fn, procs_fn
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
        bad = [t for t in self.tiers if t not in TIERS]
        if bad:
            raise ValueError(f"unknown tier(s) {bad}; known: {sorted(TIERS)}")
        self.ctxs = _ctx_list(a)
        self.fast = fast  # smoke / tests: no real waiting
        self.settle_s, self.t_wait, self.poll_s, self.unload_wait = (0.0, 0.4, 0.05, 0.4) if fast else (2.0, 45.0, 1.5, 20.0)
        self.hash = hashlib.sha256(json.dumps({"design": DESIGN_HASH, "smoke": a.smoke, "limit": a.limit}, sort_keys=True).encode()).hexdigest()
        if a.smoke:
            self.train, self.ev, self.he = data.synthetic_sets()
            self.leak = data.assert_no_leakage(self.train, self.ev)
        else:
            self.train, self.ev, self.he, self.leak = data.load_all()
        if a.limit:
            self.ev = self.ev[: a.limit]
        self.workers = max(1, a.gate_workers)
        self.pool = ThreadPoolExecutor(max_workers=max(2, self.workers))
        self._journals: Dict[str, L.Journal] = {}
        self.t_start = time.time()
        self.baseline_start: Optional[int] = None
        self.start_other_procs: List[Dict[str, Any]] = []
        self.probe_prompts = sorted(self.train + self.ev, key=lambda p: p.task_id)[:N_PROBE_PROMPTS] if a.smoke else sorted(self.ev, key=lambda p: p.task_id)[:N_PROBE_PROMPTS]

    def sleep(self, s: float) -> None:
        time.sleep(min(s, 0.01) if self.fast else s)

    def procs(self) -> List[Dict[str, Any]]:
        try:
            return self._procs()
        except Exception as exc:  # noqa: BLE001
            return [{"error": f"{type(exc).__name__}: {exc}"[:200]}]

    def journal(self, name: str, keep: bool = False) -> L.Journal:
        if name not in self._journals:
            self._journals[name] = L.Journal(self.out / f"{PREFIX}_journal_{name}.jsonl", self.a.resume or keep)
        return self._journals[name]

    def stage_path(self, stage: str) -> Path:
        return self.out / f"{PREFIX}_{stage}.json"

    def stage_done(self, stage: str) -> Optional[Dict[str, Any]]:
        p = self.stage_path(stage)
        if self.a.resume and p.exists():
            try:
                d = json.loads(p.read_text())
            except ValueError:
                return None
            if d.get("design_hash") == self.hash and d.get("status") == "ok":
                return d
        return None

    def write_stage(self, stage: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"stage": stage, "design_hash": self.hash, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
        self.stage_path(stage).write_text(json.dumps(rec, indent=2, default=str))
        return rec

    # ---- GPU helpers
    def wait_mem_at_most(self, limit: int) -> int:
        """Poll nvidia-smi until memory.used <= limit or t_wait is over; returns the last reading."""
        deadline = time.monotonic() + self.t_wait
        while True:
            m = self.mem()
            if m <= limit or time.monotonic() >= deadline:
                return m
            time.sleep(self.poll_s)

    def settle(self) -> int:
        """Read memory.used once it is stable (two reads within 16 MiB), at most t_wait seconds."""
        deadline, last = time.monotonic() + self.t_wait, self.mem()
        while time.monotonic() < deadline:
            self.sleep(max(self.poll_s, 0.5) if not self.fast else 0)
            cur = self.mem()
            if abs(cur - last) <= 16:
                return cur
            last = cur
        return last

    def project_seconds(self, remaining_phases: Sequence[Tuple[str, str, int]]) -> float:
        """Model-based projection for phases (tier, kind, n) with measured (or assumed) decode tok/s."""
        A = ESTIMATE_ASSUMED
        tot = 0.0
        pr = self.probe_tps()
        for tier, kind, n in remaining_phases:
            tps = pr.get(tier) or A["decode_tps"][tier]
            call_s = A["avg_tokens_per_call"] / tps + A["overhead_s_per_call"]
            mult = {"R": 1.0, "R8": 1.0, "L": A["extra_calls_per_problem_in_loop"], "D": 2.0}[kind]
            tot += A["load_s_per_phase"] + n * call_s * mult
        return tot

    def probe_tps(self) -> Dict[str, float]:
        j = self.journal("probes", keep=True)
        out: Dict[str, float] = {}
        for r in j.recs.values():
            if r.get("ctx") == ACC_CTX and r.get("decode_tps_median"):
                out[r["tier"]] = r["decode_tps_median"]
        return out

    def check_time(self, extra_remaining_s: float, where: str) -> None:
        proj = (time.time() - self.t_start) + extra_remaining_s
        if proj > self.a.max_hours * 3600:
            raise FailFast(f"projected total {proj / 3600:.2f} h exceeds --max-hours {self.a.max_hours} ({where})")


# ================================================================ generation helpers
def chat_opts(ctx_n: int, seed: int, sample: bool, num_predict: int = MAX_NEW) -> Dict[str, Any]:
    o: Dict[str, Any] = {"num_ctx": ctx_n, "seed": seed, "num_predict": num_predict, "temperature": SAMPLE_T if sample else 0.0}
    if sample:
        o["top_p"] = SAMPLE_TOP_P
    return o


def gen_call(c: Ctx, tier: str, messages: Sequence[Dict[str, str]], ctx_n: int, task_id: Any, idx: int, sample: bool) -> Dict[str, Any]:
    return c.client.chat(TIERS[tier], messages, chat_opts(ctx_n, seed_for(task_id, idx), sample), KEEP_ALIVE)


def eval_candidate(c: Ctx, p: data.Problem, res: Dict[str, Any], idx: int, source: str) -> Tuple[Dict[str, Any], gate_mod.GateResult]:
    """Visible gate (audit + visible test) and final judgement (all tests) of one reply."""
    text = res["text"]
    code = L.code_of(text)
    fut = c.pool.submit(gate_mod.gate, code, p.visible_tests, p.setup, 10, 2048, True)
    jd = L.judge([p], [text], 1)[0]
    vis = fut.result()
    cand = {"idx": idx, "source": source, "text": text, "call": call_summary(res), "model_s": model_seconds(res), "vis_verdict": vis.verdict, "verified": bool(vis.accepted),
            "functional": bool(jd["functional"]), "audit_clean": bool(jd["audit_clean"]), "accepted": bool(jd["accepted"]), "unverified": bool(vis.verdict == gate_mod.UNVERIFIED or jd["unverified"])}
    return cand, vis


def raw_record(tier: str, ctx_n: int, p: data.Problem, cand: Dict[str, Any]) -> Dict[str, Any]:
    return {"key": f"R{ctx_n}:{tier}:{p.key}", "problem": p.key, "tier": tier, "ctx": ctx_n, **cand, "tokens": cand["call"]["eval_count"]}


def run_raw(c: Ctx, tier: str, ctx_n: int, phase_idx: int, phases: Sequence[Tuple[str, str, int]]) -> Dict[str, Any]:
    j = c.journal("acc")
    todo = [p for p in c.ev if f"R{ctx_n}:{tier}:{p.key}" not in j.recs]
    t0 = time.time()
    for i, p in enumerate(todo):
        res = gen_call(c, tier, p.messages, ctx_n, p.task_id, 0, False)
        cand, _ = eval_candidate(c, p, res, 0, "greedy")
        j.add(raw_record(tier, ctx_n, p, cand))
        _progress(c, f"R{ctx_n} {tier}", i + 1, len(todo), t0, phases, phase_idx)
    return {"n": len(c.ev), "newly_run": len(todo)}


def run_loop_one(c: Ctx, tier: str, ctx_n: int, p: data.Problem, raw: Dict[str, Any]) -> Dict[str, Any]:
    """GWAYA loop C1 on one problem.  Candidates: greedy (the raw reply), sample 1, sample 2 (T=0.7, top_p=0.95), generated lazily (a later candidate cannot change the outcome
    once an earlier one verified); then up to 3 greedy repair rounds from the previous candidate (the first one at round 1) with the gate feedback."""
    g = {k: raw[k] for k in ("idx", "source", "text", "call", "model_s", "vis_verdict", "verified", "functional", "audit_clean", "accepted", "unverified")}
    cands = [g]
    chosen = 0 if g["verified"] else None
    j = 1
    while chosen is None and j <= N_SAMPLES:
        res = gen_call(c, tier, p.messages, ctx_n, p.task_id, j, True)
        cand, _ = eval_candidate(c, p, res, j, f"sample{j}")
        cands.append(cand)
        if cand["verified"]:
            chosen = len(cands) - 1
        j += 1
    tried, rounds = len(cands), 0
    last_text = g["text"]
    # feedback object for repair round 1: the greedy reply gated again (cheap; same verdict as the stored one)
    last_gate = gate_mod.gate(L.code_of(last_text), p.visible_tests, p.setup, 10, 2048, True) if chosen is None else None
    for r in range(1, N_REPAIR + 1):
        if chosen is not None:
            break
        res = gen_call(c, tier, L.repair_messages(p, L.code_of(last_text), last_gate.feedback()), ctx_n, p.task_id, N_SAMPLES + r, False)
        cand, vis = eval_candidate(c, p, res, N_SAMPLES + r, f"repair{r}")
        cands.append(cand)
        tried, rounds, last_text, last_gate = tried + 1, r, res["text"], vis
        if cand["verified"]:
            chosen = len(cands) - 1
    fin = cands[chosen] if chosen is not None else cands[0]
    return {"key": f"L{ctx_n}:{tier}:{p.key}", "problem": p.key, "tier": tier, "ctx": ctx_n, "candidates": cands, "candidates_tried": tried, "repair_rounds": rounds,
            "verified_visible": chosen is not None, "source": fin["source"] if chosen is not None else "fallback_first_greedy", "text": fin["text"], "functional": fin["functional"],
            "audit_clean": fin["audit_clean"], "accepted": fin["accepted"], "unverified": any(x["unverified"] for x in cands), "tokens": sum(x["call"]["eval_count"] for x in cands),
            "model_s": sum(x["model_s"] for x in cands), "extra_model_s": sum(x["model_s"] for x in cands[1:])}


def run_loop(c: Ctx, tier: str, ctx_n: int, phase_idx: int, phases: Sequence[Tuple[str, str, int]]) -> Dict[str, Any]:
    j = c.journal("acc")
    todo = [p for p in c.ev if f"L{ctx_n}:{tier}:{p.key}" not in j.recs]
    t0 = time.time()
    for i, p in enumerate(todo):
        j.add(run_loop_one(c, tier, ctx_n, p, j.recs[f"R{ctx_n}:{tier}:{p.key}"]))
        _progress(c, f"L{ctx_n} {tier}", i + 1, len(todo), t0, phases, phase_idx)
    return {"n": len(c.ev), "newly_run": len(todo)}


def run_determinism(c: Ctx, tier: str) -> Dict[str, Any]:
    j = c.journal("det")
    ps = sorted(c.ev, key=lambda p: p.task_id)[:N_DET]
    for p in ps:
        for k in (0, 1):
            key = f"DET:{tier}:{p.key}:{k}"
            if key not in j.recs:
                res = gen_call(c, tier, p.messages, ACC_CTX, p.task_id, 0, False)
                j.add({"key": key, "problem": p.key, "tier": tier, "rep": k, "text": res["text"], "eval_count": res["eval_count"]})
    return {"problems": len(ps)}


def _progress(c: Ctx, label: str, done: int, total: int, t0: float, phases: Sequence[Tuple[str, str, int]], phase_idx: int) -> None:
    if done % 25 == 0 or done == total:
        log(f"[E16 acc] {label}: {done}/{total}  {time.time() - t0:.0f}s")
    if done >= 10 and done % 10 == 0 and done < total:
        left = (time.time() - t0) / done * (total - done)
        c.check_time(left + c.project_seconds(phases[phase_idx + 1:]), label)


# ================================================================ stages
def stage_preflight(c: Ctx) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"status": "ok"}
    rec["ollama_version"] = c.client.version()
    have = c.client.tags()
    missing = [TIERS[t] for t in c.tiers if TIERS[t] not in have]
    rec["models_present"] = {TIERS[t]: TIERS[t] in have for t in c.tiers}
    if missing:
        cmds = " && ".join(f"ollama pull {m}" for m in missing)
        raise RuntimeError(f"Ollama models missing: {missing}. Pull them as the Ollama user, then re-run: {cmds}")
    rec["gpu"] = oll.gpu_info() if not c.smoke else {"name": "fake", "memory_total_mib": None, "driver": None}
    rec["idle_gpu_at_start"] = c.idle_record
    avail = gate_mod.gate_available(refresh=True)
    rec["gate_available"] = {"value": avail, "why": gate_mod.availability_reason(), "pass": avail}
    refs = c.train[:REF_CONTROL_N]
    rr = L.gate_many([(p.reference, p.all_tests, p.setup, True) for p in refs], c.workers)
    rate = sum(r.accepted for r in rr) / max(1, len(rr))
    rec["reference_control"] = {"n": len(rr), "accepted": sum(r.accepted for r in rr), "rate": rate, "threshold": REF_CONTROL_MIN, "pass": rate >= REF_CONTROL_MIN,
                                "rejected_examples": [(p.key, r.verdict) for p, r in zip(refs, rr) if not r.accepted][:5]}
    probes = gate_mod.run_containment_probes()
    rec["containment_probes"] = {"pass": bool(probes.get("all_contained")), "detail": probes}
    rec["no_leakage"] = {**c.leak, "pass": c.leak["id_overlap"] == 0 and c.leak["text_overlap"] == 0}
    rec["gate_control_pass"] = bool(avail and rec["reference_control"]["pass"] and rec["containment_probes"]["pass"])
    return rec


def idle_check(c: Ctx) -> Dict[str, Any]:
    """Registered idle-GPU control.  A model left loaded in Ollama is unloaded first, then memory.used must be <= 1500 MiB."""
    rec: Dict[str, Any] = {"threshold_mib": IDLE_MAX_MIB}
    loaded = [m["name"] for m in c.client.ps()]
    rec["loaded_at_start"] = loaded
    if loaded:
        rec["unload_result"] = c.client.unload_all(c.unload_wait)
    mem = c.wait_mem_at_most(IDLE_MAX_MIB)
    still = [m["name"] for m in c.client.ps()]
    rec.update({"memory_used_mib": mem, "still_loaded": still, "gpu_processes": c.procs()})
    rec["pass"] = bool(mem <= IDLE_MAX_MIB and not still)
    c.baseline_start = mem
    c.start_other_procs = [p for p in rec["gpu_processes"] if "pid" in p and not oll.is_ollama_process(p)]
    return rec


def probe_one(c: Ctx, tier: str, ctx_n: int) -> Dict[str, Any]:
    model = TIERS[tier]
    c.client.unload_all(c.unload_wait)
    baseline = c.settle()
    procs_before = c.procs()
    rec: Dict[str, Any] = {"key": f"{tier}:{ctx_n}", "tier": tier, "ctx": ctx_n, "model": model, "baseline_mib": baseline, "gpu_processes_before": procs_before}
    try:
        r1 = c.client.chat(model, [{"role": "user", "content": "Reply with OK."}], {"num_ctx": ctx_n, "temperature": 0.0, "seed": 1, "num_predict": 1}, KEEP_ALIVE)
        c.sleep(c.settle_s)
        m_load = c.mem()
        ps1 = next((m for m in c.client.ps() if m["name"] == model), None)
        rec["first_token_call"] = call_summary(r1)
        # ~85% context fill
        target = int(FILL_FRAC * ctx_n)
        texts = [p.user_text for p in c.ev]
        chars, fill = int(target * 3.6), {}
        for attempt in range(1, 5):
            r = c.client.chat(model, [{"role": "user", "content": build_fill_prompt(texts, chars, attempt)}], {"num_ctx": ctx_n, "temperature": 0.0, "seed": 1, "num_predict": 1}, KEEP_ALIVE)
            real = r["prompt_eval_count"]
            fill = {"target_tokens": target, "real_prompt_tokens": real, "fraction_of_ctx": real / ctx_n, "attempts": attempt, "chars": chars,
                    "prompt_eval_tps": tok_per_s(real, r["prompt_eval_duration"]), "call": call_summary(r)}
            if FILL_BAND[0] <= real / ctx_n <= FILL_BAND[1]:
                break
            chars = int(chars * (target / real if 0 < real < ctx_n - 8 else 0.8))
        c.sleep(c.settle_s)
        m_fill = c.mem()
        ps2 = next((m for m in c.client.ps() if m["name"] == model), None)
        fill["in_band"] = FILL_BAND[0] <= fill["fraction_of_ctx"] <= FILL_BAND[1]
        # decode / prompt-eval throughput over 12 fixed prompts (the probe id at the start of the system turn defeats the prompt cache), greedy, fixed seed
        dec, pe, calls = [], [], []
        for i, p in enumerate(c.probe_prompts):
            msgs = [{"role": "system", "content": f"[probe {i}] " + p.messages[0]["content"]}, p.messages[1]]
            r = c.client.chat(model, msgs, {"num_ctx": ctx_n, "temperature": 0.0, "seed": seed_for(p.task_id, 0), "num_predict": PROBE_NEW}, KEEP_ALIVE)
            calls.append(call_summary(r))
            if r["eval_count"] >= 8:
                dec.append(tok_per_s(r["eval_count"], r["eval_duration"]))
            pe.append(tok_per_s(r["prompt_eval_count"], r["prompt_eval_duration"]))
        ps3 = next((m for m in c.client.ps() if m["name"] == model), None)
        d_load, d_fill = m_load - baseline, m_fill - baseline
        gpu_pcts = [x["gpu_pct"] for x in (ps1, ps2, ps3) if x]
        est = estimator_mib(tier, ctx_n)
        measured = max(d_load, d_fill)
        rec.update({"mem_after_load_mib": m_load, "mem_after_fill_mib": m_fill, "delta_after_load_mib": d_load, "delta_after_fill_mib": d_fill, "measured_mib": measured,
                    "estimator_mib": est, "ratio_measured_over_estimate": measured / est, "ollama_ps_after_load": ps1, "ollama_ps_after_fill": ps2, "ollama_ps_after_probes": ps3,
                    "ps_size_vram_mib": ps2["size_vram_mib"] if ps2 else None, "gpu_pct": min(gpu_pcts) if gpu_pcts else None,
                    "gpu_offload_ok": bool(gpu_pcts) and all(g is not None and g >= 99.9 for g in gpu_pcts), "fill": fill,
                    "decode_tps_median": median(dec), "decode_tps_all": dec, "prompt_eval_tps_median": median(pe), "prompt_eval_tps_all": pe, "n_probe_prompts": len(c.probe_prompts),
                    "probe_calls": calls, "fill_prompt_eval_tps": fill["prompt_eval_tps"]})
    finally:
        un = c.client.unload(model, c.unload_wait)
        after = c.wait_mem_at_most(baseline + RETURN_TOL_MIB)
        rec["unload_verified"], rec["mem_after_unload_mib"] = un, after
        rec["returned_within_tol"] = bool(un and after <= baseline + RETURN_TOL_MIB)
        procs_after = c.procs()
        new_other = [p for p in procs_before if "pid" in p and not oll.is_ollama_process(p) and p["pid"] not in {q["pid"] for q in c.start_other_procs}]
        rec["contamination"] = {"baseline_drift_mib": baseline - (c.baseline_start or baseline), "new_non_ollama_gpu_processes": new_other, "gpu_processes_after": procs_after}
        rec["clean"] = bool(abs(baseline - (c.baseline_start or baseline)) <= RETURN_TOL_MIB and not new_other)
    return rec


def stage_probes(c: Ctx) -> Dict[str, Any]:
    j = c.journal("probes")
    done: List[Dict[str, Any]] = []
    for tier in c.tiers:
        for ctx_n in c.ctxs:
            key = f"{tier}:{ctx_n}"
            if key not in j.recs:
                log(f"[E16 probe] {tier} num_ctx={ctx_n} ...")
                j.add(probe_one(c, tier, ctx_n))
            r = j.recs[key]
            done.append(r)
            if tier == "7b" and r.get("decode_tps_median") is not None and r["decode_tps_median"] < FAILFAST_DECODE_TPS:
                raise FailFast(f"7B decode {r['decode_tps_median']:.2f} tok/s at num_ctx {ctx_n} is below {FAILFAST_DECODE_TPS} tok/s (CPU fallback?)")
    return {"status": "ok", "probes": done, "estimator_mib": {f"{t}:{x}": estimator_mib(t, x) for t in TIERS for x in (ACC_CTX, CHECK_CTX)}}


def stage_accuracy(c: Ctx) -> Dict[str, Any]:
    phases: List[Tuple[str, str, int]] = []
    for t in c.tiers:
        phases += [(t, "R", len(c.ev)), (t, "L", len(c.ev)), (t, "D", N_DET)]
        if t == "7b" and CHECK_CTX in c.ctxs:
            phases.append((t, "R8", len(c.ev)))
    c.check_time(c.project_seconds(phases), "accuracy projection")
    summary: Dict[str, Any] = {}
    cur_loaded: Optional[str] = None
    for i, (tier, kind, n) in enumerate(phases):
        model = TIERS[tier]
        if cur_loaded != tier:
            if cur_loaded:
                c.client.unload(TIERS[cur_loaded], c.unload_wait)
            others = [m["name"] for m in c.client.ps() if m["name"] != model]
            for o in others:
                c.client.unload(o, c.unload_wait)
            cur_loaded = tier
        if kind == "R":
            summary[f"R{ACC_CTX}:{tier}"] = run_raw(c, tier, ACC_CTX, i, phases)
        elif kind == "L":
            summary[f"L{ACC_CTX}:{tier}"] = run_loop(c, tier, ACC_CTX, i, phases)
        elif kind == "D":
            summary[f"DET:{tier}"] = run_determinism(c, tier)
        else:  # 7B raw at 8192: the context change makes Ollama reload; unload first so that the reload is clean
            c.client.unload(model, c.unload_wait)
            summary[f"R{CHECK_CTX}:{tier}"] = run_raw(c, tier, CHECK_CTX, i, phases)
    if cur_loaded:
        c.client.unload(TIERS[cur_loaded], c.unload_wait)
    return {"status": "ok", "phases": summary, "n_problems": len(c.ev)}


def final_unload(c: Ctx) -> Dict[str, Any]:
    """Unload control (never raises): every model unloaded and memory.used back within 300 MiB of the start baseline."""
    rec: Dict[str, Any] = {"status": "ok", "tolerance_mib": RETURN_TOL_MIB, "baseline_start_mib": c.baseline_start}
    try:
        rec["unload_result"] = c.client.unload_all(c.unload_wait)
        base = c.baseline_start
        mem = c.wait_mem_at_most((base if base is not None else IDLE_MAX_MIB) + RETURN_TOL_MIB)
        still = [m["name"] for m in c.client.ps()]
        rec.update({"memory_used_mib": mem, "still_loaded": still,
                    "pass": bool(base is not None and not still and mem <= base + RETURN_TOL_MIB)})
        if base is None:
            rec["pass"] = bool(not still)
            rec["note"] = "no start baseline in this invocation (idle check not run): only 'no model loaded' is verified"
    except Exception as exc:  # noqa: BLE001
        rec.update({"status": "failed", "pass": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
    return rec


# ================================================================ analysis
def _recs(c: Ctx, kind: str, tier: str, ctx_n: int) -> Optional[List[Dict[str, Any]]]:
    j = c.journal("acc", keep=True)
    rs = [j.recs.get(f"{kind}{ctx_n}:{tier}:{p.key}") for p in c.ev]
    return rs if rs and all(r is not None for r in rs) else None


def _summ(label: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n, k = len(recs), sum(bool(r["functional"]) for r in recs)
    lo, hi = L.wilson(k, n)
    out = {"label": label, "n": n, "passes": k, "pass_at_1": k / n if n else None, "wilson95": [lo, hi], "accepted_gwaya": sum(bool(r["accepted"]) for r in recs),
           "unverified_judgements": sum(bool(r["unverified"]) for r in recs), "generated_tokens": sum(r["tokens"] for r in recs), "model_seconds": sum(r["model_s"] for r in recs)}
    if recs and "candidates" in recs[0]:
        out.update({"verified_rate_visible": sum(bool(r["verified_visible"]) for r in recs) / n, "mean_candidates_tried": sum(r["candidates_tried"] for r in recs) / n,
                    "repair_rounds_histogram": {str(i): sum(1 for r in recs if r["repair_rounds"] == i) for i in range(N_REPAIR + 1)},
                    "winner_source": {s: sum(1 for r in recs if r["source"] == s) for s in sorted({r["source"] for r in recs})}})
    return out


def analyse(c: Ctx, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    tiers: Dict[str, Any] = {}
    loops: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for t in c.tiers:
        R, Lr = _recs(c, "R", t, ACC_CTX), _recs(c, "L", t, ACC_CTX)
        row: Dict[str, Any] = {}
        if R:
            row["R4096"] = _summ("R", R)
        if Lr:
            row["L4096"] = _summ("L", Lr)
            row["p2_gate_calibration"] = gate_calibration(Lr)
            loops[t] = {r["problem"]: r for r in Lr}
        if R and Lr:
            pr = L.paired([bool(r["functional"]) for r in R], [bool(r["functional"]) for r in Lr])
            row["paired_L_minus_R"] = pr
            row["H16c"], row["H16d"] = h16c(pr), h16d(pr)
            row["loop_extra_model_seconds"] = sum(r["extra_model_s"] for r in Lr)
        if t == "7b":
            R8 = _recs(c, "R", t, CHECK_CTX)
            if R8:
                row["R8192"] = _summ("R8192", R8)
                if R:
                    row["paired_R8192_minus_R4096"] = L.paired([bool(r["functional"]) for r in R], [bool(r["functional"]) for r in R8])
        dj = c.journal("det", keep=True)
        ps = sorted(c.ev, key=lambda p: p.task_id)[:N_DET]
        pairs = [(dj.recs.get(f"DET:{t}:{p.key}:0"), dj.recs.get(f"DET:{t}:{p.key}:1")) for p in ps]
        if pairs and all(a and b for a, b in pairs):
            same = sum(a["text"] == b["text"] for a, b in pairs)
            row["determinism"] = {"problems": len(pairs), "identical_repeats": same, "fraction_identical": same / len(pairs)}
        tiers[t] = row
    out: Dict[str, Any] = {"status": "ok", "tiers": tiers}
    out["H16"] = {"H16a": h16a(probes), "H16b": h16b(probes), "H16c": {t: tiers[t].get("H16c", "not_evaluated") for t in c.tiers}, "H16d": {t: tiers[t].get("H16d", "not_evaluated") for t in c.tiers},
                  "H16e": {t: tiers[t]["p2_gate_calibration"]["status_H16e"] if "p2_gate_calibration" in tiers[t] else "not_evaluated" for t in c.tiers}, "H16f_descriptive": h16f(probes)}
    out["H16"]["H16c_headline_7b"] = out["H16"]["H16c"].get("7b", "not_evaluated")
    if all(t in loops for t in ("1.5b", "3b", "7b")):
        out["escalation_ladder"] = ladder(loops["1.5b"], loops["3b"], loops["7b"])
    else:
        out["escalation_ladder"] = {"status": "not_evaluated", "reason": "needs loop runs of all three tiers"}
    return out


def build_report(c: Ctx, status: str, stage_recs: Dict[str, Dict[str, Any]], probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    pre, unl = stage_recs.get("preflight", {}), stage_recs.get("unload", {})
    idle = getattr(c, "idle_record", None) or pre.get("idle_gpu_at_start") or {}
    controls = {"idle_gpu": {"pass": bool(idle.get("pass")), **{k: idle.get(k) for k in ("memory_used_mib", "threshold_mib", "loaded_at_start")}},
                "gate": {"pass": bool(pre.get("gate_control_pass")), "reference_control": pre.get("reference_control"), "containment": (pre.get("containment_probes") or {}).get("pass")},
                "unload": {"pass": bool(unl.get("pass")), "baseline_start_mib": unl.get("baseline_start_mib"), "memory_used_mib": unl.get("memory_used_mib"), "still_loaded": unl.get("still_loaded")}}
    controls["gpu_offload"] = {"pass": all(p.get("gpu_offload_ok") for p in probes) if probes else None, "per_probe": {p["key"]: {"gpu_pct": p.get("gpu_pct"), "ok": p.get("gpu_offload_ok")} for p in probes}}
    controls["probe_contamination"] = {"clean_all": all(p.get("clean") for p in probes) if probes else None, "unload_between_ok": all(p.get("returned_within_tol") for p in probes) if probes else None}
    an = stage_recs.get("analysis") or {}
    rep = {"experiment": "E16", "title": "GWAYA-Lite on a Tesla T4 (PR #21 P1/P2): VRAM, tok/s and MBPP pass@1 of Qwen2.5-Coder Q4_K_M tiers via Ollama", "status": status, "smoke": c.smoke,
           "design_hash": c.hash, "tiers": {t: TIERS[t] for t in c.tiers}, "ctx": c.ctxs, "n_problems": len(c.ev),
           "stages": {s: {k: v for k, v in r.items() if k not in ("probes",)} for s, r in stage_recs.items()}, "probes": probes, "analysis": an,
           "hypotheses": an.get("H16", {}), "escalation_ladder": an.get("escalation_ladder"), "controls": controls,
           "run_valid": bool(controls["idle_gpu"]["pass"] and controls["gate"]["pass"] and controls["unload"]["pass"]),
           "gwaya_vendored": {"commit": "9c0001e8307d079ac834a86bc37e2d56703a4fb5", "ast_audit_sha256": gate_mod.AUDITOR_SHA256},
           "environment": {"python": platform.python_version(), "ollama_url": c.client.base, "gpu": pre.get("gpu"), "ollama_version": pre.get("ollama_version")},
           "notes": ["Numbers only; no claim is made here.", "VRAM measured = max(delta after load+1 token, delta after the ~85% prompt) over the idle baseline read right before the load (CUDA context included).",
                     "Estimator = PR #21 constants (weights 1070/1840/4466 MiB + kv_kib_per_token x ctx/1024 + 600 MiB).",
                     "Loop candidates are generated lazily (an earlier verified candidate makes later ones irrelevant); seeds from (task_id, sample_index); the greedy reply of R is the first candidate of L.",
                     "Hidden outcome = all MBPP tests (the visible test is one of them); 'VERIFIED' = zero-stub audit clean + visible test passed in the E15 sandbox (not GWAYA's Bubblewrap oracle).",
                     "Ollama greedy decoding is not guaranteed bit-stable: see the determinism repeat.", "Model seconds = Ollama total_duration minus load_duration, summed over calls."],
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rep["receipt_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in rep.items() if k != "receipt_sha256"}, sort_keys=True, default=str).encode()).hexdigest()
    return rep


# ================================================================ main
STAGE_FUNCS = {"preflight": stage_preflight, "probes": stage_probes, "accuracy": stage_accuracy}


def run_stage(c: Ctx, s: str, func: Callable[[Ctx], Dict[str, Any]]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        rec = func(c)
    except FailFast as exc:
        rec = {"status": "aborted by registered fail-fast rule", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        rec = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}
        log(f"[E16] stage {s} FAILED: {rec['error']}")
    rec["elapsed_s"] = time.time() - t0
    return c.write_stage(s, rec)


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP_DIR / "E16"), help="directory for stage JSONs and journals")
    ap.add_argument("--report", default=None, help=f"default {EXP_DIR / 'E16_gwaya_lite_t4_report.json'} (smoke: <out>/E16_gwaya_lite_t4_report.json)")
    ap.add_argument("--tiers", default="1.5b,3b,7b")
    ap.add_argument("--ctx", default=f"{ACC_CTX},{CHECK_CTX}")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N MBPP-sanitized problems (changes the design hash)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fake in-process Ollama + fake GPU reader + synthetic problems; CPU, < 60 s, writes only to --out (not evidence)")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--ollama-url", default=oll.DEFAULT_URL)
    ap.add_argument("--gate-workers", type=int, default=6)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None, _fake: Any = None) -> int:
    a = parse_args(argv)
    if a.estimate:
        print(json.dumps(estimate(a), indent=2))
        return 0
    fake = _fake
    if a.smoke:
        if a.out == str(EXP_DIR / "E16"):
            print("E16: --smoke requires an explicit --out (it must never write into artifacts/)", file=sys.stderr)
            return 2
    else:
        if not PREREG_PATH.exists():
            print(f"E16 refuses to start: the registration file {PREREG_PATH} does not exist", file=sys.stderr)
            return 2
    if not gate_mod.gate_available():
        print(f"E16 cannot start: the gate sandbox is unavailable ({gate_mod.availability_reason()}); every verdict would be UNVERIFIED", file=sys.stderr)
        return 2
    wanted = [s for s in a.stages.split(",") if s]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        print(f"E16: unknown stage(s) {unknown}; known: {list(STAGES)}", file=sys.stderr)
        return 2
    try:
        if a.smoke:
            sys.path.insert(0, str(HERE))
            import e16_fake_ollama as fk

            tr, ev, _ = data.synthetic_sets()
            if fake is None:
                fake = fk.FakeOllama({p.user_text: p.reference for p in tr + ev})
            url = fake.start() if fake.httpd is None else f"http://127.0.0.1:{fake.httpd.server_address[1]}"
            client, mem_fn, procs_fn = oll.Client(url, timeout=60.0, retries=0, sleep=lambda s: None), fake.gpu_read, fake.procs
        else:
            client = oll.Client(a.ollama_url)
            mem_fn, procs_fn = oll.gpu_memory_used_mib, oll.gpu_compute_processes
    except oll.OllamaError as exc:
        print(f"E16 cannot start: {exc}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))  # run the finally-unload on SIGTERM
    try:
        c = Ctx(a, client, mem_fn, procs_fn, fast=a.smoke)
    except ValueError as exc:
        print(f"E16 cannot start: {exc}", file=sys.stderr)
        return 2
    c.idle_record = None
    rpath = Path(a.report) if a.report else (c.out / f"{PREFIX}_report.json" if a.smoke else EXP_DIR / f"{PREFIX}_report.json")
    if a.smoke and ((PROJECT_ROOT / "artifacts") in rpath.resolve().parents or (PROJECT_ROOT / "artifacts") in c.out.resolve().parents):
        print("E16: --smoke must not write a report under artifacts/", file=sys.stderr)
        return 2
    log(f"[E16] smoke={c.smoke} tiers={c.tiers} ctx={c.ctxs} eval={len(c.ev)} design={c.hash[:12]} url={client.base}")
    stage_recs: Dict[str, Dict[str, Any]] = {}
    ok = failed = 0
    status, rc = "complete", 0
    needs_gpu = any(s in wanted for s in GPU_STAGES)
    try:
        if needs_gpu:
            c.idle_record = idle_check(c)
            log(f"[E16] idle-GPU control: memory.used={c.idle_record['memory_used_mib']} MiB, loaded_at_start={c.idle_record['loaded_at_start']} -> {'pass' if c.idle_record['pass'] else 'FAIL'}")
            if not c.idle_record["pass"]:
                rec = c.write_stage("preflight", {"status": "aborted by registered fail-fast rule", "reason": "idle-GPU control failed (memory.used > 1500 MiB or a model is still loaded)",
                                                  "idle_gpu_at_start": c.idle_record, "elapsed_s": 0.0})
                stage_recs["preflight"] = rec
                status, rc, failed = "aborted by registered fail-fast rule", 3, 1
        if rc == 0:
            for s in GPU_STAGES:
                if s not in wanted:
                    prev = c.stage_done(s)
                    if prev:
                        stage_recs[s] = prev
                    continue
                prev = c.stage_done(s)
                if prev:
                    log(f"[E16] stage {s}: resumed from {c.stage_path(s).name}")
                    stage_recs[s] = prev
                    ok += 1
                    continue
                log(f"[E16] stage {s} ...")
                rec = run_stage(c, s, STAGE_FUNCS[s])
                stage_recs[s] = rec
                if rec["status"] == "ok" and s == "preflight" and not rec.get("gate_control_pass", False):
                    log("[E16] preflight: the gate control failed; the run would be invalid")
                    status, rc, failed = "failed", 1, failed + 1
                    break
                if rec["status"] == "ok":
                    ok += 1
                elif rec["status"].startswith("aborted"):
                    failed += 1
                    status, rc = "aborted by registered fail-fast rule", 3
                    break
                else:
                    failed += 1
                    status, rc = "failed", 1
                    if s == "preflight" and "models missing" in rec.get("error", ""):
                        print(rec["error"], file=sys.stderr)
                        rc = 2
                    break
    finally:
        # a model must never stay resident for the next job on the GPU: unload on success, error, abort and SIGTERM
        if needs_gpu or "unload" in wanted:
            stage_recs["unload"] = c.write_stage("unload", {**final_unload(c), "elapsed_s": 0.0})
            log(f"[E16] unload control: pass={stage_recs['unload'].get('pass')} memory.used={stage_recs['unload'].get('memory_used_mib')} MiB (baseline {c.baseline_start})")
            if stage_recs["unload"]["status"] == "ok":
                ok += 1
        if fake is not None and a.smoke and _fake is None:
            fake.stop()
    # analysis (CPU only); probes come from the journal so that an aborted probe stage still reports what was measured
    pj = c.journal("probes", keep=True)
    probes = [r for r in pj.recs.values() if r.get("tier") in c.tiers and r.get("ctx") in c.ctxs]
    for s_ in ("preflight", "probes", "accuracy", "unload"):
        if s_ not in stage_recs and c.stage_done(s_):
            stage_recs[s_] = c.stage_done(s_)
    if "analysis" in wanted:
        try:
            stage_recs["analysis"] = c.write_stage("analysis", analyse(c, probes))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            stage_recs["analysis"] = c.write_stage("analysis", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]})
            failed += 1
            if rc == 0:
                status, rc = "failed", 1
    elif c.stage_done("analysis"):
        stage_recs["analysis"] = c.stage_done("analysis")
    report = build_report(c, status, stage_recs, probes)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(report, indent=2, default=str))
    summ = {t: {k: (r.get("pass_at_1") if isinstance(r, dict) else None) for k, r in row.items() if k in ("R4096", "L4096", "R8192")} for t, row in ((report["analysis"] or {}).get("tiers") or {}).items()}
    log(json.dumps({"status": status, "pass_at_1": summ, "hypotheses": {k: v for k, v in report["hypotheses"].items() if k in ("H16c_headline_7b",)}, "run_valid": report["run_valid"],
                    "receipt": report["receipt_sha256"]}, indent=2, default=str))
    if rc == 3:
        log(f"E16 ABORTED status='aborted by registered fail-fast rule' stages_ok={ok} stages_failed={failed} report={rpath}")
        return 3
    log(f"E16 DONE tiers={','.join(c.tiers)} stages_ok={ok} stages_failed={failed} run_valid={report['run_valid']} report={rpath}" + ("" if rc == 0 else f" status={status}"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
