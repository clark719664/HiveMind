"""Experiment E15: GWAYA-gated rejection-sampling fine-tuning (RFT) of a low-cost coding assistant that trains and runs in < 8 GB VRAM.

GWAYA (github.com/xaviercallens/GWAYA-GwenLayaOpenModel, MIT) is a fail-closed verification layer between a code-generating model and execution: a candidate is
accepted only if a zero-stub AST audit is clean AND an oracle (syntax + sandboxed tests) passes; otherwise it is UNVERIFIED.  GWAYA trains nothing; E15 adds the training.
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct, fp16 base + hand-rolled LoRA (no peft, no bitsandbytes), hard cap of 7.5 GiB on the CUDA allocator.

Stages (each writes <out>/E15_gwaya_rft_<stage>.json; per-problem journals <out>/E15_gwaya_rft_journal_<name>.jsonl; --resume skips finished work):
  baseline  C0 = base greedy on MBPP-sanitized (257) and HumanEval (164)
  loop_base C1 = base + GWAYA inference loop (greedy + 2 samples, visible-test gate, 3 repair rounds) on MBPP-sanitized
  rft       n=8 samples (T=0.8, top_p=0.95) per TRAIN problem, gated with ALL tests, <= 2 distinct accepted solutions kept; fail-fast (exit 3) if < 150 problems yield one
  train     LoRA r=16 on the kept pairs (3 epochs, lr 2e-4 cosine, warmup 10, micro-batch 4 x accum 4, max seq 768, gradient checkpointing, assistant-token loss)
  tuned     C2 = tuned greedy (MBPP-sanitized + HumanEval), C3 = tuned + GWAYA loop (MBPP-sanitized)
  controls  gate accepts reference solutions, planted stubs rejected, containment probes, no leakage, gate available
  report    statistics (Wilson CIs, paired differences with exact McNemar p), cost accounting, controls, receipt

    python3 scripts/run_experiment_E15_gwaya_rft.py --estimate
    python3 scripts/run_experiment_E15_gwaya_rft.py --smoke --out /tmp/e15_smoke        # tiny random model on CPU, plumbing only (not evidence)
    python3 scripts/run_experiment_E15_gwaya_rft.py --out artifacts/experiments --scratch ~/e15_scratch --resume   # full run (needs the prereg file)

This script reports numbers; it never claims a result.  Final line of a complete run: 'E15 DONE model=<m> stages_ok=<n> stages_failed=<n> report=<path>'.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import e15_data as data  # noqa: E402
import e15_gate as gate_mod  # noqa: E402
import e15_lib as L  # noqa: E402

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = EXP_DIR / "E15_gwaya_rft_prereg.json"
PREFIX = "E15_gwaya_rft"
STAGES = ("baseline", "loop_base", "rft", "train", "tuned", "controls")
FAIL_FAST_MIN_PROBLEMS = 150
ASSUMED_TOK_PER_S_PER_SEQ, ASSUMED_BATCH = 25.0, 16
ASSUMED_AVG_TOKENS, ASSUMED_PAIRS, ASSUMED_SEC_PER_TRAIN_STEP = 200, 650, 12.0

PLANTED_STUBS = ["def f(x):\n    pass\n", "def f(x):\n    ...\n", "def f(x):\n    raise NotImplementedError\n", "def f(x) -> int:\n    return None\n",
                 "def mock_f(x):\n    return 1\n", "def fake_f(x):\n    return 1\n", "def dummy_f(x):\n    return 1\n", "def stub_f(x):\n    return 1\n",
                 "def simulate_f(x):\n    return 1\n", 'def f(x):\n    """TODO"""\n', "def f(x):\n    raise NotImplementedError()\n",
                 "class A:\n    def g(self) -> int:\n        return None\n"]


class FailFast(Exception):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- config
def train_cfg(smoke: bool) -> Dict[str, Any]:
    cfg = dict(L.DESIGN["train"])
    if smoke:
        cfg["max_seq"] = 1024  # the byte tokenizer makes prompts ~4x longer in tokens
    cfg["grad_ckpt"] = bool(cfg["grad_ckpt"])
    return cfg


def design_hash(smoke: bool) -> str:
    return hashlib.sha256(json.dumps({"design": L.DESIGN_HASH, "smoke": smoke}, sort_keys=True).encode()).hexdigest()


class Ctx:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.smoke = a.smoke
        self.out = Path(a.out)
        self.scratch = Path(a.scratch)
        self.device = "cpu" if a.smoke else a.device
        self.kind = "tiny" if a.smoke else a.model
        self.hash = design_hash(a.smoke)
        self.max_new = 24 if a.smoke else a.max_new
        self.workers = a.gate_workers
        self.out.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        if a.smoke:
            self.train, self.ev, self.he = data.synthetic_sets()
            self.leak = data.assert_no_leakage(self.train, self.ev)
        else:
            self.train, self.ev, self.he, self.leak = data.load_all()
        self.tcfg = train_cfg(a.smoke)
        self.fail_min = 1 if a.smoke else FAIL_FAST_MIN_PROBLEMS
        self._journals: Dict[str, L.Journal] = {}
        self.t_start = time.time()
        self.stage_elapsed: Dict[str, float] = {}

    def journal(self, name: str, keep: bool = False) -> L.Journal:
        if name not in self._journals:  # keep=True (report): never wipe an existing journal
            self._journals[name] = L.Journal(self.out / f"{PREFIX}_journal_{name}.jsonl", self.a.resume or keep)
        return self._journals[name]

    def stage_path(self, stage: str) -> Path:
        return self.out / f"{PREFIX}_{stage}.json"

    def stage_done(self, stage: str) -> Optional[Dict[str, Any]]:
        p = self.stage_path(stage)
        if self.a.resume and p.exists():
            d = json.loads(p.read_text())
            if d.get("design_hash") == self.hash and d.get("status") == "ok":
                return d
        return None

    def write_stage(self, stage: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"stage": stage, "design_hash": self.hash, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
        self.stage_path(stage).write_text(json.dumps(rec, indent=2))
        return rec

    def fresh(self) -> Any:
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return L.load_base(self.kind, self.device)

    def amp(self) -> bool:
        return self.device.startswith("cuda") and not self.smoke

    def lora_path(self) -> Path:
        return self.scratch / "E15_lora.pt"


# ---------------------------------------------------------------- stage bodies
def summarise(label: str, problems: List[data.Problem], recs: List[Dict[str, Any]], extra_gen_seconds: float = 0.0) -> Dict[str, Any]:
    n = len(recs)
    k = sum(r["functional"] for r in recs)
    lo, hi = L.wilson(k, n)
    gen_s = sum(r.get("gen_seconds_share", 0.0) for r in recs) + extra_gen_seconds
    out = {"label": label, "n": n, "passes": k, "pass_at_1": k / n if n else None, "wilson95": [lo, hi], "accepted_gwaya": sum(r["accepted"] for r in recs),
           "stub_flagged_among_passing": sum(1 for r in recs if r["functional"] and not r["audit_clean"]), "unverified_judgements": sum(r["unverified"] for r in recs),
           "generated_tokens": sum(r["tokens"] for r in recs), "gen_gpu_seconds": gen_s, "correct_per_gpu_minute": (k / (gen_s / 60.0)) if gen_s > 0 else None}
    if recs and "candidates_tried" in recs[0]:
        out.update({"verified_rate_visible": sum(r["verified_visible"] for r in recs) / n, "mean_candidates_tried": sum(r["candidates_tried"] for r in recs) / n,
                    "repair_rounds_histogram": {str(i): sum(1 for r in recs if r["repair_rounds"] == i) for i in range(0, 4)},
                    "winner_source": {s: sum(1 for r in recs if r["source"] == s) for s in sorted({r["source"] for r in recs})},
                    "verified_visible_solutions_per_gpu_minute": (sum(r["verified_visible"] for r in recs) / (gen_s / 60.0)) if gen_s > 0 else None})
    return out


def stage_baseline(c: Ctx) -> Dict[str, Any]:
    model, tk = c.fresh()
    gen = L.Gen(model, tk, c.device, c.max_new, c.a.eval_bs)
    j = c.journal("eval")
    with L.Meter(c.device) as m:
        r0 = L.eval_greedy(gen, tk, c.ev, j, "C0", c.workers)
        r0h = L.eval_greedy(gen, tk, c.he, j, "C0h", c.workers)
    del model
    return {"status": "ok", "meter": m.rec, "C0": summarise("C0", c.ev, r0), "C0h": summarise("C0h", c.he, r0h), "gen": {"oom_retries": gen.oom_retries, "failed_batches": gen.failed_batches}}


def stage_loop_base(c: Ctx) -> Dict[str, Any]:
    model, tk = c.fresh()
    gen = L.Gen(model, tk, c.device, c.max_new, c.a.sample_bs)
    j = c.journal("eval")
    with L.Meter(c.device) as m:
        g0 = L.eval_greedy(gen, tk, c.ev, j, "C0", c.workers)  # reused if the baseline stage already wrote it
        r1 = L.gwaya_loop(gen, tk, c.ev, g0, j, "C1", c.workers)
    del model
    return {"status": "ok", "meter": m.rec, "C1": summarise("C1", c.ev, r1, extra_gen_seconds=sum(r.get("gen_seconds_share", 0.0) for r in g0)),
            "note": "C1 GPU seconds include the (reused) C0 greedy generation", "gen": {"oom_retries": gen.oom_retries, "failed_batches": gen.failed_batches}}


def stage_rft(c: Ctx) -> Dict[str, Any]:
    model, tk = c.fresh()
    gen = L.Gen(model, tk, c.device, c.max_new, c.a.sample_bs)
    j = c.journal("rft")

    def time_rule(done: int, todo: int, elapsed: float) -> None:
        """Registered fail-fast: projected total time (this invocation so far + remaining RFT at the measured rate + training at the assumed rate + a repeat of the baseline
        and loop stages for C2 / C3) above --max-hours."""
        if done < c.a.rft_chunk * 2 or done >= todo:
            return
        rft_left = elapsed / done * (todo - done)
        train_s = 3 * math.ceil(ASSUMED_PAIRS / 16) * ASSUMED_SEC_PER_TRAIN_STEP
        tuned_s = c.stage_elapsed.get("baseline", 0.0) + c.stage_elapsed.get("loop_base", 0.0)
        proj = (time.time() - c.t_start) + rft_left + train_s + tuned_s
        if proj > c.a.max_hours * 3600:
            raise FailFast(f"projected total {proj / 3600:.2f} h exceeds --max-hours {c.a.max_hours}")

    with L.Meter(c.device) as m:
        s = L.collect_rft(gen, tk, c.train, j, n_samples=L.DESIGN["rft"]["n_samples"], chunk_problems=c.a.rft_chunk, workers=c.workers, smoke_inject=c.smoke, progress=log,
                          after_chunk=time_rule)
    del model
    s.update({"gen_gpu_seconds": gen.seconds, "fail_fast_min_problems": c.fail_min, "smoke_reference_injected": c.smoke})
    rec = {"status": "ok", "meter": m.rec, **s}
    if s["problems_with_accepted"] < c.fail_min:
        rec["status"] = "aborted by registered fail-fast rule"
        rec["reason"] = f"only {s['problems_with_accepted']} of {s['problems']} problems have >= 1 accepted solution (< {c.fail_min})"
    return rec


def stage_train(c: Ctx) -> Dict[str, Any]:
    model, tk = c.fresh()
    pairs, pst = L.build_pairs(c.train, tk, c.journal("rft"), c.tcfg["max_seq"])
    if not pairs:
        raise RuntimeError("no training pairs")
    with L.Meter(c.device) as m:
        tr = L.train_lora(model, tk, pairs, c.device, c.amp(), c.tcfg, progress=log)
    st = L.lora_state(model)
    torch.save({"state": st, "rank": L.LORA_R, "alpha": L.LORA_ALPHA, "targets": list(L.LORA_TARGETS), "design_hash": c.hash}, c.lora_path())
    sha = hashlib.sha256(b"".join(st[k].numpy().tobytes() for k in sorted(st))).hexdigest()
    del model
    return {"status": "ok", "meter": m.rec, **pst, **tr, "lora_file": str(c.lora_path()), "lora_sha256": sha, "lora_nonzero_B": any(bool((v != 0).any()) for k, v in st.items() if k.endswith("lora_B"))}


def stage_tuned(c: Ctx) -> Dict[str, Any]:
    model, tk = c.fresh()
    blob = torch.load(c.lora_path(), map_location="cpu")
    n = L.merge_lora_into_base(model, blob["state"], blob["rank"], blob["alpha"])
    gen = L.Gen(model, tk, c.device, c.max_new, c.a.eval_bs)
    j = c.journal("eval")
    with L.Meter(c.device) as m:
        r2 = L.eval_greedy(gen, tk, c.ev, j, "C2", c.workers)
        r2h = L.eval_greedy(gen, tk, c.he, j, "C2h", c.workers)
        gen.bs = c.a.sample_bs
        r3 = L.gwaya_loop(gen, tk, c.ev, r2, j, "C3", c.workers)
    del model
    return {"status": "ok", "meter": m.rec, "merged_matrices": n, "C2": summarise("C2", c.ev, r2), "C2h": summarise("C2h", c.he, r2h),
            "C3": summarise("C3", c.ev, r3, extra_gen_seconds=sum(r.get("gen_seconds_share", 0.0) for r in r2)), "gen": {"oom_retries": gen.oom_retries, "failed_batches": gen.failed_batches}}


def stage_controls(c: Ctx) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    avail = gate_mod.gate_available(refresh=True)
    res["v_gate_available"] = {"value": avail, "why": gate_mod.availability_reason(), "pass": avail}
    ref_train = c.train[:100]
    rr = L.gate_many([(p.reference, p.all_tests, p.setup, True) for p in ref_train], c.workers)
    rate = sum(r.accepted for r in rr) / max(1, len(rr))
    res["i_mbpp_reference_accept"] = {"n": len(rr), "accepted": sum(r.accepted for r in rr), "rate": rate, "threshold": 0.95, "pass": rate >= 0.95,
                                      "rejected_examples": [(p.key, r.verdict, (r.violations or [r.test_error])[0][:100]) for p, r in zip(ref_train, rr) if not r.accepted][:5]}
    hr = L.gate_many([(p.reference, p.all_tests, p.setup, True) for p in c.he], c.workers)
    hrate = sum(r.accepted for r in hr) / max(1, len(hr))
    res["i_humaneval_canonical_accept"] = {"n": len(hr), "accepted": sum(r.accepted for r in hr), "rate": hrate, "threshold": 0.98, "pass": hrate >= 0.98,
                                           "rejected_examples": [(p.key, r.verdict, (r.violations or [r.test_error])[0][:100]) for p, r in zip(c.he, hr) if not r.accepted][:5]}
    sr = L.gate_many([(s, ["assert True"], "", True) for s in PLANTED_STUBS], c.workers)
    res["ii_planted_stubs_rejected"] = {"n": len(sr), "rejected_as_stub": sum(r.verdict == gate_mod.REJECTED_STUB for r in sr), "pass": all(r.verdict == gate_mod.REJECTED_STUB for r in sr)}
    probes = gate_mod.run_containment_probes()
    res["iii_containment_probes"] = {"all_contained": probes.get("all_contained", False), "pass": bool(probes.get("all_contained")), "detail": probes}
    res["iv_no_leakage"] = {**c.leak, "pass": c.leak["id_overlap"] == 0 and c.leak["text_overlap"] == 0}
    res["all_passed"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and "pass" in v)
    return {"status": "ok", "controls": res}


STAGE_FUNCS = {"baseline": stage_baseline, "loop_base": stage_loop_base, "rft": stage_rft, "train": stage_train, "tuned": stage_tuned, "controls": stage_controls}


# ---------------------------------------------------------------- report
def build_report(c: Ctx, status: str, stage_recs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    j = c.journal("eval", keep=True)
    conds: Dict[str, Any] = {}
    flags: Dict[str, List[bool]] = {}

    def cond(label: str, probs: List[data.Problem]) -> None:
        recs = [j.recs.get(f"{label}:{p.key}") for p in probs]
        if all(r is not None for r in recs):
            flags[label] = [bool(r["functional"]) for r in recs]
            extra = 0.0
            if label in ("C1", "C3"):
                base = "C0" if label == "C1" else "C2"
                extra = sum(j.recs[f"{base}:{p.key}"].get("gen_seconds_share", 0.0) for p in probs)
            conds[label] = summarise(label, probs, recs, extra)

    for lab, pr in (("C0", c.ev), ("C1", c.ev), ("C2", c.ev), ("C3", c.ev), ("C0h", c.he), ("C2h", c.he)):
        cond(lab, pr)
    comps = {}
    for name, (a, b) in {"C1-C0": ("C0", "C1"), "C2-C0": ("C0", "C2"), "C3-C1": ("C1", "C3"), "C3-C0": ("C0", "C3"), "C2h-C0h (HumanEval)": ("C0h", "C2h")}.items():
        if a in flags and b in flags:
            comps[name] = L.paired(flags[a], flags[b])
    peaks = {s: (r.get("meter") or {}).get("peak_reserved_gib") for s, r in stage_recs.items()}
    cuda = any(v is not None for v in peaks.values())
    controls = (stage_recs.get("controls") or {}).get("controls", {})
    vi = {"peak_reserved_gib_per_stage": peaks, "limit_gib": L.VRAM_LIMIT_GIB, "pass": (all(v <= L.VRAM_LIMIT_GIB for v in peaks.values() if v is not None) if cuda else None),
          "note": None if cuda else "CPU run: not measured"}
    rep = {"experiment": "E15", "title": "GWAYA-gated RFT of a low-cost coding assistant (<8 GB VRAM)", "status": status, "smoke": c.smoke, "design_hash": c.hash, "model": L.MODEL_REPO if not c.smoke else "tiny random-init Qwen2 (smoke)",
           "stages": {s: {k: v for k, v in r.items() if k not in ("controls",)} for s, r in stage_recs.items()}, "conditions": conds, "paired_comparisons": comps,
           "controls": {**controls, "vi_peak_reserved_le_8gib": vi}, "energy_wh_per_stage": {s: (r.get("meter") or {}).get("energy_wh") for s, r in stage_recs.items()},
           "gwaya_vendored": {"commit": "9c0001e8307d079ac834a86bc37e2d56703a4fb5", "ast_audit_sha256": gate_mod.AUDITOR_SHA256},
           "leakage": c.leak, "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": c.device},
           "notes": ["Numbers only; no claim is made here.", "Energy is whole-GPU NVML power x time and includes any other process on the GPU.",
                     "Sampling seeds are derived from the (task_id, sample_index) keys of each batch; results depend on the batch composition.",
                     "C1 / C3 reuse the C0 / C2 greedy reply as their first candidate (same prompt, same decode)."],
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rep["controls_all_passed"] = bool(controls.get("all_passed")) and vi["pass"] is not False
    payload = json.dumps({k: v for k, v in rep.items() if k != "receipt_sha256"}, sort_keys=True, default=str).encode()
    rep["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    return rep


# ---------------------------------------------------------------- estimate
def estimate(a: argparse.Namespace) -> Dict[str, Any]:
    ne, nh, nt = 257, 164, 462
    u0 = 0.30  # ASSUMED fraction of problems with no visible-gate pass among the 3 candidates
    repairs = ne * (u0 + u0 * 0.7 + u0 * 0.7 * 0.7)
    gens = {"C0 greedy": ne + nh, "C1 samples": 2 * ne, "C1 repair (assumed)": round(repairs), "RFT samples": nt * 8, "C2 greedy": ne + nh, "C3 samples": 2 * ne, "C3 repair (assumed)": round(repairs)}
    total = sum(gens.values())
    tokens = total * ASSUMED_AVG_TOKENS
    tput = ASSUMED_TOK_PER_S_PER_SEQ * ASSUMED_BATCH
    steps = 3 * math.ceil(ASSUMED_PAIRS / 16)
    gen_s = tokens / tput
    train_s = steps * ASSUMED_SEC_PER_TRAIN_STEP
    return {"planned_generations": gens, "total_generations": total, "assumed_avg_tokens_per_generation": ASSUMED_AVG_TOKENS, "planned_generated_tokens": tokens,
            "assumed_tokens_per_s_per_sequence": ASSUMED_TOK_PER_S_PER_SEQ, "assumed_batch": ASSUMED_BATCH, "est_generation_hours": round(gen_s / 3600, 2),
            "assumed_pairs": ASSUMED_PAIRS, "train_steps": steps, "assumed_sec_per_train_step": ASSUMED_SEC_PER_TRAIN_STEP, "est_training_hours": round(train_s / 3600, 2),
            "est_total_hours_excluding_gating_and_load": round((gen_s + train_s) / 3600, 2),
            "note": "ASSUMPTIONS, not measurements; gate calls (~0.06 s each, a few thousand, run in a thread pool on the CPU) add little."}


# ---------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="coder1.5b", choices=["coder1.5b"])
    ap.add_argument("--out", default=str(EXP_DIR), help="directory for stage JSONs, journals and the report")
    ap.add_argument("--scratch", default=None, help="directory for the LoRA .pt (default ~/e15_scratch; smoke: <out>/scratch)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny random Qwen2 on CPU, 6+6+2 synthetic problems with real tests; writes only to --out (not evidence)")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--rft-chunk", type=int, default=8, help="problems per generate/gate/journal chunk of the RFT stage")
    ap.add_argument("--gate-workers", type=int, default=6)
    ap.add_argument("--report", default=None)
    ap.add_argument("--max-hours", type=float, default=3.0, help="registered fail-fast: exit 3 if the projected total time exceeds this (checked during the RFT stage)")
    a = ap.parse_args(argv)
    if a.estimate:
        print(json.dumps(estimate(a), indent=2))
        return 0
    if a.smoke:
        a.device = "cpu"
        a.out = a.out if a.out != str(EXP_DIR) else ap.error("--smoke requires an explicit --out (it must never write into artifacts/)")
        a.scratch = a.scratch or str(Path(a.out) / "scratch")
        a.max_new = 24
        a.rft_chunk = 6
    else:
        if not PREREG_PATH.exists():
            print(f"E15 refuses to start: the registration file {PREREG_PATH} does not exist", file=sys.stderr)
            return 2
        a.scratch = a.scratch or os.path.expanduser("~/e15_scratch")
    torch.set_num_threads(2)
    L.lib.configure_determinism()
    if not gate_mod.gate_available():
        print(f"E15 cannot start: the gate sandbox is unavailable ({gate_mod.availability_reason()}); every verdict would be UNVERIFIED", file=sys.stderr)
        return 2
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("E15: CUDA requested but not available", file=sys.stderr)
        return 2
    total = L.cap_vram(a.device)
    c = Ctx(a)
    log(f"[E15] smoke={c.smoke} device={c.device} train={len(c.train)} eval={len(c.ev)} humaneval={len(c.he)} design={c.hash[:12]} vram_total={total}")
    wanted = [s for s in a.stages.split(",") if s]
    stage_recs: Dict[str, Dict[str, Any]] = {}
    ok = failed = 0
    status, rc = "complete", 0
    for s in STAGES:
        if s not in wanted:
            prev = c.stage_done(s)
            if prev:
                stage_recs[s] = prev
            continue
        prev = c.stage_done(s)
        if prev:
            log(f"[E15] stage {s}: resumed from {c.stage_path(s).name}")
            stage_recs[s] = prev
            c.stage_elapsed[s] = prev.get("elapsed_s", 0.0)
            ok += 1
            continue
        log(f"[E15] stage {s} ...")
        t0 = time.time()
        try:
            rec = STAGE_FUNCS[s](c)
        except FailFast as exc:
            rec = {"status": "aborted by registered fail-fast rule", "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            rec = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}
            log(f"[E15] stage {s} FAILED: {rec['error']}")
        rec["elapsed_s"] = time.time() - t0
        c.stage_elapsed[s] = rec["elapsed_s"]
        stage_recs[s] = c.write_stage(s, rec)
        if rec["status"] == "ok":
            ok += 1
        elif rec["status"].startswith("aborted"):
            failed += 1
            status, rc = "aborted by registered fail-fast rule", 3
            break
        else:
            failed += 1
            status, rc = "failed", 1
            break
    report = build_report(c, status, stage_recs)
    rpath = Path(a.report) if a.report else c.out / f"{PREFIX}_report.json"
    rpath.write_text(json.dumps(report, indent=2, default=str))
    summary = {k: report["conditions"][k]["pass_at_1"] for k in report["conditions"]}
    log(json.dumps({"status": status, "pass_at_1": summary, "controls_all_passed": report["controls_all_passed"], "receipt": report["receipt_sha256"]}, indent=2))
    if rc == 3:
        log(f"E15 ABORTED status='aborted by registered fail-fast rule' stages_ok={ok} stages_failed={failed} report={rpath}")
        return 3
    log(f"E15 DONE model={c.kind} stages_ok={ok} stages_failed={failed} report={rpath}" + ("" if rc == 0 else f" status={status}"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
