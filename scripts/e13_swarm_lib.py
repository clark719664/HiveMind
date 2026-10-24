"""E13 helpers: local swarm simulation (M worker OS processes + a coordinator over real localhost HTTP).

Everything the experiment script needs that is not a CLI or a cell definition: model registry, LoRA (PEFT-scale init),
non-IID token shards, the shared inner-loop / merge / outer-step code (used by BOTH the sequential reference S0 and the
worker processes, so that S1 == S0 is a protocol test and not a numerics lottery), the wire codec (raw fp32 or
E8/Z lattice codewords + zlib), the HTTP coordinator, and the worker process loop.

This is E8's "localhost negative control". It cannot support any WAN / volunteer claim.

Error-feedback caveat: a worker updates its EF carry as if its delta WAS applied; when the coordinator discards the delta (late, outside the grace window, stale
incarnation) the carry is therefore not exact. This is the E8-pilot behaviour and is intentionally unchanged.
E14 (opt-in S6 / S7): S7 merges late deltas (weight / (1 + staleness)) instead of discarding them, so for the deferred ones the EF assumption "the delta was applied" is
closer to true (still not exact: a carried delta is applied one or two versions late, with a reduced weight, and one that is dropped as too stale or after the last round is
still assumed applied). S6 only changes WHEN a round closes (bounded wait for repeatedly excluded workers), not which deltas are applied.

``hivemind.swarm`` cannot be imported here (its ``__init__`` needs the optional aioquic), so ``decoupled_diloco.py``
(which only needs torch) is loaded BY FILE PATH; its functions (select_quorum, AdaptiveGraceWindowController,
direct_weighted_average, radial_directional_average, token_weighted_quality_weight) are used unchanged.
"""

from __future__ import annotations

import atexit
import glob
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from e13_watchdog import start_parent_watchdog  # noqa: E402,F401  (re-exported; torch-free module)
import run_experiment_E10b_lattice_05b as e10b  # noqa: E402  (lora_params / get_flat / set_flat / TorchLatticeCodec / clean; importing is side-effect free)

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

# ---------------------------------------------------------------- constants (all registrable)
DOMAINS: Tuple[str, ...] = ("code", "wiki-fr", "pubmed", "gsm8k", "news", "fiction")  # worker i trains only on DOMAINS[i]
LORA_RANK, LORA_ALPHA = 8, 16.0
LORA_TARGETS: Tuple[str, ...] = ("q_proj", "v_proj")
INNER_LR, INNER_BETAS, INNER_WD, GRAD_CLIP = 5e-4, (0.9, 0.95), 0.01, 1.0
OUTER_LR, OUTER_MOMENTUM = 0.7, 0.9
SEQ_LEN, MICRO_BATCH = 256, 4
EVAL_SEQS_PER_DOMAIN, EVAL_BATCH = 16, 4
TRAIN_FRAC, HELDOUT_FRAC = 0.5, 0.2  # first 50% of a domain's tokens train, last 20% held out, middle 30% unused (disjoint)
LOSS_SCALE_FP16 = 256.0  # static loss scale for the fp16 autocast path (grads are divided back before clipping; a non-finite step is skipped and counted)
GRACE_TAU, GRACE_GAMMA, GRACE_EMA = 2, 0.5, 0.5  # AdaptiveGraceWindowController, as the E8 pilot
POLL_WAIT_S = 0.5  # server-side long-poll of GET /global
CLOSE_POLL_S = 0.01
E8_CODEWORD_SCALE = 2.0  # E8/Z codewords are multiples of 1/2: sent as int8(2 c)
ROUTING_TOP_FRACTION = 0.25
SMOKE_VOCAB = 256
FAILFAST_ROUND = 3  # the round-3 evaluation (learning gate); the registered evaluation versions are 0, FAILFAST_ROUND and R
STARTUP_TIMEOUT_S = 900.0  # hello / registration phase deadline (min with the cell timeout)
STALL_FLOOR_S, STALL_FACTOR, STALL_FIRST_S, HEARTBEAT_S = 600.0, 20.0, 1800.0, 60.0  # per-round stall watchdog
INT8_MAX = 127
# E14 (exploratory, opt-in cells S6 fair_quorum / S7 deferred_carry; S0-S5 never use them)
MAX_EXCLUSION = 1  # S6: a round waits for every alive worker that was not admitted in this many consecutive rounds
MAX_STALENESS = 2  # S7: a late delta older than this many global versions is discarded (counted); weight of a carried delta = normal weight / (1 + staleness)
FAIR_WAIT_CAP_FRACTION = 0.5  # S6: per-round cap of the extra wait = this x the StallWatchdog limit, so the cap always fires before the watchdog


def eval_versions(rounds: int) -> Tuple[int, ...]:
    """Versions of the global LoRA that are evaluated on the held-out mixture: 0, FAILFAST_ROUND (the learning gate) and R (smoke runs with R < 3: 0 and R)."""
    return tuple(sorted({0, min(FAILFAST_ROUND, rounds), rounds}))


def configure_determinism() -> None:
    """Identical numerics for S0 and every worker / host: deterministic cuBLAS workspace, math SDPA only (no flash / mem-efficient kernels, whose reductions are
    shape- and batch-dependent), deterministic algorithms (warn only). Must run before any CUDA use."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------- spawned worker registry (orphan protection on the coordinator side)
_SPAWNED: List[subprocess.Popen] = []
_SPAWNED_LOCK = threading.Lock()


def kill_spawned() -> int:
    """SIGKILL (by pid, one Popen at a time) and reap every worker still running. Registered with atexit by the experiment script; returns how many were killed."""
    with _SPAWNED_LOCK:
        procs = list(_SPAWNED)
        _SPAWNED.clear()
    n = 0
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
                n += 1
            p.wait(timeout=30)
        except Exception:  # noqa: BLE001 (best effort at interpreter exit)
            pass
    return n


class StallWatchdog:
    """Progress watchdog of one wait loop. ``poll(token, latencies, state)``: raises TimeoutError when ``token`` has not changed for
    max(STALL_FLOOR_S, STALL_FACTOR x median(latencies)) seconds (STALL_FIRST_S while no round has completed); prints a heartbeat every HEARTBEAT_S."""

    def __init__(self, label: str, clock: Callable[[], float] = time.monotonic, printer: Callable[[str], None] = lambda m: print(m, flush=True),
                 floor_s: float = STALL_FLOOR_S, factor: float = STALL_FACTOR, first_s: float = STALL_FIRST_S, heartbeat_s: float = HEARTBEAT_S) -> None:
        self.label, self.clock, self.printer = label, clock, printer
        self.floor_s, self.factor, self.first_s, self.heartbeat_s = floor_s, factor, first_s, heartbeat_s
        self.t0 = self.last_change = self.last_beat = clock()
        self.token: Any = object()

    def limit_s(self, latencies: Sequence[float]) -> float:
        return self.first_s if not latencies else max(self.floor_s, self.factor * statistics.median(latencies))

    def poll(self, token: Any, latencies: Sequence[float], state: Callable[[], str]) -> None:
        now = self.clock()
        if token != self.token:
            self.token, self.last_change = token, now
        if now - self.last_beat >= self.heartbeat_s:
            self.last_beat = now
            self.printer(f"[E13 hb] {self.label} t={now - self.t0:.0f}s {state()}")
        if now - self.last_change > self.limit_s(latencies):
            raise TimeoutError(f"{self.label}: no progress for {now - self.last_change:.0f}s (limit {self.limit_s(latencies):.0f}s); {state()}")


def fair_wait_cap_s(latencies: Sequence[float]) -> float:
    """S6 per-round cap of the extra (bounded-exclusion) wait: FAIR_WAIT_CAP_FRACTION x the StallWatchdog limit for the same latencies."""
    limit = STALL_FIRST_S if not latencies else max(STALL_FLOOR_S, STALL_FACTOR * statistics.median(latencies))
    return FAIR_WAIT_CAP_FRACTION * limit


# ---------------------------------------------------------------- misc
def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return sha256_hex(t.detach().cpu().contiguous().numpy().tobytes())


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


_DD: Any = None


def load_decoupled_diloco() -> Any:
    """``decoupled_diloco.py`` by file path (the ``hivemind.swarm`` package import needs aioquic, which is not installed)."""
    global _DD
    if _DD is None:
        name = "hm_decoupled_diloco_by_path"
        spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "src" / "hivemind" / "swarm" / "decoupled_diloco.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _DD = mod
    return _DD


# ---------------------------------------------------------------- run configuration
class RunCfg:
    """Sizes of one run. ``smoke`` selects synthetic tokens + the tiny CPU model."""

    def __init__(self, rounds: int, workers: int, inner: int, seq_len: int = SEQ_LEN, micro_batch: int = MICRO_BATCH, eval_seqs: int = EVAL_SEQS_PER_DOMAIN,
                 eval_batch: int = EVAL_BATCH, smoke: bool = False) -> None:
        if not 1 <= workers <= len(DOMAINS):
            raise ValueError(f"workers must be in 1..{len(DOMAINS)}")
        self.rounds, self.workers, self.inner, self.seq_len, self.micro_batch = rounds, workers, inner, seq_len, micro_batch
        self.eval_seqs, self.eval_batch, self.smoke = eval_seqs, eval_batch, smoke

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------- model registry
def _snapshot(repo: str) -> str:
    hits = sorted(glob.glob(str(HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / "*")))
    if not hits:
        raise FileNotFoundError(f"{repo} is not in the HF cache")
    return hits[0]


def load_tiny(device: str, experts_impl: Optional[str] = None) -> Tuple[nn.Module, Any, Tuple[str, ...]]:
    """Random-init tiny Qwen2 (fp32, CPU) for smoke / tests; no tokenizer (synthetic tokens)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=SMOKE_VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).to(device)
    model.train(False)
    return model, None, LORA_TARGETS


def load_qwen05(device: str, experts_impl: Optional[str] = None) -> Tuple[nn.Module, Any, Tuple[str, ...]]:
    """Qwen2.5-0.5B-Instruct from the HF cache: fp16 base on CUDA (fp32 on CPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = _snapshot(MODEL_META["qwen0.5b"]["repo"])
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device)
    model.train(False)
    return model, AutoTokenizer.from_pretrained(path), LORA_TARGETS


LOAD_INFO: Dict[str, Any] = {}  # what the last HF loader actually did (experts implementation requested / used), recorded in the S0 record


def resolve_experts_impl(model_key: str, override: Optional[str]) -> Optional[str]:
    """None / unset override -> MODEL_META's 'experts_implementation' (granite: 'eager'); 'default' -> do not pass the kwarg (HF default); else the override."""
    if override in (None, ""):
        return MODEL_META[model_key].get("experts_implementation")
    return None if override == "default" else override


def _load_hf_lm(key: str) -> Callable[..., Tuple[nn.Module, Any, Tuple[str, ...]]]:
    """Loader of an HF causal LM from the cache: fp16 base on CUDA (fp32 on CPU); q_proj / v_proj are plain bias-free nn.Linear (Granite MoE: attention-only LoRA;
    the fused 3-D experts and the router are frozen but used in every forward)."""

    def loader(device: str, experts_impl: Optional[str] = None) -> Tuple[nn.Module, Any, Tuple[str, ...]]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = _snapshot(MODEL_META[key]["repo"])
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        LOAD_INFO.clear()
        LOAD_INFO.update({"experts_impl_requested": experts_impl, "experts_impl_used": None})
        model = None
        if experts_impl:  # eager: the default grouped_mm falls back on sm75 to a Python loop with host syncs and computes weight grads for the frozen experts
            try:
                model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, experts_implementation=experts_impl)
                LOAD_INFO["experts_impl_used"] = experts_impl
            except (TypeError, ValueError) as exc:  # this model class does not take the kwarg: retry without and say so
                LOAD_INFO["experts_impl_error"] = f"{type(exc).__name__}: {exc}"
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        model = model.to(device)
        model.train(False)
        return model, AutoTokenizer.from_pretrained(path), LORA_TARGETS

    loader.__name__ = f"load_{key.replace('.', '_')}"
    return loader


def _not_included(device: str, experts_impl: Optional[str] = None) -> Tuple[nn.Module, Any, Tuple[str, ...]]:
    raise RuntimeError("not included: dense-upcycled experts are identical at init")


# key -> loader(device) -> (model, tokenizer, lora_target_module_names). Adding a model = one loader + one MODEL_META row.
LOADERS: Dict[str, Callable[..., Tuple[nn.Module, Any, Tuple[str, ...]]]] = {
    "tiny": load_tiny,
    "qwen0.5b": load_qwen05,
    "granite1b": _load_hf_lm("granite1b"),
    "granite3b": _load_hf_lm("granite3b"),
    "qwen0.5b-moe": _not_included,  # kept only to give a clear error
}
# repo: tokenizer / weights source; amp: fp16 autocast on CUDA; sec_per_step: ASSUMPTION used by --estimate and the fail-fast timing rule (to be measured; None = no assumption);
# workers: default M; hosting: default peer hosting ('thread' = one frozen model, M peer threads); grad_ckpt: default gradient checkpointing (memory)
MODEL_META: Dict[str, Dict[str, Any]] = {
    "tiny": {"repo": None, "amp": False, "sec_per_step": None, "workers": 2, "hosting": "process", "grad_ckpt": False},
    "qwen0.5b": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "amp": True, "sec_per_step": 0.45, "workers": 4, "hosting": "process", "grad_ckpt": False},  # ~2-3 GB per process
    "granite1b": {"repo": "ibm-granite/granite-3.1-1b-a400m-base", "amp": True, "sec_per_step": None, "workers": 3, "hosting": "process", "grad_ckpt": True, "experts_implementation": "eager"},  # ~3.8 GB/proc with ckpt
    "granite3b": {"repo": "ibm-granite/granite-3.1-3b-a800m-base", "amp": True, "sec_per_step": None, "workers": 4, "hosting": "thread", "grad_ckpt": False, "experts_implementation": "eager"},  # ~11 GB: ONE process
    "qwen0.5b-moe": {"repo": None, "amp": True, "sec_per_step": None, "workers": 4, "hosting": "process", "grad_ckpt": False},
}


def use_amp(model_key: str, device: str) -> bool:
    return bool(MODEL_META[model_key]["amp"]) and device.startswith("cuda")


# ---------------------------------------------------------------- LoRA (fp32 params over a frozen, possibly fp16, base)
class LoRALinear(nn.Module):
    """y = base(x) + (x A^T B^T) * alpha / r. A, B are fp32 even when the base is fp16 (autocast does the matmul in fp16)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out + (F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling).to(out.dtype)


def attach_lora(model: nn.Module, targets: Sequence[str], rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> int:
    """Freeze the base and wrap every nn.Linear whose leaf name is in ``targets``. Returns the number wrapped."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Linear) and leaf in targets:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, leaf, LoRALinear(mod, rank, alpha))
            n += 1
    if n == 0:
        raise ValueError(f"no nn.Linear named {tuple(targets)} found")
    return n


@torch.no_grad()
def reset_lora_all(model: nn.Module, init_seed: int) -> None:
    """PEFT-scale init (same arithmetic as E10b's ``reset_lora``): A ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)), B = 0; every worker draws the same."""
    gen = torch.Generator().manual_seed(init_seed)
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            r, fan_in = mod.lora_A.shape
            a = (torch.rand(r, fan_in, generator=gen) * 2.0 - 1.0) / math.sqrt(fan_in)
            mod.lora_A.copy_(a.to(mod.lora_A.device))
            mod.lora_B.zero_()


def build_model(model_key: str, device: str, seed: int, grad_ckpt: bool = False, experts_impl: Optional[str] = None) -> Tuple[nn.Module, Any, List[nn.Parameter]]:
    """Load, wrap with LoRA, PEFT-init from ``seed``. Returns (model, tokenizer, lora params in a fixed order).

    Everything runs with dropout off (``model.eval()``; Granite has attention_dropout 0.1). Gradient checkpointing in HF is only active in train mode, so with
    ``grad_ckpt`` the model is put in train mode AFTER setting every ``attention_dropout`` to 0 (numerically the same forward as eval mode, less activation memory)."""
    model, tok, targets = LOADERS[model_key](device, experts_impl)
    attach_lora(model, targets)
    reset_lora_all(model, seed)
    if grad_ckpt:
        model.config.attention_dropout = 0.0
        for m in model.modules():
            if hasattr(m, "attention_dropout"):
                m.attention_dropout = 0.0
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.train(True)
    else:
        model.train(False)
    return model, tok, e10b.lora_params(model)


# ---------------------------------------------------------------- data: non-IID domain shards
def stride_indices(n_available: int, n_pick: int) -> List[int]:
    """``n_pick`` sequence indices spread evenly over ``n_available`` consecutive sequences: floor(i * n_available / n_pick). Deterministic, strictly increasing,
    first = 0, last = n_available - ceil(n_available / n_pick)."""
    if n_available < n_pick:
        raise ValueError("fewer sequences available than requested")
    return [(i * n_available) // n_pick for i in range(n_pick)]


def _split_ids(ids: Sequence[int], seq_len: int, eval_seqs: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """train = first TRAIN_FRAC of the token stream (cut into non-overlapping sequences); held-out region = last HELDOUT_FRAC, cut into non-overlapping sequences
    of which ``eval_seqs`` are taken at an EVEN STRIDE over the whole region (``stride_indices``), not the first contiguous ones, so the evaluation sample is not
    one local stretch of a (sorted / clustered) corpus. Identical for S0 and every swarm worker (they all read the same token file)."""
    n = len(ids)
    t = torch.as_tensor(np.asarray(ids, dtype=np.int64)).to(torch.int32)
    n_train, held_start = int(TRAIN_FRAC * n), int((1.0 - HELDOUT_FRAC) * n)
    train, held = t[:n_train], t[held_start:]
    n_seq = train.numel() // seq_len
    if n_seq < 1:
        raise ValueError(f"train split has {train.numel()} tokens < one sequence of {seq_len}")
    if held.numel() < eval_seqs * seq_len:
        raise ValueError(f"held-out split has {held.numel()} tokens < {eval_seqs} x {seq_len}")
    train_seqs = train[: n_seq * seq_len].reshape(n_seq, seq_len)
    n_held_seq = held.numel() // seq_len
    eval_idx = stride_indices(n_held_seq, eval_seqs)
    eval_slice = held[: n_held_seq * seq_len].reshape(n_held_seq, seq_len)[eval_idx]
    prov = {"tokens_total": n, "train_tokens": int(train.numel()), "train_sequences": n_seq, "heldout_tokens": int(held.numel()), "unused_middle_tokens": held_start - n_train,
            "train_range": [0, n_train], "heldout_range": [held_start, n], "train_sha256": tensor_sha(train_seqs), "heldout_full_sha256": tensor_sha(held), "eval_slice_sha256": tensor_sha(eval_slice),
            "eval_sampling": f"{eval_seqs} of {n_held_seq} held-out sequences at stride floor(i*{n_held_seq}/{eval_seqs})", "eval_sequence_indices": eval_idx}
    return train_seqs, eval_slice, prov


def synthetic_ids(domain_idx: int, n_tokens: int) -> List[int]:
    g = torch.Generator().manual_seed(7000 + domain_idx)
    lo = (32 * domain_idx) % 128
    return torch.randint(lo, lo + 128, (n_tokens,), generator=g).tolist()


def load_tokenizer(model_key: str) -> Any:
    from transformers import AutoTokenizer

    repo = MODEL_META[model_key]["repo"]
    if repo is None:
        raise NotImplementedError(f"model {model_key!r} has no tokenizer repo yet")
    tok = AutoTokenizer.from_pretrained(_snapshot(repo))
    tok.model_max_length = 10**9
    return tok


def build_token_file(model_key: str, cfg: RunCfg, path: Path) -> Dict[str, Any]:
    """Tokenise DOMAINS[:M] (eval text of E11's ``dataset_texts``), split train/held-out/unused, store int32 tensors + provenance (sha256 of every tensor)."""
    train: Dict[str, torch.Tensor] = {}
    held: Dict[str, torch.Tensor] = {}
    prov: Dict[str, Any] = {}
    if cfg.smoke:
        n_tok = max(2000, 4 * cfg.seq_len * max(cfg.eval_seqs, 8)) * 2
        for i, dom in enumerate(DOMAINS[: cfg.workers]):
            train[dom], held[dom], prov[dom] = _split_ids(synthetic_ids(i, n_tok), cfg.seq_len, cfg.eval_seqs)
            prov[dom]["source"] = "synthetic (smoke)"
    else:
        import run_experiment_E11_kv_lattice as e11  # noqa: PLC0415 (importing is safe; data loaders only)

        tok = load_tokenizer(model_key)
        for dom in DOMAINS[: cfg.workers]:
            text = e11.dataset_texts(dom)[1]
            ids = tok(text, add_special_tokens=False).input_ids
            train[dom], held[dom], prov[dom] = _split_ids(ids, cfg.seq_len, cfg.eval_seqs)
            prov[dom]["source"] = f"dataset_texts({dom!r})[1] (eval text), tokenizer {MODEL_META[model_key]['repo']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train, "eval": held}, path)
    return prov


def load_token_file(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    blob = torch.load(path, map_location="cpu")
    return blob["train"], blob["eval"]


# ---------------------------------------------------------------- shared training / merge / eval code
def worker_perm(seed: int, worker: int, n_seq: int) -> np.ndarray:
    return np.random.RandomState(1000 * seed + worker).permutation(n_seq)


def batch_rows(perm: np.ndarray, rnd: int, step: int, inner: int, micro_batch: int) -> np.ndarray:
    """Rows of step ``step`` of round ``rnd`` (1-based): a pure function of (round, step), so a respawned / late worker needs no data cursor."""
    base = ((rnd - 1) * inner + step) * micro_batch
    return perm[(base + np.arange(micro_batch)) % len(perm)]


def make_opt(params: Sequence[nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(list(params), lr=INNER_LR, betas=INNER_BETAS, weight_decay=INNER_WD)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def local_train(model: nn.Module, params: List[nn.Parameter], opt: torch.optim.Optimizer, theta: torch.Tensor, shard: torch.Tensor, perm: np.ndarray, rnd: int,
                cfg: RunCfg, device: str, amp: bool, step_times: Optional[List[float]] = None) -> Dict[str, Any]:
    """H inner AdamW steps from ``theta`` (CPU fp32 flat). Used by S0 and by every worker process. Returns delta = theta - end (CPU fp32)."""
    e10b.set_flat(params, theta.to(device))
    scale = LOSS_SCALE_FP16 if amp else 1.0
    skipped, last_loss = 0, float("nan")
    _sync(device)
    t0 = time.perf_counter()
    if step_times is not None and not step_times:
        step_times.append(t0)  # start marker; one timestamp per finished step follows
    for s in range(cfg.inner):
        b = shard[torch.as_tensor(batch_rows(perm, rnd, s, cfg.inner, cfg.micro_batch))].long().to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = model(input_ids=b, labels=b, use_cache=False).loss
        opt.zero_grad(set_to_none=True)
        (loss * scale).backward()
        if scale != 1.0:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(scale)
        gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        if bool(torch.isfinite(gn)):
            opt.step()
        else:
            skipped += 1
        last_loss = float(loss)
        if step_times is not None:
            _sync(device)
            step_times.append(time.perf_counter())
    _sync(device)
    compute_s = time.perf_counter() - t0
    delta = (theta.to(device) - e10b.get_flat(params)).cpu()
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("non-finite delta: the inner loop diverged")
    return {"delta": delta, "compute_s": compute_s, "last_loss": last_loss, "skipped_steps": skipped, "tokens": cfg.inner * cfg.micro_batch * cfg.seq_len, "steps": cfg.inner}


def merge_deltas(deltas: Sequence[torch.Tensor], tokens: Sequence[int], steps: Sequence[int], mode: str) -> torch.Tensor:
    """Token-weighted ('quantity x quality') merge of per-worker deltas, in the order given (callers sort by worker id), direct or RDA."""
    dd = load_decoupled_diloco()
    w = [dd.token_weighted_quality_weight(t, s) for t, s in zip(tokens, steps)]
    fn = dd.radial_directional_average if mode == "rda" else dd.direct_weighted_average
    return fn(list(deltas), w)


def merge_deltas_weighted(deltas: Sequence[torch.Tensor], tokens: Sequence[int], steps: Sequence[int], mode: str, discounts: Sequence[float]) -> torch.Tensor:
    """``merge_deltas`` with a per-delta divisor (S7: 1 + staleness; 1.0 = a fresh delta): weight_i = token_weighted_quality_weight(tokens_i, steps_i) / discounts_i.
    (The averages normalise the weights.) ``merge_deltas`` itself is unchanged, so S0-S5 are bit-identical to before."""
    dd = load_decoupled_diloco()
    w = [dd.token_weighted_quality_weight(t, s) / float(d) for t, s, d in zip(tokens, steps, discounts)]
    fn = dd.radial_directional_average if mode == "rda" else dd.direct_weighted_average
    return fn(list(deltas), w)


def outer_nesterov(theta: torch.Tensor, buf: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """buf = mu buf + g; theta -= lr (g + mu buf)  (as torch.optim.SGD(nesterov=True), same arithmetic as E10b)."""
    buf = OUTER_MOMENTUM * buf + g
    return theta - OUTER_LR * (g + OUTER_MOMENTUM * buf), buf


@torch.no_grad()
def eval_domains(model: nn.Module, params: List[nn.Parameter], theta: torch.Tensor, held: Dict[str, torch.Tensor], bs: int, device: str, amp: bool) -> Dict[str, float]:
    e10b.set_flat(params, theta.to(device))
    out: Dict[str, float] = {}
    for dom, ids in held.items():
        tot = 0.0
        for s in range(0, ids.shape[0], bs):
            b = ids[s : s + bs].long().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                tot += float(model(input_ids=b, labels=b, use_cache=False).loss) * b.shape[0]
        out[dom] = tot / ids.shape[0]
    return out


def mixture(per_domain: Dict[str, float]) -> float:
    """Mean cross-entropy over the fixed mixture (equal sequence count per domain, so the mean of the domain means)."""
    return statistics.fmean(per_domain.values())


# ---------------------------------------------------------------- MoE routing statistics (descriptive)
def routing_stats_from_counts(counts: Sequence[torch.Tensor], top_k: int) -> Dict[str, Any]:
    """counts[l] = tokens routed (top-k selections) to each expert of layer l. Per-layer: normalised entropy, max/mean load, share of the 25% most-used experts."""
    ent, imb, top = [], [], []
    for c in counts:
        c = c.double()
        e = c.numel()
        p = c / c.sum().clamp_min(1.0)
        nz = p[p > 0]
        ent.append(float(-(nz * nz.log()).sum() / math.log(e)) if e > 1 else 0.0)
        imb.append(float(c.max() / c.mean().clamp_min(1e-12)))
        k = max(1, math.ceil(ROUTING_TOP_FRACTION * e))
        top.append(float(c.sort(descending=True).values[:k].sum() / c.sum().clamp_min(1.0)))
    return {"available": True, "top_k": top_k, "layers": len(counts), "mean_norm_entropy": statistics.fmean(ent), "mean_load_imbalance": statistics.fmean(imb),
            "mean_top25pct_expert_share": statistics.fmean(top)}


@torch.no_grad()
def routing_stats(model: nn.Module, params: List[nn.Parameter], theta: torch.Tensor, held: Dict[str, torch.Tensor], bs: int, device: str, amp: bool) -> Dict[str, Any]:
    """Per-domain expert usage of the global model on its held-out tokens, captured with forward hooks on ``layer.block_sparse_moe.router`` (each output is
    (top_k_index, top_k_weights, router_logits); ``output_router_logits=True`` is NOT used: it fails on GraniteMoe in transformers 5.x).
    ``{"available": False}`` for models without a ``block_sparse_moe`` attribute."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0 or not hasattr(layers[0], "block_sparse_moe"):
        return {"available": False, "reason": "no block_sparse_moe (dense model)"}
    cap: Dict[int, Any] = {}
    handles = [ly.block_sparse_moe.router.register_forward_hook(lambda mod, i, out, k=k: cap.__setitem__(k, out)) for k, ly in enumerate(layers)]
    try:
        e10b.set_flat(params, theta.to(device))
        res: Dict[str, Any] = {}
        top_k = 0
        for dom, ids in held.items():
            counts: Dict[int, torch.Tensor] = {}
            for s in range(0, ids.shape[0], bs):
                b = ids[s : s + bs].long().to(device)
                cap.clear()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    model(input_ids=b, use_cache=False)
                for k, out in cap.items():
                    idx, logits = out[0], out[2]
                    top_k = idx.shape[-1]
                    c = torch.bincount(idx.reshape(-1).cpu(), minlength=logits.shape[-1])
                    counts[k] = counts[k] + c if k in counts else c
            res[dom] = routing_stats_from_counts([counts[k] for k in sorted(counts)], top_k)
        return {"available": True, "per_domain": res}
    except Exception as exc:  # noqa: BLE001 (descriptive statistic; never fail the run for it)
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------- wire codec
def parse_codec_spec(spec: str) -> Tuple[Optional[str], float, bool]:
    """'none' | 'E8@2+EF' | 'Z@2' -> (kind or None, bits/dim, error feedback)."""
    if spec == "none":
        return None, 0.0, False
    body, _, ef = spec.partition("+")
    kind, _, rate = body.partition("@")
    if kind not in ("E8", "Z") or not rate or ef not in ("", "EF"):
        raise ValueError(f"bad codec spec {spec!r}")
    return kind, float(rate), ef == "EF"


class _CapturingCodec(e10b.TorchLatticeCodec):
    """E10b's calibrated lattice codec, keeping the codewords of the last call (the int8 payload)."""

    last_codewords: Optional[torch.Tensor] = None

    def calibrate(self, rows: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        s, r, c = super().calibrate(rows)
        self.last_codewords = c
        return s, r, c


class WireCodec:
    """Worker-side encoder of one delta; stateful (error-feedback carry), so a respawned worker starts with no carry."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.kind, self.rate, self.ef = parse_codec_spec(spec)
        self.codec = _CapturingCodec(self.kind, self.rate, self.ef) if self.kind else None
        self.n_int16 = 0  # how many encoded deltas needed int16 codewords

    def encode(self, delta: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        n = int(delta.numel())
        raw = 4 * n
        if self.codec is None:
            return delta.detach().cpu().contiguous().numpy().tobytes(), {"codec": "none", "n": n, "raw_bytes": raw, "ideal_bytes": float(raw)}
        q = self.codec.quantize(delta)
        c = self.codec.last_codewords
        cw = (c * E8_CODEWORD_SCALE).round()
        mx = float(cw.abs().max())
        if mx > 32767:
            raise ValueError(f"codeword {mx} exceeds the int16 range")
        if mx > INT8_MAX:  # outlier-dominated delta: int16 codewords (tag in the metadata); the int8 path below is unchanged
            cw_dtype, tdtype = "int16", torch.int16
            self.n_int16 += 1
        else:
            cw_dtype, tdtype = "int8", torch.int8
        body = zlib.compress(cw.to(tdtype).cpu().numpy().tobytes(), 6)
        npad = int(c.numel())
        meta = {"codec": self.kind, "cw_dtype": cw_dtype, "n": n, "scale": repr(float(self.codec.last_scale)), "rate": q.rate_bits_per_dim, "rel_mse": q.relative_mse, "raw_bytes": raw,
                "ideal_bytes": q.rate_bits_per_dim * npad / 8.0}
        return body, meta


def decode_body(body: bytes, meta: Dict[str, Any]) -> torch.Tensor:
    """Inverse of ``WireCodec.encode`` (bit-identical to the encoder's ``dequantized``)."""
    n = int(meta["n"])
    if meta["codec"] == "none":
        if len(body) != 4 * n:
            raise ValueError("raw body has the wrong length")
        return torch.frombuffer(bytearray(body), dtype=torch.float32).clone()
    wire_dtype = torch.int16 if meta.get("cw_dtype", "int8") == "int16" else torch.int8
    cw = torch.frombuffer(bytearray(zlib.decompress(body)), dtype=wire_dtype).to(torch.float64) / E8_CODEWORD_SCALE
    return (cw * float(meta["scale"])).reshape(-1)[:n].to(torch.float32)


# ---------------------------------------------------------------- stragglers and churn (explicit, deterministic)
def parse_straggle(spec: str, n_workers: int) -> Dict[Tuple[int, int], float]:
    """'-1:1,3,5:2.0;0:2:1.5' -> {(worker, round): factor}. Worker may be negative (python index); the delay is factor x the worker's median compute time."""
    out: Dict[Tuple[int, int], float] = {}
    if spec in ("", "off"):
        return out
    for part in spec.split(";"):
        try:
            w, rs, f = part.split(":")
            wi = _chaos_worker(int(w), n_workers, "straggle")
            for r in rs.split(","):
                out[(wi, int(r))] = float(f)
        except ValueError as exc:
            raise ValueError(f"bad straggle spec {spec!r} (part {part!r}): {exc}") from exc
    return out


def _chaos_worker(w: int, n_workers: int, what: str) -> int:
    """Resolve a (possibly negative) worker index; reject nonexistent workers and worker 0 (the registered chaos never targets worker 0)."""
    if not -n_workers <= w < n_workers:
        raise ValueError(f"{what}: worker {w} does not exist (M = {n_workers})")
    wi = w % n_workers
    if wi == 0:
        raise ValueError(f"{what}: worker 0 must not be targeted")
    return wi


def parse_churn(spec: str, n_workers: int) -> Optional[Tuple[int, int, int]]:
    """'W:KILL:RESPAWN' -> (worker, kill_after_round, respawn_after_round): SIGKILL after round KILL closes, re-spawn once round RESPAWN closes (>= KILL)."""
    if spec in ("", "off"):
        return None
    try:
        w, k, r = (int(x) for x in spec.split(":"))
    except ValueError as exc:
        raise ValueError(f"bad churn spec {spec!r}: {exc}") from exc
    if r < k:
        raise ValueError("respawn round must be >= kill round")
    return _chaos_worker(w, n_workers, "churn"), k, r


# ---------------------------------------------------------------- coordinator (holds no model)
class Coordinator:
    """Round state machine + the HTTP-level handlers (so tests can drive it without sockets). All byte counts are len() of real HTTP bodies."""

    def __init__(self, workers: int, rounds: int, inner: int, k: int, merge: str, grace: bool, clock: Callable[[], float] = time.monotonic,
                 eval_at: Optional[Sequence[int]] = None, fair_max_exclusion: int = 0, carry_max_staleness: Optional[int] = None,
                 fair_cap_fn: Optional[Callable[[Sequence[float]], float]] = None) -> None:
        """E14 options (both off = the E13 behaviour, bit for bit): ``fair_max_exclusion`` = MAX_EXCLUSION > 0 turns on S6 bounded exclusion (a round also waits for every
        alive worker not admitted in that many consecutive rounds, up to ``fair_cap_fn(round latencies)`` seconds); ``carry_max_staleness`` = MAX_STALENESS (not None) turns on
        S7 deferred-delta carry-over (late deltas are queued and merged into the next merge with weight / (1 + staleness) instead of being discarded)."""
        if not 1 <= k <= workers:
            raise ValueError("need 1 <= K <= M")
        if fair_max_exclusion < 0 or (carry_max_staleness is not None and carry_max_staleness < 0):
            raise ValueError("fair_max_exclusion and carry_max_staleness must be >= 0")
        if merge not in ("direct", "rda"):
            raise ValueError("merge must be direct or rda")
        self.dd = load_decoupled_diloco()
        self.workers, self.rounds, self.inner, self.k, self.merge, self.clock = workers, rounds, inner, k, merge, clock
        self.ctrl = self.dd.AdaptiveGraceWindowController(GRACE_TAU, GRACE_GAMMA, GRACE_EMA, 0.0) if grace else None
        self.cv = threading.Condition()
        self.theta: Optional[torch.Tensor] = None
        self.buf: Optional[torch.Tensor] = None
        self.version = 0
        self.opened = False
        self.done = False
        self.open_t = 0.0
        self.history: Dict[int, bytes] = {}
        self.alive = set(range(workers))
        self.incarn: Dict[int, int] = {w: 0 for w in range(workers)}  # +1 on every re-spawn; a delta from an older incarnation is rejected
        self.init_shas: Dict[int, List[str]] = {}
        self.inbox: Dict[int, Dict[str, Any]] = {}
        self._window: Optional[float] = None
        self._tq: Optional[float] = None
        self.evals: Dict[int, Dict[str, Any]] = {}
        self.eval_versions: Tuple[int, ...] = tuple(eval_at) if eval_at is not None else eval_versions(rounds)
        self.eval_claims: Dict[int, Tuple[int, int]] = {}  # version -> (worker, incarnation) currently evaluating it
        self.rounds_log: List[Dict[str, Any]] = []
        self.late: List[Dict[str, Any]] = []
        self.posts: List[Dict[str, Any]] = []  # one row per POSTed delta (accepted or not): bytes, ideal, raw, rate
        self.bytes_up: Dict[Tuple[int, int], int] = {}
        self.bytes_down: Dict[Tuple[int, int], int] = {}
        self.bytes_eval = 0
        self.bytes_setup = 0
        self.bytes_control = 0
        self.events: List[Dict[str, Any]] = []
        self.requests = 0
        # E14 state (inert unless enabled)
        self.fair_max = int(fair_max_exclusion)
        self.carry_max = carry_max_staleness
        self.fair_cap_fn = fair_cap_fn or fair_wait_cap_s
        self.excl: Dict[int, int] = {w: 0 for w in range(workers)}  # consecutive rounds an alive worker was not admitted
        self._fair_t0: Optional[float] = None  # when the round first met every other closing condition but still lacked a required worker
        self.fair_wait_timeouts = 0
        self.carry_queue: List[Dict[str, Any]] = []  # late deltas waiting for the next merge
        self._too_stale_round = 0
        self.carry_too_stale_total = 0

    # -- helpers
    def _add(self, d: Dict[Tuple[int, int], int], key: Tuple[int, int], n: int) -> None:
        d[key] = d.get(key, 0) + n

    def set_alive(self, worker: int, alive: bool) -> None:
        with self.cv:
            (self.alive.add if alive else self.alive.discard)(worker)
            if alive:
                self.incarn[worker] = self.incarn.get(worker, -1) + 1
            self.excl[worker] = 0  # re-spawn (new incarnation) or death: the exclusion count restarts; a dead worker is never waited for
            self.events.append({"event": "alive" if alive else "dead", "worker": worker, "after_round": self.version, "t": self.clock()})
            self.cv.notify_all()

    # -- handlers
    def hello(self, worker: int, init_sha: str, body: bytes) -> Dict[str, Any]:
        with self.cv:
            self.bytes_setup += len(body)
            self.init_shas.setdefault(worker, []).append(init_sha)
            if self.theta is None and body:
                self.theta = torch.frombuffer(bytearray(body), dtype=torch.float32).clone()
                self.buf = torch.zeros_like(self.theta)
                self.history[0] = body
            self.cv.notify_all()
            return {"ok": True}

    def wait_ready(self, timeout: float) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.theta is not None and all(w in self.init_shas for w in range(self.workers)), timeout=timeout)

    def missing_workers(self) -> List[int]:
        with self.cv:
            return [w for w in range(self.workers) if w not in self.init_shas]

    def progress_token(self) -> Tuple[Any, ...]:
        """Changes whenever the open round makes progress (a delta was accepted, a round closed, a peer died / came back)."""
        with self.cv:
            return (self.version, tuple(sorted(self.inbox)), tuple(sorted(self.alive)), self.done)

    def state_summary(self) -> str:
        with self.cv:
            return (f"round {self.version + 1}/{self.rounds} admitted-so-far {sorted(self.inbox)} of effective K {max(1, min(self.k, len(self.alive)))}, alive {sorted(self.alive)}, "
                    f"evals {sorted(self.evals)} of {list(self.eval_versions)}, late {len(self.late)}"
                    + (f", fair-exclusion counters {dict(sorted(self.excl.items()))} (max {self.fair_max})" if self.fair_max else "")
                    + (f", carry queue {len(self.carry_queue)}" if self.carry_max is not None else ""))

    def open_first(self) -> None:
        with self.cv:
            self.opened, self.open_t, self.version = True, self.clock(), 0
            self.cv.notify_all()

    def get_global(self, worker: int, have: int) -> Tuple[Dict[str, str], bytes]:
        with self.cv:
            self.requests += 1
            self.cv.wait_for(lambda: self.opened and self.version > have, timeout=POLL_WAIT_S)
            if not (self.opened and self.version > have):
                return {"X-Wait": "1", "X-Version": str(self.version), "X-Done": "1" if self.done else "0", **self._who(worker)}, b""
            body = self.history[self.version]
            self._add(self.bytes_down, (worker, self.version + 1), len(body))
            return {"X-Version": str(self.version), "X-Done": "1" if self.done else "0", **self._who(worker)}, body

    def _who(self, worker: int) -> Dict[str, str]:
        return {"X-Alive": "1" if worker in self.alive else "0", "X-Incarn": str(self.incarn.get(worker, 0))}

    def alive_map(self) -> Dict[str, int]:
        with self.cv:
            return {str(w): self.incarn[w] for w in sorted(self.alive)}

    def get_theta(self, version: int) -> bytes:
        with self.cv:
            body = self.history[version]
            self.bytes_eval += len(body)
            return body

    def eval_assignee(self, v: int) -> int:
        """Worker that evaluates version ``v``: v mod M if alive, else the lowest-id alive worker (caller holds the lock)."""
        w = v % self.workers
        if w in self.alive:
            return w
        return min(self.alive) if self.alive else w

    def eval_todo(self, worker: int, incarn: Optional[int] = None) -> List[int]:
        """Versions ``worker`` must evaluate now (assigned to it, already published, not evaluated, not held by another live peer); they are claimed for it."""
        with self.cv:
            inc = self.incarn.get(worker, 0) if incarn is None else incarn
            out: List[int] = []
            for v in self.eval_versions:
                if v in self.evals or v not in self.history or self.eval_assignee(v) != worker:
                    continue
                claim = self.eval_claims.get(v)
                if claim is not None and claim != (worker, inc) and claim[0] in self.alive and self.incarn.get(claim[0]) == claim[1]:
                    continue
                self.eval_claims[v] = (worker, inc)
                out.append(v)
            return out

    def post_eval(self, version: int, payload: Dict[str, Any], nbytes: int) -> None:
        with self.cv:
            self.bytes_eval += nbytes
            self.evals[version] = payload
            self.cv.notify_all()

    def wait_evals(self, timeout: float) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: all(v in self.evals for v in self.eval_versions), timeout=timeout)

    def post_delta(self, worker: int, meta: Dict[str, Any], body: bytes) -> Dict[str, Any]:
        arrival = self.clock()
        r = int(meta["round"])
        with self.cv:
            self.requests += 1
            self._add(self.bytes_up, (worker, r), len(body))
            cur = self.version + 1
            row = {"worker": worker, "round": r, "bytes": len(body), "raw_bytes": int(meta["raw_bytes"]), "ideal_bytes": float(meta["ideal_bytes"]), "rate": meta.get("rate"),
                   "rel_mse": meta.get("rel_mse"), "arrived_in_round": cur, "cw_dtype": meta.get("cw_dtype")}
            reason = None
            carry = False
            if not self.opened or self.done:
                reason = "not open"
            elif r != cur:
                if self.carry_max is not None and r < cur:  # S7: a late delta is kept for the next merge unless it is from a dead / old incarnation, too stale or a duplicate
                    if worker not in self.alive:
                        reason = "worker not alive"
                    elif int(meta.get("incarnation", self.incarn.get(worker, 0))) != self.incarn.get(worker, 0):
                        reason = "stale incarnation"
                    elif cur - r > self.carry_max:
                        reason = "too stale (staleness > MAX_STALENESS)"
                        self._too_stale_round += 1
                        self.carry_too_stale_total += 1
                    elif any(e["worker"] == worker and e["round"] == r for e in self.carry_queue):
                        reason = "duplicate"
                    carry = reason is None
                else:
                    reason = "late (round already closed)"
            elif worker not in self.alive:
                reason = "worker not alive"
            elif int(meta.get("incarnation", self.incarn.get(worker, 0))) != self.incarn.get(worker, 0):
                reason = "stale incarnation"
            elif worker in self.inbox:
                reason = "duplicate"
            if carry:
                self.carry_queue.append({"worker": worker, "round": r, "delta": decode_body(body, meta), "tokens": int(meta["tokens"]), "steps": int(meta["steps"]), "arrived_in_round": cur})
                row.update({"accepted": True, "carried": True})
                self.posts.append(row)
                self.cv.notify_all()
                return {"accepted": True, "carried": True, "round": cur}
            if reason is not None:
                row["accepted"] = False
                self.posts.append(row)
                self.late.append({"worker": worker, "delta_round": r, "arrived_in_round": cur, "reason": reason, "bytes": len(body)})
                return {"accepted": False, "reason": reason, "round": cur}
            self.inbox[worker] = {"delta": decode_body(body, meta), "arrival_rel": arrival - self.open_t, "compute_s": float(meta["compute_s"]), "tokens": int(meta["tokens"]),
                                  "steps": int(meta["steps"])}
            row["accepted"] = True
            self.posts.append(row)
            self.cv.notify_all()
            return {"accepted": True, "round": cur}

    # -- round closing (called by the main loop)
    def try_close_round(self, before_publish: Optional[Callable[[int], None]] = None) -> Optional[Dict[str, Any]]:
        with self.cv:
            if not self.opened or self.done:
                return None
            n, eff_k = len(self.inbox), max(1, min(self.k, len(self.alive)))
            if n < eff_k:
                return None
            now = self.clock()
            order = sorted(self.inbox.items(), key=lambda kv: (kv[1]["arrival_rel"], kv[0]))
            if self._window is None:  # quorum just reached: fix the grace window once per round
                self._tq = order[eff_k - 1][1]["arrival_rel"]
                self._window = 0.0
                if self.ctrl is not None:
                    self.ctrl.observe_step_duration(statistics.median(v["compute_s"] for _, v in order))
                    self.ctrl.observe_quorum_duration(self._tq)
                    self._window = self.ctrl.grace_window_seconds()
            if n < len(self.alive) and (now - self.open_t) < self._tq + self._window:
                return None
            # S6 bounded exclusion: every alive worker excluded for >= MAX_EXCLUSION consecutive rounds must have posted for this round (dead workers never block);
            # the extra wait is capped (fair_cap_fn) and a timeout closes the round with what is there
            required = sorted(w for w in self.alive if self.excl.get(w, 0) >= self.fair_max) if self.fair_max else []
            fair_timed_out = False
            if any(w not in self.inbox for w in required):
                if self._fair_t0 is None:
                    self._fair_t0 = now
                if now - self._fair_t0 < self.fair_cap_fn([r["latency_s"] for r in self.rounds_log]):
                    return None
                fair_timed_out = True
                self.fair_wait_timeouts += 1
            arrivals = [self.dd.LearnerArrival(str(w), v["arrival_rel"], v["steps"], v["tokens"]) for w, v in self.inbox.items()]
            sel = self.dd.select_quorum(arrivals, min_quorum_k=eff_k, grace_window_s=self._window)
            admitted = sorted(int(i) for i in sel.admitted_learner_ids)
            unadmitted = sorted(int(i) for i in sel.deferred_learner_ids)  # arrived before the close but outside the window: discarded, counted (S7: carried to the next merge)
            rnd = self.version + 1
            forced = [w for w in required if w in self.inbox and w not in admitted]  # S6: admitted although outside the grace window
            if forced:
                admitted = sorted(set(admitted) | set(forced))
                unadmitted = [w for w in unadmitted if w not in forced]
            carry_in, self.carry_queue = self.carry_queue, []
            carry_use: List[Dict[str, Any]] = []
            stale_now = 0
            for e in sorted(carry_in, key=lambda e: (e["worker"], e["round"])):
                if rnd - e["round"] > (self.carry_max if self.carry_max is not None else -1):
                    stale_now += 1
                    self.carry_too_stale_total += 1
                    self.late.append({"worker": e["worker"], "delta_round": e["round"], "arrived_in_round": e["arrived_in_round"], "reason": "too stale at merge", "bytes": 0})
                else:
                    carry_use.append(e)
            queued_next = 0
            for w in unadmitted:
                if self.carry_max is not None and rnd < self.rounds:  # S7: keep it for the next round's merge (no next round after the last one: discard)
                    v = self.inbox[w]
                    self.carry_queue.append({"worker": w, "round": rnd, "delta": v["delta"], "tokens": v["tokens"], "steps": v["steps"], "arrived_in_round": rnd})
                    queued_next += 1
                else:
                    self.late.append({"worker": w, "delta_round": rnd, "arrived_in_round": rnd, "reason": "outside grace window", "bytes": 0})
            d_fresh = [self.inbox[w]["delta"] for w in admitted]
            t_fresh, s_fresh = [self.inbox[w]["tokens"] for w in admitted], [self.inbox[w]["steps"] for w in admitted]
            if carry_use:
                g = merge_deltas_weighted(d_fresh + [e["delta"] for e in carry_use], t_fresh + [e["tokens"] for e in carry_use], s_fresh + [e["steps"] for e in carry_use], self.merge,
                                          [1.0] * len(admitted) + [1.0 + (rnd - e["round"]) for e in carry_use])
            else:
                g = merge_deltas(d_fresh, t_fresh, s_fresh, self.merge)
            self.theta, self.buf = outer_nesterov(self.theta, self.buf, g)
            rec = {"round": rnd, "latency_s": now - self.open_t, "quorum_at_s": self._tq, "grace_window_s": self._window, "effective_k": eff_k, "alive": sorted(self.alive),
                   "arrivals_s": {str(w): v["arrival_rel"] for w, v in self.inbox.items()}, "compute_s": {str(w): v["compute_s"] for w, v in self.inbox.items()},
                   "admitted": admitted, "unadmitted_at_close": unadmitted, "merged_norm": float(g.norm())}
            if self.fair_max:
                rec.update({"fair_waited_for": required, "fair_extra_wait_s": (now - self._fair_t0) if self._fair_t0 is not None else 0.0, "fair_timed_out": fair_timed_out,
                            "fair_forced_admitted": forced, "exclusion_counters_before": {str(w): c for w, c in sorted(self.excl.items())}})
                for w in self.alive:  # consecutive rounds in which an ALIVE worker was not admitted
                    self.excl[w] = 0 if w in admitted else self.excl.get(w, 0) + 1
            if self.carry_max is not None:
                hist: Dict[str, int] = {}
                for e in carry_use:
                    hist[str(rnd - e["round"])] = hist.get(str(rnd - e["round"]), 0) + 1
                rec.update({"carried_count": len(carry_use), "carried_tokens": sum(e["tokens"] for e in carry_use),
                            "carried": [{"worker": e["worker"], "delta_round": e["round"], "staleness": rnd - e["round"], "weight_divisor": 1.0 + (rnd - e["round"]), "tokens": e["tokens"]} for e in carry_use],
                            "staleness_hist": hist, "carry_too_stale_discarded": stale_now + self._too_stale_round, "carry_queued_for_next": queued_next})
                self._too_stale_round = 0
            self.rounds_log.append(rec)
            self._fair_t0 = None
            self.inbox, self._window, self._tq = {}, None, None
            if before_publish is not None:
                before_publish(rnd)  # churn: kill / respawn BEFORE the next round opens
            self.version = rnd
            self.history[rnd] = self.theta.numpy().tobytes()
            self.done = rnd >= self.rounds
            self.open_t = self.clock()
            self.cv.notify_all()
            return rec

    # -- summaries
    def e14_summary(self) -> Dict[str, Any]:
        """Totals of the E14 mechanisms for the cell record ({} when neither is on)."""
        out: Dict[str, Any] = {}
        if self.fair_max:
            rl = [r for r in self.rounds_log if "fair_waited_for" in r]
            out["fair"] = {"max_exclusion": self.fair_max, "fair_wait_timeouts": self.fair_wait_timeouts, "rounds_with_required_workers": sum(1 for r in rl if r["fair_waited_for"]),
                           "rounds_with_extra_wait": sum(1 for r in rl if r["fair_extra_wait_s"] > 0.0), "extra_wait_s_total": sum(r["fair_extra_wait_s"] for r in rl),
                           "forced_admissions_total": sum(len(r["fair_forced_admitted"]) for r in rl),
                           "per_round": [{"round": r["round"], "waited_for": r["fair_waited_for"], "extra_wait_s": r["fair_extra_wait_s"], "timed_out": r["fair_timed_out"],
                                          "forced_admitted": r["fair_forced_admitted"]} for r in rl]}
        if self.carry_max is not None:
            rl = [r for r in self.rounds_log if "carried_count" in r]
            hist: Dict[str, int] = {}
            for r in rl:
                for k_, v_ in r["staleness_hist"].items():
                    hist[k_] = hist.get(k_, 0) + v_
            out["carry"] = {"max_staleness": self.carry_max, "carried_count_total": sum(r["carried_count"] for r in rl), "carried_tokens_total": sum(r["carried_tokens"] for r in rl),
                            "staleness_hist": hist, "too_stale_discarded_total": self.carry_too_stale_total,
                            "queued_unmerged_at_end": len(self.carry_queue),
                            "per_round": [{"round": r["round"], "carried_count": r["carried_count"], "carried_tokens": r["carried_tokens"], "staleness_hist": r["staleness_hist"],
                                           "too_stale_discarded": r["carry_too_stale_discarded"]} for r in rl]}
        return out

    def per_round_bytes(self) -> Tuple[List[Dict[str, int]], List[Dict[str, int]], int]:
        up = [{str(w): self.bytes_up.get((w, r), 0) for w in range(self.workers)} for r in range(1, self.rounds + 1)]
        down = [{str(w): self.bytes_down.get((w, r), 0) for w in range(self.workers)} for r in range(1, self.rounds + 1)]
        final = sum(v for (w, r), v in self.bytes_down.items() if r == self.rounds + 1)
        return up, down, final


# ---------------------------------------------------------------- HTTP plumbing
def make_server(coord: Coordinator) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:  # silence
            del fmt, a

        def _send(self, body: bytes = b"", headers: Optional[Dict[str, str]] = None, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any) -> None:
            body = json.dumps(obj).encode()
            with coord.cv:
                coord.bytes_control += len(body)
            self._send(body, {"Content-Type": "application/json"})

        def _q(self) -> Dict[str, str]:
            return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

        def do_GET(self) -> None:
            path, q = urllib.parse.urlparse(self.path).path, self._q()
            try:
                if path == "/global":
                    headers, body = coord.get_global(int(q["worker"]), int(q["have"]))
                    self._send(body, headers)
                elif path == "/theta":
                    self._send(coord.get_theta(int(q["version"])))
                elif path == "/alive":
                    self._json(coord.alive_map())
                elif path == "/evaltodo":
                    self._json(coord.eval_todo(int(q["worker"]), int(q["incarn"]) if "incarn" in q else None))
                else:
                    self._send(code=404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            path, q = urllib.parse.urlparse(self.path).path, self._q()
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                if path == "/hello":
                    self._json(coord.hello(int(q["worker"]), self.headers.get("X-Init-Sha", ""), body))
                elif path == "/delta":
                    meta = json.loads(self.headers["X-Meta"])
                    self._json(coord.post_delta(int(q["worker"]), meta, body))
                elif path == "/eval":
                    coord.post_eval(int(q["version"]), json.loads(body), len(body))
                    self._json({"ok": True})
                else:
                    self._send(code=404)
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

    return Server(("127.0.0.1", 0), Handler)


def http_request(url: str, data: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None, timeout: float = 600.0) -> Tuple[Dict[str, str], bytes]:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET", headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return dict(r.headers.items()), r.read()


# ---------------------------------------------------------------- worker process / peer threads
class _Null:
    def __enter__(self) -> "_Null":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        del exc_type, exc_val, exc_tb
        return False


class PeerEnv:
    """Everything the peers of one host share: the (single) frozen model + LoRA params, data, config. In thread hosting the peers also share ``lock``."""

    def __init__(self, wc: Dict[str, Any], model: nn.Module, params: List[nn.Parameter], train: Dict[str, torch.Tensor], held: Dict[str, torch.Tensor]) -> None:
        self.wc, self.model, self.params, self.train, self.held = wc, model, params, train, held
        self.cfg = RunCfg(**wc["run_cfg"])
        self.device, self.model_key, self.seed, self.codec = wc["device"], wc["model"], int(wc["seed"]), wc["codec"]
        self.amp = use_amp(self.model_key, self.device)
        self.base = f"http://127.0.0.1:{wc['port']}"
        self.straggle = {(int(a), int(b)): f for a, b, f in wc["straggle"]}
        self.thread_mode = wc.get("hosting", "process") == "thread"
        self.init_theta = e10b.get_flat(params).cpu()  # taken before any training: every (re)started peer reports this init
        self.init_sha = tensor_sha(self.init_theta)
        self.lock = threading.Lock() if self.thread_mode else _Null()  # serialises forward/backward + eval on the one shared model
        self.done = threading.Event()
        self.errors: List[BaseException] = []


def peer_loop(env: PeerEnv, wid: int, inc: int) -> str:
    """One peer: GET global -> H inner AdamW steps on DOMAINS[wid] -> POST (optionally compressed) delta -> repeat. Evaluation duty: after each POST and at the end the
    peer asks ``/evaltodo`` and evaluates the held-out set at the versions the coordinator assigned to it (version v goes to worker v mod M if alive, else the lowest
    alive id), so no single peer (in particular not the code-domain worker 0) carries all evaluations.
    Own LoRA flat vector (swapped in with set_flat inside ``local_train``), AdamW state, shard, codec/EF carry. Returns 'done' or 'stopped' (thread hosting churn)."""
    cfg, base, device, amp = env.cfg, env.base, env.device, env.amp
    shard = env.train[DOMAINS[wid]]
    perm = worker_perm(env.seed, wid, shard.shape[0])
    opt = make_opt(env.params)
    wire = WireCodec(env.codec)
    init_body = env.init_theta.numpy().tobytes() if (wid == 0 and inc == 0) else b""
    http_request(f"{base}/hello?worker={wid}", init_body, {"X-Init-Sha": env.init_sha, "Content-Type": "application/octet-stream"})
    compute_hist: List[float] = []
    held_m = {d: env.held[d] for d in DOMAINS[: cfg.workers]}

    def do_evals() -> None:
        for v in json.loads(http_request(f"{base}/evaltodo?worker={wid}&incarn={inc}")[1]):
            theta = torch.frombuffer(bytearray(http_request(f"{base}/theta?version={v}")[1]), dtype=torch.float32).clone()
            t = time.perf_counter()
            with env.lock:
                per = eval_domains(env.model, env.params, theta, held_m, cfg.eval_batch, device, amp)
                routing = routing_stats(env.model, env.params, theta, held_m, cfg.eval_batch, device, amp) if v == cfg.rounds else None
            payload = {"version": v, "per_domain": per, "mixture": mixture(per), "eval_s": time.perf_counter() - t, "evaluator": wid}
            if routing is not None:
                payload["routing"] = routing
            http_request(f"{base}/eval?version={v}", json.dumps(payload).encode(), {"Content-Type": "application/json"})

    have = -1
    while True:
        try:
            headers, body = http_request(f"{base}/global?worker={wid}&have={have}", timeout=120.0)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                return "done"  # coordinator gone: the cell is over
            raise
        if env.thread_mode and (headers.get("X-Alive") == "0" or int(headers.get("X-Incarn", inc)) != inc):
            return "stopped"  # churn: the coordinator declared this incarnation dead; its optimizer / EF state is discarded with the thread
        if headers.get("X-Wait") == "1":
            if headers.get("X-Done") == "1" and have >= 0:
                do_evals()
                env.done.set()
                return "done"
            continue
        v, done = int(headers["X-Version"]), headers["X-Done"] == "1"
        if done:
            do_evals()
            env.done.set()
            return "done"
        theta = torch.frombuffer(bytearray(body), dtype=torch.float32).clone()
        rnd = v + 1
        with env.lock:
            res = local_train(env.model, env.params, opt, theta, shard, perm, rnd, cfg, device, amp)
        compute_hist.append(res["compute_s"])
        if (wid, rnd) in env.straggle:
            time.sleep(env.straggle[(wid, rnd)] * statistics.median(compute_hist))
        payload, meta = wire.encode(res["delta"])
        meta.update({"round": rnd, "compute_s": res["compute_s"], "tokens": res["tokens"], "steps": res["steps"], "skipped_steps": res["skipped_steps"], "incarnation": inc})
        http_request(f"{base}/delta?worker={wid}", payload, {"X-Meta": json.dumps(meta), "Content-Type": "application/octet-stream"})
        have = v
        do_evals()


def run_worker(cfg_path: str) -> int:
    """Process hosting: this process is ONE peer (``worker_id``). Thread hosting: this process is the host of all M peers (one frozen model, peers are threads
    talking to the coordinator over the same HTTP protocol)."""
    configure_determinism()  # before any CUDA use, identical to S0
    wc = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    start_parent_watchdog(1.0, wc.get("parent_pid"))  # coordinator died (SIGKILL / OOM): exit(3) instead of becoming an orphan that holds GPU memory
    if wc["device"] == "cpu":
        torch.set_num_threads(int(wc.get("threads", 2)))
    model, _, params = build_model(wc["model"], wc["device"], int(wc["seed"]), bool(wc.get("grad_ckpt", False)), wc.get("experts_impl"))
    train, held = load_token_file(wc["tokens_file"])
    env = PeerEnv(wc, model, params, train, held)
    if not env.thread_mode:
        peer_loop(env, int(wc["worker_id"]), int(wc["incarnation"]))
        return 0
    threads: Dict[int, Tuple[int, threading.Thread]] = {}

    def run_peer(w: int, inc: int) -> None:
        try:
            peer_loop(env, w, inc)
        except BaseException as exc:  # noqa: BLE001
            env.errors.append(exc)

    while True:
        if env.errors:
            raise env.errors[0]
        try:
            alive = json.loads(http_request(f"{env.base}/alive")[1])
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                return 0
            raise
        if not env.done.is_set():
            for w_s, inc in alive.items():
                w = int(w_s)
                if w not in threads or threads[w][0] != inc:  # not started yet, or re-spawned (new incarnation): start a fresh peer
                    t = threading.Thread(target=run_peer, args=(w, int(inc)), daemon=True)
                    threads[w] = (int(inc), t)
                    t.start()
        elif not any(t.is_alive() for _, t in threads.values()):
            return 0
        time.sleep(0.05)


def spawn_worker(script: str, cfg_path: Path, log_path: Path, env: Dict[str, str]) -> subprocess.Popen:
    """Start one worker / host process (own pid; not a new session or process group: the coordinator kills each worker by pid and ``kill_spawned`` is the atexit net)."""
    with open(log_path, "ab") as log:  # the child inherits its own duplicate of the descriptor; the parent's handle is closed here
        p = subprocess.Popen([sys.executable, script, "--worker", "--worker-config", str(cfg_path)], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=False)
    with _SPAWNED_LOCK:
        _SPAWNED[:] = [q for q in _SPAWNED if q.poll() is None]
        _SPAWNED.append(p)
    return p


def log_tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


