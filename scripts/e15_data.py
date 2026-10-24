"""E15 data: MBPP (train = full train + validation, eval = sanitized test) and HumanEval (eval only) from the Hugging Face cache; prompt format; extraction.

Splits (registered): TRAIN = MBPP ``full`` train + validation (task_ids 511-974, 464 problems); EVAL = MBPP ``sanitized`` test (257 problems, task_ids within 11-510);
HumanEval 164 (eval only).  ``assert_no_leakage`` checks task_ids and normalised prompt texts.  Visible test = ``test_list[0]`` (shown in the prompt and the only test the
INFERENCE-time gate may use); the final judgement uses ALL tests; RFT data collection on TRAIN problems uses ALL tests.
"""

from __future__ import annotations

import ast
import glob
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e15_gate import extract_code  # noqa: E402,F401  (re-exported)

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
MBPP_REPO, HE_REPO = "google-research-datasets/mbpp", "openai/openai_humaneval"
SYSTEM_PROMPT = "You are a careful Python programming assistant. Reply with exactly one ```python code block containing the complete solution."
EXPECTED = {"train": 464, "eval": 257, "humaneval": 164}
MBPP_FILES = {"full_train": "full/train-00000-of-00001.parquet", "full_validation": "full/validation-00000-of-00001.parquet",
              "sanitized_test": "sanitized/test-00000-of-00001.parquet"}
HE_FILE = "openai_humaneval/test-00000-of-00001.parquet"


@dataclass
class Problem:
    key: str  # unique string id, e.g. "mbpp-601" / "HumanEval/3" / "syn-train-0"
    source: str  # "mbpp-train" | "mbpp-sanitized" | "humaneval" | "synthetic-*"
    task_id: Any
    user_text: str  # problem statement WITHOUT the example test
    visible_tests: List[str]
    all_tests: List[str]
    setup: str = ""
    reference: str = ""  # reference solution (complete code)
    entry_point: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)  # chat messages for the first attempt (system + user)


def _cached_file(repo: str, rel: str) -> Path:
    kind = "datasets--" + repo.replace("/", "--")
    hits = sorted(glob.glob(str(HUB / kind / "snapshots" / "*" / rel)))
    if hits:
        return Path(hits[0])
    from huggingface_hub import hf_hub_download  # network is allowed; only the missing split is fetched

    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=rel))


def _read(repo: str, rel: str):
    import pandas as pd

    return pd.read_parquet(_cached_file(repo, rel))


def build_messages(user_text: str, visible_test: Optional[str]) -> List[Dict[str, str]]:
    user = user_text.strip()
    if visible_test:
        user += "\nYour code should pass this test:\n" + visible_test.strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def normalise_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _mbpp_problem(row: Any, source: str, text_col: str) -> Problem:
    tests = [str(t) for t in row["test_list"]]
    setup = str(row["test_setup_code"]) if "test_setup_code" in row and row["test_setup_code"] else ""
    if "test_imports" in row:
        setup = "\n".join(str(x) for x in row["test_imports"])
    tid = int(row["task_id"])
    p = Problem(key=f"mbpp-{tid}", source=source, task_id=tid, user_text=str(row[text_col]).strip(), visible_tests=tests[:1], all_tests=tests, setup=setup,
                reference=str(row["code"]).replace("\r\n", "\n"))
    p.messages = build_messages(p.user_text, p.visible_tests[0])
    return p


def load_mbpp_train() -> List[Problem]:
    rows = []
    for k in ("full_train", "full_validation"):
        df = _read(MBPP_REPO, MBPP_FILES[k])
        rows += [_mbpp_problem(r, "mbpp-train", "text") for _, r in df.iterrows()]
    return sorted(rows, key=lambda p: p.task_id)


def load_mbpp_eval() -> List[Problem]:
    df = _read(MBPP_REPO, MBPP_FILES["sanitized_test"])
    return sorted((_mbpp_problem(r, "mbpp-sanitized", "prompt") for _, r in df.iterrows()), key=lambda p: p.task_id)


def humaneval_setup(prompt: str) -> str:
    """Top-level imports of the HumanEval prompt (a model that rewrites the function may drop e.g. ``from typing import List``)."""
    try:
        tree = ast.parse(prompt)
        return "\n".join(ast.get_source_segment(prompt, n) or "" for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    except SyntaxError:
        return "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))


def load_humaneval() -> List[Problem]:
    df = _read(HE_REPO, HE_FILE)
    out = []
    for _, r in df.iterrows():
        p = Problem(key=str(r["task_id"]), source="humaneval", task_id=str(r["task_id"]), user_text=str(r["prompt"]), visible_tests=[], setup=humaneval_setup(str(r["prompt"])),
                    all_tests=[str(r["test"]), f"check({r['entry_point']})"], reference=str(r["prompt"]) + str(r["canonical_solution"]), entry_point=str(r["entry_point"]))
        p.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p.user_text}]  # the user turn is the prompt itself
        out.append(p)
    return out


def decontaminate(train: Sequence[Problem], evals: Sequence[Problem]) -> Tuple[List[Problem], List[str]]:
    """Drop TRAIN problems whose normalised prompt text equals an EVAL prompt (MBPP has near-duplicate pairs, e.g. eval 104 / train 569 and eval 427 / train 928)."""
    e_txt = {normalise_text(p.user_text) for p in evals}
    kept = [p for p in train if normalise_text(p.user_text) not in e_txt]
    return kept, [p.key for p in train if normalise_text(p.user_text) in e_txt]


def assert_no_leakage(train: Sequence[Problem], evals: Sequence[Problem]) -> Dict[str, Any]:
    """No EVAL task_id and no normalised EVAL prompt text occurs in TRAIN (raises AssertionError)."""
    t_ids = {p.task_id for p in train}
    e_ids = {p.task_id for p in evals}
    assert not (t_ids & e_ids), f"task_id overlap: {sorted(t_ids & e_ids)[:5]}"
    t_txt = {normalise_text(p.user_text) for p in train}
    e_txt = {normalise_text(p.user_text) for p in evals}
    assert not (t_txt & e_txt), "normalised prompt text overlap"
    t_ref = {normalise_text(p.reference) for p in train if p.reference}
    e_ref = {normalise_text(p.reference) for p in evals if p.reference}
    ref_overlap = len(t_ref & e_ref)
    return {"train_ids": len(t_ids), "eval_ids": len(e_ids), "id_overlap": 0, "text_overlap": 0, "reference_solution_overlap": ref_overlap,
            "train_id_range": [min(t_ids), max(t_ids)], "eval_id_range": [min(e_ids), max(e_ids)]}


def load_all(check_counts: bool = True) -> Tuple[List[Problem], List[Problem], List[Problem], Dict[str, Any]]:
    """(train after decontamination, eval, humaneval, leakage report).  The raw TRAIN has 464 problems; ``removed_for_text_overlap`` lists the dropped ones."""
    train, ev, he = load_mbpp_train(), load_mbpp_eval(), load_humaneval()
    if check_counts:
        assert (len(train), len(ev), len(he)) == (EXPECTED["train"], EXPECTED["eval"], EXPECTED["humaneval"]), (len(train), len(ev), len(he))
        assert all(511 <= p.task_id <= 974 for p in train) and all(11 <= p.task_id <= 510 for p in ev)
    train, removed = decontaminate(train, ev)
    rep = assert_no_leakage(train, ev)
    rep.update({"train_raw": EXPECTED["train"] if check_counts else None, "removed_for_text_overlap": removed, "train_after_decontamination": len(train)})
    return train, ev, he, rep


# ---------------------------------------------------------------- synthetic problems (smoke / unit tests; real tests, no network, no cache)
_SYN = [
    ("Write a function add_two(a, b) that returns the sum of a and b.", "add_two", "def add_two(a, b):\n    return a + b\n", ["assert add_two(1, 2) == 3", "assert add_two(-1, 1) == 0", "assert add_two(0, 0) == 0"]),
    ("Write a function mul_two(a, b) that returns the product of a and b.", "mul_two", "def mul_two(a, b):\n    return a * b\n", ["assert mul_two(2, 3) == 6", "assert mul_two(0, 5) == 0"]),
    ("Write a function biggest(xs) that returns the largest element of a list.", "biggest", "def biggest(xs):\n    return max(xs)\n", ["assert biggest([1, 5, 2]) == 5", "assert biggest([-3, -1]) == -1"]),
    ("Write a function rev(s) that reverses a string.", "rev", "def rev(s):\n    return s[::-1]\n", ["assert rev('abc') == 'cba'", "assert rev('') == ''"]),
    ("Write a function is_even(n) that tells whether n is even.", "is_even", "def is_even(n):\n    return n % 2 == 0\n", ["assert is_even(4) is True", "assert is_even(3) is False"]),
    ("Write a function total(xs) that sums a list of numbers.", "total", "def total(xs):\n    return sum(xs)\n", ["assert total([1, 2, 3]) == 6", "assert total([]) == 0"]),
    ("Write a function square(n) returning n squared.", "square", "def square(n):\n    return n * n\n", ["assert square(3) == 9", "assert square(-2) == 4"]),
    ("Write a function first_char(s) returning the first character.", "first_char", "def first_char(s):\n    return s[0]\n", ["assert first_char('xy') == 'x'"]),
    ("Write a function count_a(s) counting the letter a.", "count_a", "def count_a(s):\n    return s.count('a')\n", ["assert count_a('banana') == 3", "assert count_a('xyz') == 0"]),
    ("Write a function minus(a, b) returning a - b.", "minus", "def minus(a, b):\n    return a - b\n", ["assert minus(5, 3) == 2", "assert minus(0, 1) == -1"]),
    ("Write a function twice(s) returning s repeated two times.", "twice", "def twice(s):\n    return s * 2\n", ["assert twice('ab') == 'abab'"]),
    ("Write a function last(xs) returning the last element of a list.", "last", "def last(xs):\n    return xs[-1]\n", ["assert last([1, 2, 3]) == 3", "assert last(['a']) == 'a'"]),
]


def synthetic_sets() -> Tuple[List[Problem], List[Problem], List[Problem]]:
    """(train 6, eval 6, humaneval-style 2).  Disjoint by construction; the reference solutions pass the real gate."""
    def mk(i: int, src: str, base: int) -> Problem:
        text, name, ref, tests = _SYN[i]
        p = Problem(key=f"syn-{src}-{i}", source=f"synthetic-{src}", task_id=base + i, user_text=text, visible_tests=tests[:1], all_tests=tests, reference=ref, entry_point=name)
        p.messages = build_messages(text, tests[0])
        return p

    train = [mk(i, "train", 1000) for i in range(0, 6)]
    ev = [mk(i, "eval", 2000) for i in range(6, 12)]
    he = []
    for i in (0, 3):
        text, name, ref, tests = _SYN[i]
        prompt = f"def {name}(*args):\n    \"\"\"{text}\"\"\"\n"
        p = Problem(key=f"syn-he-{i}", source="synthetic-humaneval", task_id=f"SynHE/{i}", user_text=prompt, visible_tests=[], setup="",
                    all_tests=["def check(candidate):\n    " + "\n    ".join(t.replace(name, "candidate", 1) for t in tests), f"check({name})"], reference=ref, entry_point=name)
        p.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        he.append(p)
    return train, ev, he
