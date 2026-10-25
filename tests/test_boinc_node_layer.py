"""CPU-only tests of the T4 node layer: multi-domain workunit data, per-domain evaluation, the governor's night window, idle wait."""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hivemind.boinc.real_client import moe_work  # noqa: E402
from hivemind.boinc.real_client.boinc_server_scripts import wait_for_idle  # noqa: E402
from hivemind.boinc.real_client.moe_app import eval_by_domain  # noqa: E402


def test_domain_registry_is_consistent_and_excludes_benchmarks() -> None:
    assert set(moe_work.DOMAIN_ORDER) == set(moe_work._SOURCES)
    joined = " ".join(s["ds"] for s in moe_work._SOURCES.values()).lower()
    assert "mbpp" not in joined and "humaneval" not in joined  # the repository's evaluation benchmarks are never training data


def test_rows_slices_across_row_groups_and_from_the_end(tmp_path: Path) -> None:
    f = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": [f"row{i}" for i in range(2500)]}), f, row_group_size=1000)
    assert [r["text"] for r in moe_work._rows(f, ["text"], 995, 10)] == [f"row{i}" for i in range(995, 1005)]  # crosses a group boundary
    assert [r["text"] for r in moe_work._rows(f, ["text"], -3, 3)] == ["row2497", "row2498", "row2499"]
    assert len(moe_work._rows(f, ["text"], 2495, 50)) == 5  # asking past the end returns what exists


def test_ids_stops_when_enough_tokens_and_skips_blank_docs() -> None:
    tok = lambda s: SimpleNamespace(input_ids=list(range(len(s))))  # noqa: E731 - one token per character
    ids = moe_work._ids(tok, ["", "  ", "a" * 10, "b" * 10, "c" * 10], need=15)
    assert 15 <= len(ids) < 40  # stopped after the second real document, did not read the third


class _Stub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels):  # a deterministic "loss": mean token id
        return SimpleNamespace(loss=input_ids.float().mean())


def test_eval_by_domain_per_domain_and_single_set_fallback() -> None:
    m = _Stub()
    per = eval_by_domain(m, {"eval_sets": {"a": [[1, 1]], "b": [[3, 3]], "empty": []}})
    assert per == {"a": 1.0, "b": 3.0}  # an empty domain is skipped, not divided by zero
    assert eval_by_domain(m, {"eval_tokens": [[2, 2]]}) == {"general": 2.0}  # workunits created before multi-domain evaluation


def test_governor_night_window_wraps_midnight() -> None:
    from t4_governor import in_night

    def at(h: int) -> float:  # 2026-10-25 is after the end of daylight saving: Paris = UTC+1
        return dt.datetime(2026, 10, 25, h, 0, tzinfo=dt.timezone.utc).timestamp()

    assert [in_night(at(h), "Europe/Paris", 22, 7) for h in (12, 20, 21, 23, 2, 5, 6)] == [False, False, True, True, True, True, False]
    assert in_night(at(10), "Europe/Paris", 8, 18) is True  # a non-wrapping window


def test_wait_for_idle_does_not_block_when_there_is_no_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIVEMIND_IDLE_URL", "http://127.0.0.1:9/nginx_status")  # nothing listens here
    t = time.time()
    wait_for_idle(quiet_s=5, max_wait_s=30)
    assert time.time() - t < 5  # a box without nginx must not stall validation
