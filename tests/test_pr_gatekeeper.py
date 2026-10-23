"""Pure-logic tests for scripts/pr_gatekeeper.py: trust policy, decision rule and forbidden-pattern scan (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pr_gatekeeper as g  # noqa: E402


def _pr(login: str, emails: list[str]) -> dict:
    return {"author": {"login": login}, "commits": [{"oid": "a" * 40, "authors": [{"email": e, "login": login}]} for e in emails]}


def test_owner_login_and_owner_email_are_required_together() -> None:
    assert g.trust_report(_pr("xaviercallens", ["callensxavier@gmail.com"]))["owner_authored"] is True
    assert g.trust_report(_pr("xaviercallens", ["bot@socrateai.com"]))["owner_authored"] is False
    assert g.trust_report(_pr("someoneelse", ["callensxavier@gmail.com"]))["owner_authored"] is False
    assert g.trust_report(_pr("xaviercallens", ["callensxavier@gmail.com", "clark719664@gmail.com"]))["owner_authored"] is False
    assert g.trust_report(_pr("xaviercallens", []))["owner_authored"] is False  # no commits -> nothing to trust


@settings(max_examples=60, deadline=None)
@given(st.text(min_size=1, max_size=40))
def test_variants_of_the_owner_address_are_never_the_owner(suffix: str) -> None:
    for email in (f"callensxavier+{suffix}@gmail.com", f"callens.xavier{suffix}@gmail.com", f"callensxavier@gmail.com <{suffix}@evil.com>"):
        assert g.trust_report(_pr("xaviercallens", [email]))["owner_authored"] is False


def test_decision_rule_orders_trust_state_and_checks() -> None:
    ok = {"security": True, "forbidden": True, "gwaya_audit": True, "pytest": True, "lean": True, "pipeline": True}
    assert g.decide(True, False, "MERGEABLE", ok) == "ACCEPT"
    assert g.decide(False, False, "MERGEABLE", ok).startswith("HOLD (not owner-authored")
    assert g.decide(True, True, "MERGEABLE", ok) == "HOLD (draft)"
    assert g.decide(True, False, "CONFLICTING", ok) == "HOLD (merge conflict)"
    bad = dict(ok, pytest=False, lean=False)
    assert g.decide(True, False, "MERGEABLE", bad) == "REFUSE (pytest, lean)"


def test_forbidden_scan_flags_only_added_lines() -> None:
    diff = "+++ b/x.py\n-GWAYA_ALLOW_UNISOLATED=1\n+os.environ['GWAYA_ALLOW_UNISOLATED'] = '1'\n+print('ok')\n+ referee_score = 9.4\n"
    hits = g.forbidden_scan(diff)
    whys = {h["why"] for h in hits}
    assert "never run GWAYA unisolated" in whys
    assert "a script may not emit a review score or claim peer review" in whys
    assert len(hits) == 2  # the removed line and the harmless print are not hits


def test_forbidden_scan_is_clean_on_ordinary_code() -> None:
    diff = "+def f(x: int) -> int:\n+    return x + 1\n+# is_peer_review stays false\n"
    assert g.forbidden_scan(diff) == []
