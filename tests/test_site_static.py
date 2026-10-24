"""Static checks of the community site in docs/site/ (CPU only, no browser): no external requests, local references exist, honest-status and
non-monetary wording present, no forbidden claims, accessibility basics, and (when node is installed) the DOM-free parts of app.js / game.js.

These checks are structural. They do not render the pages: real browser rendering is not verified by this test.
"""

from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import pytest

SITE = Path(__file__).resolve().parent.parent / "docs" / "site"
PAGES = ["index.html", "game.html"]
ALLOWED_LINK_PREFIX = "https://github.com/xaviercallens/HiveMind"
NODE = shutil.which("node")
LEGAL = SITE.parent.parent / "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md"


def legal_phrases() -> list[str]:
    """Verbatim wording taken from the LEGAL file at test time (markdown emphasis removed), so the pages cannot drift from it."""
    text = re.sub(r"\s+", " ", LEGAL.read_text(encoding="utf-8").replace("*", "").replace("`", ""))
    money = re.search(r"NO MONEY, FIAT CURRENCY[^.]*?OR PAID\.", text)
    units = re.search(r"internal, non-monetary, non-transferable accounting units measuring contributed and reciprocal computing power", text)
    assert money and units, "LEGAL file wording changed: update the site notice and this test"
    return [money.group(0), units.group(0)]


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.text: list[str] = []
        self.style_blocks: list[str] = []
        self._in_style = False
        self._stack: list[tuple[str, dict[str, str]]] = []
        self.banner_text: list[str] = []
        self._banner_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        self.tags.append((tag, a))
        if tag == "style":
            self._in_style = True
        if self._banner_depth:
            self._banner_depth += 1
        elif a.get("id") == "status-banner":
            self._banner_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False
        if self._banner_depth:
            self._banner_depth -= 1

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self._in_style:
            self.style_blocks.append(data)
        if self._banner_depth:
            self.banner_text.append(data)


def parse(name: str) -> Collector:
    c = Collector()
    c.feed((SITE / name).read_text(encoding="utf-8"))
    return c


def refs(c: Collector) -> list[tuple[str, str, str]]:
    out = []
    for tag, a in c.tags:
        for attr in ("src", "href", "action", "poster", "data", "srcset", "formaction"):
            if attr in a:
                out.append((tag, attr, a[attr]))
    return out


@pytest.mark.parametrize("page", PAGES)
def test_no_external_resources(page: str) -> None:
    c = parse(page)
    for tag, attr, value in refs(c):
        parsed = urlparse(value)
        is_remote = parsed.scheme in ("http", "https") or value.startswith("//")
        if tag == "a" and attr == "href" and is_remote:
            assert value.startswith(ALLOWED_LINK_PREFIX), f"{page}: external link {value}"
        else:
            assert not is_remote, f"{page}: external resource <{tag} {attr}={value}>"
            assert parsed.scheme in ("", "mailto"), f"{page}: unexpected scheme in {value}"
    for block in c.style_blocks:
        assert "url(" not in block.replace(" ", "").lower() or "http" not in block
    for _, a in c.tags:
        assert "http" not in a.get("style", "").lower()


@pytest.mark.parametrize("page", PAGES)
def test_every_local_reference_exists(page: str) -> None:
    c = parse(page)
    checked = 0
    for tag, attr, value in refs(c):
        parsed = urlparse(value)
        if parsed.scheme or value.startswith("#") or not parsed.path:
            continue
        assert (SITE / parsed.path).resolve().exists(), f"{page}: missing local file {value}"
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize("page", PAGES)
def test_site_files_make_no_network_calls_and_use_storage_safely(page: str) -> None:
    for name in ("swarm_lab.js", "app.js", "game.js", "data.js", "style.css", page):
        text = (SITE / name).read_text(encoding="utf-8")
        for bad in ("fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "EventSource", "importScripts", "import(", "document.cookie", "@import", "navigator.serviceWorker"):
            assert bad not in text, f"{name} uses {bad}"
        urls = re.findall(r"https?://[^\s\"'<>)]+", text)
        assert all(u.startswith(ALLOWED_LINK_PREFIX) for u in urls), f"{name}: {urls}"
    for name in ("app.js", "game.js"):
        text = (SITE / name).read_text(encoding="utf-8")
        for m in re.finditer(r"localStorage\.(?:get|set)Item", text):
            assert "try" in text[max(0, m.start() - 120): m.start()], f"{name}: localStorage outside try/catch near {m.start()}"


@pytest.mark.parametrize("page", PAGES)
def test_honest_status_banner_and_nonmonetary_wording(page: str) -> None:
    c = parse(page)
    banner = " ".join(c.banner_text)
    assert banner.strip(), f"{page}: no #status-banner"
    low = re.sub(r"\s+", " ", banner).lower()
    assert "educational toy" in low and "not a reproduction" in low and "no peer review of record exists" in low
    full = re.sub(r"\s+", " ", " ".join(c.text))
    for phrase in legal_phrases():
        assert phrase in full, f"{page}: missing verbatim LEGAL wording: {phrase}"
    assert "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md" in full
    assert "learning score" in full.lower()


@pytest.mark.parametrize("page", PAGES)
def test_no_forbidden_claims_in_site_files(page: str) -> None:
    for name in ("swarm_lab.js", "app.js", "game.js", "data.js", page):
        low = (SITE / name).read_text(encoding="utf-8").lower()
        assert "peer-reviewed" not in low and "peer reviewed" not in low, name
        assert "licen" not in low, f"{name}: no licence statement may appear"
        assert "appid" not in low and "app id" not in low and "steam" not in low, name
        assert "price" not in low and "store page" not in low, name


def test_index_structure_and_accessibility() -> None:
    c = parse("index.html")
    tags = [t for t, _ in c.tags]
    assert tags.count("h1") == 1 and tags.count("main") == 1 and tags.count("footer") == 1
    html = next(a for t, a in c.tags if t == "html")
    assert html.get("lang") == "en"
    assert any(t == "meta" and a.get("name") == "viewport" for t, a in c.tags)
    ids = {a["id"] for _, a in c.tags if "id" in a}
    assert len(ids) == len([1 for _, a in c.tags if "id" in a]), "duplicate ids"
    labelled = {a["for"] for t, a in c.tags if t == "label" and "for" in a}
    for t, a in c.tags:
        if t in ("input", "select", "textarea") and a.get("type") != "hidden":
            assert a.get("id") in labelled or "aria-label" in a, f"unlabelled control {a}"
    for section in ("about", "lab", "real", "contribute"):
        assert section in ids
    for needed in ("lab-form", "lab-chart", "lab-domains", "lab-strip", "lab-compare", "real-results", "theme-toggle", "nonmonetary-notice"):
        assert needed in ids
    text = re.sub(r"\s+", " ", " ".join(c.text))
    for needle in ("CONTRIBUTING.md", "docs/community/SWARM_LAB.md", "python3 scripts/swarm_lab.py simulate --preset persistent-straggler", "github.com/xaviercallens/HiveMind"):
        assert needle in text or needle in " ".join(a.get("href", "") for _, a in c.tags)
    scripts = [a.get("src") for t, a in c.tags if t == "script"]
    assert scripts == ["swarm_lab.js", "data.js", "app.js"]  # engine first, data before the renderer


def test_game_structure() -> None:
    c = parse("game.html")
    ids = {a["id"] for _, a in c.tags if "id" in a}
    for needed in ("level-nav", "cards-K", "cards-policy", "cards-grace", "cards-merge", "run", "result", "debrief", "star-rule", "theme-toggle", "game-chart", "game-strip"):
        assert needed in ids
    assert [a.get("src") for t, a in c.tags if t == "script"] == ["swarm_lab.js", "app.js", "game.js"]
    low = re.sub(r"\s+", " ", " ".join(c.text)).lower()
    assert "not pts or $hive" in low and "no multiplayer" in low


def test_style_has_dark_mode_reduced_motion_and_responsive_rules() -> None:
    css = (SITE / "style.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css and ':root[data-theme="dark"]' in css and "prefers-reduced-motion: reduce" in css
    assert "focus-visible" in css and "max-width: 480px" in css


def test_data_js_is_valid_and_tagged() -> None:
    text = (SITE / "data.js").read_text(encoding="utf-8")
    assert "window.HM_DATA = " in text
    data = json.loads(text.split("window.HM_DATA = ", 1)[1].rstrip().rstrip(";"))
    assert data["schema"] == "hivemind-site-data/v1" and {"e13", "e14", "e15"} <= set(data)
    for blk in (data["e13"]["registered"], data["e13"]["rotating_straggler"], data["e13"]["m2_process_identity"], data["e14"], data["e15"]):
        assert isinstance(blk["available"], bool) and blk.get("source")
        if blk["available"]:
            assert blk.get("generated_utc") and ("receipt12" in blk or "file_sha256_12" in blk)


# ---------------------------------------------------------------------------------------------- node smoke tests (DOM-free logic)
NODE_SCRIPT = r"""
const path = require("path");
const dir = process.argv[1];
const Lab = require(path.join(dir, "swarm_lab.js"));
const Site = require(path.join(dir, "app.js"));
const Game = require(path.join(dir, "game.js"));
const sc = Lab.presetScenario("churn");
const res = Lab.simulate(sc), ref = Lab.simulate(Site.referenceScenario(sc));
const out = {
  svgs: [Site.lossChartSvg(res, ref), Site.lossChartSvg(res, ref, 3), Site.domainBarsSvg(res, ref), Site.stripSvg(res), Site.stripSvg(res, 2)],
  tables: [Site.roundsTableHtml(res), Site.compareTableHtml(Lab.comparePolicies(sc, Lab.POLICIES, [sc.straggle]), "fair")],
  cli: Site.cliCommand(sc),
  levels: Game.LEVELS.map((lv) => {
    const t = Game.thresholds(lv);
    const plays = Lab.POLICIES.map((p) => Game.play(lv, { K: 3, policy: p, grace: 0.1, merge: "direct" }));
    return { id: lv.id, stars: t.stars, best: t.best, strategies: t.strategies,
             plays: plays.map((p) => ({ score: p.score.score, stars: p.stars, lines: p.debrief.length, time: p.score.total_time })), rule: Game.starRuleText(lv) };
  }),
  results_null: Site.resultsHtml(null),
};
const fs = require("fs");
global.window = global;
eval(fs.readFileSync(path.join(dir, "data.js"), "utf8"));
out.results = Site.resultsHtml(global.HM_DATA);
out.results_empty = Site.resultsHtml({ generated_utc: "x", e13: {}, e14: { available: false, prereg: { available: false } }, e15: { available: false } });
out.results_e15 = Site.resultsHtml({ generated_utc: "x", e13: {}, e14: { available: false }, e15: { available: true, source: "E15_gwaya_rft_report.json", receipt12: "abc", run_status: "complete", smoke: true,
  conditions: { C0: { n: 10, passes: 4, pass_at_1: 0.4, wilson95: [0.1, 0.7] } }, paired_comparisons: { "C2-C0": { n: 10, diff: 0.1, b_only: 2, a_only: 1, mcnemar_p: 1 } },
  verdicts_note: "not computed here", report_excerpt: { controls_all_passed: true } } });
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_logic() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("node is not installed: the DOM-free JavaScript checks need `node` (>= 18) on PATH")
    proc = subprocess.run([NODE, "-e", NODE_SCRIPT, str(SITE)], capture_output=True, text=True, timeout=50, check=False, env={"PATH": os.environ.get("PATH", "")})
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_svg_builders_emit_well_formed_xml(node_logic: dict[str, Any]) -> None:
    for svg in node_logic["svgs"]:
        root = ET.fromstring(svg.replace("<svg ", "<svg xmlns=\"http://www.w3.org/2000/svg\" ", 1))
        assert root.tag.endswith("svg") and root.get("role") == "img"
        assert any(ch.tag.endswith("title") for ch in root) and any(ch.tag.endswith("desc") for ch in root)
    for table in node_logic["tables"]:
        ET.fromstring("<root>" + table.replace("&nbsp;", " ") + "</root>")
    assert node_logic["cli"] == "python3 scripts/swarm_lab.py simulate --policy fair --straggle persistent --churn-start 3 --churn-end 6"


def test_game_levels_and_star_thresholds(node_logic: dict[str, Any]) -> None:
    assert [lv["id"] for lv in node_logic["levels"]] == [1, 2, 3, 4, 5]
    for lv in node_logic["levels"]:
        s = lv["stars"]
        assert s["3"] >= s["2"] >= s["1"] > 0 and lv["best"] >= s["3"] and lv["strategies"] >= 54
        assert "computed from the engine" in lv["rule"]
        assert all(0 <= p["score"] <= 100 and 0 <= p["stars"] <= 3 and p["lines"] >= 4 for p in lv["plays"])
    by = {lv["id"]: lv["plays"] for lv in node_logic["levels"]}
    discard2, fair2, deferred2 = by[2]
    assert discard2["score"] < fair2["score"] and discard2["score"] < deferred2["score"]  # the E13 effect: dropping the persistent straggler's domain hurts
    discard5, fair5, deferred5 = by[5]
    assert fair5["time"] > discard5["time"] and fair5["score"] < deferred5["score"]  # on the time-budget level fairness costs points
    assert all(p["stars"] == 3 for p in by[1])  # IID data: every K=3 policy is fine


def test_results_panel_uses_data_and_degrades_gracefully(node_logic: dict[str, Any]) -> None:
    html = node_logic["results"]
    assert "E13_swarm_local_predictions.json" in html and "receipt" in html and "What this does not show" in html
    assert "supported" in html and "refuted" in html
    assert "did not load" in node_logic["results_null"]
    assert "No data available" in node_logic["results_empty"]
    e15 = node_logic["results_e15"]
    assert "SMOKE RUN" in e15 and "McNemar" in e15 and "What this does not show" in e15 and "E15_gwaya_rft_report.json" in e15
    assert "derived by the builder, registered rule" in html  # mapped E14 words are never shown as if the report said them
