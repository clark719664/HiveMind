"""PR #24: the Geo-Ring panel must contain a real WebGL renderer, not just a JSON descriptor."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from hivemind.game.steam_edition import create_hack_miner_fastapi_app

    return TestClient(create_hack_miner_fastapi_app())


def test_hud_has_canvas_and_real_webgl_script(client) -> None:
    html = client.get("/").text
    assert 'id="webgl-georing-map"' in html
    assert 'id="webgl-georing-canvas"' in html
    for needle in (
        "getContext('webgl2'",
        "getContext('webgl', opts)",
        "gl.createShader",
        "gl.drawArrays",
        "/api/georing-topology",
        "/ws/swarm-telemetry",
        "devicePixelRatio",
        "__georingStats",
        "WEBGL UNAVAILABLE",
    ):
        assert needle in html, needle


def test_topology_peers_have_lat_lon_for_the_renderer(client) -> None:
    topo = client.get("/api/georing-topology").json()
    assert topo["renderer"] == "webgl_3d_georing_globe_v3"
    assert len(topo["nodes"]) >= 3
    conts = set()
    for node in topo["nodes"]:
        assert -90 <= node["lat"] <= 90
        assert -180 <= node["lon"] <= 180
        assert node["node_id"] and node["nat_mode"] and node["rtt_ms"] > 0
        conts.add(node["region"].split("-")[0])
    assert {"EU", "US", "APAC"} <= conts


def test_hud_has_no_external_script_or_style_urls(client) -> None:
    html = client.get("/").text
    assert not re.findall(r"<script[^>]+src\s*=\s*[\"']?(?:https?:)?//", html, re.I)
    assert not re.findall(r"<link[^>]+href\s*=\s*[\"']?(?:https?:)?//", html, re.I)
    assert not re.findall(r"@import\s+url\(\s*[\"']?(?:https?:)?//", html, re.I)


def _embedded_snapshot(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="georing-topology-snapshot">(.*?)</script>', html, re.S)
    assert match, "HUD has no embedded topology snapshot for file:// use"
    return json.loads(match.group(1))


def test_hud_embeds_the_api_topology_for_offline_use(client) -> None:
    api_nodes = client.get("/api/georing-topology").json()["nodes"]
    assert _embedded_snapshot(client.get("/").text)["nodes"] == api_nodes
    assert "location.protocol === 'file:'" in client.get("/").text


def test_generated_hud_embeds_the_snapshot() -> None:
    from hivemind.game.hf_hack_miner import build_georing_3d_topology_snapshot
    from hivemind.game.steam_edition import _generate_cyberpunk_hud_html

    html = _generate_cyberpunk_hud_html({})
    assert _embedded_snapshot(html)["nodes"] == build_georing_3d_topology_snapshot()["nodes"]
    assert html.index('id="georing-topology-snapshot"') < html.index('id="webgl-georing-map"')
