"""Real headless-browser render check for the Geo-Ring WebGL panel (skips without Edge/Chrome + Pillow)."""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("uvicorn")
Image = pytest.importorskip("PIL.Image")

_CANDIDATES = [
    os.environ.get("HIVEMIND_TEST_BROWSER", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    shutil.which("msedge") or "",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]
BROWSER = next((c for c in _CANDIDATES if c and Path(c).exists()), None)
pytestmark = pytest.mark.skipif(BROWSER is None, reason="no headless Chromium/Edge available")

FLAGS = [
    "--headless=new",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--disable-gpu-sandbox",
    "--window-size=1000,700",
]


@pytest.fixture(scope="module")
def server_url():
    import uvicorn

    from hivemind.game.steam_edition import create_hack_miner_fastapi_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_hack_miner_fastapi_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    assert server.started
    yield f"http://127.0.0.1:{port}/?georing=solo"
    server.should_exit = True
    thread.join(timeout=10)


def _dom(url: str, budget: int) -> str:
    out = subprocess.run(
        [BROWSER, *FLAGS, f"--virtual-time-budget={budget}", "--dump-dom", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    return out.stdout


def _shot(url: str, budget: int, path: Path) -> None:
    subprocess.run(
        [BROWSER, *FLAGS, f"--virtual-time-budget={budget}", f"--screenshot={path.as_posix()}", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert path.exists()


def _attrs(dom: str) -> dict:
    tag = re.search(r'<div class="card" id="webgl-georing-map"[^>]*>', dom)
    assert tag, "georing panel missing from rendered DOM"
    return dict(re.findall(r'data-([a-z-]+)="([^"]*)"', tag.group(0)))


def _crop(path: Path, rect: str):
    x, y, w, h = (int(v) for v in rect.split(","))
    return Image.open(path).convert("RGB").crop((x, y, x + w, y + h))


def test_headless_browser_renders_rotating_continent_colored_globe(server_url, tmp_path) -> None:
    attrs = _attrs(_dom(server_url, 8000))
    if attrs["webgl"] in ("unavailable", "pending"):
        pytest.skip("headless browser could not create a WebGL context")
    assert attrs["webgl"] in ("webgl1", "webgl2")
    assert attrs["errors"] == "0"
    assert int(attrs["nodes"]) >= 3 and int(attrs["arcs"]) >= 1 and int(attrs["frames"]) >= 1

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _shot(server_url, 8000, a)
    _shot(server_url, 40000, b)
    ca, cb = _crop(a, attrs["canvas-rect"]), _crop(b, attrs["canvas-rect"])

    colors = ca.getcolors(maxcolors=1 << 20)
    assert colors is not None and len(colors) > 20, "canvas looks blank"

    def count(img, pred):
        return sum(n for n, (r, g, bl) in img.getcolors(maxcolors=1 << 20) if pred(r, g, bl))

    eu = lambda r, g, bl: bl > 180 and r < 120 and g > 120 and bl > r + 80
    us = lambda r, g, bl: r > 200 and 90 < g < 180 and bl < 110
    apac = lambda r, g, bl: g > 180 and r < 130 and bl < 160 and g > bl + 40

    # The initial view contains EU, US and APAC peers; peer markers are drawn last so
    # the continent hue is not painted over by the guild node sharing a home node.
    apac_view = tmp_path / "c.png"
    _shot(server_url + "&georing_yaw=3.3", 8000, apac_view)
    cc = _crop(apac_view, attrs["canvas-rect"])
    assert count(ca, eu) > 0, "no EU (blue) pixels"
    assert count(ca, us) > 0, "no US (orange) pixels"
    assert count(cc, apac) > 0, "no APAC (green) pixels at the APAC-facing yaw"

    from PIL import ImageChops

    diff = ImageChops.difference(ca, cb)
    assert diff.getbbox() is not None, "globe did not rotate between captures"
