"""Dependency-light checks for generated HUD assets and standalone shortcuts.

Load the template module itself, avoiding the game package's eager ML imports.
These contract checks do not substitute for real-browser layout or Windows cmd
execution. Opt in to the browser checks with HIVEMIND_BROWSER_EXECUTABLE.
"""
from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / 'steam_dist/HiveMind_Steam_FreeEdition/web_ui/index.html'


@pytest.fixture(scope='module')
def hud():
    spec = importlib.util.spec_from_file_location('hud_distribution_assets', ROOT / 'src/hivemind/game/steam_edition.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Markup(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


@pytest.mark.parametrize('shipped', [False, True])
def test_hud_embeds_a_valid_offline_favicon(hud, shipped):
    html = SHIPPED.read_text() if shipped else hud._generate_cyberpunk_hud_html({})
    icon, = [attrs for tag, attrs in Markup(html).tags if tag == 'link' and attrs.get('rel') == 'icon']
    assert icon['type'] == 'image/svg+xml'
    assert icon['href'].startswith('data:image/svg+xml,')
    svg = ET.fromstring(unquote(icon['href'].split(',', 1)[1]))
    assert svg.tag == '{http://www.w3.org/2000/svg}svg'
    assert svg.attrib['viewBox'] == '0 0 32 32'


@pytest.mark.parametrize('shipped', [False, True])
def test_hud_layout_contract_keeps_wide_table_in_keyboard_scroll_region(hud, shipped):
    html = SHIPPED.read_text() if shipped else hud._generate_cyberpunk_hud_html({})
    assert 'minmax(min(400px, 100%), 1fr)' in html
    assert re.search(r'\.card\s*\{\s*min-width:\s*0;', html)
    assert re.search(r'\.table-scroll\s*\{[^}]*overflow-x:\s*auto;', html)
    assert not re.search(r'(?:body|html)\s*\{[^}]*overflow(?:-x)?:\s*hidden', html)
    assert 'overflow-wrap: anywhere;' in html
    tags = Markup(html).tags
    table_index = next(i for i, (tag, attrs) in enumerate(tags) if attrs.get('id') == 'leaderboard-table')
    tag, region = tags[table_index - 1]
    assert tag == 'div' and region['class'] == 'table-scroll'
    assert region['role'] == 'region' and region['tabindex'] == '0'
    assert 'scroll horizontally' in region['aria-label']
    assert len([tag for tag, _ in tags if tag == 'th']) == 9
    # The layout-only patch must ship exactly the source renderer.
    renderer = re.search(r'<script id="georing-webgl-renderer">(.*?)</script>', html, re.S)
    assert renderer[1].strip() == hud._GEORING_WEBGL_SCRIPT.strip()


def test_generated_and_shipped_layout_match(hud):
    generated = hud._generate_cyberpunk_hud_html({})
    assert re.search(r'<head>(.*?)</head>', generated, re.S)[1] == re.search(r'<head>(.*?)</head>', SHIPPED.read_text(), re.S)[1]


@pytest.mark.parametrize('shipped', [False, True])
def test_layout_patch_preserves_valid_javascript(hud, shipped, tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to check the HUD JavaScript syntax')
    html = SHIPPED.read_text() if shipped else hud._generate_cyberpunk_hud_html({})
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.S)
    assert len(scripts) == 2
    for index, script in enumerate(scripts):
        file = tmp_path / f'hud-{index}.js'
        file.write_text(script)
        result = subprocess.run([node, '--check', str(file)], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr


def test_source_distribution_keeps_checkout_launchers(hud, tmp_path):
    package = hud.build_free_steam_edition_package({}, output_root=tmp_path / 'source')
    campaign = Path(package['windows_launcher_bat']).read_text()
    web = Path(package['windows_hud_launcher_bat']).read_text()
    assert '-m hivemind.cli.main play-hack-game --season all' in campaign
    assert '-m hivemind.cli.main serve-hack-hud --port 8090' in web
    assert '.venv\\Scripts\\python.exe' in campaign and '.venv\\Scripts\\python.exe' in web


def test_standalone_shortcuts_preserve_exact_executable_and_data_paths(hud, monkeypatch, tmp_path):
    executable = 'C:\\HiveMind Release\\caf\u00e9 & 100% !\\HiveMindHUD.exe'
    data = Path('D:\\HiveMind Data\\user ! & 50%')
    monkeypatch.setattr(hud.sys, 'executable', executable)
    package = hud.build_free_steam_edition_package({}, output_root=tmp_path / 'frozen', standalone_data_dir=data)
    for key in ('windows_launcher_bat', 'windows_hud_launcher_bat'):
        launcher = Path(package[key]).read_text()
        assert f'"{executable.replace("%", "%%")}" --data-dir "{str(data).replace("%", "%%")}"' in launcher
        assert 'setlocal DisableDelayedExpansion' in launcher
        assert 'chcp 65001 >nul' in launcher
        assert 'exit /b %errorlevel%' in launcher
        assert 'python' not in launcher and 'cd /d' not in launcher
        assert '--port' not in launcher and 'start "" "http' not in launcher
    campaign = Path(package['windows_launcher_bat']).read_text()
    assert 'CLI campaign requires the source distribution' in campaign
    assert 'Opening the standalone HUD instead' in campaign


@pytest.mark.parametrize('bad', ['C:\\bad"path', 'C:\\bad\npath', 'C:\\bad\rpath'])
def test_standalone_shortcuts_reject_invalid_batch_paths(hud, monkeypatch, bad):
    monkeypatch.setattr(hud.sys, 'executable', bad)
    with pytest.raises(ValueError, match='Invalid Windows launcher path'):
        hud._generate_standalone_hud_launcher(Path('C:/data'))


def test_standalone_data_dir_is_explicitly_passed_through_to_package():
    # The API imports the optional ML runtime; keep this wiring check static in
    # the dependency-light suite and exercise the full app in the regular suite.
    source = (ROOT / 'src/hivemind/cli/standalone.py').read_text()
    assert 'create_hack_miner_fastapi_app(standalone_data_dir=data_dir if is_frozen() else None)' in source
    source = (ROOT / 'src/hivemind/game/steam_edition.py').read_text()
    assert 'game_report=game_report, standalone_data_dir=standalone_data_dir,' in source


@pytest.mark.skipif(not os.environ.get('HIVEMIND_BROWSER_EXECUTABLE'), reason='Opt in with an installed browser path; no browser downloads')
@pytest.mark.parametrize('shipped', [False, True])
def test_real_browser_narrow_layout_and_embedded_favicon(hud, shipped):
    """Real browser against static HUD + test API fixtures, not the frozen server."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from playwright.sync_api import sync_playwright

    html = SHIPPED.read_text() if shipped else hud._generate_cyberpunk_hud_html(json.loads(
        re.search(r'let GAME_DATA = (.*?);\n', SHIPPED.read_text(), re.S)[1]))
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path == '/':
                body, content_type = html.encode(), 'text/html; charset=utf-8'
            elif self.path.startswith('/api/'):
                body, content_type = b'{"nodes":[],"ring_links":[],"contracts":[],"missions":[]}', 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=os.environ['HIVEMIND_BROWSER_EXECUTABLE'], headless=True)
            for dpr in (1, 2, 3):
                context = browser.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=dpr)
                page = context.new_page()
                page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='load')
                page.wait_for_selector('#leaderboard-table tbody tr')
                for width in (412, 320, 800, 1280, 412):
                    page.set_viewport_size({'width': width, 'height': 915})
                    geometry = page.evaluate('''() => ({
                        viewport: document.documentElement.clientWidth,
                        page: document.documentElement.scrollWidth,
                        cards: [...document.querySelectorAll('.card')].map(el => el.getBoundingClientRect().right),
                        overflow: getComputedStyle(document.body).overflowX,
                    })''')
                    assert geometry['page'] == geometry['viewport'], (dpr, width, geometry)
                    assert all(right <= width for right in geometry['cards'])
                    assert geometry['overflow'] != 'hidden'
                region = page.get_by_role('region', name='Leaderboard, scroll horizontally to view all columns')
                region.focus()
                page.keyboard.press('ArrowRight')
                page.wait_for_function("() => document.querySelector('.table-scroll').scrollLeft > 0")
                assert page.evaluate("() => document.activeElement.classList.contains('table-scroll')")
                context.close()
            browser.close()
        assert '/favicon.ico' not in requests, 'The embedded icon must prevent fallback favicon requests'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_full_sequence_trace_is_bounded_and_preserves_webgl_calls():
    import ast
    tree = ast.parse((ROOT / 'tools/browser_evidence/hud_browser_evidence.py').read_text())
    script = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'TRACE_SCRIPT' for target in node.targets))
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the executable trace test')
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const script = require('node:fs').readFileSync(0, 'utf8');
const handlers = [];
const canvas = {id:'webgl-georing-canvas', getBoundingClientRect:()=>({x:1,y:2,width:300,height:320}), hasPointerCapture:()=>true};
let calls = 0, lost = false;
class GL {
  constructor() { this.canvas = canvas; }
  getUniformLocation(program, name) { return {name}; }
  uniform2f(location, x, y) { calls++; return 'unchanged'; }
  isContextLost() { return lost; }
}
const document = {getElementById:()=>canvas, activeElement:canvas, visibilityState:'visible'};
const window = {scrollX:0,scrollY:12, WebGLRenderingContext:GL, WebGL2RenderingContext:null,
  __hudEvidence:{raf:{run:3},draws:{georing:9}},
  addEventListener:(type, handler, capture)=>handlers.push({type,handler,capture})};
vm.runInNewContext(script,{window,document,WeakMap,performance:{now:()=>1}});
const trace=window.__hudInteractionTrace, gl=new GL();
const angles=gl.getUniformLocation({},'angles'), other=gl.getUniformLocation({},'other');
assert.equal(gl.uniform2f(angles,0.8,-0.03),'unchanged');
assert.equal(calls,1);
assert.equal(JSON.stringify(trace.rendered_angles),'[0.8,-0.03]');
assert.equal(trace.records.length,1);
gl.uniform2f(other,9,9);
assert.equal(trace.records.length,1,'non-angle uniforms must not be logged');
const event={target:canvas,pointerId:1,clientX:20,clientY:30,buttons:1,defaultPrevented:false};
for(const h of handlers.filter(h=>h.type==='pointermove')) h.handler(event);
assert.equal(trace.records.length,3);
assert.equal(trace.records[1].phase,'capture');
assert.equal(trace.records[2].phase,'bubble');
assert.equal(trace.records[2].target_has_capture,true);
assert.equal(trace.records[2].raf_run,3);
assert.equal(trace.records[2].draws,9);
lost=true;
for(const h of handlers.filter(h=>h.type==='webglcontextlost')) h.handler({target:canvas});
assert.equal(trace.context_epoch,1);
assert.equal(trace.records.at(-1).context_lost,true);
trace.limit=trace.records.length;
trace.mark('over limit');
assert.equal(trace.dropped,1);
assert.equal(trace.records.length,trace.limit);
assert.equal(calls,2,'observer must not add WebGL submissions');
console.log('bounded trace preserves calls and captures pointer/frame/context evidence');
"""
    result = subprocess.run([node, '-e', harness], input=script, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_standalone_shortcut_quotes_drive_root_without_escaping_closing_quote(hud, monkeypatch):
    monkeypatch.setattr(hud.sys, 'executable', 'C:\\Apps\\HiveMindHUD.exe')
    launcher = hud._generate_standalone_hud_launcher('D:\\')
    assert '--data-dir "D:\\\\"' in launcher
