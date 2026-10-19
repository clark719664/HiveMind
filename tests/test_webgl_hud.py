"""Execute HUD JavaScript lifecycle regressions without requiring a GPU/browser."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hivemind.game.steam_edition import _GEORING_WEBGL_SCRIPT, _generate_cyberpunk_hud_html

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / 'steam_dist/HiveMind_Steam_FreeEdition/web_ui/index.html'

# This instrumented WebGL/DOM test double verifies JS behavior, not raster output.
NODE_HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
class Element {
  constructor() { this.listeners = {}; this.dataset = {}; this.children = []; this.width = 920; this.height = 320; this.value = '-1'; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  getBoundingClientRect() { return { left: 0, top: 0, width: this.cssWidth || 920, height: 320 }; }
  setPointerCapture() {}
  replaceChildren(...items) { this.children = items; }
  add(item) { this.children.push(item); }
}
const ids = Object.fromEntries(['webgl-georing-canvas','georing-terminal','georing-refresh','georing-reset','georing-node'].map(id=>[id,new Element()]));
const canvas = ids['webgl-georing-canvas'], terminal = ids['georing-terminal'];
let lost = false, draws = 0, uploads = 0, requests = 0, frameId = 0, angleValues = [];
const frames = new Map(), timers = new Map(), windowEvents = {};
const gl = { VERTEX_SHADER:1, FRAGMENT_SHADER:2, COMPILE_STATUS:3, LINK_STATUS:4, ARRAY_BUFFER:5, STATIC_DRAW:6, LINES:7, POINTS:8, COLOR_BUFFER_BIT:16, DEPTH_BUFFER_BIT:32, DEPTH_TEST:9, FLOAT:10,
 createShader:()=>({}), shaderSource(){}, compileShader(){}, getShaderParameter:()=>input.mode !== 'shader_error', getShaderInfoLog:()=> 'fixture shader error', deleteShader(){},
 createProgram:()=>({}), attachShader(){}, linkProgram(){}, getProgramParameter:()=>true, deleteProgram(){}, getAttribLocation:()=>0, getUniformLocation:()=>({}),
 createBuffer:()=>({}), bindBuffer(){}, bufferData(){uploads++}, deleteBuffer(){}, isContextLost:()=>lost,
 viewport(){}, clearColor(){}, clear(){}, useProgram(){}, enable(){}, uniform2f(loc,x,y){angleValues.push([x,y])}, uniform1f(){}, enableVertexAttribArray(){}, vertexAttribPointer(){}, uniform4fv(){}, uniform1i(){}, drawArrays(){draws++}
};
canvas.getContext=()=>input.mode === 'no_webgl' ? null : gl;
const nodes = [
 {node_id:'a',region:'Paris',xyz:[0.6,0.7,0.1],hosted_experts:['Expert A'],rtt_ms:1},
 {node_id:'b',region:'Tokyo',xyz:[-0.6,0.5,0.5],hosted_experts:[],rtt_ms:2}
];
const context = { document:{getElementById:id=>ids[id]}, window:{devicePixelRatio:1,addEventListener:(name,fn)=>windowEvents[name]=fn},
 Option:class {constructor(text,value){this.text=text;this.value=value}},
 AbortController, Float32Array, console,
 requestAnimationFrame:fn=>{frames.set(++frameId,fn);return frameId}, cancelAnimationFrame:id=>frames.delete(id),
 setTimeout:fn=>{timers.set(++frameId,fn);return frameId},clearTimeout:id=>timers.delete(id),
 fetch:async()=>{requests++;return {ok:input.mode !== 'http_error',status:503,json:async()=>({nodes,ring_links:[{source:'a',target:input.mode==='bad_link'?'missing':'b'}]})}}
};
function flush(){for(const [id,fn] of [...frames]){frames.delete(id);fn()}}
(async()=>{
 vm.runInNewContext(input.script,context,{timeout:3000});
 await new Promise(resolve=>setImmediate(resolve));flush();
 if(input.mode==='no_webgl'||input.mode==='shader_error') {
   assert.match(terminal.textContent,/WEBGL UNAVAILABLE/);assert.equal(draws,0);assert.equal(requests,0);
 } else if(input.mode==='http_error'||input.mode==='bad_link') {
   assert.match(terminal.textContent,/TOPOLOGY UNAVAILABLE/);assert.equal(ids['georing-refresh'].disabled,false);
 } else {
   assert.match(terminal.textContent,/DEMO FIXTURE.*2 nodes \/ 1 links/);assert.ok(draws>=3);assert.ok(uploads>=3);assert.equal(requests,1);
   ids['georing-node'].listeners.change({target:{value:'0'}});assert.match(terminal.textContent,/Paris.*a/);
   input.mode='http_error';ids['georing-refresh'].listeners.click();
   await new Promise(resolve=>setImmediate(resolve));flush();
   assert.match(terminal.textContent,/TOPOLOGY UNAVAILABLE/);assert.equal(ids['georing-node'].children.length,1);
   input.mode='success';ids['georing-refresh'].listeners.click();
   await new Promise(resolve=>setImmediate(resolve));flush();
   assert.match(terminal.textContent,/2 nodes \/ 1 links/);assert.equal(requests,3);
   const original=angleValues.at(-1);
   canvas.listeners.pointerdown({clientX:10,clientY:10,pointerId:1});
   canvas.listeners.pointermove({clientX:60,clientY:20});canvas.listeners.pointerup({clientX:60,clientY:20});flush();
   assert.notDeepEqual(angleValues.at(-1),original);
   canvas.listeners.keydown({key:'ArrowRight',preventDefault(){}});flush();
   canvas.cssWidth=600;windowEvents.resize();flush();assert.equal(canvas.width,600);
   lost=true;canvas.listeners.webglcontextlost({preventDefault(){}});assert.match(terminal.textContent,/CONTEXT LOST/);const count=draws;
   flush();assert.equal(draws,count);
   lost=false;canvas.listeners.webglcontextrestored();flush();assert.ok(draws>count);assert.match(terminal.textContent,/DEMO FIXTURE/);
   windowEvents.pagehide({persisted:false});const final=draws;windowEvents.resize();flush();assert.equal(draws,final);
 }
 assert.equal(timers.size,0,'fetch timeout must be cleared');
 console.log(JSON.stringify({mode:input.mode,draws,uploads,requests,status:terminal.textContent}));
})().catch(error=>{console.error(error);process.exit(1)});
"""


def node() -> str:
    executable = shutil.which('node')
    if executable is None:
        pytest.skip('Node is required for executable JavaScript HUD tests')
    return executable


@pytest.mark.parametrize('mode', ['success', 'no_webgl', 'shader_error', 'http_error', 'bad_link'])
def test_renderer_lifecycle_in_node(mode: str) -> None:
    result = subprocess.run([node(), '-e', NODE_HARNESS], input=json.dumps({'script': _GEORING_WEBGL_SCRIPT, 'mode': mode}), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['mode'] == mode


@pytest.mark.parametrize('shipped', [False, True])
def test_hud_javascript_has_valid_syntax(tmp_path: Path, shipped: bool) -> None:
    html = SHIPPED.read_text() if shipped else _generate_cyberpunk_hud_html({})
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.S)
    assert len(scripts) == 2
    for index, source in enumerate(scripts):
        path = tmp_path / f'hud_{index}.js'
        path.write_text(source)
        result = subprocess.run([node(), '--check', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert '<<<<<<<' not in html and '>>>>>>>' not in html
    assert 'id="scientific-swarm-card"' in html
    assert 'id="season4-missions-panel"' in html


def test_shipped_renderer_matches_generated_source() -> None:
    html = SHIPPED.read_text()
    renderer = re.search(r'<script id="georing-webgl-renderer">(.*?)</script>', html, re.S)
    assert renderer
    assert renderer[1].strip() == _GEORING_WEBGL_SCRIPT.strip()
    assert 'DEMO FIXTURE' in html
    assert 'WEBGL 3D GEO-RING ACTIVE' not in html


def test_embedded_report_cannot_close_script_element() -> None:
    attack = '</script><script>window.injected=true</script><img src=x onerror=alert(1)>'
    html = _generate_cyberpunk_hud_html({'attacker': attack})
    assert attack not in html
    assert r'\u003c/script>' in html
    assert len(re.findall(r'<script(?:\s|>)', html)) == 2


def test_dynamic_html_escape_handles_report_strings() -> None:
    html = _generate_cyberpunk_hud_html({})
    helper = re.search(r'function escapeHtml\(value\) \{.*?\n    \}', html, re.S)[0]
    assertion = "const assert=require('node:assert/strict');assert.equal(escapeHtml('<img onerror=\"bad\">&'), '&lt;img onerror=&quot;bad&quot;&gt;&amp;');"
    result = subprocess.run([node(), '-e', helper + '\n' + assertion], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for expression in ['g.gamer_tag', 'g.rank_title', 'g.rich_presence_status', 'g.steam_leaderboard_receipt']:
        assert '${escapeHtml(' + expression + ')}' in html


def test_fixture_provenance_and_training_are_explicit(monkeypatch) -> None:
    from hivemind.game import hf_hack_miner as game

    monkeypatch.setattr(game, 'fetch_hf_domain_subset', lambda **kwargs: ['example'])
    result = game.execute_interactive_lora_micro_forge(contract_id='CONTRACT-HF-13-TRITON-KERNEL-SYNTH')
    assert result['source'] == 'demo_fixture'
    assert result['training_executed'] is False
    assert result['runtime_verified'] is False
    assert result['spot_check_passed'] is None
    assert result['loss_is_simulated'] is True
    assert result['rewards_simulated'] is True
    assert game.build_georing_3d_topology_snapshot()['live_peer_discovery'] is False
    assert game.build_swarm_telemetry_frame()['runtime_verified'] is False


def test_cpu_diagnostic_is_telemetry_not_certification(monkeypatch, tmp_path: Path) -> None:
    import torch
    from fastapi.testclient import TestClient
    from hivemind.game import steam_edition as hud

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(hud, 'build_free_steam_edition_package', lambda **kwargs: {
        'interactive_hud_html': str(tmp_path / 'index.html'),
        'steam_store_manifest': str(tmp_path / 'manifest.json'),
        'steam_cloud_save_file': str(tmp_path / 'cloud.json'),
    })
    client = TestClient(hud.create_hack_miner_fastapi_app(game_report={}))
    result = client.post('/api/run-rtx-diagnostic', json={}).json()
    assert result['cuda_available'] is False
    assert result['diagnostic_passed'] is None
    assert result['diagnostic_status'] == 'telemetry_only'
    assert result['gpu_workload_tested'] is False
    assert result['profile_tested'] is None
    topology = client.get('/api/georing-topology').json()
    assert topology['source'] == 'demo_fixture'
    assert topology['live_peer_discovery'] is False
