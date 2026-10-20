"""Real-browser evidence for the HiveMind WebGL Geo-Ring HUD.

Drives an INSTALLED Edge or Chrome (Playwright ``channel``), headed by default so
the browser's normal GPU path is used. It never downloads browsers.

Usage (source server, launched and stopped by this script):
    python tools/browser_evidence/hud_browser_evidence.py --server-python <venv python>
Usage (an already running HUD, e.g. the frozen exe):
    python tools/browser_evidence/hud_browser_evidence.py --url http://127.0.0.1:8090/

Requires playwright, pillow and numpy in the interpreter running this script, and
ffmpeg on PATH for the recording. Pixel checks use element screenshots, never
``canvas.dataset.rendered`` (that attribute is recorded for context only).
"""
from __future__ import annotations

import argparse
import getpass
import io
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version as metadata_version
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
CANVAS = '#webgl-georing-canvas'
TERMINAL = '#georing-terminal'
CARD = '#webgl-georing-map'
BG = np.array([5, 8, 18], dtype=np.int16)
PALETTE = {'grid': (13, 77, 97), 'links': (255, 51, 148), 'nodes': (26, 255, 179)}
SOFTWARE_MARKERS = ('swiftshader', 'warp', 'basic render', 'llvmpipe', 'softpipe', 'software')
EDGE_CROP = 6
NONBLANK_MIN = 0.01
CHANGED_MIN = 0.01
SAME_MAX = 0.002
DRAG = (150, 40)
SIZES = [(1280, 800), (1024, 768), (800, 600), (412, 915)]
DPR_CASES = [((412, 915), 2), ((412, 915), 3)]
READY_JS = "() => /^\\[DEMO FIXTURE\\] \\d+ nodes/.test(document.getElementById('georing-terminal').textContent)"

INIT_SCRIPT = r"""
(() => {
  const ev = window.__hudEvidence = {
    errors: [], rejections: [], listeners: [],
    raf: { requested: 0, run: 0, cancelled: 0 }, draws: { georing: 0, other: 0 },
  };
  window.addEventListener('error', e => ev.errors.push({
    message: String(e.message), source: String(e.filename || ''), line: e.lineno, col: e.colno,
    t: Math.round(performance.now()) }), true);
  window.addEventListener('unhandledrejection', e => ev.rejections.push({
    reason: String((e.reason && (e.reason.stack || e.reason.message)) || e.reason),
    t: Math.round(performance.now()) }));
  const raf = window.requestAnimationFrame.bind(window);
  const caf = window.cancelAnimationFrame.bind(window);
  ev.rawRaf = raf;
  window.requestAnimationFrame = function (cb) {
    ev.raf.requested++;
    return raf(ts => { ev.raf.run++; return cb(ts); });
  };
  window.cancelAnimationFrame = function (id) { ev.raf.cancelled++; return caf(id); };
  const add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, listener, options) {
    const target = this === window ? 'window' : this === document ? 'document'
      : (this && this.id) ? '#' + this.id : ((this && this.constructor && this.constructor.name) || 'unknown');
    ev.listeners.push({ target, type });
    return add.call(this, type, listener, options);
  };
  for (const ctor of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!ctor) continue;
    const drawArrays = ctor.prototype.drawArrays;
    ctor.prototype.drawArrays = function (...args) {
      if (this.canvas && this.canvas.id === 'webgl-georing-canvas') ev.draws.georing++; else ev.draws.other++;
      return drawArrays.apply(this, args);
    };
  }
})();
"""


# Opt-in diagnostic only: these observers/wrappers never change HUD handlers or
# state, but their logging overhead can affect timing. Rendered angles are the
# actual shader uniform submissions, not access to the renderer's private state.
TRACE_SCRIPT = r"""
(() => {
  const trace = window.__hudInteractionTrace = {
    limit: 10000, dropped: 0, records: [], context_epoch: 0, rendered_angles: null,
  };
  let context = null;
  const uniforms = new WeakMap();
  const state = () => {
    const canvas = document.getElementById('webgl-georing-canvas');
    const box = canvas && canvas.getBoundingClientRect();
    const evidence = window.__hudEvidence;
    return {
      context_epoch: trace.context_epoch,
      context_lost: context ? context.isContextLost() : null,
      last_rendered_angles: trace.rendered_angles,
      raf_run: evidence ? evidence.raf.run : null,
      draws: evidence ? evidence.draws.georing : null,
      scroll: [window.scrollX, window.scrollY],
      focus: document.activeElement && document.activeElement.id,
      visibility: document.visibilityState,
      canvas_box: box ? [box.x, box.y, box.width, box.height] : null,
    };
  };
  const record = (type, details = {}) => {
    if (trace.records.length >= trace.limit) { trace.dropped++; return; }
    trace.records.push({ t: performance.now(), type, ...details, ...state() });
  };
  trace.mark = label => record('marker', { label });
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel',
      'gotpointercapture', 'lostpointercapture', 'keydown', 'focus', 'blur',
      'scroll', 'resize', 'visibilitychange', 'webglcontextlost', 'webglcontextrestored']) {
    for (const capture of [true, false]) window.addEventListener(type, event => {
      const target = event.target;
      if (type.startsWith('webgl') && capture) trace.context_epoch++;
      record(type, {
        phase: capture ? 'capture' : 'bubble',
        target: target === window ? 'window' : target === document ? 'document' : target && (target.id || target.tagName),
        pointer_id: event.pointerId, x: event.clientX, y: event.clientY,
        buttons: event.buttons, key: event.key, default_prevented: event.defaultPrevented,
        target_has_capture: event.pointerId !== undefined && target && target.hasPointerCapture
          ? target.hasPointerCapture(event.pointerId) : null,
      });
    }, capture);
  }
  for (const ctor of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!ctor) continue;
    const getUniformLocation = ctor.prototype.getUniformLocation;
    ctor.prototype.getUniformLocation = function (program, name) {
      const location = getUniformLocation.call(this, program, name);
      if (this.canvas && this.canvas.id === 'webgl-georing-canvas' && location && name === 'angles') {
        context = this;
        uniforms.set(location, true);
      }
      return location;
    };
    const uniform2f = ctor.prototype.uniform2f;
    ctor.prototype.uniform2f = function (location, yaw, pitch) {
      const result = uniform2f.call(this, location, yaw, pitch);
      if (this.canvas && this.canvas.id === 'webgl-georing-canvas' && uniforms.has(location)) {
        context = this;
        trace.rendered_angles = [yaw, pitch];
        record('rendered_angles');
      }
      return result;
    };
  }
})();
"""

GPU_TEXT_JS = r"""
() => {
  const out = [];
  const block = /^(DIV|P|LI|TR|H1|H2|H3|H4|SECTION|TABLE|UL|BR)$/;
  const walk = node => {
    if (node.nodeType === 3) { out.push(node.textContent); return; }
    if (node.nodeType !== 1 && node.nodeType !== 11) return;
    if (node.nodeType === 1 && /^(SCRIPT|STYLE)$/.test(node.tagName)) return;
    if (node.shadowRoot) walk(node.shadowRoot);
    for (const child of node.childNodes) walk(child);
    if (node.nodeType === 1 && node.tagName === 'TD') out.push('\t');
    if (node.nodeType === 1 && block.test(node.tagName)) out.push('\n');
  };
  walk(document.body);
  return out.join('');
}
"""

WEBGL_INFO_JS = r"""
() => {
  const canvas = document.getElementById('webgl-georing-canvas');
  const gl = canvas && canvas.getContext('webgl');
  if (!gl) return { available: false };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const exts = gl.getSupportedExtensions() || [];
  return {
    available: true,
    vendor: gl.getParameter(gl.VENDOR), renderer: gl.getParameter(gl.RENDERER),
    unmasked_vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : null,
    unmasked_renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null,
    debug_renderer_info_available: !!dbg,
    version: gl.getParameter(gl.VERSION), glsl: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
    context_attributes: gl.getContextAttributes(),
    lose_context_available: exts.includes('WEBGL_lose_context'),
    extension_count: exts.length,
  };
}
"""

CAVEAT_JS = r"""
() => {
  const strict = document.createElement('canvas').getContext('webgl', { failIfMajorPerformanceCaveat: true });
  const result = { fail_if_major_performance_caveat_context: !!strict };
  if (strict) { const ext = strict.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); }
  return result;
}
"""

COUNTERS_JS = r"""
() => { const e = window.__hudEvidence; return { raf_requested: e.raf.requested, raf_run: e.raf.run,
  raf_cancelled: e.raf.cancelled, draws: e.draws.georing, listeners: e.listeners.length }; }
"""

LISTENERS_JS = r"""
() => { const counts = {}; for (const l of window.__hudEvidence.listeners) {
  const key = l.target + ' ' + l.type; counts[key] = (counts[key] || 0) + 1; } return counts; }
"""

GEOMETRY_JS = r"""
() => { const c = document.getElementById('webgl-georing-canvas'); const r = c.getBoundingClientRect();
  return { dpr: window.devicePixelRatio, border_box: [r.width, r.height], content_box: [c.clientWidth, c.clientHeight],
    backing: [c.width, c.height], dataset_rendered: c.dataset.rendered || null,
    page_scroll_width: document.documentElement.scrollWidth, page_client_width: document.documentElement.clientWidth }; }
"""


OVERFLOW_JS = r"""
() => {
  const limit = document.documentElement.clientWidth + 1;
  const hits = [...document.querySelectorAll('body *')].filter(el => el.getBoundingClientRect().right > limit);
  const leaves = hits.filter(el => !hits.some(other => other !== el && el.contains(other)));
  return leaves.slice(0, 6).map(el => {
    const r = el.getBoundingClientRect();
    const cls = typeof el.className === 'string' && el.className.trim() ? '.' + el.className.trim().split(/\s+/).join('.') : '';
    return { element: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + cls, right: Math.round(r.right), width: Math.round(r.width) };
  });
}
"""


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def leftover_processes(port: int) -> int:
    if os.name == 'nt':
        query = ("@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and "
                 f"$_.CommandLine -like '*serve-hack-hud*' -and $_.CommandLine -like '*--port {port}*' }}).Count")
        out = subprocess.run(['powershell', '-NoProfile', '-Command', query], capture_output=True, text=True)
        return int(out.stdout.strip() or 0)
    out = subprocess.run(['pgrep', '-f', f'serve-hack-hud.*--port {port}'], capture_output=True, text=True)
    return len(out.stdout.split())


class SourceServer:
    def __init__(self, python: str, log_path: Path, source_root: Path = REPO):
        self.python, self.log_path, self.root = python, log_path, source_root.resolve()
        self.port = free_port()
        self.env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(self.root / 'src'), str(self.root)]), PYTHONUNBUFFERED='1')
        self.proc: subprocess.Popen | None = None
        self.info: dict = {'command': f'python -m hivemind.cli.main serve-hack-hud --host 127.0.0.1 --port {self.port}'}

    def start(self) -> str:
        probe = subprocess.run([self.python, '-c', 'import hivemind; print(hivemind.__file__)'], cwd=self.root,
                               env=self.env, capture_output=True, text=True, timeout=180)
        module = Path(probe.stdout.strip()).resolve()
        inside = self.root in module.parents
        self.info['hivemind_module'] = module.relative_to(REPO.resolve()).as_posix() if inside else 'OUTSIDE_SOURCE_TREE'
        if not inside:
            raise SystemExit('hivemind does not resolve inside the source tree under test; refusing to produce evidence')
        self.log = open(self.log_path, 'w', encoding='utf-8')
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        self.proc = subprocess.Popen([self.python, '-m', 'hivemind.cli.main', 'serve-hack-hud', '--host', '127.0.0.1',
                                      '--port', str(self.port)], cwd=self.root, env=self.env, stdout=self.log,
                                     stderr=subprocess.STDOUT, creationflags=flags)
        url = f'http://127.0.0.1:{self.port}/'
        started = time.monotonic()
        while time.monotonic() - started < 240:
            if self.proc.poll() is not None:
                raise SystemExit(f'HUD server exited early with code {self.proc.returncode}; see {self.log_path}')
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        self.info['startup_seconds'] = round(time.monotonic() - started, 1)
                        return url
            except OSError:
                time.sleep(0.5)
        raise SystemExit('HUD server did not answer within 240 s')

    def stop(self) -> dict:
        if not self.proc:
            return self.info
        sent = 'CTRL_BREAK_EVENT' if os.name == 'nt' else 'SIGINT'
        self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
        forced = False
        try:
            code = self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            forced = True
            self.proc.kill()
            code = self.proc.wait(timeout=10)
        self.log.close()
        log = self.log_path.read_text(encoding='utf-8', errors='replace')
        time.sleep(0.5)
        self.info.update({
            'stop_signal': sent, 'force_killed': forced, 'exit_code': code,
            'log_has_shutdown_lines': 'Shutting down' in log and 'Finished server process' in log,
            'port_still_open': port_open(self.port), 'leftover_matching_processes': leftover_processes(self.port),
        })
        if os.name == 'nt' and code == 3 and self.info['log_has_shutdown_lines']:
            self.info['exit_code_note'] = ('uvicorn shut down gracefully, then re-raised the captured SIGBREAK '
                                           '(signal.raise_signal); the Windows CRT default handler exits with code 3')
        return self.info


def to_array(png: bytes, crop: int = EDGE_CROP) -> np.ndarray:
    arr = np.asarray(Image.open(io.BytesIO(png)).convert('RGB')).astype(np.int16)
    return arr[crop:-crop, crop:-crop] if crop else arr


def render_stats(arr: np.ndarray) -> dict:
    distance = np.abs(arr - BG).max(axis=2)
    mask = distance > 24
    luma = arr @ np.array([0.299, 0.587, 0.114])
    stats = {
        'analysed_size': [int(arr.shape[1]), int(arr.shape[0])],
        'non_background_pixels': int(mask.sum()), 'non_background_fraction': round(float(mask.mean()), 5),
        'luma_std': round(float(luma.std()), 3),
        'unique_colors': int(len(np.unique(arr.reshape(-1, 3), axis=0))),
    }
    for name, rgb in PALETTE.items():
        stats[f'{name}_color_pixels'] = int((np.abs(arr - np.array(rgb)).max(axis=2) < 40).sum())
    if mask.any():
        ys, xs = np.nonzero(mask)
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        stats['content_bbox_wh'] = [w, h]
        stats['content_bbox_aspect'] = round(w / h, 4)
    return stats


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {'shape_mismatch': [list(a.shape), list(b.shape)], 'changed_fraction': 1.0}
    delta = np.abs(a - b)
    changed = delta.max(axis=2) > 16
    return {'changed_pixels': int(changed.sum()), 'changed_fraction': round(float(changed.mean()), 5),
            'mean_abs_diff': round(float(delta.mean()), 4)}


def scrub(value, secrets: list[tuple[re.Pattern, str]]):
    if isinstance(value, str):
        for pattern, replacement in secrets:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v, secrets) for k, v in value.items()}
    return value


def build_scrubbers() -> list[tuple[re.Pattern, str]]:
    home = str(Path.home())
    items = []
    for variant in {home, home.replace('\\', '/'), home.replace('\\', '\\\\'), str(REPO), str(REPO).replace('\\', '/')}:
        items.append((re.compile(re.escape(variant), re.IGNORECASE), '<REDACTED_PATH>'))
    items.append((re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '<REDACTED_EMAIL>'))
    user = getpass.getuser()
    if len(user) >= 3:
        items.append((re.compile(re.escape(user), re.IGNORECASE), '<REDACTED_USER>'))
    host = socket.gethostname()
    if len(host) >= 3:
        items.append((re.compile(re.escape(host), re.IGNORECASE), '<REDACTED_HOST>'))
    return items


class Recorder:
    W, H, BAR = 1280, 720, 44

    def __init__(self, folder: Path):
        self.folder = folder
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        self.count = 0
        try:
            self.font = ImageFont.load_default(size=22)
        except TypeError:
            self.font = ImageFont.load_default()

    def capture(self, page, caption: str) -> None:
        shot = Image.open(io.BytesIO(page.locator(CARD).screenshot())).convert('RGB')
        scale = min(self.W / shot.width, (self.H - self.BAR) / shot.height, 1.0)
        shot = shot.resize((max(1, int(shot.width * scale)), max(1, int(shot.height * scale))), Image.LANCZOS)
        frame = Image.new('RGB', (self.W, self.H), (8, 10, 18))
        frame.paste(shot, ((self.W - shot.width) // 2, self.BAR + (self.H - self.BAR - shot.height) // 2))
        ImageDraw.Draw(frame).text((14, 10), caption, fill=(230, 240, 255), font=self.font)
        frame.save(self.folder / f'f_{self.count:04d}.png')
        self.count += 1

    def encode(self, target: Path, fps: int = 3) -> dict:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg or not self.count:
            return {'status': 'not_established', 'reason': 'ffmpeg not on PATH or no frames'}
        for crf in (30, 36, 42):
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i',
                            str(self.folder / 'f_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf),
                            '-preset', 'slow', '-movflags', '+faststart', str(target)], check=True)
            if target.stat().st_size < 5 * 1024 * 1024:
                break
        return {'file': target.name, 'frames': self.count, 'fps': fps, 'crf': crf, 'bytes': target.stat().st_size}


class Evidence:
    def __init__(self, args, out: Path, scratch: Path):
        self.args, self.out, self.scratch = args, out, scratch
        self.checks: list[dict] = []
        self.console: dict[str, list] = {k: [] for k in (
            'console', 'pageerror', 'init_script_errors', 'unhandled_rejections', 'requestfailed', 'http_error_responses')}
        self.ws: list[dict] = []
        self.interaction_traces: list[dict] = []
        self.recorder = Recorder(scratch / 'frames')
        self.shots: dict[str, str] = {}

    def check(self, cid: str, name: str, status: str, **evidence) -> None:
        self.checks.append({'id': cid, 'name': name, 'status': status, 'evidence': evidence})
        print(f'[{status:>15}] {cid}', flush=True)

    def attach(self, page, label: str) -> None:
        def on_console(msg):
            loc = msg.location or {}
            self.console['console'].append({'page': label, 'type': msg.type, 'text': msg.text,
                                            'url': loc.get('url'), 'line': loc.get('lineNumber')})

        def on_ws(ws):
            record = {'page': label, 'url': ws.url, 'frames': [], 'closed': False}
            self.ws.append(record)
            ws.on('framereceived', lambda payload: record['frames'].append(
                payload if isinstance(payload, str) else payload.decode('utf-8', 'replace')))
            ws.on('close', lambda _: record.__setitem__('closed', True))

        page.on('console', on_console)
        page.on('pageerror', lambda err: self.console['pageerror'].append({'page': label, 'error': str(err)}))
        page.on('websocket', on_ws)
        page.on('requestfailed', lambda req: self.console['requestfailed'].append(
            {'page': label, 'url': req.url, 'failure': req.failure}))
        page.on('response', lambda resp: self.console['http_error_responses'].append(
            {'page': label, 'url': resp.url, 'status': resp.status}) if resp.status >= 400 else None)

    def harvest(self, page, label: str) -> None:
        data = page.evaluate('() => ({ errors: window.__hudEvidence.errors, rejections: window.__hudEvidence.rejections })')
        self.console['init_script_errors'] += [dict(e, page=label) for e in data['errors']]
        self.console['unhandled_rejections'] += [dict(r, page=label) for r in data['rejections']]
        if self.args.trace_events:
            trace = page.evaluate('() => ({ limit: window.__hudInteractionTrace.limit, dropped: window.__hudInteractionTrace.dropped, records: window.__hudInteractionTrace.records })')
            self.interaction_traces.append(dict(trace, page=label))

    def save_png(self, name: str, png: bytes) -> None:
        (self.out / name).write_bytes(png)
        self.shots[name] = name


def wait_frames(page, n: int = 3) -> None:
    page.evaluate('async n => { for (let i = 0; i < n; i++) await new Promise(r => window.__hudEvidence.rawRaf(r)); }', n)


def terminal(page) -> str:
    return page.locator(TERMINAL).text_content() or ''


def canvas_png(page) -> bytes:
    wait_frames(page)
    return page.locator(CANVAS).screenshot()


def counters(page) -> dict:
    return page.evaluate(COUNTERS_JS)


def delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


def canvas_box(page) -> dict:
    page.locator(CANVAS).scroll_into_view_if_needed()
    return page.locator(CANVAS).bounding_box()


def trace_mark(page, ev, label: str) -> None:
    if ev and ev.args.trace_events:
        page.evaluate('label => window.__hudInteractionTrace.mark(label)', label)


def drag(page, ev: Evidence | None, dx: float, dy: float, caption: str = '') -> None:
    trace_mark(page, ev, f'drag start: {caption or "probe"}, dx={dx}, dy={dy}')
    box = canvas_box(page)
    cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, 5):
        page.mouse.move(cx + dx * i / 4, cy + dy * i / 4, steps=5)
        if ev and caption:
            ev.recorder.capture(page, f'{caption} ({i}/4)')
        trace_mark(page, ev, f'drag segment {i}/4 complete')
    page.mouse.up()
    wait_frames(page)


def reset_view(page) -> None:
    page.click('#georing-reset')
    wait_frames(page)


def project(xyz, yaw: float, pitch: float, box: dict) -> tuple[float, float, float]:
    x, y, z = xyz
    rx = math.cos(yaw) * x + math.sin(yaw) * z
    rz = -math.sin(yaw) * x + math.cos(yaw) * z
    ry = math.cos(pitch) * y - math.sin(pitch) * rz
    depth = math.sin(pitch) * y + math.cos(pitch) * rz
    zoom = 1.25 / (1.7 - depth * 0.3)
    return (box['x'] + box['width'] / 2 + rx * zoom * box['height'] / 2,
            box['y'] + box['height'] / 2 - ry * zoom * box['height'] / 2, depth)


def click_pick_target(page, nodes: list[dict]) -> tuple[int, float, float]:
    box = canvas_box(page)
    points = [project(n['xyz'], -0.4, -0.35, box) for n in nodes]
    best, best_gap = 0, -1.0
    for i, (px, py, _depth) in enumerate(points):
        gap = min((math.hypot(px - qx, py - qy) for j, (qx, qy, _) in enumerate(points) if j != i), default=999)
        if gap > best_gap:
            best, best_gap = i, gap
    return best, points[best][0], points[best][1]


def gpu_page_excerpt(context, channel: str) -> dict:
    page = context.new_page()
    attempts = []
    for url in ([f'{"edge" if channel == "msedge" else "chrome"}://gpu', 'chrome://gpu']):
        try:
            page.goto(url, wait_until='load', timeout=20000)
            attempts.append({'url': url, 'ok': True})
            break
        except Exception as exc:  # noqa: BLE001 - recorded as evidence
            attempts.append({'url': url, 'ok': False, 'error': str(exc).splitlines()[0]})
    page.wait_for_timeout(3000)
    lines = [re.sub(r'\s+', ' ', line).strip() for line in page.evaluate(GPU_TEXT_JS).splitlines()]
    lines = [line for line in lines if line]
    page.close()
    feature: list[str] = []
    if 'Graphics Feature Status' in lines:
        for line in lines[lines.index('Graphics Feature Status') + 1:]:
            if set(line) == {'='}:
                continue
            line = line.lstrip('* ').strip()
            if ':' not in line or len(feature) > 30:
                break
            feature.append(line)
    keys = ('GL_RENDERER', 'GL_VENDOR', 'GL_VERSION', 'GPU0', 'GPU1', 'Driver vendor', 'Driver version', 'Driver date')
    extras = [line for line in lines if line.startswith(keys)]
    webgl = next((line for line in feature if line.startswith('WebGL:')), None)
    return {'attempts': attempts, 'graphics_feature_status': feature, 'gpu_lines': extras[:12], 'webgl_line': webgl,
            'lines_scanned': len(lines)}


def run(args) -> int:
    sha = args.source_commit or git('rev-parse', 'HEAD')
    out = Path(args.out) if args.out else REPO / 'docs' / 'evidence' / 'browser' / sha[:12]
    out.mkdir(parents=True, exist_ok=True)
    scratch = REPO / 'build' / 'browser_evidence'
    scratch.mkdir(parents=True, exist_ok=True)
    ev = Evidence(args, out, scratch)
    results: dict = {
        'schema': 'hivemind.hud_browser_evidence/v1',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'commit': {'sha': sha, 'sha12': sha[:12], 'harness_head': git('rev-parse', 'HEAD'),
                   'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                   'source_tree': (Path(args.source_tree).resolve().relative_to(REPO.resolve()).as_posix()
                                   + f' (git archive of {sha[:12]})') if args.source_tree else 'worktree',
                   'dirty_paths_at_start': git('status', '--porcelain').splitlines()},
        'limitations': [],
    }
    server = None
    if args.url:
        url = args.url
        results['target'] = {'mode': 'external_url', 'url': url}
    else:
        server = SourceServer(args.server_python or sys.executable, scratch / 'server.log',
                              Path(args.source_tree) if args.source_tree else REPO)
        url = server.start()
        results['target'] = {'mode': 'source_server', 'url': url}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel=args.channel, headless=args.headless)
            results['browser'] = {'channel': args.channel, 'version': browser.version, 'headless': args.headless,
                                  'playwright': metadata_version('playwright'),
                                  'extra_launch_args': []}
            gpu_ctx = browser.new_context()
            results['gpu_page'] = gpu_page_excerpt(gpu_ctx, args.channel)
            blank = gpu_ctx.new_page()
            blank.set_content('<canvas></canvas>')
            results['webgl_caveat_probe'] = blank.evaluate(CAVEAT_JS)
            gpu_ctx.close()

            context = browser.new_context(viewport={'width': 1280, 'height': 800})
            context.add_init_script(INIT_SCRIPT + (TRACE_SCRIPT if args.trace_events else ''))
            page = context.new_page()
            ev.attach(page, 'hud')
            run_hud(page, url, ev, results)
            context.close()

            for (w, h), dpr in DPR_CASES:
                ctx = browser.new_context(viewport={'width': w, 'height': h}, device_scale_factor=dpr)
                ctx.add_init_script(INIT_SCRIPT + (TRACE_SCRIPT if args.trace_events else ''))
                p2 = ctx.new_page()
                ev.attach(p2, f'dpr{dpr}')
                p2.goto(url, wait_until='load')
                p2.wait_for_function(READY_JS, polling=100, timeout=30000)
                resize_case(p2, ev, results, f'{w}x{h}@dpr{dpr}', None)
                ev.harvest(p2, f'dpr{dpr}')
                ctx.close()
            browser.close()
    finally:
        if server:
            results['target']['server'] = server.stop()
    finish(ev, results, server)
    return 0


def run_hud(page, url: str, ev: Evidence, results: dict) -> None:
    started = time.monotonic()
    with page.expect_response(lambda r: '/api/georing-topology' in r.url) as topo_resp:
        page.goto(url, wait_until='load')
    page.wait_for_function(READY_JS, polling=100, timeout=30000)
    topology = topo_resp.value.json()
    results['time_to_ready_seconds'] = round(time.monotonic() - started, 2)
    nodes = topology.get('nodes', [])
    results['navigator'] = page.evaluate(
        "async () => ({ user_agent: navigator.userAgent, platform: navigator.platform, hardware_concurrency: navigator.hardwareConcurrency,"
        " ua_full_version_list: navigator.userAgentData ? (await navigator.userAgentData.getHighEntropyValues(['fullVersionList'])).fullVersionList : null })")
    webgl = page.evaluate(WEBGL_INFO_JS)
    results['webgl'] = webgl
    hardware_accel(ev, results)

    init_png = canvas_png(page)
    ev.save_png('01_initial_canvas.png', init_png)
    init = to_array(init_png)
    stats = render_stats(init)
    geom = page.evaluate(GEOMETRY_JS)
    ok = stats['non_background_fraction'] > NONBLANK_MIN and all(stats[f'{k}_color_pixels'] > 20 for k in PALETTE)
    ev.check('initial_render', 'Initial render produces non-blank pixels with grid, link and node colours',
             'pass' if ok else 'fail', pixel_stats=stats, geometry=geom, terminal_text=terminal(page),
             note='dataset.rendered is recorded in geometry for context only and is not used as evidence')
    ev.recorder.capture(page, 'Initial render (demo fixture topology)')

    before_text = terminal(page)
    before_px = to_array(canvas_png(page))
    options = page.evaluate("() => [...document.getElementById('georing-node').options].map(o => [o.value, o.text])")
    page.select_option('#georing-node', '0')
    after_text = terminal(page)
    after_px = to_array(canvas_png(page))
    ok = (nodes and 'Example RTT' not in before_text and nodes[0]['node_id'] in after_text and 'Example RTT' in after_text)
    ev.check('node_selection_selector', 'Node selector (#georing-node change) updates #georing-terminal',
             'pass' if ok else 'fail', options=options, selected_value='0', expected_node_id=nodes[0]['node_id'] if nodes else None,
             terminal_before=before_text, terminal_after=after_text,
             canvas_diff_after_selection=diff_stats(before_px, after_px),
             note='selection only changes the terminal text; the canvas does not highlight the selected node')
    ev.recorder.capture(page, 'Node selected via #georing-node')
    page.select_option('#georing-node', '-1')
    click_pick(page, ev, nodes, 'node_selection_click', 'Clicking a projected node on the canvas selects it')

    reset_view(page)
    base_png = canvas_png(page)
    base = to_array(base_png)
    c0 = counters(page)
    drag(page, ev, *DRAG, caption='Pointer drag rotation')
    c1 = counters(page)
    drag_png = canvas_png(page)
    ev.save_png('02_after_drag_canvas.png', drag_png)
    dragged = to_array(drag_png)
    d = diff_stats(base, dragged)
    ev.check('drag_rotation', f'Pointer drag {DRAG} rotates the globe (pixel diff)', 'pass' if d['changed_fraction'] > CHANGED_MIN else 'fail',
             diff_before_vs_after=d, counters_delta=delta(c0, c1), after_stats=render_stats(dragged))

    reset_view(page)
    page.locator(CANVAS).focus()
    k0 = to_array(canvas_png(page))
    scroll0 = page.evaluate('() => window.scrollY')
    ck0 = counters(page)
    for key in ['ArrowRight'] * 5 + ['ArrowUp'] * 3:
        page.keyboard.press(key)
    wait_frames(page)
    ck1 = counters(page)
    k1 = to_array(canvas_png(page))
    scroll1 = page.evaluate('() => window.scrollY')
    ev.recorder.capture(page, 'Keyboard: ArrowRight x5, ArrowUp x3 on focused canvas')
    d = diff_stats(k0, k1)
    focused = page.evaluate("() => document.activeElement && document.activeElement.id")
    ev.check('keyboard_rotation', 'Arrow keys on the focused canvas rotate the globe (handler found in JS: canvas keydown)',
             'pass' if d['changed_fraction'] > CHANGED_MIN and focused == 'webgl-georing-canvas' else 'fail',
             implemented=True, focused_element=focused, keys=['ArrowRight'] * 5 + ['ArrowUp'] * 3, diff_before_vs_after=d,
             counters_delta=delta(ck0, ck1), page_scroll_y_before_after=[scroll0, scroll1])

    page.click('#georing-reset')
    wait_frames(page)
    reset_px = to_array(canvas_png(page))
    ev.recorder.capture(page, 'Reset view')
    d_base, d_rot = diff_stats(base, reset_px), diff_stats(k1, reset_px)
    ev.check('reset_view', '#georing-reset restores the default view', 'pass'
             if d_base['changed_fraction'] <= SAME_MAX and d_rot['changed_fraction'] > CHANGED_MIN else 'fail',
             diff_reset_vs_default=d_base, diff_reset_vs_rotated=d_rot)

    ev.harvest(page, 'hud')
    page.reload(wait_until='load')
    page.wait_for_function(READY_JS, polling=100, timeout=30000)
    reload_png = canvas_png(page)
    reloaded = to_array(reload_png)
    s = render_stats(reloaded)
    d = diff_stats(init, reloaded)
    ev.check('reload_rerender', 'Page reload re-renders the same default view', 'pass'
             if s['non_background_fraction'] > NONBLANK_MIN and d['changed_fraction'] <= SAME_MAX else 'fail',
             pixel_stats=s, diff_vs_initial=d, terminal_text=terminal(page))
    ev.recorder.capture(page, 'After page reload')

    context_loss(page, ev, results, nodes)

    resize_results = []
    for w, h in SIZES:
        page.set_viewport_size({'width': w, 'height': h})
        resize_results.append(resize_case(page, ev, results, f'{w}x{h}@dpr1', w))
    page.set_viewport_size({'width': 1280, 'height': 800})
    wait_frames(page)

    telemetry_and_labels(page, ev, results, topology)
    ev.harvest(page, 'hud')


def click_pick(page, ev: Evidence, nodes: list[dict], cid: str, name: str) -> dict:
    reset_view(page)
    before = terminal(page)
    index, x, y = click_pick_target(page, nodes)
    page.mouse.click(x, y)
    wait_frames(page)
    after = terminal(page)
    selected_value = page.evaluate("() => document.getElementById('georing-node').value")
    ok = nodes[index]['node_id'] in after and selected_value == str(index) and nodes[index]['node_id'] not in before
    ev.check(cid, name, 'pass' if ok else 'fail', target_index=index, click_xy=[round(x, 1), round(y, 1)],
             expected_node_id=nodes[index]['node_id'], selector_value_after=selected_value,
             terminal_before=before, terminal_after=after)
    ev.recorder.capture(page, f'Canvas click selects node {nodes[index]["node_id"]}')
    page.select_option('#georing-node', '-1')
    return {'ok': ok}


def drag_probe(page, ev: Evidence, dx: float, caption: str = '') -> tuple[np.ndarray, dict]:
    reset_view(page)
    c0 = counters(page)
    drag(page, ev, dx, DRAG[1], caption)
    c1 = counters(page)
    return to_array(canvas_png(page)), delta(c0, c1)


def idle_rafs(page, ms: int = 1500) -> dict:
    c0 = counters(page)
    page.wait_for_timeout(ms)
    return delta(c0, counters(page))


def context_loss(page, ev: Evidence, results: dict, nodes: list[dict]) -> None:
    available = page.evaluate("() => !!document.getElementById('webgl-georing-canvas').getContext('webgl').getExtension('WEBGL_lose_context')")
    if not available:
        for cid in ('context_loss_pause', 'context_restore_resume', 'no_duplicate_loops', 'no_duplicate_handlers'):
            ev.check(cid, 'WEBGL_lose_context unavailable', 'not_established', reason='extension not exposed by this browser')
        return
    reset_view(page)
    pre_png = canvas_png(page)
    ev.save_png('03_before_context_loss_canvas.png', pre_png)
    pre = to_array(pre_png)
    idle_pre = idle_rafs(page)
    drag_pre, drag_pre_counts = drag_probe(page, ev, DRAG[0])
    drag_2x, _ = drag_probe(page, ev, DRAG[0] * 2)
    reset_view(page)
    listeners_pre = page.evaluate(LISTENERS_JS)
    cycles = []
    for cycle in (1, 2):
        trace_mark(page, ev, f'context-loss cycle {cycle} start')
        lost_c0 = counters(page)
        page.evaluate("() => { const gl = document.getElementById('webgl-georing-canvas').getContext('webgl');"
                      " window.__loseExt = gl.getExtension('WEBGL_lose_context'); window.__loseExt.loseContext(); }")
        page.wait_for_function("() => document.getElementById('georing-terminal').textContent.startsWith('[WEBGL CONTEXT LOST]')",
                               polling=100, timeout=10000)
        lost_text = terminal(page)
        is_lost = page.evaluate("() => document.getElementById('webgl-georing-canvas').getContext('webgl').isContextLost()")
        lost_png = page.locator(CANVAS).screenshot()
        lost_card = page.locator(CARD).screenshot()
        if cycle == 1:
            ev.save_png('04_context_lost_card.png', lost_card)
        ev.recorder.capture(page, f'Cycle {cycle}: loseContext() -> paused UI')
        lost_c1 = counters(page)
        drag(page, None, *DRAG)
        page.locator(CANVAS).focus()
        page.keyboard.press('ArrowRight')
        page.click('#georing-reset')
        page.select_option('#georing-node', '1')
        page.wait_for_timeout(1000)
        lost_c2 = counters(page)
        lost_after_png = page.locator(CANVAS).screenshot()
        while_lost = delta(lost_c1, lost_c2)
        lost_stats = render_stats(to_array(lost_png))
        page.evaluate('() => window.__loseExt.restoreContext()')
        page.wait_for_function("() => document.getElementById('georing-terminal').textContent.startsWith('[DEMO FIXTURE]')",
                               polling=100, timeout=10000)
        page.select_option('#georing-node', '-1')
        restored_png = canvas_png(page)
        restored = to_array(restored_png)
        if cycle == 1:
            ev.save_png('05_after_restore_canvas.png', restored_png)
        ev.recorder.capture(page, f'Cycle {cycle}: restoreContext() -> rendering resumed')
        idle_post = idle_rafs(page)
        listeners_post = page.evaluate(LISTENERS_JS)
        drag_post, drag_post_counts = drag_probe(page, ev, DRAG[0], f'Cycle {cycle}: drag after restore')
        reset_view(page)
        sel = click_pick(page, ev, nodes, f'post_restore_click_selection_cycle{cycle}',
                         f'Cycle {cycle}: canvas click selection works after restore')
        page.select_option('#georing-node', '2')
        sel_text = terminal(page)
        page.select_option('#georing-node', '-1')
        cycles.append({
            'cycle': cycle, 'terminal_while_lost': lost_text, 'is_context_lost': is_lost,
            'counters_lose_event': delta(lost_c0, lost_c1), 'counters_while_lost_during_input': while_lost,
            'lost_canvas_stats': lost_stats, 'lost_canvas_diff_during_input': diff_stats(to_array(lost_png), to_array(lost_after_png)),
            'restored_stats': render_stats(restored), 'restored_vs_pre_loss': diff_stats(pre, restored),
            'idle_counters_pre': idle_pre, 'idle_counters_post': idle_post,
            'listeners_pre': listeners_pre, 'listeners_post': listeners_post,
            'drag_counters_pre': drag_pre_counts, 'drag_counters_post': drag_post_counts,
            'drag_post_vs_drag_pre': diff_stats(drag_pre, drag_post), 'drag_post_vs_double_drag_control': diff_stats(drag_2x, drag_post),
            'drag_pre_vs_double_drag_control': diff_stats(drag_pre, drag_2x),
            'post_restore_click_selection_ok': sel['ok'], 'post_restore_selector_text': sel_text,
        })
    results['context_loss_cycles'] = cycles
    paused = all(c['is_context_lost'] and c['terminal_while_lost'].startswith('[WEBGL CONTEXT LOST]')
                 and c['counters_while_lost_during_input']['draws'] == 0 and c['counters_while_lost_during_input']['raf_run'] == 0
                 and c['lost_canvas_diff_during_input']['changed_fraction'] <= SAME_MAX for c in cycles)
    ev.check('context_loss_pause', 'loseContext(): paused UI text shown, zero rAF callbacks and zero drawArrays during input',
             'pass' if paused else 'fail', cycles=[{k: c[k] for k in ('cycle', 'terminal_while_lost', 'is_context_lost',
             'counters_while_lost_during_input', 'lost_canvas_stats', 'lost_canvas_diff_during_input')} for c in cycles])
    resumed = all(c['restored_stats']['non_background_fraction'] > NONBLANK_MIN
                  and c['restored_vs_pre_loss']['changed_fraction'] <= SAME_MAX for c in cycles)
    ev.check('context_restore_resume', 'restoreContext(): rendering resumes with the same view as before loss',
             'pass' if resumed else 'fail', cycles=[{k: c[k] for k in ('cycle', 'restored_stats', 'restored_vs_pre_loss')} for c in cycles])
    loops_ok = all(c['idle_counters_post']['raf_run'] == c['idle_counters_pre']['raf_run'] == 0
                   and c['drag_counters_post']['draws'] == 3 * c['drag_counters_post']['raf_run']
                   and c['drag_counters_pre']['draws'] == 3 * c['drag_counters_pre']['raf_run']
                   and 0.5 <= (c['drag_counters_post']['raf_run'] / max(1, c['drag_counters_pre']['raf_run'])) <= 1.5
                   for c in cycles)
    ev.check('no_duplicate_loops', 'No duplicate animation loop after restore (on-demand renderer: idle rAF stays 0; 3 draws per frame)',
             'pass' if loops_ok else 'fail', cycles=[{k: c[k] for k in ('cycle', 'idle_counters_pre', 'idle_counters_post',
             'drag_counters_pre', 'drag_counters_post')} for c in cycles])
    handlers_ok = all(c['listeners_post'] == c['listeners_pre']
                      and c['drag_post_vs_drag_pre']['changed_fraction'] <= SAME_MAX
                      and c['drag_post_vs_double_drag_control']['changed_fraction'] > CHANGED_MIN for c in cycles)
    ev.check('no_duplicate_handlers', 'No duplicate handlers after restore (listener registrations unchanged; same drag gives same rotation, not 2x)',
             'pass' if handlers_ok else 'fail', cycles=[{k: c[k] for k in ('cycle', 'listeners_pre', 'listeners_post',
             'drag_post_vs_drag_pre', 'drag_post_vs_double_drag_control', 'drag_pre_vs_double_drag_control')} for c in cycles])
    post_ok = all(c['post_restore_click_selection_ok'] and 'Example RTT' in c['post_restore_selector_text'] for c in cycles)
    ev.check('post_restore_interaction', 'Selector and canvas-click selection plus drag work after restore',
             'pass' if post_ok and handlers_ok else 'fail',
             selector_text_after_restore=[c['post_restore_selector_text'] for c in cycles])


def resize_case(page, ev: Evidence, results: dict, label: str, frame_width: int | None) -> dict:
    expected_js = ("() => { const c = document.getElementById('webgl-georing-canvas'); const r = Math.min(window.devicePixelRatio || 1, 2);"
                   " const b = c.getBoundingClientRect(); const fit = (w, h) => c.width === Math.max(1, Math.round(w * r)) &&"
                   " c.height === Math.max(1, Math.round(h * r)); return fit(c.clientWidth, c.clientHeight) || fit(b.width, b.height); }")
    settled = True
    try:
        page.wait_for_function(expected_js, polling=100, timeout=5000)
    except Exception:  # noqa: BLE001 - recorded as a failure below
        settled = False
    page.locator(CANVAS).scroll_into_view_if_needed()
    wait_frames(page, 4)
    geom = page.evaluate(GEOMETRY_JS)
    png = page.locator(CANVAS).screenshot()
    stats = render_stats(to_array(png, EDGE_CROP * max(1, round(geom['dpr']))))
    ratio = min(geom['dpr'] or 1, 2)
    content_expected = [round(geom['content_box'][0] * ratio), round(geom['content_box'][1] * ratio)]
    content_match = geom['backing'] == content_expected
    aspect = stats.get('content_bbox_aspect', 0)
    ok = content_match and stats['non_background_fraction'] > NONBLANK_MIN and abs(aspect - 1) < 0.08
    overflow = page.evaluate(OVERFLOW_JS) if geom['page_scroll_width'] > geom['page_client_width'] else []
    case = {'label': label, 'backing_settled': settled, 'geometry': geom, 'effective_ratio': ratio,
            'page_horizontal_overflow_px': geom['page_scroll_width'] - geom['page_client_width'], 'overflowing_elements': overflow,
            'backing_expected_from_content_box': content_expected, 'backing_matches_content_box_x_ratio': content_match,
            'backing_minus_content_box_x_ratio': [geom['backing'][0] - content_expected[0], geom['backing'][1] - content_expected[1]],
            'pixel_stats': stats, 'globe_bbox_aspect': aspect, 'status': 'pass' if ok else 'fail'}
    results.setdefault('resize_cases', []).append(case)
    if label.startswith('412x915'):
        ev.save_png(f'06_resize_{label.replace("@", "_")}_canvas.png', png)
    if frame_width:
        ev.recorder.capture(page, f'Resize {label}: backing {geom["backing"]} for content box {geom["content_box"]}')
    return case


def hardware_accel(ev: Evidence, results: dict) -> None:
    webgl, gpu = results['webgl'], results['gpu_page']
    renderer = (webgl.get('unmasked_renderer') or webgl.get('renderer') or '')
    software = any(m in renderer.lower() for m in SOFTWARE_MARKERS)
    gpu_says_hw = bool(gpu.get('webgl_line') and 'hardware accelerated' in gpu['webgl_line'].lower())
    caveat = results.get('webgl_caveat_probe', {}).get('fail_if_major_performance_caveat_context')
    established = bool(renderer) and not software and gpu_says_hw
    ev.check('hardware_acceleration', 'WebGL renderer is a hardware GPU and the GPU page reports WebGL hardware accelerated',
             'pass' if established else 'not_established', renderer=renderer, software_renderer_marker_found=software,
             gpu_page_webgl_line=gpu.get('webgl_line'), fail_if_major_performance_caveat_context_created=caveat,
             rule='claimed only if renderer is not SwiftShader/WARP/Basic Render AND the GPU page WebGL line says Hardware accelerated')


def telemetry_and_labels(page, ev: Evidence, results: dict, topology: dict) -> None:
    page.wait_for_timeout(2500)
    frames, bad = [], 0
    for record in ev.ws:
        for raw in record['frames']:
            try:
                frames.append(json.loads(raw))
            except ValueError:
                bad += 1
    sources = sorted({str(f.get('source')) for f in frames})
    ws_urls = sorted({r['url'] for r in ev.ws})
    sample = {k: frames[0].get(k) for k in ('stream', 'source', 'seq', 'num_nodes', 'active_ring_peers', 'mean_ring_rtt_ms',
                                            'compute_backend')} if frames else {}
    ev.check('telemetry_ws_fixture', 'WebSocket /ws/swarm-telemetry frames are labelled source=demo_fixture',
             'pass' if frames and sources == ['demo_fixture'] and not bad else 'fail', websocket_urls=ws_urls,
             connections=len(ev.ws), frames_received=len(frames), unparsable_frames=bad, sources=sources,
             seq_range=[min(f.get('seq', 0) for f in frames), max(f.get('seq', 0) for f in frames)] if frames else None,
             first_frame_sample=sample,
             statement='A WebSocket connection to the local HUD server is NOT evidence of live distributed peers; '
                       'these frames are server-generated demo fixtures.')
    api = page.request.get(urljoin(page.url, '/api/georing-topology')).json()
    topo_ok = (topology.get('source') == 'demo_fixture' and topology.get('live_peer_discovery') is False
               and api.get('source') == 'demo_fixture')
    ev.check('rest_topology_fixture', 'REST /api/georing-topology reports source=demo_fixture, live_peer_discovery=false',
             'pass' if topo_ok else 'fail', in_page_response={'source': topology.get('source'),
             'live_peer_discovery': topology.get('live_peer_discovery'), 'nodes': len(topology.get('nodes', [])),
             'ring_links': len(topology.get('ring_links', [])), 'node_ids': [n.get('node_id') for n in topology.get('nodes', [])]},
             independent_request={'source': api.get('source'), 'live_peer_discovery': api.get('live_peer_discovery')})
    labels = page.evaluate("""() => ({
      card_heading: document.querySelector('#webgl-georing-map h2').textContent,
      canvas_aria_label: document.getElementById('webgl-georing-canvas').getAttribute('aria-label'),
      georing_terminal: document.getElementById('georing-terminal').textContent,
      telemetry_terminal: document.getElementById('swarm-telemetry-terminal').textContent })""")
    ok = ('DEMO FIXTURE' in labels['card_heading'] and labels['georing_terminal'].startswith('[DEMO FIXTURE]')
          and 'no live peer discovery' in labels['georing_terminal'] and labels['telemetry_terminal'].startswith('[SIMULATED TELEMETRY]'))
    ev.check('ui_fixture_labels', 'UI labels the globe and telemetry as demo fixture / simulated', 'pass' if ok else 'fail', **labels)
    ev.recorder.capture(page, 'Fixture labels: DEMO FIXTURE globe, SIMULATED TELEMETRY stream')


def finish(ev: Evidence, results: dict, server: SourceServer | None) -> None:
    sizes = results.get('resize_cases', [])
    ev.check('resize', 'Resize: backing store = content box x min(DPR,2), non-blank, undistorted globe at every size',
             'pass' if sizes and all(c['status'] == 'pass' for c in sizes) else 'fail',
             cases=[{k: c[k] for k in ('label', 'status', 'geometry', 'backing_expected_from_content_box',
                    'backing_minus_content_box_x_ratio', 'globe_bbox_aspect')} for c in sizes],
             note='DPR is capped at 2 by the renderer (Math.min(devicePixelRatio, 2)); the 3x case checks that cap')
    con = ev.console
    errors = [m for m in con['console'] if m['type'] == 'error']
    resource_errors = [m for m in errors if m['text'].startswith('Failed to load resource')]
    script_errors = [m for m in errors if m not in resource_errors]
    warnings = [m for m in con['console'] if m['type'] == 'warning']
    webgl_warnings = [m for m in con['console'] if 'webgl' in m['text'].lower()]
    total = len(errors) + len(con['pageerror']) + len(con['init_script_errors']) + len(con['unhandled_rejections'])
    ev.check('console_clean', 'No console errors, page errors or unhandled rejections during the whole run',
             'pass' if total == 0 else 'fail', console_errors=len(errors), console_warnings=len(warnings),
             pageerrors=len(con['pageerror']), init_script_errors=len(con['init_script_errors']),
             unhandled_rejections=len(con['unhandled_rejections']), webgl_console_messages=len(webgl_warnings),
             resource_load_errors=resource_errors, http_error_responses=con['http_error_responses'],
             requestfailed=len(con['requestfailed']))
    script_total = len(script_errors) + len(con['pageerror']) + len(con['init_script_errors']) + len(con['unhandled_rejections'])
    ev.check('no_script_errors', 'No JavaScript errors: console errors other than resource loads, pageerror, window error, unhandledrejection',
             'pass' if script_total == 0 else 'fail', script_console_errors=script_errors, pageerrors=con['pageerror'],
             init_script_errors=con['init_script_errors'], unhandled_rejections=con['unhandled_rejections'])
    overflow = [{'label': c['label'], 'overflow_px': c['page_horizontal_overflow_px'], 'elements': c['overflowing_elements']}
                for c in sizes if c['page_horizontal_overflow_px'] > 0]
    ev.check('page_no_horizontal_overflow', 'Whole HUD page has no horizontal scroll at the tested sizes (outside the Geo-Ring card)',
             'pass' if not overflow else 'fail', overflowing_cases=overflow)
    ev.check('no_webgl_console_messages', 'No WebGL warnings/errors in the console (e.g. stale objects after restore)',
             'pass' if not webgl_warnings else 'fail', messages=webgl_warnings[:20])
    if server:
        info = results['target']['server']
        clean = (not info['force_killed'] and not info['port_still_open'] and info['leftover_matching_processes'] == 0)
        ev.check('server_shutdown_clean', 'Source HUD server stopped by CTRL_BREAK, port closed, no leftover process',
                 'pass' if clean else 'fail', **{k: v for k, v in info.items() if k != 'command'})
    results['generated_tracked_paths_after_run'] = git('status', '--porcelain').splitlines()
    results['recording'] = ev.recorder.encode(ev.out / 'hud_key_sequence.mp4')
    summary: dict[str, int] = {}
    for c in ev.checks:
        summary[c['status']] = summary.get(c['status'], 0) + 1
    results['summary'] = summary
    results['checks'] = ev.checks
    results['screenshots'] = sorted(ev.shots)
    results['console_counts'] = {k: len(v) for k, v in con.items()}
    scrubbers = build_scrubbers()
    if ev.args.trace_events:
        results['interaction_trace'] = {
            'file': 'interaction_trace.json',
            'records': sum(len(trace['records']) for trace in ev.interaction_traces),
            'dropped': sum(trace['dropped'] for trace in ev.interaction_traces),
            'limitation': 'Diagnostic logging may affect timing. Angles are shader submissions, not private renderer state.',
        }
        (ev.out / 'interaction_trace.json').write_text(
            json.dumps(scrub(ev.interaction_traces, scrubbers), indent=2), encoding='utf-8')
    (ev.out / 'results.json').write_text(json.dumps(scrub(results, scrubbers), indent=2), encoding='utf-8')
    (ev.out / 'console.json').write_text(json.dumps(scrub(con, scrubbers), indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', help='HUD URL to test; omit to launch the source server')
    parser.add_argument('--server-python', help='interpreter with HiveMind deps used to launch the source server')
    parser.add_argument('--channel', default='msedge', choices=['msedge', 'chrome'])
    parser.add_argument('--trace-events', action='store_true', help='Record bounded pointer/capture, rendered-angle, frame and context diagnostics for the full sequence')
    parser.add_argument('--headless', action='store_true', help='headless mode (may use a software renderer)')
    parser.add_argument('--out', help='output folder (default docs/evidence/browser/<sha12>)')
    parser.add_argument('--source-tree', help='serve from this exported tree (e.g. a git archive of an older commit)')
    parser.add_argument('--source-commit', help='commit the --source-tree was exported from (recorded as the commit)')
    args = parser.parse_args()
    if bool(args.source_tree) != bool(args.source_commit):
        parser.error('--source-tree and --source-commit must be given together')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
