"""Diagnose the intermittent truncated post-restore drag seen in the frozen-exe browser run.

    <evidence-venv python> tools/browser_evidence/diagnose_drag_truncation.py --url http://127.0.0.1:PORT/ --out results.json

Repeats the harness's 4x5-step drag under several conditions and logs every
pointer/focus/scroll event the page sees, in the capture phase, so a truncated
drag can be attributed to a specific event rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

CANVAS = '#webgl-georing-canvas'
CARD = '#webgl-georing-map'
DRAG = (150, 40)

INIT = r"""
(() => {
  const log = [];
  window.__diag = { log, moves: 0, rafs: 0 };
  const raw = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = cb => { window.__diag.rafs++; return raw(cb); };
  window.__diag.rawRaf = raw;
  const types = ['pointerdown','pointerup','pointercancel','lostpointercapture','gotpointercapture',
                 'pointerleave','blur','focus','visibilitychange','scroll','resize'];
  for (const t of types) {
    window.addEventListener(t, e => log.push({ t: t, at: performance.now(),
      target: e.target && e.target.id ? e.target.id : (e.target === window ? 'window' : (e.target && e.target.nodeName) || ''),
      x: e.clientX, y: e.clientY, buttons: e.buttons, scrollY: window.scrollY }), true);
  }
  window.addEventListener('pointermove', e => {
    if (e.target && e.target.id === 'webgl-georing-canvas') window.__diag.moves++;
  }, true);
})();
"""


def frames(page, n=3):
    page.evaluate('async n => { for (let i = 0; i < n; i++) await new Promise(r => window.__diag.rawRaf(r)); }', n)


def trial(page, mode: str) -> dict:
    page.click('#georing-reset')
    frames(page)
    page.evaluate('() => { window.__diag.log.length = 0; window.__diag.moves = 0; window.__diag.rafs = 0; }')
    page.locator(CANVAS).scroll_into_view_if_needed()
    box = page.locator(CANVAS).bounding_box()
    cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    scroll_marks = []
    for i in range(1, 5):
        page.mouse.move(cx + DRAG[0] * i / 4, cy + DRAG[1] * i / 4, steps=5)
        if mode == 'card_screenshot':
            page.locator(CARD).screenshot()
        elif mode == 'page_screenshot':
            page.screenshot()
        scroll_marks.append(page.evaluate('() => window.scrollY'))
    page.mouse.up()
    frames(page)
    d = page.evaluate('() => ({ log: window.__diag.log.slice(), moves: window.__diag.moves, rafs: window.__diag.rafs })')
    box_after = page.locator(CANVAS).bounding_box()
    kinds = [e['t'] for e in d['log']]
    return {
        'mode': mode, 'canvas_moves': d['moves'], 'rafs': d['rafs'],
        'truncated': d['moves'] < 20,
        'box_before': box, 'box_after': box_after, 'scrollY_after_segments': scroll_marks,
        'event_counts': {k: kinds.count(k) for k in sorted(set(kinds))},
        'events': [e for e in d['log'] if e['t'] not in ('scroll',)][:40],
    }


def lose_restore(page):
    page.evaluate("() => { const gl = document.getElementById('webgl-georing-canvas').getContext('webgl');"
                  " window.__ext = gl.getExtension('WEBGL_lose_context'); window.__ext.loseContext(); }")
    page.wait_for_function("() => document.getElementById('georing-terminal').textContent.startsWith('[WEBGL CONTEXT LOST]')", timeout=10000)
    page.evaluate('() => window.__ext.restoreContext()')
    page.wait_for_function("() => document.getElementById('georing-terminal').textContent.startsWith('[DEMO FIXTURE]')", timeout=10000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--channel', default='msedge')
    args = ap.parse_args()
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=False)
        ctx = browser.new_context(viewport={'width': 1280, 'height': 800})
        ctx.add_init_script(INIT)
        page = ctx.new_page()
        page.goto(args.url)
        page.wait_for_function("() => document.getElementById('webgl-georing-canvas').dataset.rendered === 'true'", timeout=60000)
        version = browser.version
        for restored in (False, True):
            for mode in ('none', 'card_screenshot', 'page_screenshot'):
                for i in range(args.trials):
                    if restored:
                        lose_restore(page)
                    r = trial(page, mode)
                    r.update({'after_context_restore': restored, 'index': i})
                    results.append(r)
        browser.close()
    summary = {}
    for r in results:
        key = f"{'restored' if r['after_context_restore'] else 'fresh'}/{r['mode']}"
        s = summary.setdefault(key, {'trials': 0, 'truncated': 0, 'cancel_or_lost_capture': 0})
        s['trials'] += 1
        s['truncated'] += int(r['truncated'])
        s['cancel_or_lost_capture'] += int(r['event_counts'].get('pointercancel', 0) > 0 or
                                            (r['event_counts'].get('lostpointercapture', 0) > 0 and r['truncated']))
    args.out.write_text(json.dumps({'browser': version, 'url': args.url, 'summary': summary, 'trials': results}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
