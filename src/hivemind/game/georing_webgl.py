"""Self-contained WebGL renderer for the HUD's Geo-Ring topology panel (PR #24).

The panel is plain HTML + inline JS: raw WebGL (WebGL2 when available, WebGL1
otherwise), no CDN, no external requests. It draws a lat/long wireframe globe,
one glowing point per peer from ``/api/georing-topology`` (colored by continent),
and arcs between peers and their continental bridge and between bridges.
"""

from __future__ import annotations

GEORING_PANEL_HTML = r"""
    <div class="card" id="webgl-georing-map" style="grid-column: 1 / -1;" data-nodes="0" data-arcs="0" data-frames="0" data-errors="0" data-webgl="pending" data-yaw="0">
      <h2>SEASON 3 INTERACTIVE WEBGL 3D GEO-RING TOPOLOGY MAP &amp; MISSIONS (9-12)</h2>
      <div id="georing-stage" style="position:relative;">
        <canvas id="webgl-georing-canvas" width="920" height="360" style="display:block; width:100%; height:360px; background:#050811; border:1px solid #00f0ff; border-radius:6px; cursor:grab; touch-action:none;"></canvas>
        <div id="georing-label" style="display:none; position:absolute; pointer-events:none; background:rgba(5,8,17,0.92); border:1px solid #00f0ff; border-radius:4px; padding:4px 8px; font-size:0.78rem; color:#e2e8f0; white-space:pre; z-index:2;"></div>
        <div id="georing-fallback" style="display:none; padding:40px 16px; text-align:center; border:1px solid #ff3860; border-radius:6px; color:#ff9eb1; background:#050811;"></div>
      </div>
      <div class="terminal" id="georing-terminal">[WEBGL 3D GEO-RING] Loading /api/georing-topology ... Drag to orbit, wheel to zoom, hover a peer for details.</div>
      <div style="font-size:0.75rem; margin-top:4px;">
        <span style="color:#38bdf8;">&#9679; EU</span> &nbsp; <span style="color:#fb923c;">&#9679; US</span> &nbsp; <span style="color:#4ade80;">&#9679; APAC</span>
      </div>
    </div>
    <script>
""" + r"""
(function () {
  'use strict';
  var panel = document.getElementById('webgl-georing-map');
  var canvas = document.getElementById('webgl-georing-canvas');
  var term = document.getElementById('georing-terminal');
  var labelEl = document.getElementById('georing-label');
  var fallback = document.getElementById('georing-fallback');
  if (!panel || !canvas) return;
  if (/[?&]georing=solo\b/.test(location.search)) {
    for (var node = panel; node && node !== document.body; node = node.parentElement) {
      Array.prototype.forEach.call(node.parentElement.children, function (sib) {
        if (sib !== node && sib.tagName !== 'SCRIPT') sib.style.display = 'none';
      });
    }
  }
  var stats = { nodes: 0, arcs: 0, frames: 0, errors: 0, webgl: 'pending' };
  var yaw = 0.6, pitch = 0.35, dist = 3.4;
  window.__georingStats = stats;
  var qy = /[?&]georing_yaw=(-?[0-9.]+)/.exec(location.search);
  if (qy) yaw = parseFloat(qy[1]);
  function publish() {
    panel.setAttribute('data-nodes', stats.nodes);
    panel.setAttribute('data-arcs', stats.arcs);
    panel.setAttribute('data-frames', stats.frames);
    panel.setAttribute('data-errors', stats.errors);
    panel.setAttribute('data-webgl', stats.webgl);
    panel.setAttribute('data-yaw', yaw.toFixed(4));
    var r = canvas.getBoundingClientRect();
    panel.setAttribute('data-canvas-rect', [r.left + window.scrollX, r.top + window.scrollY, r.width, r.height].map(Math.round).join(','));
  }
  function fail(msg) {
    stats.errors++;
    term.textContent = '[GEORING ERROR] ' + msg;
    publish();
  }

  var CONT = {
    EU: { rgb: [0.22, 0.74, 0.97] },
    US: { rgb: [0.98, 0.57, 0.24] },
    APAC: { rgb: [0.29, 0.87, 0.50] }
  };
  function continentOf(n) {
    var r = String(n.region || n.node_id || '').toUpperCase();
    if (r.indexOf('EU') === 0) return 'EU';
    if (r.indexOf('US') === 0 || r.indexOf('NA') === 0) return 'US';
    if (r.indexOf('APAC') === 0 || r.indexOf('AP-') === 0) return 'APAC';
    var lon = Number(n.lon);
    if (lon < -30) return 'US';
    if (lon > 60) return 'APAC';
    return 'EU';
  }

  var m4 = {
    ident: function () { return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; },
    mul: function (a, b) {
      var o = new Array(16);
      for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
        var s = 0;
        for (var k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
    persp: function (fovy, asp, n, f) {
      var t = 1 / Math.tan(fovy / 2);
      return [t / asp,0,0,0, 0,t,0,0, 0,0,(f + n) / (n - f),-1, 0,0,2 * f * n / (n - f),0];
    },
    trans: function (x, y, z) { var m = m4.ident(); m[12] = x; m[13] = y; m[14] = z; return m; },
    rotY: function (a) { var c = Math.cos(a), s = Math.sin(a); return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]; },
    rotX: function (a) { var c = Math.cos(a), s = Math.sin(a); return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]; },
    apply: function (m, v) {
      var x = v[0], y = v[1], z = v[2];
      return [
        m[0]*x + m[4]*y + m[8]*z + m[12],
        m[1]*x + m[5]*y + m[9]*z + m[13],
        m[2]*x + m[6]*y + m[10]*z + m[14],
        m[3]*x + m[7]*y + m[11]*z + m[15]
      ];
    }
  };

  function llToXyz(lat, lon, r) {
    var la = lat * Math.PI / 180, lo = lon * Math.PI / 180;
    return [r * Math.cos(la) * Math.sin(lo), r * Math.sin(la), r * Math.cos(la) * Math.cos(lo)];
  }
  function arcPoints(a, b, lift, steps) {
    var da = Math.hypot(a[0], a[1], a[2]), db = Math.hypot(b[0], b[1], b[2]);
    var ua = [a[0]/da, a[1]/da, a[2]/da], ub = [b[0]/db, b[1]/db, b[2]/db];
    var dot = Math.max(-1, Math.min(1, ua[0]*ub[0] + ua[1]*ub[1] + ua[2]*ub[2]));
    var om = Math.acos(dot), so = Math.sin(om) || 1e-6, out = [];
    for (var i = 0; i <= steps; i++) {
      var t = i / steps, w1 = Math.sin((1 - t) * om) / so, w2 = Math.sin(t * om) / so;
      var rr = 1.0 + lift * Math.sin(Math.PI * t) * Math.min(1, om / 1.5);
      out.push([(ua[0]*w1 + ub[0]*w2) * rr, (ua[1]*w1 + ub[1]*w2) * rr, (ua[2]*w1 + ub[2]*w2) * rr]);
    }
    return out;
  }

  var gl = null, isGL2 = false, prog = null, loc = {};
  var dragging = false, moved = false, lastX = 0, lastY = 0;
  var hover = -1, pinned = -1, mouse = null, peers = [], arcs = [];
  var batches = { grid: null, arcs: null, pts: null };
  var lastT = 0, telemetry = null;

  function initGL() {
    var opts = { antialias: true, alpha: false };
    gl = canvas.getContext('webgl2', opts);
    isGL2 = !!gl;
    if (!gl) gl = canvas.getContext('webgl', opts) || canvas.getContext('experimental-webgl', opts);
    if (!gl) return false;
    stats.webgl = isGL2 ? 'webgl2' : 'webgl1';
    var vs = 'attribute vec3 aPos; attribute vec4 aCol; attribute float aSize;' +
      'uniform mat4 uMVP; uniform mat4 uModel; varying vec4 vCol;' +
      'void main(){ vec4 w = uModel * vec4(aPos,1.0); float front = w.z > -0.15 ? 1.0 : 0.22;' +
      'vCol = vec4(aCol.rgb, aCol.a * front); gl_Position = uMVP * vec4(aPos,1.0); gl_PointSize = aSize * front; }';
    var fs = 'precision mediump float; varying vec4 vCol; uniform float uRound;' +
      'void main(){ if (uRound > 0.5) { float d = distance(gl_PointCoord, vec2(0.5)); if (d > 0.5) discard;' +
      'gl_FragColor = vec4(vCol.rgb, vCol.a * smoothstep(0.5, 0.05, d)); } else gl_FragColor = vCol; }';
    function sh(type, src) {
      var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
      return s;
    }
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
    loc.aPos = gl.getAttribLocation(prog, 'aPos');
    loc.aCol = gl.getAttribLocation(prog, 'aCol');
    loc.aSize = gl.getAttribLocation(prog, 'aSize');
    loc.uMVP = gl.getUniformLocation(prog, 'uMVP');
    loc.uModel = gl.getUniformLocation(prog, 'uModel');
    loc.uRound = gl.getUniformLocation(prog, 'uRound');
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    return true;
  }

  function makeBatch(data, ranges, old) {
    if (old && old.buf) gl.deleteBuffer(old.buf);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    return { buf: buf, ranges: ranges };
  }
  function pushV(d, p, c, a, s) { d.push(p[0], p[1], p[2], c[0], c[1], c[2], a, s); }

  function buildGrid() {
    var d = [], ranges = [], col = [0.10, 0.45, 0.62], i, j, start;
    for (var lat = -60; lat <= 60; lat += 30) {
      start = d.length / 8;
      for (i = 0; i <= 72; i++) pushV(d, llToXyz(lat, i * 5 - 180, 1), lat === 0 ? [0.2, 0.7, 0.9] : col, 0.55, 1);
      ranges.push([start, 73]);
    }
    for (var lon = -180; lon < 180; lon += 30) {
      start = d.length / 8;
      for (j = 0; j <= 36; j++) pushV(d, llToXyz(j * 5 - 90, lon, 1), col, 0.55, 1);
      ranges.push([start, 37]);
    }
    return makeBatch(d, ranges, null);
  }

  function rebuild(topo) {
    var nodes = (topo && (topo.nodes || topo.peers)) || [];
    peers = nodes.filter(function (n) { return isFinite(Number(n.lat)) && isFinite(Number(n.lon)); }).map(function (n) {
      var cont = continentOf(n);
      return { id: n.node_id || n.id || n.region, region: n.region, cont: cont, lat: Number(n.lat), lon: Number(n.lon),
               rtt: n.rtt_ms, nat: n.nat_mode, pos: llToXyz(Number(n.lat), Number(n.lon), 1.0), bridge: false };
    });
    var bridgeOf = {};
    peers.forEach(function (p) {
      if (!bridgeOf[p.cont] || p.rtt < bridgeOf[p.cont].rtt) bridgeOf[p.cont] = p;
    });
    Object.keys(bridgeOf).forEach(function (k) { bridgeOf[k].bridge = true; });
    arcs = [];
    peers.forEach(function (p) {
      var b = bridgeOf[p.cont];
      if (b && b !== p) arcs.push({ a: p, b: b, kind: 'peer' });
    });
    var bs = Object.keys(bridgeOf).map(function (k) { return bridgeOf[k]; });
    for (var i = 0; i < bs.length; i++) for (var j = i + 1; j < bs.length; j++) arcs.push({ a: bs[i], b: bs[j], kind: 'bridge' });
    stats.nodes = peers.length; stats.arcs = arcs.length;
    if (!gl) return;
    var d = [], ranges = [];
    arcs.forEach(function (e) {
      var c = e.kind === 'bridge' ? [1.0, 0.85, 0.35] : CONT[e.a.cont].rgb;
      var pts = arcPoints(e.a.pos, e.b.pos, e.kind === 'bridge' ? 0.32 : 0.14, 40);
      var start = d.length / 8;
      pts.forEach(function (p) { pushV(d, p, c, e.kind === 'bridge' ? 0.95 : 0.7, 1); });
      ranges.push([start, pts.length]);
    });
    batches.arcs = makeBatch(d, ranges, batches.arcs);
    var pd = [];
    peers.forEach(function (p) {
      var c = CONT[p.cont].rgb;
      pushV(pd, p.pos, c, 0.25, p.bridge ? 34 : 26);
      pushV(pd, p.pos, c, 1.0, p.bridge ? 16 : 11);
      pushV(pd, p.pos, [1, 1, 1], 0.9, 5);
    });
    batches.pts = makeBatch(pd, [[0, peers.length * 3]], batches.pts);
  }

  function bind(batch) {
    gl.bindBuffer(gl.ARRAY_BUFFER, batch.buf);
    gl.enableVertexAttribArray(loc.aPos); gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(loc.aCol); gl.vertexAttribPointer(loc.aCol, 4, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(loc.aSize); gl.vertexAttribPointer(loc.aSize, 1, gl.FLOAT, false, 32, 28);
  }

  var cssW = 1, cssH = 1;
  function resize() {
    var dpr = window.devicePixelRatio || 1;
    var r = canvas.getBoundingClientRect();
    cssW = Math.max(1, r.width); cssH = Math.max(1, r.height);
    var w = Math.max(1, Math.round(cssW * dpr)), h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function matrices() {
    var model = m4.mul(m4.rotX(pitch), m4.rotY(yaw));
    var view = m4.trans(0, 0, -dist);
    var proj = m4.persp(0.8, canvas.width / canvas.height, 0.1, 20);
    return { model: model, mvp: m4.mul(proj, m4.mul(view, model)) };
  }
  function project(p, M) {
    var c = m4.apply(M.mvp, p);
    var w = m4.apply(M.model, p);
    return { x: (c[0] / c[3] * 0.5 + 0.5) * cssW, y: (1 - (c[1] / c[3] * 0.5 + 0.5)) * cssH, front: w[2] > -0.15 };
  }

  function frame(t) {
    var dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0;
    lastT = t;
    if (!dragging) yaw += dt * 0.18;
    try {
      resize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.02, 0.03, 0.07, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(prog);
      var M = matrices();
      gl.uniformMatrix4fv(loc.uMVP, false, new Float32Array(M.mvp));
      gl.uniformMatrix4fv(loc.uModel, false, new Float32Array(M.model));
      gl.uniform1f(loc.uRound, 0);
      [batches.grid, batches.arcs].forEach(function (b) {
        if (!b) return;
        bind(b);
        b.ranges.forEach(function (r) { gl.drawArrays(gl.LINE_STRIP, r[0], r[1]); });
      });
      if (batches.pts && peers.length) {
        gl.uniform1f(loc.uRound, 1);
        bind(batches.pts);
        gl.drawArrays(gl.POINTS, 0, peers.length * 3);
      }
      updateLabel(M);
      stats.frames++;
    } catch (e) { fail('render: ' + e.message); return; }
    publish();
    requestAnimationFrame(frame);
  }

  function pick(M) {
    if (!mouse) return -1;
    var best = -1, bd = 18 * 18;
    peers.forEach(function (p, i) {
      var s = project(p.pos, M);
      if (!s.front) return;
      var d2 = (s.x - mouse.x) * (s.x - mouse.x) + (s.y - mouse.y) * (s.y - mouse.y);
      if (d2 < bd) { bd = d2; best = i; }
    });
    return best;
  }
  function updateLabel(M) {
    hover = dragging ? -1 : pick(M);
    var i = hover >= 0 ? hover : pinned;
    if (i < 0 || !peers[i]) { labelEl.style.display = 'none'; return; }
    var p = peers[i], s = project(p.pos, M);
    labelEl.textContent = p.id + (p.bridge ? '  [bridge]' : '') + '\n' + p.cont + ' | ' + p.region +
      '\nRTT ' + p.rtt + ' ms | NAT ' + p.nat;
    labelEl.style.display = 'block';
    labelEl.style.left = Math.min(s.x + 14, Math.max(0, cssW - 230)) + 'px';
    labelEl.style.top = Math.max(0, s.y - 20) + 'px';
  }

  function events() {
    function rel(e) { var r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    canvas.addEventListener('pointerdown', function (e) {
      dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
      canvas.style.cursor = 'grabbing';
      try { canvas.setPointerCapture(e.pointerId); } catch (x) {}
    });
    canvas.addEventListener('pointermove', function (e) {
      mouse = rel(e);
      if (!dragging) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      yaw += dx * 0.008; pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.008));
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', function (e) {
      if (dragging && !moved) { mouse = rel(e); pinned = pick(matrices()); }
      dragging = false; canvas.style.cursor = 'grab';
    });
    canvas.addEventListener('pointercancel', function () { dragging = false; });
    canvas.addEventListener('pointerleave', function () { mouse = null; });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      dist = Math.max(1.8, Math.min(8, dist * Math.exp(e.deltaY * 0.0012)));
    }, { passive: false });
    window.addEventListener('resize', resize);
  }

  function status() {
    var s = '[WEBGL 3D GEO-RING ' + stats.webgl.toUpperCase() + '] ' + stats.nodes + ' peers / ' + stats.arcs +
      ' arcs. Drag to orbit, wheel to zoom, hover or click a peer.';
    if (telemetry) s += ' | live telemetry frame #' + telemetry.seq;
    term.textContent = s;
  }
  function load() {
    fetch('/api/georing-topology', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (topo) { rebuild(topo); status(); publish(); })
      .catch(function (e) { fail('topology fetch failed: ' + e.message); });
  }
  function ws() {
    if (!window.WebSocket || location.protocol === 'file:') return;
    try {
      var s = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/swarm-telemetry');
      s.onmessage = function (m) { try { telemetry = JSON.parse(m.data); } catch (e) {} };
      s.onclose = function () { setTimeout(ws, 5000); };
    } catch (e) {}
  }

  try {
    if (!initGL()) {
      stats.webgl = 'unavailable';
      canvas.style.display = 'none';
      fallback.style.display = 'block';
      fallback.textContent = '[WEBGL UNAVAILABLE] This browser or GPU could not create a WebGL context, so the 3D Geo-Ring globe cannot be drawn. Peer data is still available at /api/georing-topology.';
      term.textContent = fallback.textContent;
      publish();
      return;
    }
    batches.grid = buildGrid();
    events(); resize();
    load(); setInterval(load, 2000); ws();
    requestAnimationFrame(frame);
  } catch (e) {
    stats.webgl = 'error';
    canvas.style.display = 'none';
    fallback.style.display = 'block';
    fallback.textContent = '[WEBGL ERROR] ' + e.message;
    fail(e.message);
  }
})();
</script>
"""
