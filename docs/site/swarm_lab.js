/*
 * HiveMind Swarm Lab ("SwarmLab-1"): JavaScript port of src/hivemind/lab/swarm_lab.py.
 *
 * WHAT THIS IS NOT: an EDUCATIONAL, QUALITATIVE ANALOGY of an effect measured in the real E13/E14 swarm experiments. It is NOT a reproduction of them,
 * its defaults were chosen to make the effect visible, and its numbers must never be quoted as results about real LLM training.
 *
 * PACKAGING: one classic-script UMD file with no dependencies, no network and no DOM.
 *   - <script src="swarm_lab.js"></script>        -> global `SwarmLab`
 *   - node: const SwarmLab = require("./swarm_lab.js")      (or: import SwarmLab from "./swarm_lab.js")
 *
 * CONTRACT: the Python module docstring is normative. Every floating-point operation below is performed in the same order as in Python, so that
 * simulate() matches the Python results to <= 1e-9 (in practice ~1e-15; the only difference is the libm implementation of log/cos).
 * Field names are snake_case on purpose: scenario JSON is interchangeable with the Python CLI/`simulate`.
 */
(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.SwarmLab = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : (typeof self !== "undefined" ? self : this), function () {
  "use strict";

  var MAX_EXCLUSION = 1; // 'fair': a worker excluded this many consecutive rounds is required next round
  var MAX_STALENESS = 2; // 'deferred': a carried delta older than this many rounds is dropped
  var POLICIES = ["discard", "fair", "deferred"];
  var STRAGGLES = ["none", "persistent", "rotating"];
  var MERGES = ["direct", "rda"];
  var NONMONETARY_NOTICE =
    "Notice: $HIVE / PTS are internal, non-monetary, non-transferable accounting units; no money or financial payment is or will ever be involved. " +
    "(See LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md.)";

  // ------------------------------------------------------------------ PRNG (mulberry32 + Box-Muller)
  function Rng(seed) {
    this.state = Number(seed) >>> 0; // seeds are taken mod 2^32 (ToUint32 is exactly that)
  }
  Rng.prototype.uniform = function () {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    var t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  Rng.prototype.gauss = function () {
    var u1 = this.uniform();
    var u2 = this.uniform();
    return Math.sqrt(-2.0 * Math.log(Math.max(u1, 1e-300))) * Math.cos(2.0 * Math.PI * u2);
  };

  // ------------------------------------------------------------------ Scenario
  var SCENARIO_DEFAULTS = {
    M: 4, K: 3, rounds: 12, H: 10, eta: 0.1, sigma: 0.2, dim: 16, spread: 2.0, offset: 0.0, seed: 1,
    merge: "direct", policy: "discard", straggle: "none", penalty: 2.0, straggle_every: 1, grace: 0.1,
    churn_start: 0, churn_end: 0, outer_lr: 0.7, outer_momentum: 0.9
  };
  var INT_FIELDS = ["M", "K", "rounds", "H", "dim", "seed", "straggle_every", "churn_start", "churn_end"];

  function copy(obj) {
    var out = {};
    for (var k in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, k)) out[k] = obj[k];
    }
    return out;
  }

  /** Fill defaults, reject unknown fields, validate exactly like Scenario.validate() in Python. Returns a new plain object. */
  function makeScenario(partial) {
    var sc = copy(SCENARIO_DEFAULTS);
    var given = partial || {};
    for (var key in given) {
      if (!Object.prototype.hasOwnProperty.call(given, key)) continue;
      if (!Object.prototype.hasOwnProperty.call(SCENARIO_DEFAULTS, key)) throw new Error("unknown scenario field: " + key);
      sc[key] = given[key];
    }
    for (var f in sc) {
      if (typeof SCENARIO_DEFAULTS[f] === "number") {
        if (typeof sc[f] !== "number" || !isFinite(sc[f])) throw new Error(f + " must be a finite number");
      }
    }
    INT_FIELDS.forEach(function (name) {
      if (Math.floor(sc[name]) !== sc[name]) throw new Error(name + " must be an integer");
    });
    if (!(sc.M >= 1 && sc.M <= 6)) throw new Error("M must be in 1..6");
    if (!(sc.K >= 1 && sc.K <= sc.M)) throw new Error("K must be in 1..M");
    if (sc.rounds < 1 || sc.H < 1 || sc.straggle_every < 1) throw new Error("rounds, H and straggle_every must be >= 1");
    if (!(sc.dim >= 2 && sc.dim <= 64)) throw new Error("dim must be in 2..64");
    if (sc.eta <= 0 || sc.sigma < 0 || sc.spread < 0 || sc.offset < 0 || sc.penalty < 0 || sc.grace < 0) {
      throw new Error("eta must be > 0 and sigma, spread, penalty, grace must be >= 0");
    }
    if (sc.outer_lr <= 0 || !(sc.outer_momentum >= 0 && sc.outer_momentum < 1)) {
      throw new Error("outer_lr must be > 0 and outer_momentum in [0, 1)");
    }
    if (MERGES.indexOf(sc.merge) < 0) throw new Error("merge must be 'direct' or 'rda'");
    if (POLICIES.indexOf(sc.policy) < 0) throw new Error("policy must be one of " + POLICIES.join(", "));
    if (STRAGGLES.indexOf(sc.straggle) < 0) throw new Error("straggle must be one of " + STRAGGLES.join(", "));
    if (sc.straggle !== "none" && sc.M < 2) throw new Error("a straggler needs M >= 2");
    if (sc.churn_start !== 0 || sc.churn_end !== 0) {
      if (sc.M < 2 || !(sc.churn_start >= 1 && sc.churn_start <= sc.churn_end && sc.churn_end <= sc.rounds)) {
        throw new Error("churn needs M >= 2 and 1 <= churn_start <= churn_end <= rounds (both 0 = off)");
      }
    }
    return sc;
  }

  var PRESETS = {
    "iid-sync": { M: 4, K: 4, spread: 0.0, offset: 3.0, straggle: "none", policy: "discard" },
    "persistent-straggler": { M: 4, K: 3, straggle: "persistent", policy: "discard" },
    "rotating-straggler": { M: 4, K: 3, straggle: "rotating", policy: "discard" },
    "churn": { M: 4, K: 3, straggle: "persistent", policy: "fair", churn_start: 3, churn_end: 6 }
  };

  // ------------------------------------------------------------------ model
  function norm(v) {
    var acc = 0.0;
    for (var i = 0; i < v.length; i++) acc += v[i] * v[i];
    return Math.sqrt(acc);
  }

  function unit(v) {
    var n = norm(v);
    if (n < 1e-300) return v;
    var out = new Array(v.length);
    for (var i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  function buildDomains(sc) {
    var rng = new Rng(sc.seed);
    var d = sc.dim;
    var k, i;
    var shared = [];
    for (k = 0; k < d; k++) shared.push(rng.gauss());
    var aAll = [], cAll = [];
    for (i = 0; i < sc.M; i++) {
      var a = [];
      for (k = 0; k < d; k++) a.push(0.5 + 1.5 * rng.uniform());
      var v = [];
      for (k = 0; k < d; k++) v.push(rng.gauss());
      var u = unit(v);
      var c = [];
      for (k = 0; k < d; k++) c.push(shared[k] + sc.spread * u[k]);
      aAll.push(a);
      cAll.push(c);
    }
    var vv = [];
    for (k = 0; k < d; k++) vv.push(rng.gauss());
    var w = unit(vv);
    var start = [];
    for (k = 0; k < d; k++) start.push(shared[k] + sc.offset * w[k]);
    return { a: aAll, c: cAll, shared: start };
  }

  function domainLoss(theta, a, c) {
    var acc = 0.0;
    for (var k = 0; k < theta.length; k++) {
      var e = theta[k] - c[k];
      acc += a[k] * e * e;
    }
    return 0.5 * acc;
  }

  /** Returns {mean, per}: global loss (mean of domain losses, ascending order) and the per-domain losses. */
  function losses(theta, aAll, cAll) {
    var per = [];
    for (var i = 0; i < aAll.length; i++) per.push(domainLoss(theta, aAll[i], cAll[i]));
    var tot = 0.0;
    for (var j = 0; j < per.length; j++) tot += per[j];
    return { mean: tot / per.length, per: per };
  }

  /** items: array of {delta, w}. 'rda' multiplies the weighted mean by r = sum w|D| / |sum w D|. */
  function mergeDeltas(items, mode) {
    var d = items[0].delta.length;
    var wsum = 0.0, num = 0.0, s = [], k;
    for (k = 0; k < d; k++) s.push(0.0);
    for (var j = 0; j < items.length; j++) {
      var delta = items[j].delta, w = items[j].w;
      wsum += w;
      num += w * norm(delta);
      for (k = 0; k < d; k++) s[k] += w * delta[k];
    }
    var mean = [];
    for (k = 0; k < d; k++) mean.push(s[k] / wsum);
    if (mode === "direct") return mean;
    var sn = norm(s);
    var r = sn < 1e-300 ? 1.0 : num / sn;
    var out = [];
    for (k = 0; k < d; k++) out.push(r * mean[k]);
    return out;
  }

  /** Queue entries usable in round rnd: {keep (with .w), dropped}, ordered by (base round, worker). */
  function carriedWeights(queue, rnd) {
    var sorted = queue.slice().sort(function (x, y) { return x.base - y.base || x.worker - y.worker; });
    var keep = [], dropped = 0;
    sorted.forEach(function (e) {
      var s = rnd - e.base;
      if (s > MAX_STALENESS) dropped += 1;
      else keep.push({ delta: e.delta, base: e.base, worker: e.worker, w: 1.0 / (1.0 + s) });
    });
    return { keep: keep, dropped: dropped };
  }

  function penalised(sc, rnd) {
    if (sc.straggle === "none" || (rnd - 1) % sc.straggle_every !== 0) return -1;
    if (sc.straggle === "persistent") return sc.M - 1;
    return sc.M - 1 - (Math.floor((rnd - 1) / sc.straggle_every) % (sc.M - 1));
  }

  /** E13's rho = (L0 - L_cell) / (L0 - L_sync); null when the synchronous reference did not improve on L0. */
  function retention(l0, lFinal, lSync) {
    var den = l0 - lSync;
    return Math.abs(den) <= 1e-12 ? null : (l0 - lFinal) / den;
  }

  function contains(list, x) {
    return list.indexOf(x) >= 0;
  }

  function run(sc) {
    var dom = buildDomains(sc);
    var aAll = dom.a, cAll = dom.c;
    var d = sc.dim, M = sc.M, i, k;
    var rngs = [];
    for (i = 0; i < M; i++) rngs.push(new Rng(sc.seed + 1 + i));
    var theta = dom.shared.slice();
    var buf = [];
    for (k = 0; k < d; k++) buf.push(0.0);
    var first = losses(theta, aAll, cAll);
    var l0 = first.mean, per0 = first.per;
    var curve = [l0], rows = [], queue = [], excl = [], totalT = 0.0;
    for (i = 0; i < M; i++) excl.push(0);
    for (var rnd = 1; rnd <= sc.rounds; rnd++) {
      var absent = (sc.churn_end && sc.churn_start <= rnd && rnd <= sc.churn_end) ? 1 : -1;
      var alive = [];
      for (i = 0; i < M; i++) if (i !== absent) alive.push(i);
      var slow = penalised(sc, rnd);
      var arr = {}, deltas = {};
      alive.forEach(function (w) {
        var rng = rngs[w];
        arr[w] = 1.0 + 0.2 * rng.uniform() + (w === slow ? sc.penalty : 0.0);
        var th = theta.slice();
        for (var step = 0; step < sc.H; step++) {
          for (var kk = 0; kk < d; kk++) {
            var g = aAll[w][kk] * (th[kk] - cAll[w][kk]) + sc.sigma * rng.gauss();
            th[kk] = th[kk] - sc.eta * g;
          }
        }
        var dl = [];
        for (var q = 0; q < d; q++) dl.push(theta[q] - th[q]);
        deltas[w] = dl;
      });
      var order = alive.slice().sort(function (x, y) { return arr[x] - arr[y] || x - y; });
      var tK = arr[order[Math.min(sc.K, alive.length) - 1]];
      var maxArr = -Infinity;
      alive.forEach(function (w) { if (arr[w] > maxArr) maxArr = arr[w]; });
      var close = Math.min(tK * (1.0 + sc.grace), maxArr);
      var required = alive.filter(function (w) { return sc.policy === "fair" && excl[w] >= MAX_EXCLUSION && arr[w] > close; });
      required.forEach(function (w) { close = Math.max(close, arr[w]); });
      var admitted = alive.filter(function (w) { return arr[w] <= close; });
      var late = alive.filter(function (w) { return arr[w] > close; });
      var items = admitted.map(function (w) { return { delta: deltas[w], w: 1.0 }; });
      var nCarried = 0, dropped = 0;
      if (sc.policy === "deferred") {
        var cw = carriedWeights(queue, rnd);
        dropped = cw.dropped;
        cw.keep.forEach(function (e) { items.push({ delta: e.delta, w: e.w }); });
        nCarried = cw.keep.length;
        queue = late.map(function (w) { return { delta: deltas[w], base: rnd, worker: w }; });
      }
      for (i = 0; i < M; i++) excl[i] = (contains(admitted, i) || !contains(alive, i)) ? 0 : excl[i] + 1;
      var gvec = mergeDeltas(items, sc.merge);
      for (k = 0; k < d; k++) {
        buf[k] = sc.outer_momentum * buf[k] + gvec[k];
        theta[k] = theta[k] - sc.outer_lr * (gvec[k] + sc.outer_momentum * buf[k]);
      }
      totalT += close;
      var cur = losses(theta, aAll, cAll);
      curve.push(cur.mean);
      rows.push({ round: rnd, loss: cur.mean, domain_losses: cur.per, alive: alive, admitted: admitted, late: late, required: required,
                  carried: nCarried, dropped_stale: dropped, round_time: close });
    }
    return { initial_loss: l0, initial_domain_losses: per0, loss_curve: curve, rounds: rows, total_time: totalT, final_loss: curve[curve.length - 1],
             final_domain_losses: rows[rows.length - 1].domain_losses };
  }

  // ------------------------------------------------------------------ SHA-256 (digest of the loss curve, no dependency)
  var K256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  ];

  /** SHA-256 hex digest of an ASCII string (the digest input only contains digits, '-', '.', ','). */
  function sha256Hex(ascii) {
    var bytes = [], i;
    for (i = 0; i < ascii.length; i++) bytes.push(ascii.charCodeAt(i) & 0xff);
    var bitLen = bytes.length * 8;
    bytes.push(0x80);
    while (bytes.length % 64 !== 56) bytes.push(0);
    var hi = Math.floor(bitLen / 4294967296), lo = bitLen >>> 0;
    bytes.push((hi >>> 24) & 255, (hi >>> 16) & 255, (hi >>> 8) & 255, hi & 255, (lo >>> 24) & 255, (lo >>> 16) & 255, (lo >>> 8) & 255, lo & 255);
    var h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    var w = new Array(64);
    function rotr(x, n) { return (x >>> n) | (x << (32 - n)); }
    for (var off = 0; off < bytes.length; off += 64) {
      for (i = 0; i < 16; i++) {
        w[i] = ((bytes[off + 4 * i] << 24) | (bytes[off + 4 * i + 1] << 16) | (bytes[off + 4 * i + 2] << 8) | bytes[off + 4 * i + 3]) >>> 0;
      }
      for (i = 16; i < 64; i++) {
        var s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
        var s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
      }
      var a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
      for (i = 0; i < 64; i++) {
        var S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        var ch = (e & f) ^ (~e & g);
        var t1 = (hh + S1 + ch + K256[i] + w[i]) >>> 0;
        var S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        var maj = (a & b) ^ (a & c) ^ (b & c);
        var t2 = (S0 + maj) >>> 0;
        hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
      }
      h[0] = (h[0] + a) >>> 0; h[1] = (h[1] + b) >>> 0; h[2] = (h[2] + c) >>> 0; h[3] = (h[3] + d) >>> 0;
      h[4] = (h[4] + e) >>> 0; h[5] = (h[5] + f) >>> 0; h[6] = (h[6] + g) >>> 0; h[7] = (h[7] + hh) >>> 0;
    }
    return h.map(function (x) { return ("00000000" + x.toString(16)).slice(-8); }).join("");
  }

  function digestOf(curve) {
    return sha256Hex(curve.map(function (v) { return v.toFixed(9); }).join(",")).slice(0, 16);
  }

  // ------------------------------------------------------------------ public API
  /** Run one scenario (object with any subset of the Scenario fields). Same result shape as Python simulate(). */
  function simulate(scenario) {
    var sc = makeScenario(scenario);
    var res = run(sc);
    var ref = copy(sc);
    ref.K = sc.M; ref.straggle = "none"; ref.policy = "discard"; ref.churn_start = 0; ref.churn_end = 0;
    var lSync = run(ref).final_loss;
    res.scenario = sc;
    res.sync_final_loss = lSync;
    res.retention = retention(res.initial_loss, res.final_loss, lSync);
    res.digest = digestOf(res.loss_curve);
    return res;
  }

  /** Rows for policy x straggle (straggle-major, like Python): retention, final loss, total simulated time, late and carried counts. */
  function comparePolicies(base, policies, straggles) {
    var sc = makeScenario(base || {});
    var pols = policies || POLICIES, sts = straggles || STRAGGLES;
    var rows = [];
    sts.forEach(function (st) {
      pols.forEach(function (pol) {
        var s = copy(sc);
        s.straggle = st; s.policy = pol;
        var r = simulate(s);
        rows.push({
          policy: pol, straggle: st, retention: r.retention, final_loss: r.final_loss, total_time: r.total_time,
          late_total: r.rounds.reduce(function (a, x) { return a + x.late.length; }, 0),
          carried_total: r.rounds.reduce(function (a, x) { return a + x.carried; }, 0)
        });
      });
    });
    return rows;
  }

  /** A preset merged with explicit overrides, as a validated full scenario. */
  function presetScenario(name, overrides) {
    if (!Object.prototype.hasOwnProperty.call(PRESETS, name)) throw new Error("unknown preset: " + name);
    var merged = copy(PRESETS[name]);
    var o = overrides || {};
    for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) merged[k] = o[k];
    return makeScenario(merged);
  }

  return {
    ENGINE: "SwarmLab-1", Rng: Rng, SCENARIO_DEFAULTS: SCENARIO_DEFAULTS, PRESETS: PRESETS, POLICIES: POLICIES, STRAGGLES: STRAGGLES, MERGES: MERGES,
    MAX_EXCLUSION: MAX_EXCLUSION, MAX_STALENESS: MAX_STALENESS, NONMONETARY_NOTICE: NONMONETARY_NOTICE,
    makeScenario: makeScenario, presetScenario: presetScenario, buildDomains: buildDomains, domainLoss: domainLoss, losses: losses,
    mergeDeltas: mergeDeltas, carriedWeights: carriedWeights, penalised: penalised, retention: retention,
    simulate: simulate, comparePolicies: comparePolicies, sha256Hex: sha256Hex, digestOf: digestOf
  };
});
