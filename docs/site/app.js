/*
 * HiveMind Swarm Lab site: UI glue and shared renderers (also loaded by game.html).
 *
 * - Pure helpers (SVG string builders, tables, the results panel) have no DOM dependency and are exposed as `SwarmSite`
 *   (browser global and node `require`), so they can be exercised under node.
 * - DOM wiring only runs when the matching elements exist (#lab-form on index.html, #real-results, the theme button).
 * - No network access, no cookies; localStorage is only used inside try/catch for UI preferences.
 *
 * EDUCATIONAL TOY: nothing computed here is a result about real LLM training.
 */
(function (root) {
  "use strict";

  var Lab = root.SwarmLab || (typeof require === "function" ? require("./swarm_lab.js") : null);

  // ---------------------------------------------------------------- small utilities
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmt(x, nd) {
    if (x === null || x === undefined || (typeof x === "number" && !isFinite(x))) return "n/a";
    return Number(x).toFixed(nd === undefined ? 3 : nd);
  }
  function pm(mean, half, nd) {
    return fmt(mean, nd) + " ± " + fmt(half, nd);
  }
  function storeGet(key) {
    try { return root.localStorage.getItem(key); } catch (e) { return null; }
  }
  function storeSet(key, value) {
    try { root.localStorage.setItem(key, value); } catch (e) { /* storage unavailable: the page works without it */ }
  }
  function prefersReducedMotion() {
    try { return !!(root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches); } catch (e) { return false; }
  }

  /** The CLI command that reproduces a scenario (only fields that differ from the defaults). */
  function cliCommand(sc) {
    var d = Lab.SCENARIO_DEFAULTS, parts = ["python3 scripts/swarm_lab.py simulate"];
    Object.keys(d).forEach(function (k) {
      if (sc[k] !== d[k]) parts.push("--" + k.replace(/_/g, "-") + " " + sc[k]);
    });
    return parts.join(" ");
  }

  // ---------------------------------------------------------------- SVG builders (strings; numbers only, so no injection risk)
  var W = 640;

  function niceMax(v) {
    if (!(v > 0)) return 1;
    var p = Math.pow(10, Math.floor(Math.log(v) / Math.LN10));
    var f = v / p;
    var n = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
    return n * p;
  }

  /**
   * Global loss per round. `res` = simulate() result, `ref` = synchronous reference result. `upto` limits the drawn rounds (animation).
   * Solid line + circles = this run, dashed line + squares = synchronous reference (shape and dash, not only colour, tell them apart).
   */
  function lossChartSvg(res, ref, upto) {
    var R = res.loss_curve.length - 1, last = upto === undefined ? R : Math.max(0, Math.min(R, upto));
    var H = 300, L = 52, Rm = 14, T = 14, B = 38, pw = W - L - Rm, ph = H - T - B;
    var ymax = niceMax(Math.max.apply(null, res.loss_curve.concat(ref.loss_curve)));
    function X(i) { return L + (R === 0 ? 0 : pw * i / R); }
    function Y(v) { return T + ph * (1 - v / ymax); }
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-labelledby="lc-t lc-d">';
    s += '<title id="lc-t">Global loss per round</title>';
    s += '<desc id="lc-d">Global loss goes from ' + fmt(res.loss_curve[0]) + " at round 0 to " + fmt(res.final_loss) + " after round " + R +
      ". The synchronous reference ends at " + fmt(ref.final_loss) + ". Toy model, not a real training run.</desc>";
    var i;
    for (i = 0; i <= 4; i++) {
      var yv = ymax * i / 4;
      s += '<line class="grid" x1="' + L + '" x2="' + (W - Rm) + '" y1="' + Y(yv).toFixed(1) + '" y2="' + Y(yv).toFixed(1) + '"/>';
      s += '<text x="' + (L - 6) + '" y="' + (Y(yv) + 4).toFixed(1) + '" text-anchor="end">' + (yv >= 10 ? yv.toFixed(0) : yv.toFixed(2)) + "</text>";
    }
    var step = Math.max(1, Math.ceil(R / 8));
    for (i = 0; i <= R; i += step) {
      s += '<text x="' + X(i).toFixed(1) + '" y="' + (H - B + 16) + '" text-anchor="middle">' + i + "</text>";
    }
    s += '<line class="axis" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (T + ph) + '"/><line class="axis" x1="' + L + '" x2="' + (W - Rm) + '" y1="' + (T + ph) + '" y2="' + (T + ph) + '"/>';
    s += '<text class="muted" x="' + (L + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">round</text>';
    s += '<text class="muted" transform="translate(12 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">global loss (mean of domains)</text>';
    function path(curve, style, marker) {
      var pts = [], k, out = "";
      for (k = 0; k <= last; k++) pts.push(X(k).toFixed(1) + "," + Y(curve[k]).toFixed(1));
      out += '<polyline fill="none" style="' + style + '" points="' + pts.join(" ") + '"/>';
      if (R <= 30) {
        for (k = 0; k <= last; k++) {
          var cx = X(k), cy = Y(curve[k]);
          out += marker === "circle"
            ? '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="3" style="fill:var(--c1)"/>'
            : '<rect x="' + (cx - 2.5).toFixed(1) + '" y="' + (cy - 2.5).toFixed(1) + '" width="5" height="5" style="fill:var(--c2)"/>';
        }
      }
      return out;
    }
    s += path(ref.loss_curve, "stroke:var(--c2);stroke-width:2;stroke-dasharray:7 4", "square");
    s += path(res.loss_curve, "stroke:var(--c1);stroke-width:2.6", "circle");
    return s + "</svg>";
  }

  function lossLegendHtml() {
    return '<ul class="legend" aria-label="chart legend">' +
      '<li><svg width="34" height="10" aria-hidden="true"><line x1="0" y1="5" x2="34" y2="5" style="stroke:var(--c1);stroke-width:2.6"/></svg>this run (solid)</li>' +
      '<li><svg width="34" height="10" aria-hidden="true"><line x1="0" y1="5" x2="34" y2="5" style="stroke:var(--c2);stroke-width:2;stroke-dasharray:7 4"/></svg>synchronous reference, K=M, no stragglers (dashed)</li></ul>';
  }

  /** Per-domain final loss: start (grey), this run (blue), synchronous reference (amber, outlined). */
  function domainBarsSvg(res, ref) {
    var M = res.initial_domain_losses.length, LBL = 96, barH = 9, gap = 3, grp = 3 * barH + 2 * gap + 10, H = M * grp + 8;
    var vmax = niceMax(Math.max.apply(null, res.initial_domain_losses.concat(res.final_domain_losses, ref.final_domain_losses)));
    var pw = W - LBL - 60;
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-labelledby="db-t db-d"><title id="db-t">Final loss per data domain</title>';
    var d = "Per domain (start, this run, synchronous reference): ";
    for (var i = 0; i < M; i++) {
      d += "domain " + i + ": " + fmt(res.initial_domain_losses[i], 2) + ", " + fmt(res.final_domain_losses[i], 2) + ", " + fmt(ref.final_domain_losses[i], 2) + "; ";
    }
    s += '<desc id="db-d">' + esc(d) + "</desc>";
    for (i = 0; i < M; i++) {
      var y0 = 4 + i * grp;
      s += '<text x="' + (LBL - 8) + '" y="' + (y0 + 2 * barH) + '" text-anchor="end">domain ' + i + "</text>";
      var vals = [[res.initial_domain_losses[i], "fill:var(--c3)", "start"], [res.final_domain_losses[i], "fill:var(--c1)", "this run"],
                  [ref.final_domain_losses[i], "fill:none;stroke:var(--c2);stroke-width:2", "sync"]];
      for (var j = 0; j < 3; j++) {
        var w = Math.max(1, pw * vals[j][0] / vmax), y = y0 + j * (barH + gap);
        s += '<rect x="' + LBL + '" y="' + y + '" width="' + w.toFixed(1) + '" height="' + barH + '" style="' + vals[j][1] + '"/>';
        s += '<text x="' + (LBL + w + 5).toFixed(1) + '" y="' + (y + barH - 1) + '">' + fmt(vals[j][0], 2) + "</text>";
      }
    }
    return s + "</svg>";
  }

  function domainLegendHtml() {
    return '<ul class="legend" aria-label="bar legend">' +
      '<li><svg width="16" height="10" aria-hidden="true"><rect width="16" height="10" style="fill:var(--c3)"/></svg>start</li>' +
      '<li><svg width="16" height="10" aria-hidden="true"><rect width="16" height="10" style="fill:var(--c1)"/></svg>this run, final</li>' +
      '<li><svg width="16" height="10" aria-hidden="true"><rect x="1" y="1" width="14" height="8" style="fill:none;stroke:var(--c2);stroke-width:2"/></svg>synchronous reference, final</li></ul>';
  }

  /**
   * Which worker was admitted (A), required by the fair policy and admitted (R), late (L) or absent (-) in each round.
   * Under 'deferred' a late delta (L) is merged one round later: the bottom row counts the carried deltas merged in each round.
   */
  function stripSvg(res, upto) {
    var rows = res.rounds, R = rows.length, M = res.scenario.M, last = upto === undefined ? R : Math.max(0, Math.min(R, upto));
    var LBL = 70, cw = Math.min(40, Math.floor((W - LBL - 8) / R)), ch = 24, H = (M + 2) * ch + 30;
    var s = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-labelledby="st-t st-d"><title id="st-t">Worker status per round</title>';
    s += '<desc id="st-d">Rows are workers, columns are rounds. A means admitted, L late, R admitted because the fair policy required it, a dash absent. ' +
      "The same information is in the per-round table below.</desc>";
    var r, w;
    for (r = 0; r < R; r++) {
      s += '<text x="' + (LBL + r * cw + cw / 2) + '" y="14" text-anchor="middle" class="muted">' + (r + 1) + "</text>";
    }
    for (w = 0; w < M; w++) {
      s += '<text x="' + (LBL - 8) + '" y="' + (30 + w * ch + 16) + '" text-anchor="end">worker ' + w + "</text>";
      for (r = 0; r < last; r++) {
        var row = rows[r], x = LBL + r * cw, y = 24 + w * ch;
        var st = row.alive.indexOf(w) < 0 ? "-" : row.required.indexOf(w) >= 0 ? "R" : row.admitted.indexOf(w) >= 0 ? "A" : "L";
        var style = st === "A" ? "fill:var(--c1)" : st === "R" ? "fill:var(--c1);stroke:var(--focus);stroke-width:3" : st === "L" ? "fill:none;stroke:var(--bad);stroke-width:2;stroke-dasharray:4 2" : "fill:none;stroke:var(--border)";
        s += '<rect x="' + (x + 1) + '" y="' + (y + 1) + '" width="' + (cw - 2) + '" height="' + (ch - 2) + '" rx="3" style="' + style + '"/>';
        s += '<text x="' + (x + cw / 2) + '" y="' + (y + 16) + '" text-anchor="middle" style="' + (st === "A" || st === "R" ? "fill:var(--accent-ink)" : "") + '">' + (st === "-" ? "–" : st) + "</text>";
      }
    }
    var yb = 24 + M * ch + 6;
    s += '<text x="' + (LBL - 8) + '" y="' + (yb + 14) + '" text-anchor="end">carried in</text><text x="' + (LBL - 8) + '" y="' + (yb + 14 + ch) + '" text-anchor="end">round time</text>';
    for (r = 0; r < last; r++) {
      s += '<text x="' + (LBL + r * cw + cw / 2) + '" y="' + (yb + 14) + '" text-anchor="middle">' + rows[r].carried + "</text>";
      s += '<text x="' + (LBL + r * cw + cw / 2) + '" y="' + (yb + 14 + ch) + '" text-anchor="middle">' + rows[r].round_time.toFixed(1) + "</text>";
    }
    return s + "</svg>";
  }

  function stripLegendHtml() {
    return '<p class="hint">A = admitted, L = late (dashed), R = admitted because the fair policy required it (thick ring), – = absent (churn). ' +
      "Bottom rows: delayed deltas merged this round (deferred policy) and the simulated round time.</p>";
  }

  function roundsTableHtml(res) {
    var h = '<table><caption>Per-round data</caption><thead><tr><th scope="col" class="num">Round</th><th scope="col" class="num">Global loss</th><th scope="col">Admitted</th><th scope="col">Late</th>' +
      '<th scope="col">Required (fair)</th><th scope="col" class="num">Carried in</th><th scope="col" class="num">Dropped stale</th><th scope="col" class="num">Round time</th></tr></thead><tbody>';
    res.rounds.forEach(function (r) {
      h += '<tr><td class="num">' + r.round + '</td><td class="num">' + fmt(r.loss, 4) + "</td><td>" + (r.admitted.join(", ") || "-") + "</td><td>" + (r.late.join(", ") || "-") +
        "</td><td>" + (r.required.join(", ") || "-") + '</td><td class="num">' + r.carried + '</td><td class="num">' + r.dropped_stale + '</td><td class="num">' + r.round_time.toFixed(2) + "</td></tr>";
    });
    return h + "</tbody></table>";
  }

  /** The three policies for the current setting (same scenario, only the policy changes). */
  function compareTableHtml(rows, currentPolicy) {
    var h = '<table><caption>The three late-delta policies for this setting (straggle mode: ' + esc(rows.length ? rows[0].straggle : "") + ')</caption><thead><tr><th scope="col">Policy</th>' +
      '<th scope="col" class="num">Retention</th><th scope="col" class="num">Final loss</th><th scope="col" class="num">Simulated total time</th><th scope="col" class="num">Late deltas</th><th scope="col" class="num">Carried deltas</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      h += '<tr' + (r.policy === currentPolicy ? ' class="current" aria-current="true"' : "") + "><th scope=\"row\">" + esc(r.policy) + (r.policy === currentPolicy ? " (selected)" : "") +
        '</th><td class="num">' + fmt(r.retention, 3) + '</td><td class="num">' + fmt(r.final_loss, 4) + '</td><td class="num">' + fmt(r.total_time, 2) + '</td><td class="num">' + r.late_total +
        '</td><td class="num">' + r.carried_total + "</td></tr>";
    });
    return h + "</tbody></table>";
  }

  /** Synchronous reference scenario for a given scenario (same as the Python simulate()). */
  function referenceScenario(sc) {
    var ref = {};
    Object.keys(sc).forEach(function (k) { ref[k] = sc[k]; });
    ref.K = sc.M; ref.straggle = "none"; ref.policy = "discard"; ref.churn_start = 0; ref.churn_end = 0;
    return ref;
  }

  // ---------------------------------------------------------------- results panel (fed ONLY by data.js -> window.HM_DATA)
  function statusClass(s) {
    var w = String(s || "").toLowerCase().split(/[\s(]/)[0];
    return ["supported", "refuted", "inconclusive", "descriptive", "invalid"].indexOf(w) >= 0 ? w : "";
  }
  function statusHtml(s) {
    return s === undefined || s === null ? '<span class="na">n/a</span>' : '<span class="status ' + statusClass(s) + '">' + esc(s) + "</span>";
  }
  function tagHtml(block) {
    var srcs = [].concat(block.source || []), rec = [].concat(block.receipt12 === undefined ? [] : block.receipt12);
    var out = "";
    srcs.forEach(function (s, i) {
      var r = rec[i];
      out += '<span class="tag">source: ' + esc(s) + "</span>" + (r ? '<span class="tag">receipt ' + esc(r) + "</span>" : "");
    });
    if (block.file_sha256_12) out += '<span class="tag">file sha256 ' + esc(block.file_sha256_12) + "</span>";
    if (block.generated_utc) out += '<span class="tag">data built ' + esc(block.generated_utc) + "</span>";
    return out;
  }
  function unavailableHtml(title, block, extra) {
    return '<article class="panel"><h3>' + esc(title) + '</h3><p class="na">No data available' + (block && block.reason ? ": " + esc(block.reason) : "") + ".</p>" +
      (block && block.source ? '<p><span class="tag">expected source: ' + esc([].concat(block.source).join(", ")) + "</span></p>" : "") + (extra || "") + "</article>";
  }
  function notShow(text) {
    return '<p class="notshow"><strong>What this does not show.</strong> ' + esc(text) + "</p>";
  }
  function preregHtml(pre, label) {
    if (!pre || !pre.available) return "";
    var items = pre.hypotheses_and_rules || pre.hypotheses || {};
    var h = '<details><summary>Registered before the run: ' + esc(label) + " (" + esc(pre.registered_utc || "date n/a") + ')</summary><ul>';
    Object.keys(items).forEach(function (k) { h += "<li><strong>" + esc(k) + ".</strong> " + esc(items[k]) + "</li>"; });
    return h + '</ul><p><span class="tag">source: ' + esc(pre.source) + '</span><span class="tag">file sha256 ' + esc(pre.file_sha256_12) + "</span></p></details>";
  }

  function e13RegisteredHtml(b) {
    if (!b || !b.available) return unavailableHtml("E13 registered result", b);
    var h = '<article class="panel"><h3>E13: a K=2-of-4 swarm with a persistent straggler (registered predictions)</h3><p>' + tagHtml(b) + "</p>";
    h += '<div class="tablewrap"><table><caption>Registered predictions per model (status strings copied verbatim; numbers rounded to 3 decimals for display)</caption><thead><tr><th scope="col">Model</th>' +
      '<th scope="col">P13a (S1 equals S0)</th><th scope="col">P13b (retention of the K=2 swarm, bar 0.5)</th><th scope="col">P13c (upload bytes vs raw fp32, bar 0.30)</th><th scope="col">Controls</th></tr></thead><tbody>';
    Object.keys(b.models || {}).forEach(function (name) {
      var m = b.models[name], p13b = m.P13b || {}, p13c = m.P13c || {};
      var pb = statusHtml(p13b.status);
      if (p13b.mean !== undefined) pb += " mean " + pm(p13b.mean, p13b.half_width) + (p13b.interval ? " (interval [" + fmt(p13b.interval[0]) + ", " + fmt(p13b.interval[1]) + "])" : "");
      else if (p13b.per_seed) pb += " per seed: " + Object.keys(p13b.per_seed).map(function (k) { return fmt(p13b.per_seed[k]); }).join(", ");
      var pc = statusHtml(p13c.status) + (p13c.upload_ratio_actual !== undefined ? " " + fmt(p13c.upload_ratio_actual, 4) + " of raw" : "");
      h += "<tr><th scope=\"row\">" + esc(name) + "</th><td>" + statusHtml((m.P13a || {}).status) + "</td><td>" + pb + "</td><td>" + pc + "</td><td>" +
        (m.valid === false ? "failed: " + esc((m.controls_failed || []).join(", ")) : m.valid ? "valid" : "n/a") + "</td></tr>";
    });
    h += "</tbody></table></div>";
    var q = (b.models || {})["qwen0.5b"];
    if (q && q.P13d && Object.keys(q.P13d).length) {
      h += '<div class="tablewrap"><table><caption>qwen0.5b ablations (descriptive, no status): each swarm mechanism switched off</caption><thead><tr><th scope="col">Cell</th><th scope="col">Name</th>' +
        '<th scope="col" class="num">Mean retention</th><th scope="col" class="num">Admitted fraction</th></tr></thead><tbody>';
      Object.keys(q.P13d).forEach(function (c) {
        var x = q.P13d[c];
        h += "<tr><th scope=\"row\">" + esc(c) + "</th><td>" + esc(x.name) + '</td><td class="num">' + fmt(x.mean_retention) + '</td><td class="num">' + fmt(x.admitted_fraction_mean) + "</td></tr>";
      });
      h += "</tbody></table></div>";
    }
    h += notShow("Everything here ran on one time-shared GPU with peers on localhost (separate OS processes where the memory preflight allowed it, threads otherwise; see the hosting fields of each report): it says nothing about wide-area networks, volunteers or churn in the wild. " +
      "Three seeds per model (fewer for granite3b), so the intervals are wide. The toy simulator above does not reproduce these numbers.");
    return h + (b.rules ? '<p class="hint">Analysis rules: ' + esc(b.rules) + "</p>" : "") + "</article>";
  }

  function e13RotatingHtml(b) {
    if (!b || !b.available) return unavailableHtml("E13 rotating-straggler diagnostic", b);
    var h = '<article class="panel"><h3>E13 diagnostic: rotating straggler (cells S0 and S2 only)</h3><p>' + tagHtml(b) + "</p><p>Status: " + statusHtml(b.status) + " <span class=\"hint\">(wording from " + esc(b.status_source || "n/a") + "; " + esc(b.status_note || "") + ")</span></p>";
    Object.keys(b.models || {}).forEach(function (name) {
      var m = b.models[name], p = m.paired_S2_minus_S0 || {}, rd = m.retention_S2_derived, s2 = ((m.cells || {}).S2) || {};
      h += '<div class="tablewrap"><table><caption>' + esc(name) + "</caption><tbody>";
      h += "<tr><th scope=\"row\">final loss, S2 minus S0, mean ± 95% interval</th><td>" + (p.mean !== undefined ? pm(p.mean, p.ci95) + " (" + esc(p.direction) + ")" : statusHtml(p.status)) + "</td></tr>";
      if (rd) h += "<tr><th scope=\"row\">retention of S2 (derived by the builder with the registered formula)</th><td>" + fmt(rd.mean) + " <span class=\"tag\">derived</span> per seed: " + Object.keys(rd.per_seed).map(function (k) { return fmt(rd.per_seed[k]); }).join(", ") + "</td></tr>";
      h += "<tr><th scope=\"row\">admitted fraction (S2)</th><td>" + fmt(s2.admitted_fraction_mean) + "</td></tr>";
      h += "<tr><th scope=\"row\">report flag controls_all_passed</th><td>" + esc(b.controls_all_passed) + (m.controls_not_evaluated && m.controls_not_evaluated.length ? " (controls not evaluated in this job: " + esc(m.controls_not_evaluated.join(", ")) + ")" : "") + "</td></tr>";
      h += "</tbody></table></div>";
    });
    return h + notShow("A two-cell diagnostic without a registered prediction: one model, three seeds, a single straggle schedule. It is not a verdict on rotating stragglers in general.") + "</article>";
  }

  function e13M2Html(b) {
    if (!b || !b.available) return unavailableHtml("E13 M=2 process identity", b);
    var h = '<article class="panel"><h3>E13 control: two real worker processes reproduce the synchronous baseline (M=2)</h3><p>' + tagHtml(b) + "</p><p>Status: " + statusHtml(b.status) + ' <span class="hint">(wording from ' + esc(b.status_source || "n/a") + "; " + esc(b.status_note || "") + ")</span></p>";
    Object.keys(b.models || {}).forEach(function (name) {
      var id = (b.models[name] || {}).s1_equals_s0 || {}, row = (id.rows || [])[0] || {};
      h += '<div class="tablewrap"><table><caption>' + esc(name) + '</caption><tbody><tr><th scope="row">control s1_equals_s0 (passed)</th><td>' + esc(id.passed) + "</td></tr>" +
        '<tr><th scope="row">max absolute difference of the final loss</th><td>' + esc(row.max_abs_diff) + " (tolerance " + esc(id.tolerance) + ")</td></tr>" +
        '<tr><th scope="row">bit identical</th><td>' + esc(row.bit_identical) + "</td></tr></tbody></table></div>";
    });
    return h + notShow("One seed, one model, two workers: it shows the multi-process plumbing does not change the result for M=2, not that it does for more workers, other seeds or other models.") + "</article>";
  }

  function e14Html(b) {
    var title = "E14: fair quorum (S6) and deferred carry (S7) against the plain K=2 swarm (S2)";
    if (!b || !b.available) {
      return unavailableHtml(title, b, preregHtml((b || {}).prereg, "E14")) + "";
    }
    var h = '<article class="panel"><h3>' + esc(title) + "</h3><p>" + tagHtml(b) + "</p>";
    Object.keys(b.models || {}).forEach(function (name) {
      var m = b.models[name], e = m.e14 || {}, cells = m.cells || {};
      h += '<div class="tablewrap"><table><caption>' + esc(name) + " (exploratory; numbers rounded to 3 decimals)</caption><thead><tr><th scope=\"col\">Cell</th><th scope=\"col\" class=\"num\">Retention (mean ± 95% interval)</th>" +
        '<th scope="col">Hypothesis (S2 minus cell, final loss)</th><th scope="col" class="num">Round latency p50 (s, descriptive)</th><th scope="col" class="num">Late discards</th></tr></thead><tbody>';
      ["S2", "S6", "S7"].forEach(function (c) {
        var ret = ((e.retention_vs_S0 || {})[c]) || {}, hyp = null;
        Object.keys(e.hypotheses || {}).forEach(function (k) { if (e.hypotheses[k].cell === c) hyp = e.hypotheses[k]; });
        var cell = cells[c] || {}, lat = cell.round_latency_s || {};
        h += "<tr><th scope=\"row\">" + c + " " + esc(cell.name || "") + '</th><td class="num">' + (ret.mean !== undefined ? pm(ret.mean, ret.ci95) : "n/a") + "</td><td>" +
          (hyp ? Object.keys(e.hypotheses).filter(function (k) { return e.hypotheses[k] === hyp; }).join("") + ": " + (hyp.mean !== undefined ? "mean " + pm(hyp.mean, hyp.ci95) + ", report direction: " + statusHtml(hyp.direction) : statusHtml(hyp.status)) +
            (hyp.hypothesis_status ? ' <span class="tag" title="' + esc(hyp.hypothesis_status_rule || "") + '">derived by the builder, registered rule: ' + esc(hyp.hypothesis_status) + "</span>" : "") : '<span class="na">reference</span>') +
          '</td><td class="num">' + fmt(lat.p50, 1) + '</td><td class="num">' + (cell.late_discards_total === undefined ? "n/a" : cell.late_discards_total) + "</td></tr>";
      });
      h += "</tbody></table></div><p class=\"hint\">Report flag controls_all_passed for this model: " + esc(m.controls_all_passed) +
        (m.controls ? " (controls not evaluated in this job: " + esc(Object.keys(m.controls).filter(function (k) { return m.controls[k] === null; }).join(", ") || "none") + ")" : "") + ". " + esc(e.note || "") + "</p>";
    });
    h += notShow("An exploratory follow-up written after the E13 result (not a confirmatory test): one or two models, three seeds, fixed constants not tuned, one time-shared GPU. Waiting for the excluded worker (S6) costs wall time; the latency column is descriptive only.");
    return h + preregHtml(b.prereg, "E14") + "</article>";
  }

  function e15Html(b) {
    var title = "E15: GWAYA-gated fine-tuning of a small coding assistant";
    if (!b || !b.available) return unavailableHtml(title, b, preregHtml((b || {}).prereg, "E15"));
    var h = '<article class="panel"><h3>' + esc(title) + "</h3><p>" + tagHtml(b) + "</p>";
    h += "<p>Run status (the runner's own word): " + statusHtml(b.run_status) + (b.smoke ? ' <strong>SMOKE RUN with a tiny random model: not a result.</strong>' : "") + " Report flag controls_all_passed: " + esc((b.report_excerpt || {}).controls_all_passed) + "</p>";
    h += '<div class="tablewrap"><table><caption>Conditions (pass@1, Wilson 95% interval)</caption><thead><tr><th scope="col">Condition</th><th scope="col" class="num">n</th><th scope="col" class="num">passes</th><th scope="col" class="num">pass@1</th><th scope="col">interval</th></tr></thead><tbody>';
    Object.keys(b.conditions || {}).forEach(function (k) {
      var c = b.conditions[k];
      h += "<tr><th scope=\"row\">" + esc(k) + '</th><td class="num">' + esc(c.n) + '</td><td class="num">' + esc(c.passes) + '</td><td class="num">' + fmt(c.pass_at_1) + "</td><td>" + (c.wilson95 ? "[" + fmt(c.wilson95[0]) + ", " + fmt(c.wilson95[1]) + "]" : "n/a") + "</td></tr>";
    });
    h += '</tbody></table></div><div class="tablewrap"><table><caption>Paired comparisons (B minus A, exact McNemar p)</caption><thead><tr><th scope="col">Comparison</th><th scope="col" class="num">n</th><th scope="col" class="num">difference</th><th scope="col" class="num">B only / A only</th><th scope="col" class="num">McNemar p</th></tr></thead><tbody>';
    Object.keys(b.paired_comparisons || {}).forEach(function (k) {
      var c = b.paired_comparisons[k];
      h += "<tr><th scope=\"row\">" + esc(k) + '</th><td class="num">' + esc(c.n) + '</td><td class="num">' + fmt(c.diff) + '</td><td class="num">' + esc(c.b_only) + " / " + esc(c.a_only) + '</td><td class="num">' + fmt(c.mcnemar_p, 4) + "</td></tr>";
    });
    h += "</tbody></table></div><p class=\"hint\">" + esc(b.verdicts_note || "") + ".</p>";
    h += notShow("The report states numbers only. No verdict on H15a to H15e is shown here; the registered rules (thresholds, McNemar) decide them, and a single model, one benchmark split and one run cannot speak for coding assistants in general.");
    return h + preregHtml(b.prereg, "E15") + "</article>";
  }

  /** Whole "What the real experiments measured" panel from a HM_DATA object (or a placeholder when there is none). */
  function resultsHtml(data) {
    if (!data || typeof data !== "object") {
      return '<p class="na">data.js did not load, so no measured results can be shown. Regenerate it with <code>python3 scripts/build_site_data.py</code>.</p>';
    }
    var e13 = data.e13 || {};
    return '<p class="hint">Data file built ' + esc(data.generated_utc || "n/a") + " by scripts/build_site_data.py from artifacts/experiments/*.json. Values and status strings are copied from those files; " +
      'a block with no source file says so instead of showing a number.</p><h3>E13: the real swarm (models qwen0.5b, granite1b, granite3b as named in the data; LoRA; 4 workers, one data domain each)</h3>' +
      e13RegisteredHtml(e13.registered) + e13RotatingHtml(e13.rotating_straggler) + e13M2Html(e13.m2_process_identity) +
      "<h3>E14: admission policies that address the straggler</h3>" + e14Html(data.e14) +
      "<h3>E15: a GWAYA-gated coding assistant</h3>" + e15Html(data.e15);
  }

  // ---------------------------------------------------------------- DOM wiring
  function applyTheme(theme) {
    var el = root.document.documentElement;
    if (theme === "light" || theme === "dark") el.setAttribute("data-theme", theme); else el.removeAttribute("data-theme");
    var btn = root.document.getElementById("theme-toggle");
    if (btn) {
      var dark = theme === "dark" || (theme !== "light" && root.matchMedia && root.matchMedia("(prefers-color-scheme: dark)").matches);
      btn.setAttribute("aria-pressed", dark ? "true" : "false");
      btn.textContent = dark ? "Dark theme: on" : "Dark theme: off";
    }
  }
  function initTheme() {
    var doc = root.document, saved = storeGet("hm-theme");
    applyTheme(saved === "light" || saved === "dark" ? saved : null);
    var btn = doc.getElementById("theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var dark = btn.getAttribute("aria-pressed") === "true";
      var next = dark ? "light" : "dark";
      storeSet("hm-theme", next);
      applyTheme(next);
    });
    if (root.matchMedia) {
      try { root.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { if (!storeGet("hm-theme")) applyTheme(null); }); } catch (e) { /* old browser */ }
    }
  }

  var LAB_FIELDS = ["M", "K", "policy", "straggle", "straggle_every", "churn_start", "churn_end", "merge", "rounds", "seed", "grace"];
  var FIELD_ID = { M: "lab-M", K: "lab-K", policy: "lab-policy", straggle: "lab-straggle", straggle_every: "lab-every", churn_start: "lab-churn-start", churn_end: "lab-churn-end", merge: "lab-merge", rounds: "lab-rounds", seed: "lab-seed", grace: "lab-grace" };

  function initLab() {
    var doc = root.document;
    var form = doc.getElementById("lab-form");
    if (!form || !Lab) return;
    function el(id) { return doc.getElementById(id); }
    function setScenario(sc) {
      LAB_FIELDS.forEach(function (f) { el(FIELD_ID[f]).value = String(sc[f]); });
      fillK(sc.M, sc.K);
    }
    function fillK(M, K) {
      var sel = el("lab-K"), html = "";
      for (var k = 1; k <= M; k++) html += '<option value="' + k + '">' + k + (k === M ? " (all workers: synchronous)" : "") + "</option>";
      sel.innerHTML = html;
      sel.value = String(Math.min(K, M));
    }
    function readScenario() {
      var sc = {};
      LAB_FIELDS.forEach(function (f) {
        var v = el(FIELD_ID[f]).value;
        sc[f] = (f === "policy" || f === "straggle" || f === "merge") ? v : Number(v);
      });
      return sc;
    }
    function show(sc) {
      var err = el("lab-error");
      var res, ref;
      try {
        sc = Lab.makeScenario(sc);
        res = Lab.simulate(sc);
        ref = Lab.simulate(referenceScenario(sc));
      } catch (e) {
        err.textContent = "Cannot run this setting: " + e.message;
        err.hidden = false;
        return;
      }
      err.hidden = true;
      el("lab-chart").innerHTML = lossChartSvg(res, ref);
      el("lab-legend").innerHTML = lossLegendHtml();
      el("lab-domains").innerHTML = domainBarsSvg(res, ref);
      el("lab-domain-legend").innerHTML = domainLegendHtml();
      el("lab-strip").innerHTML = stripSvg(res) + stripLegendHtml();
      el("lab-rounds-table").innerHTML = roundsTableHtml(res);
      el("lab-compare").innerHTML = compareTableHtml(Lab.comparePolicies(sc, Lab.POLICIES, [sc.straggle]), sc.policy);
      el("stat-retention").textContent = fmt(res.retention, 3);
      el("stat-time").textContent = fmt(res.total_time, 2);
      el("stat-loss").textContent = fmt(res.initial_loss, 3) + " → " + fmt(res.final_loss, 3);
      el("stat-late").textContent = String(res.rounds.reduce(function (a, r) { return a + r.late.length; }, 0));
      el("lab-cli").textContent = cliCommand(sc);
      el("lab-json").textContent = JSON.stringify(sc, null, 1);
      el("lab-digest").textContent = res.digest;
      el("lab-summary").textContent = "Retention " + fmt(res.retention, 3) + " of the synchronous improvement; final loss " + fmt(res.final_loss, 3) + " against " + fmt(ref.final_loss, 3) +
        " for the synchronous reference; simulated total time " + fmt(res.total_time, 2) + ".";
      storeSet("hm-lab-scenario", JSON.stringify(sc));
    }
    var presetSel = el("lab-preset");
    var opts = '<option value="custom">custom</option>';
    Object.keys(Lab.PRESETS).forEach(function (n) { opts += '<option value="' + esc(n) + '">' + esc(n) + "</option>"; });
    presetSel.innerHTML = opts;

    var start = Lab.presetScenario("persistent-straggler");
    var saved = storeGet("hm-lab-scenario");
    if (saved) { try { start = Lab.makeScenario(JSON.parse(saved)); presetSel.value = "custom"; } catch (e) { /* ignore a stale or hand-edited value */ } }
    else presetSel.value = "persistent-straggler";
    setScenario(start);
    show(start);

    presetSel.addEventListener("change", function () {
      if (presetSel.value === "custom") return;
      var sc = Lab.presetScenario(presetSel.value, { rounds: Number(el("lab-rounds").value) || 12, seed: Number(el("lab-seed").value) || 1 });
      setScenario(sc);
      show(sc);
    });
    form.addEventListener("change", function (ev) {
      if (ev.target === presetSel) return;
      presetSel.value = "custom";
      if (ev.target.id === "lab-M") fillK(Number(el("lab-M").value), Number(el("lab-K").value));
      show(readScenario());
    });
    form.addEventListener("submit", function (ev) { ev.preventDefault(); show(readScenario()); });
    el("lab-random-seed").addEventListener("click", function () {
      el("lab-seed").value = String(1 + Math.floor(Math.random() * 99999));
      presetSel.value = "custom";
      show(readScenario());
    });
  }

  function initResults() {
    var c = root.document.getElementById("real-results");
    if (c) c.innerHTML = resultsHtml(root.HM_DATA);
  }

  function init() {
    initTheme();
    initLab();
    initResults();
  }

  var api = {
    esc: esc, fmt: fmt, cliCommand: cliCommand, lossChartSvg: lossChartSvg, lossLegendHtml: lossLegendHtml, domainBarsSvg: domainBarsSvg, domainLegendHtml: domainLegendHtml,
    stripSvg: stripSvg, stripLegendHtml: stripLegendHtml, roundsTableHtml: roundsTableHtml, compareTableHtml: compareTableHtml, referenceScenario: referenceScenario,
    resultsHtml: resultsHtml, storeGet: storeGet, storeSet: storeSet, prefersReducedMotion: prefersReducedMotion, applyTheme: applyTheme, init: init
  };
  root.SwarmSite = api;
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root.document && typeof root.document.addEventListener === "function") {
    if (root.document.readyState === "loading") root.document.addEventListener("DOMContentLoaded", init); else init();
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
