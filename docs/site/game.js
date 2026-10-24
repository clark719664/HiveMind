/*
 * Quorum Quest: an educational game on the shared Swarm Lab engine (swarm_lab.js). You are the swarm coordinator.
 *
 * - GameLogic (levels, scoring, star thresholds, debrief) is pure and has no DOM dependency; it is exposed as `GameLogic`
 *   (browser global and node `require`) so that it can be checked under node.
 * - The learning score is a teaching number. It is NOT PTS or $HIVE and has no monetary or reward value of any kind.
 * - No network, no accounts, no multiplayer. Best scores are kept in localStorage (inside try/catch) when the browser allows it.
 *
 * SCORING RULE
 *   learning score = round(100 * clamp(retention, 0, 1)) - time penalty, never below 0.
 *   time penalty   = 0 on levels without a time budget; on the budget level, PENALTY_PER_UNIT points for every simulated time unit
 *                    above the budget (budget = BUDGET_PER_ROUND * rounds).
 * STAR RULE (computed from the engine for the level's scenario and seed)
 *   Every strategy the cards allow (K x policy x grace x merge) is simulated on THRESHOLD_SEEDS consecutive seeds starting at the level's seed;
 *   best = the highest mean learning score among the strategies (a mean, so that one lucky seed of an erratic strategy cannot set the bar).
 *   3 stars: your score >= 95% of best.  2 stars: >= 80% of best.  1 star: >= 50% of best.  0 stars below that.
 */
(function (root) {
  "use strict";

  var Lab = root.SwarmLab || (typeof require === "function" ? require("./swarm_lab.js") : null);
  var Site = root.SwarmSite || (typeof require === "function" ? require("./app.js") : null);

  var GRACES = [0, 0.1, 0.5];
  var MERGES = ["direct", "rda"];
  var STAR_RULE = [[3, 0.95], [2, 0.8], [1, 0.5]];
  var BUDGET_PER_ROUND = 1.5;
  var PENALTY_PER_UNIT = 5;
  var THRESHOLD_SEEDS = 3;

  var LEVELS = [
    {
      id: 1, title: "IID data: any quorum works", seed: 1, rounds: 12, maxK: 3,
      scenario: { M: 4, spread: 0.0, offset: 3.0, straggle: "persistent" },
      story: "All four workers hold statistically identical data, and one of them is always slow. Every worker's delta points the same way, so leaving the slow one out costs nothing.",
      goal: "Find out how little you need to wait for. Then try the fair policy and see what waiting costs for no benefit.",
      lesson: "With IID data a missing delta is replaceable, so a small quorum is fine. This is why quorum rules are harmless in many classic setups and dangerous only when the data differ.",
      real: { label: "E13 control: the synchronous baseline (S0)", href: "index.html#real" }
    },
    {
      id: 2, title: "Non-IID data and a persistent straggler", seed: 1, rounds: 12, maxK: 3,
      scenario: { M: 4, straggle: "persistent" },
      story: "Each worker owns a different data domain, and the last worker is slow in every round. If you never wait for it, nobody ever teaches the swarm its domain.",
      goal: "Keep as much of the synchronous improvement as you can. Compare discarding, waiting (fair) and carrying late deltas over (deferred).",
      lesson: "A quorum smaller than the swarm silently drops one domain every round when the same worker is always last. This is the E13 effect: the swarm kept almost none of the synchronous improvement.",
      real: { label: "E13: K=2-of-4 with a persistent straggler", href: "index.html#real" }
    },
    {
      id: 3, title: "A rotating straggler", seed: 1, rounds: 12, maxK: 3,
      scenario: { M: 4, straggle: "rotating" },
      story: "A different worker (never worker 0) is slow each round. Nobody is starved for long, but every round still loses a delta.",
      goal: "See which policies still matter when the late worker keeps changing.",
      lesson: "A rotating straggler is much less harmful than a persistent one. The fair policy cannot help much, because the worker it waits for next round is a different one; carrying late deltas over can recover some of what each round lost.",
      real: { label: "E13 diagnostic: rotating straggler", href: "index.html#real" }
    },
    {
      id: 4, title: "A worker churns out and back", seed: 1, rounds: 12, maxK: 3,
      scenario: { M: 4, straggle: "persistent", churn_start: 3, churn_end: 6 },
      story: "Worker 1 disappears after round 2 and returns for round 7, while the last worker is still slow. The coordinator must not wait for someone who is not there.",
      goal: "Keep the swarm learning through the outage.",
      lesson: "A fair policy must never wait for a worker that is absent, otherwise one departure stalls the round. Absent workers contribute nothing, so their domain pauses until they return.",
      real: { label: "E13 churn schedule (a worker killed and re-spawned)", href: "index.html#real" }
    },
    {
      id: 5, title: "A time budget", seed: 1, rounds: 12,
      scenario: { M: 4, straggle: "persistent" },
      budgetPerRound: BUDGET_PER_ROUND,
      story: "Same persistent straggler, but now the clock matters: the run has a budget of " + BUDGET_PER_ROUND + " simulated time units per round, and every unit over the budget costs " + PENALTY_PER_UNIT + " points.",
      goal: "Keep the learning without paying for too much waiting.",
      lesson: "Fairness is not free: waiting for the slow worker every other round adds simulated time, as the fair-quorum cell of E14 added round latency. In this toy, carrying late deltas over keeps most of the learning at almost no time cost; in the real E14 runs the two mechanisms gave different retention, so do not read this level as a prediction.",
      real: { label: "E14: fair quorum (S6) and deferred carry (S7)", href: "index.html#real" }
    }
  ];

  var POLICY_CARDS = [
    { value: "discard", title: "Discard", text: "Drop late deltas. Fast, but the late worker's work is lost." },
    { value: "fair", title: "Fair", text: "A worker excluded last round must be waited for now. Costs time." },
    { value: "deferred", title: "Deferred", text: "Keep late deltas and merge them next round, down-weighted." }
  ];
  var GRACE_CARDS = [
    { value: 0, title: "No grace", text: "Close the round right at the K-th arrival." },
    { value: 0.1, title: "Short grace", text: "Wait 10% longer than the K-th arrival." },
    { value: 0.5, title: "Long grace", text: "Wait 50% longer than the K-th arrival." }
  ];
  var MERGE_CARDS = [
    { value: "direct", title: "Direct mean", text: "Average the deltas you have." },
    { value: "rda", title: "RDA merge", text: "Average, then rescale by sum|D| / |sum D|. Can be erratic in this toy." }
  ];
  var K_TEXT = { 1: "first arrival only", 2: "half the swarm", 3: "all but one", 4: "everyone: fully synchronous" };

  // ---------------------------------------------------------------- scoring
  function levelScenario(level, choice, seed) {
    var sc = {};
    Object.keys(level.scenario).forEach(function (k) { sc[k] = level.scenario[k]; });
    sc.rounds = level.rounds;
    sc.seed = seed === undefined ? level.seed : seed;
    sc.K = choice.K; sc.policy = choice.policy; sc.grace = choice.grace; sc.merge = choice.merge;
    return Lab.makeScenario(sc);
  }

  function timeBudget(level) {
    return level.budgetPerRound ? level.budgetPerRound * level.rounds : null;
  }

  /** Learning score (integer, 0..100) and its parts for one finished run. */
  function scoreRun(level, res) {
    var rho = res.retention;
    var base = rho === null ? 0 : Math.round(100 * Math.min(1, Math.max(0, rho)));
    var budget = timeBudget(level);
    var penalty = budget === null ? 0 : Math.round(PENALTY_PER_UNIT * Math.max(0, res.total_time - budget));
    return { retention: rho, base: base, penalty: penalty, budget: budget, total_time: res.total_time, score: Math.max(0, base - penalty) };
  }

  /** Every strategy the cards allow: K (up to the level's maxK) x policy x grace x merge. */
  function allChoices(level) {
    var out = [], maxK = level.maxK || level.scenario.M;
    for (var k = 1; k <= maxK; k++) {
      Lab.POLICIES.forEach(function (policy) {
        GRACES.forEach(function (grace) {
          MERGES.forEach(function (merge) { out.push({ K: k, policy: policy, grace: grace, merge: merge }); });
        });
      });
    }
    return out;
  }

  var thresholdCache = {};

  /** Star thresholds for a level and seed, computed by simulating every strategy the cards allow (see the STAR RULE above). */
  function thresholds(level, seed) {
    var s = seed === undefined ? level.seed : seed, key = level.id + ":" + s;
    if (thresholdCache[key]) return thresholdCache[key];
    var best = -1, bestChoice = null, n = 0, worst = Infinity;
    allChoices(level).forEach(function (choice) {
      var total = 0;
      for (var j = 0; j < THRESHOLD_SEEDS; j++) total += score(level, choice, s + j).score;
      var mean = total / THRESHOLD_SEEDS;
      n += 1;
      if (mean > best) { best = mean; bestChoice = choice; }
      if (mean < worst) worst = mean;
    });
    best = Math.round(best * 10) / 10;
    var t = { best: best, worst: Math.round(worst * 10) / 10, bestChoice: bestChoice, strategies: n, seed: s, seeds: THRESHOLD_SEEDS, stars: {} };
    STAR_RULE.forEach(function (r) { t.stars[r[0]] = Math.ceil(r[1] * best - 1e-9); });
    thresholdCache[key] = t;
    return t;
  }

  function score(level, choice, seed) {
    return scoreRun(level, Lab.simulate(levelScenario(level, choice, seed)));
  }

  function starsFor(level, scoreValue, seed) {
    var t = thresholds(level, seed);
    for (var i = 0; i < STAR_RULE.length; i++) {
      if (scoreValue >= t.stars[STAR_RULE[i][0]]) return STAR_RULE[i][0];
    }
    return 0;
  }

  function starRuleText(level, seed) {
    var t = thresholds(level, seed);
    return "Stars are computed from the engine: all " + t.strategies + " strategies on the cards were simulated on seeds " + t.seed + " to " + (t.seed + t.seeds - 1) + "; the best one has a mean score of " + t.best +
      ". 3 stars need at least 95% of that (" + t.stars[3] + "), 2 stars 80% (" + t.stars[2] + "), 1 star 50% (" + t.stars[1] + ").";
  }

  // ---------------------------------------------------------------- debrief (generated from the engine's per-round data)
  function rangeText(rounds) {
    return rounds.length === 0 ? "" : rounds.length === 1 ? "round " + rounds[0] : "rounds " + rounds[0] + " to " + rounds[rounds.length - 1];
  }

  function describeChoice(c) {
    return "K=" + c.K + ", " + c.policy + ", grace " + c.grace + ", " + c.merge + " merge";
  }

  /** Short plain-language lines explaining what happened in a run. */
  function debrief(level, choice, res, ref, sc) {
    var lines = [];
    var M = res.scenario.M, rows = res.rounds, R = rows.length;
    lines.push("Your strategy (" + describeChoice(choice) + ") kept " + (sc.retention === null ? "n/a" : Math.round(100 * sc.retention) + "%") + " of the improvement a fully synchronous swarm makes. Global loss went from " +
      res.initial_loss.toFixed(2) + " to " + res.final_loss.toFixed(2) + " (synchronous: " + ref.final_loss.toFixed(2) + ").");
    // who was left out
    var late = [], present = [], w, r;
    for (w = 0; w < M; w++) { late.push(0); present.push(0); }
    rows.forEach(function (row) {
      row.alive.forEach(function (i) { present[i] += 1; });
      row.late.forEach(function (i) { late[i] += 1; });
    });
    var gaps = res.final_domain_losses.map(function (v, i) { return v - ref.final_domain_losses[i]; });
    var worst = 0;
    for (w = 1; w < M; w++) if (gaps[w] > gaps[worst]) worst = w;
    var totalLate = late.reduce(function (a, b) { return a + b; }, 0);
    var mostLate = 0;
    for (w = 1; w < M; w++) if (late[w] > late[mostLate]) mostLate = w;
    if (totalLate === 0) {
      lines.push("Nobody was ever late under your settings, so every present worker contributed in every round.");
    } else if (gaps[worst] > 0.15 * Math.max(ref.final_loss, 0.1) && gaps[worst] > 0.2) {
      lines.push("Domain " + worst + " was starved: it ended at loss " + res.final_domain_losses[worst].toFixed(2) + " against " + ref.final_domain_losses[worst].toFixed(2) + " in the synchronous run. Worker " + worst +
        " was late in " + late[worst] + " of the " + present[worst] + " rounds it took part in.");
    } else {
      lines.push("No domain was left badly behind: the worst gap to the synchronous run is " + gaps[worst].toFixed(2) + " (domain " + worst + "). Worker " + mostLate + " was late most often (" + late[mostLate] + " of " + present[mostLate] + " rounds).");
    }
    // policy-specific
    var dropped = rows.reduce(function (a, row) { return a + row.dropped_stale; }, 0);
    var carried = rows.reduce(function (a, row) { return a + row.carried; }, 0);
    var required = rows.filter(function (row) { return row.required.length > 0; }).map(function (row) { return row.round; });
    if (choice.policy === "discard") {
      lines.push(totalLate + " late delta" + (totalLate === 1 ? " was" : "s were") + " discarded and the work in them was lost.");
    } else if (choice.policy === "fair") {
      var alt = Lab.simulate(Site ? Site.referenceScenario(res.scenario) : res.scenario);
      var disc = Lab.simulate(Object.assign({}, res.scenario, { policy: "discard" }));
      lines.push("The fair policy forced the coordinator to wait in " + required.length + " round" + (required.length === 1 ? "" : "s") + (required.length ? " (" + required.slice(0, 6).join(", ") + (required.length > 6 ? ", ..." : "") + ")" : "") +
        ". Simulated total time was " + res.total_time.toFixed(1) + " instead of " + disc.total_time.toFixed(1) + " with discard, a cost of " + (res.total_time - disc.total_time).toFixed(1) + " time units. (Synchronous reference time: " + alt.total_time.toFixed(1) + ".)");
    } else {
      lines.push(carried + " delayed delta" + (carried === 1 ? " was" : "s were") + " merged one or two rounds late with weight 1/(1+staleness)" + (dropped ? "; " + dropped + " more were too stale (older than " + Lab.MAX_STALENESS + " rounds) and dropped" : "") + ".");
    }
    // churn
    if (res.scenario.churn_end) {
      var away = rows.filter(function (row) { return row.alive.indexOf(1) < 0; }).map(function (row) { return row.round; });
      lines.push("Worker 1 was away in " + rangeText(away) + ", so its domain made no progress then" +
        (choice.policy === "fair" ? "; the fair policy did not wait for the absent worker, so no round stalled." : "."));
    }
    if (res.scenario.churn_end && choice.K >= M - 1) {
      lines.push("While worker 1 was away, K=" + choice.K + " meant waiting for every worker that was still present, including the straggler.");
    }
    // time budget
    if (sc.budget !== null) {
      lines.push("Time: " + sc.total_time.toFixed(1) + " simulated units against a budget of " + sc.budget.toFixed(1) + (sc.penalty ? ", a penalty of " + sc.penalty + " points." : ", so no penalty."));
    }
    lines.push(level.lesson);
    return lines;
  }

  /** Everything the page needs for one run. */
  function play(level, choice, seed) {
    var sc = levelScenario(level, choice, seed);
    var res = Lab.simulate(sc);
    var ref = Lab.simulate(Site ? Site.referenceScenario(sc) : Object.assign({}, sc, { K: sc.M, straggle: "none", policy: "discard", churn_start: 0, churn_end: 0 }));
    var s = scoreRun(level, res);
    var stars = starsFor(level, s.score, sc.seed);
    return { scenario: sc, res: res, ref: ref, score: s, stars: stars, debrief: debrief(level, choice, res, ref, s), thresholds: thresholds(level, sc.seed) };
  }

  var GameLogic = {
    LEVELS: LEVELS, POLICY_CARDS: POLICY_CARDS, GRACE_CARDS: GRACE_CARDS, MERGE_CARDS: MERGE_CARDS, K_TEXT: K_TEXT, STAR_RULE: STAR_RULE, BUDGET_PER_ROUND: BUDGET_PER_ROUND,
    PENALTY_PER_UNIT: PENALTY_PER_UNIT, THRESHOLD_SEEDS: THRESHOLD_SEEDS, levelScenario: levelScenario, timeBudget: timeBudget, scoreRun: scoreRun, allChoices: allChoices, thresholds: thresholds, starsFor: starsFor,
    starRuleText: starRuleText, debrief: debrief, play: play, describeChoice: describeChoice
  };
  root.GameLogic = GameLogic;
  if (typeof module === "object" && module.exports) module.exports = GameLogic;

  // ---------------------------------------------------------------- DOM
  function initGame() {
    var doc = root.document, host = doc.getElementById("game-root");
    if (!host || !Lab || !Site) return;
    var esc = Site.esc;
    var state = { level: 0, seed: null, timer: null, last: null };
    function el(id) { return doc.getElementById(id); }

    function bestKey(i) { return "hm-quest-best-" + LEVELS[i].id; }
    function loadBest(i) {
      var raw = Site.storeGet(bestKey(i));
      try { var v = raw ? JSON.parse(raw) : null; return v && typeof v.score === "number" ? v : null; } catch (e) { return null; }
    }
    function saveBest(i, score, stars) {
      var prev = loadBest(i);
      if (!prev || score > prev.score) Site.storeSet(bestKey(i), JSON.stringify({ score: score, stars: stars }));
    }
    function starsText(n) { return new Array(n + 1).join("★") + new Array(4 - n).join("☆"); }

    function renderNav() {
      var total = 0, html = "";
      LEVELS.forEach(function (lv, i) {
        var b = loadBest(i);
        if (b) total += b.score;
        html += '<li><button type="button" data-level="' + i + '"' + (i === state.level ? ' aria-current="true"' : "") + ">Level " + lv.id + (b ? " (" + starsText(b.stars) + " " + b.score + ")" : "") + "</button></li>";
      });
      el("level-nav").innerHTML = html;
      el("total-best").textContent = String(total);
    }

    function cards(name, list, selected, labelFn) {
      return list.map(function (c, i) {
        var id = name + "-" + i;
        return '<label class="card" for="' + id + '"><input type="radio" name="' + name + '" id="' + id + '" value="' + esc(c.value) + '"' + (String(c.value) === String(selected) ? " checked" : "") + (c.disabled ? " disabled" : "") + '>' +
          '<span class="t">' + esc(labelFn ? labelFn(c) : c.title) + '</span><span class="d">' + esc(c.text) + "</span></label>";
      }).join("");
    }

    function renderLevel() {
      var lv = LEVELS[state.level], M = lv.scenario.M;
      state.seed = lv.seed;
      el("level-title").textContent = "Level " + lv.id + ": " + lv.title;
      el("level-story").textContent = lv.story;
      el("level-goal").textContent = "Goal: " + lv.goal;
      var kcards = [];
      var maxK = lv.maxK || M;
      for (var k = 1; k <= M; k++) {
        kcards.push({ value: k, title: "K = " + k, disabled: k > maxK,
          text: k > maxK ? "Not available on this level: waiting for everyone is the synchronous reference you are measured against." : "Wait for " + (k === M ? "all " + M : "the first " + k) + " (" + K_TEXT[k] + ")." });
      }
      el("cards-K").innerHTML = cards("K", kcards, M - 1);
      el("cards-policy").innerHTML = cards("policy", POLICY_CARDS, "discard");
      el("cards-grace").innerHTML = cards("grace", GRACE_CARDS, 0.1);
      el("cards-merge").innerHTML = cards("merge", MERGE_CARDS, "direct");
      el("seed-text").textContent = String(state.seed);
      el("star-rule").textContent = starRuleText(lv, state.seed);
      el("result").hidden = true;
      el("run-status").textContent = "";
      var real = el("real-link");
      real.textContent = lv.real.label;
      real.setAttribute("href", lv.real.href);
      renderNav();
    }

    function readChoice() {
      function val(name) { var n = doc.querySelector('input[name="' + name + '"]:checked'); return n ? n.value : null; }
      return { K: Number(val("K")), policy: val("policy"), grace: Number(val("grace")), merge: val("merge") };
    }

    function draw(p, upto) {
      el("game-chart").innerHTML = Site.lossChartSvg(p.res, p.ref, upto);
      el("game-strip").innerHTML = Site.stripSvg(p.res, upto);
    }

    function finish(p) {
      var lv = LEVELS[state.level];
      draw(p);
      el("result").hidden = false;
      el("score-line").textContent = "Learning score " + p.score.score + " / 100";
      el("stars").textContent = starsText(p.stars);
      el("stars").setAttribute("aria-label", p.stars + " of 3 stars");
      el("score-parts").textContent = "100 x retention (capped at 1) = " + p.score.base + (p.score.budget !== null ? "; time penalty " + p.score.penalty + " (" + p.score.total_time.toFixed(1) + " units against a budget of " + p.score.budget.toFixed(1) + ")" : "; no time penalty on this level") +
        ". Best strategy found by the engine (mean over " + p.thresholds.seeds + " seeds): " + describeChoice(p.thresholds.bestChoice) + " (mean score " + p.thresholds.best + ")." +
        (p.res.retention !== null && p.res.retention > 1.0001 ? " Your run went past the synchronous result (retention " + p.res.retention.toFixed(2) + ") and the cap hides the excess: the RDA merge can overshoot in this toy." : "");
      el("debrief").innerHTML = p.debrief.map(function (l) { return "<li>" + esc(l) + "</li>"; }).join("");
      el("run-status").textContent = "Run finished: learning score " + p.score.score + ", " + p.stars + " of 3 stars.";
      saveBest(state.level, p.score.score, p.stars);
      renderNav();
      el("run").disabled = false;
      el("skip").hidden = true;
    }

    function run() {
      var lv = LEVELS[state.level], choice = readChoice();
      var p;
      try { p = play(lv, choice, state.seed); } catch (e) { el("run-status").textContent = "Cannot run: " + e.message; return; }
      if (state.timer) { root.clearTimeout(state.timer); state.timer = null; }
      state.last = p;
      el("result").hidden = false;
      el("score-line").textContent = "";
      el("stars").textContent = "";
      el("score-parts").textContent = "";
      el("debrief").innerHTML = "";
      if (Site.prefersReducedMotion()) { finish(p); return; }
      el("run").disabled = true;
      el("skip").hidden = false;
      el("run-status").textContent = "Running the rounds...";
      var round = 0, R = p.res.rounds.length;
      draw(p, 0);
      (function tick() {
        round += 1;
        draw(p, round);
        if (round >= R) { state.timer = null; finish(p); return; }
        state.timer = root.setTimeout(tick, 380);
      })();
    }

    el("skip").addEventListener("click", function () {
      if (state.timer) { root.clearTimeout(state.timer); state.timer = null; }
      if (state.last) finish(state.last);
    });
    el("run").addEventListener("click", run);
    el("new-seed").addEventListener("click", function () {
      state.seed = 1 + Math.floor(Math.random() * 99999);
      el("seed-text").textContent = String(state.seed);
      el("star-rule").textContent = starRuleText(LEVELS[state.level], state.seed);
      el("result").hidden = true;
    });
    el("level-nav").addEventListener("click", function (ev) {
      var b = ev.target.closest ? ev.target.closest("button[data-level]") : null;
      if (!b) return;
      if (state.timer) { root.clearTimeout(state.timer); state.timer = null; }
      state.level = Number(b.getAttribute("data-level"));
      el("run").disabled = false;
      el("skip").hidden = true;
      renderLevel();
      el("level-title").focus();
    });
    renderLevel();
  }

  if (root.document && typeof root.document.addEventListener === "function") {
    if (root.document.readyState === "loading") root.document.addEventListener("DOMContentLoaded", initGame); else initGame();
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
