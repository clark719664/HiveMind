/*
 * Parity driver for tests/test_swarm_lab_js_parity.py (node only, not used by the web pages).
 *
 *   node docs/site/parity_driver.js spec.json   # {"scenarios": [partial Scenario objects, snake_case like the Python CLI], "invalid": [objects that must be rejected]}
 *
 * Prints one JSON document: the pinned PRNG values and, per scenario, the fields the Python test compares with simulate().
 */
"use strict";
const fs = require("fs");
const path = require("path");
const SwarmLab = require(path.join(__dirname, "swarm_lab.js"));

const out = { engine: SwarmLab.ENGINE, prng: {}, results: [] };

const u = new SwarmLab.Rng(12345);
out.prng.uniform_12345 = [0, 1, 2, 3, 4].map(() => u.uniform());
const g = new SwarmLab.Rng(12345);
out.prng.gauss_12345 = [0, 1, 2].map(() => g.gauss());
out.prng.state_wrap = [new SwarmLab.Rng(2 ** 32 + 12345).state, new SwarmLab.Rng(12345).state];
out.prng.sha256_abc = SwarmLab.sha256Hex("abc");

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
out.invalid_rejected = (spec.invalid || []).map((sc) => {
  try {
    SwarmLab.simulate(sc);
    return false;
  } catch (err) {
    return true;
  }
});
for (const sc of spec.scenarios) {
  const r = SwarmLab.simulate(sc);
  out.results.push({
    loss_curve: r.loss_curve,
    admitted: r.rounds.map((x) => x.admitted),
    late: r.rounds.map((x) => x.late),
    required: r.rounds.map((x) => x.required),
    carried: r.rounds.map((x) => x.carried),
    dropped_stale: r.rounds.map((x) => x.dropped_stale),
    round_time: r.rounds.map((x) => x.round_time),
    domain_losses_final: r.final_domain_losses,
    total_time: r.total_time,
    retention: r.retention,
    sync_final_loss: r.sync_final_loss,
    digest: r.digest,
  });
}
process.stdout.write(JSON.stringify(out));
