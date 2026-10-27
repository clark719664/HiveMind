# hivemind-solver: the Solver Bee command line (wave 1, W1-1 and W1-2)

Status: built and tested on one host (x86-64 Linux). Cross-CPU reproducibility is **not yet measured** (that is experiment S-1). The verifier (PR-B) and the sandbox/service (PR-C) are separate pull requests and are not wired in here.

`hivemind-solver` integrates ordinary-differential-equation initial-value problems that are given as **data** (a JSON spec). It is a thin, strictly validated front end over the CVODE BDF/Adams solver of rusty-SUNDIALS, pinned to one git revision.

## Build

```
cd solvers/hivemind-solver
CARGO_TARGET_DIR=/path/outside/the/repo nice -n 10 cargo build --release --locked -j2
cargo test --locked          # unit tests + CLI integration tests
```

Rust 1.85 or newer (the rusty-SUNDIALS crates are edition 2024). `Cargo.lock` is committed. The build fetches rusty-SUNDIALS from GitHub at the pinned revision (network is needed once at build time, never at run time).

Release profile: `opt-level = 3`, `lto = false`, `codegen-units = 1`. No fast-math (not available in stable rustc), no `target-cpu=native`, no `RUSTFLAGS` anywhere.

## Commands

| Command | Behaviour |
|---|---|
| `hivemind-solver run < spec.json > result.json` | Solve one spec. Exit 0 for `status:"ok"` and for a clean `status:"error"` (invalid input, solver failure). Exit 2 only for an unknown subcommand. A crash is the only other non-zero exit. |
| `hivemind-solver version` | `{"crate_version":"0.1.0","solver_rev":"<40-char sha>"}` |
| `hivemind-solver families` | JSON list of named families, parameters, defaults, equations and limits |

The program never opens the network, reads no file other than stdin and writes only stdout/stderr. stdin is read through a bounded reader (at most 64 KiB + 1 byte are buffered). The pinned solver still contains a few `println!` diagnostics on failure paths, which would corrupt the JSON, so on Unix the real stdout is duplicated and fd 1 is pointed at stderr while the solver runs; only the result JSON reaches the real stdout.

## Spec: `hivemind-solver-spec/1`

```
{"schema":"hivemind-solver-spec/1", "id":"[A-Za-z0-9_.-]{1,64}",
 "problem": {"kind":"named","family":"...","params":{...}}
          | {"kind":"linear","A":[[...]],"b":[...]}                          // x' = A x + b, n <= 20
          | {"kind":"mass_action","species":["A",..],"reactions":[{"reactants":{"A":1},"products":{"B":1},"k":0.5}]},
 "y0":[...], "t0":0.0, "t_out":[strictly increasing, > t0, <= 1000 entries],
 "rtol":1e-8, "atol":1e-10, "method":"bdf|adams", "max_steps":100000}
```

Unknown fields are rejected everywhere (`deny_unknown_fields`). `y0` is always required and its length must match the problem. Numbers must be finite (JSON has no NaN; numbers that overflow f64 are rejected). `0 < rtol < 1`, `atol > 0`. `atol` may be an array of n numbers **only if all entries are equal**: the pinned solver supports a scalar `atol` only, so unequal entries return `status:"error"` instead of being silently approximated. `max_steps` (1..=1,000,000) is passed to CVODE, which applies it per output time. `mass_action`: at most 20 species and 50 reactions, species names `[A-Za-z0-9_+-]{1,32}`, stoichiometric coefficients integers 1..=6, `k` in [0, 1e12]; rate = k times the product of reactant concentrations to their coefficients. Parameters are limited to magnitude 1e12.

### Named families (defaults in brackets)

| family | state | equations | parameters |
|---|---|---|---|
| `robertson` | 3 | y1'=-k1 y1+k3 y2 y3, y2'=k1 y1-k3 y2 y3-k2 y2^2, y3'=k2 y2^2 | k1 [0.04], k2 [3e7], k3 [1e4] (all >= 0) |
| `van_der_pol` | 2 | x'=v, v'=mu(1-x^2)v-x | mu [1] (>= 0) |
| `brusselator` | 2 | u'=a+u^2 v-(b+1)u, v'=b u-u^2 v | a [1], b [3] (>= 0) |
| `lotka_volterra` | 2 | x'=alpha x-beta x y, y'=delta x y-gamma y | alpha [1.5], beta [1], delta [1], gamma [3] |
| `harmonic` | 2 | x'=v, v'=-omega^2 x-2 zeta omega v | omega [1], zeta [0] (zeta >= 0) |
| `stiff_chain` | n | reflecting diffusion chain y_i'=k(y_{i-1}-2y_i+y_{i+1}), ends reflecting, sum(y) conserved | n [10] (integer 2..=20), k [100] (>= 0) |

All right-hand sides come with analytic Jacobians (checked against central differences in a unit test).

## Result: `hivemind-solver-result/1`

```
{"schema":"hivemind-solver-result/1","id":"..","status":"ok","t":[..],"y":[[..],..],
 "stats":{"steps":N,"rhs_evals":N,"jac_evals":null,"last_order":q},
 "solver":{"name":"rusty-sundials/cvode","rev":"<sha>","method":"bdf","crate_version":"0.1.0"},
 "input_sha256":"<hex of the exact stdin bytes>"}
```

On failure: `{"schema":..,"id":<id or null>,"status":"error","error":"message","input_sha256":".."}` (no `t`/`y`/`stats`/`solver`). `jac_evals` is always `null`: the pinned solver exposes no Jacobian-evaluation counter. `input_sha256` is the sha256 of the exact bytes received, not of a re-serialised form. Floats are written with serde_json's shortest round-trip representation. A non-finite right-hand side or solution becomes `status:"error"`; panics are caught and also reported as `status:"error"`.

### Invariant kinds in the problem library

The contract names `linear` and leaves the other kinds' fields open; this library uses (for the verifier, PR-B):

| kind | fields | meaning |
|---|---|---|
| `linear` | `weights` (n numbers), `value`, `tol` | sum(weights_i y_i) = value |
| `quadratic_form` | `matrix` (n x n), `value`, `tol` | y^T M y = value (used for the undamped oscillator energy) |
| `lv_first_integral` | `params` {alpha,beta,delta,gamma}, `value`, `tol` | delta x - gamma ln x + beta y - alpha ln y = value |

`tol` is absolute. It follows a rule fixed before any solver output was inspected: linear invariants `10*rtol*max(1,|value|)`, nonlinear invariants `1000*rtol*max(1,|value|)`. (Extra, non-contract fields in a problem file: top-level `description`, and `reference.cross_check`.)

## Determinism choices

* All right-hand sides and Jacobians use only `+ - *` and integer powers by repeated multiplication. IEEE-754 rounds these identically on every conforming CPU, so this crate calls **no** system libm function (hence no `libm` dependency was added: there is nothing in the right-hand sides to replace).
* Fixed iteration order everywhere (`BTreeMap` for names, fixed summation order), no threads (serial vectors), no randomness, no time or environment dependence.
* The pinned CVODE internals do call `f64::powf` and `sqrt` (step-size and error-norm control). `sqrt` is correctly rounded by IEEE-754; `powf` goes to the system libm, so its last-bit result can in principle differ between libm versions. This is the main known source of possible cross-host difference and is exactly what experiment S-1 must measure; it is **not** claimed to be absent.
* Measured here: ten in-process runs and ten separate process runs of the same spec give byte-identical output (tests `determinism_ten_runs_byte_identical`, `repeated_process_runs_are_byte_identical`). One host only.

## Pinned dependency

rusty-SUNDIALS (`https://github.com/xaviercallens/rusty-SUNDIALS`, same owner as HiveMind), crates `cvode` and `nvector` as git dependencies at `rev = "4e3e32de9ff188bace35c9e80e433b4e795a9d76"` (the main-branch tip on 2026-10-27, verified to exist on the remote with `gh api repos/xaviercallens/rusty-SUNDIALS/commits/<sha>`). The same string is the Rust constant `SOLVER_REV` and a unit test (`rev_constant_matches_cargo_toml`) asserts that both `Cargo.toml` lines carry it.

**Why not the older `af4886f`:** that revision was tried first. On a plain harmonic oscillator its BDF failed with "too many error test failures" for `rtol <= 1e-8` and its Adams stayed at order 1 (tens of thousands of steps). The later revision includes the upstream fixes "cvode tight-tolerance fix" (d22cf01) and the Adams order work (#69/#71); with it both methods integrate the same problem at `rtol` down to 1e-10.

Licence: BSD-3-Clause per the dependency's `LICENSE` and `Cargo.toml` (Copyright (c) 2026 Xavier Callens / SocrateAI Lab); it is not vendored into this repository. **Open point for the owner:** its `NOTICE` file (same revision) says the code is distributed under "Apache License 2.0" with "Apache-2.0 AND CC-BY-4.0" and demands a citation; the LICENSE file and the GitHub licence detection say BSD-3-Clause. The citation is reproduced in the crate README. The inconsistency should be resolved in the rusty-SUNDIALS repository before any redistribution of binaries.

## Limits

64 KiB input; <= 1000 output times; <= 20 state variables; <= 50 reactions; `max_steps` <= 1,000,000. No time limit inside the binary (CPU time and memory limits belong to the sandbox, PR-C). Scalar absolute tolerance only. The solver is a fixed-leading-coefficient CVODE translation: expect global error to be one to a few hundred times `rtol` on long oscillatory problems (see the table).

## Problem library (`solvers/problems/`, 15 problems)

Schema `hivemind-solver-problem/1` (see the contract). Regenerate every reference with `python3 solvers/tools/make_references.py` (needs numpy, scipy, mpmath; records tool versions, method and a cross-check per file); validate with `python3 solvers/tools/check_problems.py`; run the table below with `python3 solvers/tools/run_problems.py --bin <path> --markdown`.

Reference methods (all independent of rusty-SUNDIALS, each cross-checked by a second method whose disagreement is stored in `reference.cross_check`; `reference.max_abs_error_bound` is that disagreement, a conservative bound for the reference itself):

* linear systems, harmonic oscillators, first-order mass-action networks and the stiff chain: exact matrix exponential of `[[A,b],[0,0]]` in 40-digit mpmath arithmetic; cross-check: closed form (oscillators) or `scipy.linalg.expm`;
* nonlinear non-stiff problems (dimerisation, Michaelis-Menten, Van der Pol mu=1, Brusselator, Lotka-Volterra): mpmath Taylor series (`odefun`, 30 digits); cross-check: scipy DOP853 at rtol 1e-13;
* stiff nonlinear problems (Robertson, Van der Pol mu=100): scipy Radau at rtol 1e-13, atol 1e-20; cross-check: scipy LSODA (mpmath Taylor is impractical at these stiffness ratios).

Every reference passes the contract rule "at least 100x more accurate than the spec tolerance" (the generator refuses to write a file otherwise).

### Error table (release binary, `bdf` unless stated, this host)

Metric: max over all elements and output times of `|y - y_ref| / (rtol*|y_ref| + atol)`; the tolerance policy accepts a value of at most 100 (`max_rel_err_factor`). Invariant drift is the worst `|invariant - value| / tol`; it must be at most 1.

| problem | method | steps | rhs evals | error ratio | allowed | invariant drift / tol | verdict |
|---|---|---|---|---|---|---|---|
| brusselator | bdf | 1354 | 3825 | 28.6 | 100 | none defined | within policy |
| brusselator_adams | adams | 139 | 419 | 1.04 | 100 | none defined | within policy |
| decay_chain | bdf | 379 | 1055 | 6.32 | 100 | 2.55e-06 | within policy |
| dimerisation | bdf | 373 | 1014 | 3.15 | 100 | 1.55e-08 | within policy |
| harmonic_damped | bdf | 952 | 2683 | **132** | 100 | none defined | **OUTSIDE POLICY** |
| harmonic_undamped | bdf | 711 | 1967 | 80.3 | 100 | 0.143 | within policy |
| linear_2x2 | bdf | 659 | 1866 | 72.6 | 100 | none defined | within policy |
| linear_5x5 | bdf | 433 | 1212 | 16.2 | 100 | none defined | within policy |
| lotka_volterra | bdf | 2093 | 5941 | **165** | 100 | 0.0602 | **OUTSIDE POLICY** |
| michaelis_menten | bdf | 649 | 1840 | 7.56 | 100 | 1.78e-08 | within policy |
| reversible_network | bdf | 379 | 1038 | 3.26 | 100 | 4.77e-08 | within policy |
| robertson | bdf | 639 | 1826 | 12.7 | 100 | 3.01e-05 | within policy |
| stiff_chain | bdf | 425 | 1183 | 3.12 | 100 | 1.04e-08 | within policy |
| van_der_pol_mu1 | bdf | 1975 | 5531 | **481** | 100 | none defined | **OUTSIDE POLICY** |
| van_der_pol_mu100 | bdf | 1252 | 3504 | 2.25 | 100 | none defined | within policy |

**Three of 15 problems are outside the stated policy** (damped oscillator 132, Lotka-Volterra 165, Van der Pol mu=1 481 against a limit of 100). Nothing was loosened to hide this: the references are doubly cross-checked, and the solver error shrinks steadily when the tolerance is tightened (probe, not part of the test suite, max absolute error at `rtol` 1e-8 / 1e-10 / 1e-12: Van der Pol mu=1 2.5e-7 / 8.5e-9 / 1.4e-10; Lotka-Volterra 1.4e-6 / 3.6e-8 / 6.9e-10; damped oscillator 1.3e-7 / 3.1e-9 / 6.8e-11). So this is accumulation of global error over long oscillatory horizons by the solver at the stated tolerance, not a wrong reference or a wrong right-hand side. Whether the policy factor 100 is the right acceptance rule for these problems is a decision for the owner/verifier design (PR-B); the library keeps the contract's value.

All declared invariants (including the Lotka-Volterra first integral, at 6% of its tolerance) stay within their pre-registered tolerances.
