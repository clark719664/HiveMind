# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Family catalogue and data-only spec builders for the Science Bee (stdlib only).

Two kinds of building blocks produce ``hivemind-solver-spec/1`` dictionaries:

* NAMED families that the Rust CLI knows (``pendulum``, ``projectile_drag``, ``kepler``, ``logistic`` and the original six); the builders here only
  check the arguments and wrap them in a spec.
* PRESETS that are expressed with the generic ``mass_action`` and ``linear`` problem kinds (SIR, SEIR, decay chain, series RLC, spring-mass-damper).

Nothing here evaluates anything: every function returns plain JSON-compatible data. Whether a spec describes a real situation is never checked.

State orders are fixed here once and are part of the contract:

* series RLC: ``[q, i]`` (capacitor charge, loop current), ``A = [[0, 1], [-1/(L C), -R/L]]``, ``b = [0, V/L]``.
* spring-mass-damper: ``[x, v]``, ``A = [[0, 1], [-k/m, -c/m]]``, ``b = [0, F/m]``.
* SIR species ``[S, I, R]``; SEIR species ``[S, E, I, R]`` (fractions of the population summing to 1).
* decay chain species ``[A, B, C, ...]`` with ``A -> B -> C ...`` and rate ``k_i`` for step i.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

SPEC_SCHEMA = "hivemind-solver-spec/1"
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LIMITS_SENTENCE = "The numbers solve the equations that were set up; whether those equations describe your real situation is not checked."
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10


def _state(name: str, meaning: str, unit: str) -> Dict[str, str]:
    return {"name": name, "meaning": meaning, "unit": unit}


# name -> catalogue entry. ``params`` maps parameter name -> (default, plain-language meaning); ``dim`` is the state size (None = depends on a parameter).
FAMILIES: Dict[str, Dict[str, Any]] = {
    "pendulum": {
        "summary": "a pendulum: theta'' = -g_over_l*sin(theta) - damping*omega",
        "dim": 2,
        "params": {"g_over_l": (9.81, "gravity divided by the pendulum length (1/time^2)"), "damping": (0.0, "linear friction coefficient (1/time)")},
        "state": [_state("theta", "angle from the downward vertical", "radians"), _state("omega", "angular velocity", "radians per time unit")],
        "invariants": ["energy (only when damping is 0)"],
    },
    "projectile_drag": {
        "summary": "a projectile with quadratic air drag a = -k*|v|*v and gravity g (no ground-hit event: you choose the times)",
        "dim": 4,
        "params": {"g": (9.81, "gravitational acceleration (length/time^2)"), "k": (0.0, "quadratic drag coefficient (1/length); 0 means no drag")},
        "state": [_state("x", "horizontal position", "length unit"), _state("y", "height", "length unit"),
                  _state("vx", "horizontal velocity", "length unit per time unit"), _state("vy", "vertical velocity", "length unit per time unit")],
        "invariants": ["energy (only when k is 0)"],
    },
    "kepler": {
        "summary": "planar two-body motion around a fixed centre with gravitational parameter mu",
        "dim": 4,
        "params": {"mu": (1.0, "gravitational parameter G*M (length^3/time^2)")},
        "state": [_state("x", "x position", "length unit"), _state("y", "y position", "length unit"),
                  _state("vx", "x velocity", "length unit per time unit"), _state("vy", "y velocity", "length unit per time unit")],
        "invariants": ["energy", "angular momentum"],
    },
    "logistic": {
        "summary": "logistic growth N' = r*N*(1-N/K)",
        "dim": 1,
        "params": {"r": (1.0, "growth rate (1/time)"), "K": (1.0, "carrying capacity (same unit as N)")},
        "state": [_state("N", "population size", "population unit")],
        "invariants": [],
    },
    "robertson": {
        "summary": "the stiff Robertson chemical kinetics problem",
        "dim": 3,
        "params": {"k1": (0.04, "rate"), "k2": (3.0e7, "rate"), "k3": (1.0e4, "rate")},
        "state": [_state("y1", "concentration 1", "concentration unit"), _state("y2", "concentration 2", "concentration unit"), _state("y3", "concentration 3", "concentration unit")],
        "invariants": ["total mass y1+y2+y3"],
    },
    "van_der_pol": {
        "summary": "the Van der Pol oscillator x'' = mu*(1-x^2)*x' - x",
        "dim": 2,
        "params": {"mu": (1.0, "nonlinearity / stiffness parameter")},
        "state": [_state("x", "position", "length unit"), _state("v", "velocity", "length unit per time unit")],
        "invariants": [],
    },
    "brusselator": {
        "summary": "the Brusselator chemical oscillator",
        "dim": 2,
        "params": {"a": (1.0, "feed rate"), "b": (3.0, "feed rate")},
        "state": [_state("u", "concentration u", "concentration unit"), _state("v", "concentration v", "concentration unit")],
        "invariants": [],
    },
    "lotka_volterra": {
        "summary": "predator-prey Lotka-Volterra dynamics",
        "dim": 2,
        "params": {"alpha": (1.5, "prey growth rate"), "beta": (1.0, "predation rate"), "delta": (1.0, "predator growth per prey eaten"), "gamma": (3.0, "predator death rate")},
        "state": [_state("x", "prey population", "population unit"), _state("y", "predator population", "population unit")],
        "invariants": ["Lotka-Volterra first integral"],
    },
    "harmonic": {
        "summary": "a damped harmonic oscillator x'' = -omega^2*x - 2*zeta*omega*x'",
        "dim": 2,
        "params": {"omega": (1.0, "natural angular frequency (1/time)"), "zeta": (0.0, "damping ratio (dimensionless)")},
        "state": [_state("x", "displacement", "length unit"), _state("v", "velocity", "length unit per time unit")],
        "invariants": ["energy (only when zeta is 0)"],
    },
    "stiff_chain": {
        "summary": "a diffusion chain of n cells with rate k (reflecting ends)",
        "dim": None,
        "params": {"n": (10, "number of cells (integer 2..20)"), "k": (100.0, "exchange rate")},
        "state": [],
        "invariants": ["total amount (sum of all cells)"],
    },
}
NEW_FAMILIES = ("pendulum", "projectile_drag", "kepler", "logistic")
# Parameters that must be integers.
INTEGER_PARAMS = {"stiff_chain": {"n"}}

PRESETS: Dict[str, str] = {
    "sir": "epidemic SIR as mass_action: S+I->2I (k=beta), I->R (k=gamma); species [S, I, R], fractions summing to 1",
    "seir": "epidemic SEIR as mass_action: S+I->E+I (k=beta), E->I (k=sigma), I->R (k=gamma); species [S, E, I, R]",
    "decay_chain": "radioactive / first-order chain A->B->C... as mass_action with one rate per step",
    "series_rlc": "series R-L-C circuit as a linear system, state [q, i]",
    "spring_mass_damper": "spring-mass-damper as a linear system, state [x, v]",
}


class FamilyError(ValueError):
    """A builder argument is invalid."""


# ------------------------------------------------------------------ validation helpers
def _num(value: Any, name: str, *, positive: bool = False, nonneg: bool = False, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FamilyError(f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise FamilyError(f"{name} must be finite")
    if abs(v) > 1e150:
        raise FamilyError(f"{name} is too large")
    if positive and not v > 0:
        raise FamilyError(f"{name} must be > 0")
    if nonneg and v < 0:
        raise FamilyError(f"{name} must be >= 0")
    if lo is not None and v < lo:
        raise FamilyError(f"{name} must be >= {lo}")
    if hi is not None and v > hi:
        raise FamilyError(f"{name} must be <= {hi}")
    return v


def _times(t_out: Any, t0: float = 0.0) -> List[float]:
    if not isinstance(t_out, (list, tuple)) or not 1 <= len(t_out) <= 1000:
        raise FamilyError("t_out must be a list of 1..1000 numbers")
    out: List[float] = []
    prev = _num(t0, "t0")
    for i, t in enumerate(t_out):
        v = _num(t, f"t_out[{i}]")
        if not v > prev:
            raise FamilyError("t_out must be strictly increasing and greater than t0")
        out.append(v)
        prev = v
    return out


def _sid(spec_id: str) -> str:
    if not isinstance(spec_id, str) or not ID_RE.match(spec_id):
        raise FamilyError("id must match [A-Za-z0-9_.-]{1,64}")
    return spec_id


def _finish(spec_id: str, problem: Dict[str, Any], y0: Sequence[float], t_out: Any, t0: float, rtol: Optional[float], atol: Optional[float],
            method: Optional[str]) -> Dict[str, Any]:
    t0v = _num(t0, "t0")
    spec: Dict[str, Any] = {"schema": SPEC_SCHEMA, "id": _sid(spec_id), "problem": problem, "y0": [float(v) for v in y0], "t0": t0v, "t_out": _times(t_out, t0v)}
    if rtol is not None:
        spec["rtol"] = _num(rtol, "rtol", positive=True, hi=1.0)
    if atol is not None:
        spec["atol"] = _num(atol, "atol", positive=True, hi=1.0)
    if method is not None:
        if method not in ("bdf", "adams"):
            raise FamilyError("method must be 'bdf' or 'adams'")
        spec["method"] = method
    return spec


def list_families() -> List[str]:
    return sorted(FAMILIES)


def family_dim(family: str, params: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """State dimension of a named family (stiff_chain: the n parameter, default 10)."""
    entry = FAMILIES.get(family)
    if entry is None:
        raise FamilyError(f"unknown family {family!r}")
    if entry["dim"] is not None:
        return int(entry["dim"])
    n = (params or {}).get("n", entry["params"]["n"][0])
    return int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) and float(n).is_integer() else None


def check_params(family: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameter names and values of a named family; returns a clean copy (numbers as float, integer params as int)."""
    entry = FAMILIES.get(family)
    if entry is None:
        raise FamilyError(f"unknown family {family!r}; expected one of {', '.join(list_families())}")
    if not isinstance(params, dict):
        raise FamilyError("params must be an object")
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in entry["params"]:
            raise FamilyError(f"family {family!r} has no parameter {str(k)[:40]!r}; allowed: {', '.join(entry['params'])}")
        val = _num(v, f"params.{k}")
        if k in INTEGER_PARAMS.get(family, ()):
            if not val.is_integer() or not 2 <= val <= 20:
                raise FamilyError(f"params.{k} must be an integer in 2..20")
            out[k] = int(val)
            continue
        if family == "pendulum" and k == "damping" and val < 0:
            raise FamilyError("params.damping must be >= 0")
        if (family, k) in {("pendulum", "g_over_l"), ("projectile_drag", "g"), ("kepler", "mu"), ("logistic", "K"), ("harmonic", "omega")} and not val > 0:
            raise FamilyError(f"params.{k} must be > 0")
        if (family, k) in {("projectile_drag", "k")} and val < 0:
            raise FamilyError("params.k must be >= 0")
        out[k] = val
    return out


def named_spec(family: str, params: Optional[Dict[str, Any]], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee", t0: float = 0.0,
               rtol: Optional[float] = None, atol: Optional[float] = None, method: Optional[str] = None) -> Dict[str, Any]:
    """A named-family spec after checking parameter names, values and the length of y0."""
    clean = check_params(family, dict(params or {}))
    dim = family_dim(family, clean)
    if not isinstance(y0, (list, tuple)) or (dim is not None and len(y0) != dim):
        raise FamilyError(f"y0 must have {dim} entries for family {family!r}")
    ys = [_num(v, f"y0[{i}]") for i, v in enumerate(y0)]
    if family == "logistic" and ys[0] < 0:
        raise FamilyError("y0[0] (N) must be >= 0")
    return _finish(spec_id, {"kind": "named", "family": family, "params": clean}, ys, t_out, t0, rtol, atol, method)


# ------------------------------------------------------------------ thin builders for the named families
def pendulum_spec(theta0: float, omega0: float, t_out: Sequence[float], *, g_over_l: float = 9.81, damping: float = 0.0, **kw: Any) -> Dict[str, Any]:
    return named_spec("pendulum", {"g_over_l": g_over_l, "damping": damping}, [theta0, omega0], t_out, **kw)


def projectile_drag_spec(x0: float, y0: float, vx0: float, vy0: float, t_out: Sequence[float], *, g: float = 9.81, k: float = 0.0, **kw: Any) -> Dict[str, Any]:
    return named_spec("projectile_drag", {"g": g, "k": k}, [x0, y0, vx0, vy0], t_out, **kw)


def kepler_spec(x0: float, y0: float, vx0: float, vy0: float, t_out: Sequence[float], *, mu: float = 1.0, **kw: Any) -> Dict[str, Any]:
    return named_spec("kepler", {"mu": mu}, [x0, y0, vx0, vy0], t_out, **kw)


def logistic_spec(n0: float, t_out: Sequence[float], *, r: float = 1.0, K: float = 1.0, **kw: Any) -> Dict[str, Any]:  # noqa: N803 - K is the conventional name
    return named_spec("logistic", {"r": r, "K": K}, [n0], t_out, **kw)


# ------------------------------------------------------------------ mass_action presets
def mass_action_spec(species: Sequence[str], reactions: Sequence[Dict[str, Any]], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee",
                     t0: float = 0.0, rtol: Optional[float] = None, atol: Optional[float] = None, method: Optional[str] = None) -> Dict[str, Any]:
    """Generic mass-action spec (<= 20 species, <= 50 reactions, small positive integer stoichiometry, k >= 0)."""
    sp = list(species)
    if not 1 <= len(sp) <= 20 or len(set(sp)) != len(sp) or not all(isinstance(s, str) and 1 <= len(s) <= 64 for s in sp):
        raise FamilyError("species must be 1..20 distinct strings")
    if not 1 <= len(reactions) <= 50:
        raise FamilyError("reactions must have 1..50 entries")
    rx = []
    for r in reactions:
        if not isinstance(r, dict) or set(r) - {"reactants", "products", "k"}:
            raise FamilyError("each reaction needs reactants, products and k only")
        sides = {}
        for side in ("reactants", "products"):
            m = r.get(side)
            if not isinstance(m, dict) or len(m) > 20:
                raise FamilyError(f"reaction {side} must be an object")
            for s, c in m.items():
                if s not in sp or isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 10:
                    raise FamilyError("stoichiometric coefficients must be integers 1..10 over declared species")
            sides[side] = dict(m)
        rx.append({"reactants": sides["reactants"], "products": sides["products"], "k": _num(r.get("k"), "reaction k", nonneg=True)})
    if len(y0) != len(sp):
        raise FamilyError("y0 must have one entry per species")
    ys = [_num(v, f"y0[{i}]", nonneg=True) for i, v in enumerate(y0)]
    return _finish(spec_id, {"kind": "mass_action", "species": sp, "reactions": rx}, ys, t_out, t0, rtol, atol, method)


def _fractions(values: Sequence[float], names: Sequence[str]) -> List[float]:
    vs = [_num(v, n, nonneg=True, hi=1.0) for v, n in zip(values, names)]
    if abs(sum(vs) - 1.0) > 1e-9:
        raise FamilyError("the initial fractions (" + ", ".join(names) + ") must sum to 1")
    return vs


def sir_spec(beta: float, gamma: float, s0: float, i0: float, t_out: Sequence[float], *, r0: Optional[float] = None, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:
    """SIR epidemic on population fractions: S+I->2I (k=beta), I->R (k=gamma)."""
    b, g = _num(beta, "beta", nonneg=True), _num(gamma, "gamma", nonneg=True)
    rr = (1.0 - s0 - i0) if r0 is None else r0
    y0 = _fractions([s0, i0, rr], ["s0", "i0", "r0"])
    rx = [{"reactants": {"S": 1, "I": 1}, "products": {"I": 2}, "k": b}, {"reactants": {"I": 1}, "products": {"R": 1}, "k": g}]
    return mass_action_spec(["S", "I", "R"], rx, y0, t_out, spec_id=spec_id, **kw)


def seir_spec(beta: float, sigma: float, gamma: float, s0: float, e0: float, i0: float, t_out: Sequence[float], *, r0: Optional[float] = None,
              spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:
    """SEIR epidemic: S+I->E+I (k=beta), E->I (k=sigma), I->R (k=gamma)."""
    b, s, g = _num(beta, "beta", nonneg=True), _num(sigma, "sigma", nonneg=True), _num(gamma, "gamma", nonneg=True)
    rr = (1.0 - s0 - e0 - i0) if r0 is None else r0
    y0 = _fractions([s0, e0, i0, rr], ["s0", "e0", "i0", "r0"])
    rx = [{"reactants": {"S": 1, "I": 1}, "products": {"E": 1, "I": 1}, "k": b}, {"reactants": {"E": 1}, "products": {"I": 1}, "k": s},
          {"reactants": {"I": 1}, "products": {"R": 1}, "k": g}]
    return mass_action_spec(["S", "E", "I", "R"], rx, y0, t_out, spec_id=spec_id, **kw)


def decay_chain_spec(rates: Sequence[float], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:
    """First-order chain A->B->C...: len(rates) steps, len(rates)+1 species (letters A, B, C, ...)."""
    if not isinstance(rates, (list, tuple)) or not 1 <= len(rates) <= 10:
        raise FamilyError("rates must be a list of 1..10 numbers")
    names = [chr(ord("A") + i) for i in range(len(rates) + 1)]
    rx = [{"reactants": {names[i]: 1}, "products": {names[i + 1]: 1}, "k": _num(k, f"rates[{i}]", nonneg=True)} for i, k in enumerate(rates)]
    return mass_action_spec(names, rx, y0, t_out, spec_id=spec_id, **kw)


# ------------------------------------------------------------------ linear presets
def linear_spec(A: Sequence[Sequence[float]], b: Sequence[float], y0: Sequence[float], t_out: Sequence[float], *, spec_id: str = "science-bee", t0: float = 0.0,  # noqa: N803
                rtol: Optional[float] = None, atol: Optional[float] = None, method: Optional[str] = None) -> Dict[str, Any]:
    n = len(A)
    if not 1 <= n <= 20 or any(len(row) != n for row in A) or len(b) != n or len(y0) != n:
        raise FamilyError("A must be n x n (n <= 20) and b, y0 of length n")
    mat = [[_num(v, f"A[{i}][{j}]") for j, v in enumerate(row)] for i, row in enumerate(A)]
    bb = [_num(v, f"b[{i}]") for i, v in enumerate(b)]
    ys = [_num(v, f"y0[{i}]") for i, v in enumerate(y0)]
    return _finish(spec_id, {"kind": "linear", "A": mat, "b": bb}, ys, t_out, t0, rtol, atol, method)


def series_rlc_spec(R: float, L: float, C: float, V: float, q0: float, i0: float, t_out: Sequence[float], *, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:  # noqa: N803
    """Series RLC driven by a constant voltage V: state [q, i], q' = i, i' = (V - R i - q/C)/L."""
    r, l_, c, v = _num(R, "R", nonneg=True), _num(L, "L", positive=True), _num(C, "C", positive=True), _num(V, "V")
    return linear_spec([[0.0, 1.0], [-1.0 / (l_ * c), -r / l_]], [0.0, v / l_], [q0, i0], t_out, spec_id=spec_id, **kw)


def spring_mass_damper_spec(m: float, c: float, k: float, F: float, x0: float, v0: float, t_out: Sequence[float], *, spec_id: str = "science-bee", **kw: Any) -> Dict[str, Any]:  # noqa: N803
    """Spring-mass-damper with constant force F: state [x, v], x' = v, v' = (F - c v - k x)/m."""
    mm, cc, kk, ff = _num(m, "m", positive=True), _num(c, "c", nonneg=True), _num(k, "k", nonneg=True), _num(F, "F")
    return linear_spec([[0.0, 1.0], [-kk / mm, -cc / mm]], [0.0, ff / mm], [x0, v0], t_out, spec_id=spec_id, **kw)


# ------------------------------------------------------------------ catalogue text for prompts and explanations
def describe_family(name: str) -> str:
    e = FAMILIES[name]
    params = ", ".join(f"{k} (default {d:g}: {m})" for k, (d, m) in e["params"].items())
    state = ", ".join(f"{s['name']} = {s['meaning']} [{s['unit']}]" for s in e["state"]) or "n cells, each an amount"
    inv = "; ".join(e["invariants"]) or "none"
    return f"- {name}: {e['summary']}. params: {params}. state y: [{state}]. invariants: {inv}."


def catalogue_text(only_new: bool = False) -> str:
    names = [n for n in FAMILIES if (n in NEW_FAMILIES or not only_new)]
    return "\n".join(describe_family(n) for n in names)


def presets_text() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in PRESETS.items())
