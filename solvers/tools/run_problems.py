# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Run the hivemind-solver binary on every problem in solvers/problems and report the error table (stdlib only).

    python3 solvers/tools/run_problems.py --bin PATH/TO/hivemind-solver [--markdown] [--dir DIR]

For each problem: max over all elements of |y - y_ref| / (rtol*|y_ref| + atol) ("ratio"; the tolerance
policy accepts ratio <= max_rel_err_factor), the max invariant drift as a fraction of each invariant's tol,
solver statistics. Exit status 1 if any problem is outside its policy or errors. Nothing is loosened here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_problems import eval_invariant  # noqa: E402


def run_solver(binary: str, spec_bytes: bytes, timeout: float = 600.0) -> dict:
    p = subprocess.run([binary, "run"], input=spec_bytes, capture_output=True, timeout=timeout, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"solver exited with {p.returncode}: {p.stderr.decode(errors='replace')[:200]}")
    return json.loads(p.stdout)


def evaluate(doc: dict, result: dict) -> dict:
    spec, ref = doc["spec"], doc["reference"]
    out = {"id": doc["id"], "status": result.get("status")}
    if result.get("status") != "ok":
        out["error"] = result.get("error")
        out["ok"] = False
        return out
    ratio = 0.0
    for yr, ys in zip(ref["y"], result["y"]):
        for a, b in zip(yr, ys):
            ratio = max(ratio, abs(b - a) / (spec["rtol"] * abs(a) + spec["atol"]))
    drift = 0.0
    worst_inv = ""
    for inv in doc.get("invariants", []):
        for row in result["y"]:
            d = abs(eval_invariant(inv, row) - inv["value"]) / inv["tol"]
            if d > drift:
                drift, worst_inv = d, inv["name"]
    factor = doc["tolerance_policy"]["max_rel_err_factor"]
    out.update(
        ratio=ratio,
        factor=factor,
        inv_drift=drift,
        inv_name=worst_inv,
        steps=result["stats"]["steps"],
        rhs=result["stats"]["rhs_evals"],
        method=result["solver"]["method"],
        ok=ratio <= factor and drift <= 1.0,
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent.parent / "problems")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(args.dir.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        spec_bytes = (json.dumps(doc["spec"], separators=(",", ":")) + "\n").encode()
        rows.append(evaluate(doc, run_solver(args.bin, spec_bytes)))
    if args.markdown:
        print("| problem | method | steps | rhs evals | max error / (rtol*|y|+atol) | allowed | invariant drift / tol | verdict |")
        print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] != "ok":
            line = f"{r['id']}: solver status {r['status']}: {r.get('error')}"
            print(f"| {r['id']} | - | - | - | - | - | - | ERROR: {r.get('error')} |" if args.markdown else line)
            continue
        verdict = "within policy" if r["ok"] else "OUTSIDE POLICY"
        if args.markdown:
            print(f"| {r['id']} | {r['method']} | {r['steps']} | {r['rhs']} | {r['ratio']:.3g} | {r['factor']:g} | "
                  f"{r['inv_drift']:.3g} | {verdict} |")
        else:
            print(f"{r['id']:22s} ratio={r['ratio']:10.4g} (allowed {r['factor']:g}) inv_drift/tol={r['inv_drift']:.3g} "
                  f"steps={r['steps']} {verdict}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
