#!/usr/bin/env python3
"""Run the HM-S6-01 synthetic error-feedback compression benchmark (CPU, a few seconds).

Usage:
    python scripts/run_ef21_error_feedback_benchmark.py
    python scripts/run_ef21_error_feedback_benchmark.py --seed 1 --output artifacts/ef21_error_feedback_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm.ef_compression_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
