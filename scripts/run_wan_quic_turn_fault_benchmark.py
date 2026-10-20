#!/usr/bin/env python3
"""Run the HM-S5-05 WAN QUIC 0-RTT & RFC 5766 TURN fault-injection benchmark.

The benchmark replays the real `aioquic` 0-RTT link, the RFC 5766 TURN relay (data
indications and channels) and the libp2p WebSocket fallback through a deterministic
packet-loss / reordering / jitter proxy, then writes the structured telemetry report to
`artifacts/wan_quic_turn_fault_report.json`.

Usage:
    python scripts/run_wan_quic_turn_fault_benchmark.py
    python scripts/run_wan_quic_turn_fault_benchmark.py --loss-rates 0,0.01,0.03,0.05
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.swarm.wan_fault_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
