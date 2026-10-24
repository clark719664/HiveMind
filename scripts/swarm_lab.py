"""Standalone runner for the HiveMind Swarm Lab (qualitative, GPU-free toy): ``python3 scripts/swarm_lab.py simulate --preset persistent-straggler``."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hivemind.lab.swarm_lab import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
