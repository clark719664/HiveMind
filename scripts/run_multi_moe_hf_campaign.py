"""Runner script for Operation Polyglot-Singularity-128: Multi-MoE & 8-Domain Hugging Face GPU Campaign."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hivemind.cli.multi_moe_hf_campaign import main


if __name__ == "__main__":
    raise SystemExit(main())
