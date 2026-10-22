"""Runner script for the Project HiveMind Campaign Calendar, .ics Export & Quorum Coordinator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hivemind.cli.campaign_calendar_quorum import main


if __name__ == "__main__":
    raise SystemExit(main())
