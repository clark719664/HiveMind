#!/usr/bin/env python3
"""Runner script for the 3-Hour Autonomous PR Review, Merge & Discussion Orchestrator."""

from __future__ import annotations

import sys

from hivemind.cli.community_orchestrator import main

if __name__ == "__main__":
    sys.exit(main())
