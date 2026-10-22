#!/usr/bin/env python3
"""Runner script for the opt-in Project HiveMind Community GPU Benchmark Suite."""

from __future__ import annotations

import sys

from hivemind.cli.community_gpu_benchmark import main

if __name__ == "__main__":
    sys.exit(main())
