"""E13: dependency-free parent-death watchdog (no torch import, so it can be tested in a fraction of a second).

A worker / thread-host process polls ``os.getppid()`` every second; when the parent (the experiment coordinator) is gone (SIGKILL, OOM-kill, crash) the process
``os._exit(3)``s so that no orphan keeps holding GPU memory in an unattended queue.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

EXIT_ORPHAN = 3


def start_parent_watchdog(poll_s: float = 1.0, parent_pid: Optional[int] = None, exit_fn: Callable[[int], None] = os._exit,
                          getppid: Callable[[], int] = os.getppid) -> threading.Thread:
    """Daemon thread: ``exit_fn(3)`` as soon as ``getppid()`` differs from the parent pid (``parent_pid`` if given, else the parent at start)."""
    expected = int(parent_pid) if parent_pid else getppid()

    def loop() -> None:
        while True:
            time.sleep(poll_s)
            if getppid() != expected:
                exit_fn(EXIT_ORPHAN)
                return

    t = threading.Thread(target=loop, name="e13-parent-watchdog", daemon=True)
    t.start()
    return t
