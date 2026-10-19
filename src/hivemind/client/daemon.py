"""Windows Zero-Intrusion Daemon (GetLastInputInfo & Active Game Process Monitor).

Monitors OS input hooks (GetLastInputInfo on Windows via ctypes) and active process tables.
If another Steam game launches or mouse/keyboard activity occurs within the 30-second idle
window, requests a training pause and CPU offload. Disk latency and GPU memory
release must be measured on the target hardware; neither is guaranteed here.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import math
import sys
from dataclasses import dataclass
import psutil


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.UINT),
        ("dwTime", ctypes.wintypes.DWORD),
    ]


KNOWN_GAME_EXECUTABLES = {
    "cs2.exe",
    "dota2.exe",
    "cyberpunk2077.exe",
    "eldenring.exe",
    "bg3.exe",
    "helldivers2.exe",
    "unrealengine.exe",
    "unity.exe",
}


from enum import Enum


class SETIAgentState(str, Enum):
    """Autonomous 3-State SETI-Swarm Agent Lifecycle."""

    STATE_1_GAMING_SEED_ONLY = "STATE_1_GAMING_SEED_ONLY"
    STATE_2_IDLE_SETI_MINER = "STATE_2_IDLE_SETI_MINER"
    STATE_3_SWARM_CONSUMER = "STATE_3_SWARM_CONSUMER"


@dataclass
class DaemonTelemetry:
    """Live snapshot of system idle time, active game process detection, and SETI-Swarm state."""

    os_idle_seconds: float
    idle_threshold_seconds: float
    active_game_detected: str | None
    should_yield_vram: bool
    reason: str
    seti_agent_state: str = SETIAgentState.STATE_2_IDLE_SETI_MINER.value
    os_idle_available: bool = True


def _query_windows_idle_seconds() -> float | None:
    """Return None when a reliable Win32 idle reading is unavailable."""
    if sys.platform != "win32":
        return None
    try:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            tick_count = ctypes.windll.kernel32.GetTickCount()
            elapsed_ms = (tick_count - lii.dwTime) & 0xFFFFFFFF
            return round(elapsed_ms / 1000.0, 3)
    except Exception:
        pass
    return None


def get_windows_idle_seconds() -> float:
    """Return measured idle seconds, or a conservative zero when unavailable.

    The daemon keeps a separate availability flag; unavailable telemetry never
    grants permission for unattended background training.
    """
    idle = _query_windows_idle_seconds()
    return idle if idle is not None else 0.0


def detect_running_game_process(extra_watchlist: set[str] | None = None) -> str | None:
    """Scan the process table for known games that should preempt training."""
    watchlist = set(KNOWN_GAME_EXECUTABLES)
    if extra_watchlist:
        watchlist.update(n.lower() for n in extra_watchlist)

    for proc in psutil.process_iter(["name"]):
        try:
            pname = (proc.info.get("name") or "").lower()
            if pname in watchlist:
                return pname
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ZeroIntrusionDaemon:
    """Monitors user input and game launches to protect gaming framerate and yield VRAM."""

    def __init__(
        self,
        idle_threshold_seconds: float = 30.0,
        enforce_os_idle_hook: bool = True,
        simulated_idle_seconds: float | None = None,
        simulated_active_game: str | None = None,
        consumer_query_active: bool = False,
    ) -> None:
        if not math.isfinite(idle_threshold_seconds) or idle_threshold_seconds < 0:
            raise ValueError("Idle threshold must be finite and nonnegative.")
        if simulated_idle_seconds is not None and (not math.isfinite(simulated_idle_seconds) or simulated_idle_seconds < 0):
            raise ValueError("Simulated idle time must be finite and nonnegative.")
        self.idle_threshold_seconds = idle_threshold_seconds
        self.enforce_os_idle_hook = enforce_os_idle_hook
        self.simulated_idle_seconds = simulated_idle_seconds
        self.simulated_active_game = simulated_active_game
        self.consumer_query_active = consumer_query_active

    def poll(self) -> DaemonTelemetry:
        idle_sec = (
            self.simulated_idle_seconds
            if self.simulated_idle_seconds is not None
            else _query_windows_idle_seconds()
        )
        idle_available = (
            isinstance(idle_sec, (int, float)) and not isinstance(idle_sec, bool)
            and math.isfinite(idle_sec) and idle_sec >= 0
        )
        if not idle_available:
            idle_sec = 0.0
        active_game = (
            self.simulated_active_game
            if self.simulated_active_game is not None
            else detect_running_game_process()
        )

        if active_game is not None:
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                os_idle_available=idle_available,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=active_game,
                should_yield_vram=True,
                reason=f"Steam game launched ({active_game}); requesting training pause and CPU offload.",
                seti_agent_state=SETIAgentState.STATE_1_GAMING_SEED_ONLY.value,
            )

        if self.consumer_query_active:
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                os_idle_available=idle_available,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=None,
                should_yield_vram=False,
                reason="Swarm Consumer mode active — querying P2P Sparse MoE ring with $HIVE credits.",
                seti_agent_state=SETIAgentState.STATE_3_SWARM_CONSUMER.value,
            )

        if self.enforce_os_idle_hook and not idle_available:
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                os_idle_available=idle_available,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=None,
                should_yield_vram=True,
                reason="OS idle telemetry unavailable; pausing background training conservatively.",
                seti_agent_state=SETIAgentState.STATE_1_GAMING_SEED_ONLY.value,
            )

        if self.enforce_os_idle_hook and idle_sec < self.idle_threshold_seconds:
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                os_idle_available=idle_available,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=None,
                should_yield_vram=True,
                reason=(
                    f"User input activity detected (idle {idle_sec:.1f}s < {self.idle_threshold_seconds:.1f}s)."
                ),
                seti_agent_state=SETIAgentState.STATE_1_GAMING_SEED_ONLY.value,
            )

        return DaemonTelemetry(
            os_idle_seconds=idle_sec,
            os_idle_available=idle_available,
            idle_threshold_seconds=self.idle_threshold_seconds,
            active_game_detected=None,
            should_yield_vram=False,
            reason="System idle / clear for background DiLoCo training & MoE expert hosting.",
            seti_agent_state=SETIAgentState.STATE_2_IDLE_SETI_MINER.value,
        )

    def resolve_seti_agent_state(self, consumer_query_active: bool | None = None) -> SETIAgentState:
        """Resolve the current 3-state SETI-Swarm lifecycle state."""
        if consumer_query_active is not None:
            self.consumer_query_active = consumer_query_active
        telemetry = self.poll()
        return SETIAgentState(telemetry.seti_agent_state)

    def should_preempt(self) -> bool:
        return self.poll().should_yield_vram

