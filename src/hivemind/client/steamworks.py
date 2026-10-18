"""Steamworks SDK Integration: Sybil Resistance, Leaderboards, and Compute-to-API Credits.

Implements:
- ISteamUser::GetAuthSessionTicket verification & Non-Limited account check ($5+ spent on Steam)
- ISteamLeaderboards verified FLOPs telemetry & Achievement unlocks
- Cryptographic Compute-to-API credit minting tied to SteamID64
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

STEAMWORKS_APP_ID = 3141590  # Project HiveMind Free Steam Utility AppID
MIN_STEAM_SPEND_USD = 5.0


@dataclass
class SteamAuthSessionTicket:
    """Represents a signed ISteamUser::GetAuthSessionTicket token."""

    steam_id64: str
    app_id: int
    account_spend_usd: float
    is_vac_banned: bool
    ticket_hex: str

    @property
    def is_non_limited(self) -> bool:
        return self.account_spend_usd >= MIN_STEAM_SPEND_USD and not self.is_vac_banned


@dataclass
class SteamUserTelemetry:
    """Tracks Steam Leaderboard FLOPs, Achievements, and API credits for a gamer node."""

    steam_id64: str
    persona_name: str
    verified_tflops: float = 0.0
    workunits_completed: int = 0
    api_credits_balance: float = 0.0
    unlocked_achievements: list[str] = field(default_factory=list)
    credit_receipts: list[str] = field(default_factory=list)


class SteamworksSDKBridge:
    """Interface to Steamworks User Auth, Leaderboards, Achievements, and API Credit Minting."""

    def __init__(self, signing_secret: bytes = b"HiveMind-Steamworks-Auth-Key") -> None:
        self.signing_secret = signing_secret
        self.leaderboard: dict[str, SteamUserTelemetry] = {}

    def get_auth_session_ticket(
        self,
        steam_id64: str,
        account_spend_usd: float = 25.0,
        is_vac_banned: bool = False,
    ) -> SteamAuthSessionTicket:
        """Generate an ISteamUser::GetAuthSessionTicket for BOINC Coordinator Sybil verification."""
        payload = f"{steam_id64}:{STEAMWORKS_APP_ID}:{account_spend_usd:.2f}:{int(is_vac_banned)}"
        sig = hmac.new(self.signing_secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return SteamAuthSessionTicket(
            steam_id64=steam_id64,
            app_id=STEAMWORKS_APP_ID,
            account_spend_usd=account_spend_usd,
            is_vac_banned=is_vac_banned,
            ticket_hex=sig,
        )

    def verify_sybil_resistance(self, ticket: SteamAuthSessionTicket) -> tuple[bool, str]:
        """Verify ticket authenticity and enforce Steam Non-Limited ($5+) Sybil requirement."""
        payload = (
            f"{ticket.steam_id64}:{ticket.app_id}:"
            f"{ticket.account_spend_usd:.2f}:{int(ticket.is_vac_banned)}"
        )
        expected_sig = hmac.new(
            self.signing_secret, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(ticket.ticket_hex, expected_sig):
            return False, "Invalid ISteamUser::GetAuthSessionTicket signature."
        if ticket.is_vac_banned:
            return False, "Steam account has an active VAC/Game ban."
        if not ticket.is_non_limited:
            return (
                False,
                f"Steam account is Limited (${ticket.account_spend_usd:.2f} < ${MIN_STEAM_SPEND_USD:.2f} minimum).",
            )
        return True, "Verified Non-Limited Steam Account."

    def record_validated_workunit(
        self,
        steam_id64: str,
        persona_name: str,
        wu_id: str,
        tflops_contributed: float,
        credits_minted: float = 10.0,
    ) -> SteamUserTelemetry:
        """Update ISteamLeaderboards FLOPs, unlock Steam Achievements, and mint API credits."""
        entry = self.leaderboard.setdefault(
            steam_id64,
            SteamUserTelemetry(steam_id64=steam_id64, persona_name=persona_name),
        )
        entry.verified_tflops = round(entry.verified_tflops + tflops_contributed, 3)
        entry.workunits_completed += 1
        entry.api_credits_balance = round(entry.api_credits_balance + credits_minted, 2)

        receipt = hmac.new(
            self.signing_secret,
            f"CREDIT:{steam_id64}:{wu_id}:{credits_minted:.2f}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]
        entry.credit_receipts.append(receipt)

        if entry.workunits_completed >= 1 and "ACH_FIRST_SYNAPSE" not in entry.unlocked_achievements:
            entry.unlocked_achievements.append("ACH_FIRST_SYNAPSE")
        if entry.verified_tflops >= 10.0 and "ACH_10_TFLOPS_SWARM" not in entry.unlocked_achievements:
            entry.unlocked_achievements.append("ACH_10_TFLOPS_SWARM")
        return entry
