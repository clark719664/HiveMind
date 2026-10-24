"""Verification of Google ID tokens ("Sign in with Google"): the generic OIDC engine in ``oidc_auth`` with Google's settings.

The checks (RS256 only, signature, issuer, audience, expiry, freshness, verified e-mail, single use) and their negative
tests are described in ``oidc_auth``. Nothing about the user except the opaque ``sub`` is kept; the e-mail is never stored.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import httpx

from hivemind.router import oidc_auth
from hivemind.router.oidc_auth import (
    DEFAULT_LEEWAY_S,
    DEFAULT_MAX_AGE_S,
    JwksFetcher,
    OidcAuthError,
    OidcIdentity,
    OidcIdTokenVerifier,
    OidcProvider,
)

GOOGLE_JWKS_URL: str = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS: frozenset[str] = frozenset({"https://accounts.google.com", "accounts.google.com"})
CLIENT_ID_RE = re.compile(r"^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$")

GoogleAuthError = OidcAuthError
GoogleIdentity = OidcIdentity


def fetch_google_jwks(client: httpx.Client | None = None):  # type: ignore[no-untyped-def]
    """Download Google's current signing keys (cache lifetime from ``Cache-Control``, clamped)."""
    return oidc_auth.fetch_jwks(GOOGLE_JWKS_URL, client)


class GoogleIdTokenVerifier(OidcIdTokenVerifier):
    """Verify Google ID tokens for one OAuth client id. Thread-safe; ``clock`` and ``jwks_fetcher`` are injectable."""

    def __init__(
        self,
        client_id: str,
        *,
        jwks_fetcher: JwksFetcher = fetch_google_jwks,
        clock: Callable[[], float] = time.time,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        leeway_s: float = DEFAULT_LEEWAY_S,
    ) -> None:
        if not CLIENT_ID_RE.match(client_id):
            raise ValueError("client_id must look like 123456789-abc.apps.googleusercontent.com")
        provider = OidcProvider("google", GOOGLE_ISSUERS, GOOGLE_JWKS_URL, client_id)
        super().__init__(provider, jwks_fetcher=jwks_fetcher, clock=clock, max_age_s=max_age_s, leeway_s=leeway_s)
