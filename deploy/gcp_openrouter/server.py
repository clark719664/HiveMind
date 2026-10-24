"""Cloud Run entrypoint for the HiveMind OpenRouter gateway.

Configuration (environment):
    HIVEMIND_ROUTER_SECRET          required; HMAC secret for API keys and account ids (>= 32 bytes).
                                    Mount from Secret Manager, never bake into the image.
    HIVEMIND_ROUTER_DB              SQLite path, default /tmp/router.db. On Cloud Run /tmp is
                                    ephemeral: ledger and accounts vanish on instance restart.
                                    Production needs Cloud SQL or Firestore (task HM-S13-03).
    HIVEMIND_ROUTER_ADMIN_KEY_HASH  optional; HMAC hex matching the admin key, enables admin grants.
    HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS
                                    reverse proxies in front of the app that append to X-Forwarded-For (Cloud Run: 1).
                                    The registration limiter keys on the entry that many hops from the right, which a
                                    client cannot forge. 0 (default) keys on the TCP peer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

DEFAULT_DB_PATH: str = "/tmp/router.db"
MEMORY_DB: str = ":memory:"


def build_app(env: Mapping[str, str]) -> FastAPI:
    """Construct the gateway app from environment variables. Fails closed when the secret is absent."""
    secret_text = env.get("HIVEMIND_ROUTER_SECRET", "")
    if not secret_text:
        raise RuntimeError("HIVEMIND_ROUTER_SECRET is not set; refusing to start without it")
    db_setting = env.get("HIVEMIND_ROUTER_DB", DEFAULT_DB_PATH)
    db_target: Path | str = db_setting if db_setting == MEMORY_DB else Path(db_setting)
    admin_key_hash = env.get("HIVEMIND_ROUTER_ADMIN_KEY_HASH", "") or None
    store = RouterStore(db_target, secret_text.encode("utf-8"))
    hops = int(env.get("HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS", "0") or "0")
    if hops < 0:
        raise RuntimeError("HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS must be >= 0")
    return create_router_app(store, admin_key_hash=admin_key_hash, trusted_proxy_hops=hops)


app: FastAPI = build_app(os.environ)
