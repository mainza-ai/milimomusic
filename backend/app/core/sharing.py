"""
Ownership scoping and signed share links (decision D2).

D2 is "single-user, multi-tenant-ready": there is one owner today, but every
user-owned row carries an `owner_id` (default `"local"`) and all reads go through
`scoped()`, so a hosted mode later is a migration rather than a rewrite.

Sharing is the part users actually ask for. `mint_share_token()` produces a
signed, expiring, read-only token for one track; `verify_share_token()` checks it
without needing a database round-trip. Media served through a share token never
requires the API token, which is exactly what makes a link shareable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional

LOCAL_OWNER_ID = "local"
DEFAULT_SHARE_TTL_HOURS = 72


def owner_id() -> str:
    """The owner all local rows belong to.

    A hosted deployment would derive this from the authenticated principal; the
    single-user install has exactly one.
    """
    return (os.environ.get("MILIMO_OWNER_ID") or LOCAL_OWNER_ID).strip() or LOCAL_OWNER_ID


def scoped(statement, model):
    """Apply the owner filter to a SELECT for `model` when it has `owner_id`.

    Used as `scoped(select(Job), Job)` so every listing is already tenant-safe if
    the column exists, and unchanged if it does not (older databases).
    """
    column = getattr(model, "owner_id", None)
    if column is None:
        return statement
    try:
        return statement.where(column == owner_id())
    except Exception:  # pragma: no cover - defensive
        return statement


def _secret() -> bytes:
    """HMAC key for share tokens.

    Defaults to a stable derivation so links survive a restart; set
    `MILIMO_SHARE_SECRET` to rotate them independently of the auth token.
    """
    raw = (os.environ.get("MILIMO_SHARE_SECRET") or "").strip()
    if raw:
        return raw.encode("utf-8")
    seed = (os.environ.get("MILIMO_AUTH_TOKEN") or "").strip() or "milimo-local-share"
    return hashlib.sha256(f"share:{seed}".encode("utf-8")).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def mint_share_token(job_id: str, *, ttl_hours: Optional[float] = None, kind: str = "track") -> Dict[str, Any]:
    """Create an expiring, read-only token for one track."""
    ttl = DEFAULT_SHARE_TTL_HOURS if ttl_hours is None else float(ttl_hours)
    expires_at = int(time.time() + max(60, ttl * 3600))
    payload = {"j": str(job_id), "k": kind, "e": expires_at, "n": secrets.token_hex(4)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64(hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest())
    return {
        "token": f"{body}.{signature}",
        "expires_at": expires_at,
        "url": f"/share/{body}.{signature}",
        "job_id": str(job_id),
    }


def verify_share_token(token: str) -> Optional[Dict[str, Any]]:
    """Return the payload for a valid, unexpired token — otherwise None."""
    if not token or "." not in token:
        return None
    body, _, signature = token.partition(".")
    if not body or not signature:
        return None
    expected = _b64(hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        if int(payload.get("e", 0)) < time.time():
            return None
    except (TypeError, ValueError):
        return None
    return payload
