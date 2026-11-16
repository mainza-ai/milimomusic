"""
Idempotency for GPU-spending requests.

A retried `POST /generate/music` (network blip, double-click, SSE reconnect logic)
used to create a second Job and a second full generation — real GPU minutes and,
on a metered provider, real money.

`Idempotency-Key` is stored with the response the first call produced (in the same
SQLite database, so it survives restarts); a repeat with the same key replays the
stored response and is marked `idempotent_replay: true`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlmodel import Session

logger = logging.getLogger("milimo.idempotency")

DEFAULT_TTL_HOURS = 24
_lock = threading.Lock()


def _fingerprint(payload: Any) -> str:
    """Stable hash of the request body so a key reused for *different* work is caught."""
    try:
        blob = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = str(payload)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class IdempotencyStore:
    """SQLite-backed key → (fingerprint, response) map with a TTL."""

    def __init__(self, engine, ttl_hours: int = DEFAULT_TTL_HOURS):
        self.engine = engine
        self.ttl = timedelta(hours=ttl_hours)
        self._ready = False

    def ensure_table(self) -> None:
        if self._ready:
            return
        with _lock:
            with Session(self.engine) as session:
                session.exec(text(
                    """
                    CREATE TABLE IF NOT EXISTS idempotency_key (
                        key TEXT PRIMARY KEY,
                        fingerprint TEXT NOT NULL,
                        response_json TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                ))
                session.commit()
            self._ready = True

    def lookup(self, key: str, payload: Any) -> Optional[Dict[str, Any]]:
        """Return the stored response for `key`, or None.

        A key reused with a *different* body returns `{"__conflict__": True}` so the
        caller can answer 409 instead of replaying an unrelated response.
        """
        if not key:
            return None
        self.ensure_table()
        fingerprint = _fingerprint(payload)
        with Session(self.engine) as session:
            row = session.execute(
                text("SELECT fingerprint, response_json, created_at FROM idempotency_key WHERE key = :k"),
                {"k": key},
            ).first()
            if not row:
                return None
            created = row[2]
            try:
                created_dt = datetime.fromisoformat(created)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                expired = datetime.now(timezone.utc) - created_dt > self.ttl
            except (TypeError, ValueError):
                expired = False
            if expired:
                session.execute(text("DELETE FROM idempotency_key WHERE key = :k"), {"k": key})
                session.commit()
                return None

            if row[0] != fingerprint:
                return {"__conflict__": True}
            try:
                return json.loads(row[1]) if row[1] else {"__conflict__": True}
            except (TypeError, ValueError):
                return {"__conflict__": True}

    def remember(self, key: str, payload: Any, response: Dict[str, Any]) -> None:
        if not key:
            return
        self.ensure_table()
        with Session(self.engine) as session:
            session.execute(
                text(
                    "INSERT OR REPLACE INTO idempotency_key "
                    "(key, fingerprint, response_json, created_at) VALUES (:k, :f, :r, :c)"
                ),
                {
                    "k": key,
                    "f": _fingerprint(payload),
                    "r": json.dumps(response, default=str),
                    "c": datetime.now(timezone.utc).isoformat(),
                },
            )
            session.commit()

    def prune(self, older_than_hours: int = DEFAULT_TTL_HOURS * 7) -> int:
        self.ensure_table()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()
        with Session(self.engine) as session:
            result = session.execute(
                text("DELETE FROM idempotency_key WHERE created_at < :c"), {"c": cutoff}
            )
            session.commit()
            return getattr(result, "rowcount", 0) or 0
