"""
Canonical id enforcement at the persistence boundary (plan 2.8).

A `field_validator` only runs on pydantic construction; assigning
`release.profile_id = "…-…"` on an ORM instance bypasses it, and so do direct
relationship writes. Foreign keys compare raw strings, so one hyphenated value in
any code path (or test fixture) is an IntegrityError at flush time.

A `before_flush` listener normalises every UUID-shaped FK attribute on every
dirty/new instance, which makes the 32-hex form the only thing that can be stored
no matter who writes it.
"""

from __future__ import annotations

import logging

from sqlalchemy import event
from sqlalchemy.orm import Session

logger = logging.getLogger("milimo.ids")

# Only association columns that participate in a foreign key.
FK_ATTRIBUTES = (
    "project_id",
    "session_id",
    "release_id",
    "artist_profile_id",
    "profile_id",
    "job_id",
    "parent_job_id",
    "active_job_id",
    "voice_profile_id",
)

_installed = False


def _canonical(value):
    if not isinstance(value, str) or not value:
        return value
    if "/" in value or "?" in value:
        return value
    try:
        from uuid import UUID

        return UUID(value).hex
    except (ValueError, AttributeError, TypeError):
        return value


def install() -> None:
    """Register the listener once per process."""
    global _installed
    if _installed:
        return

    @event.listens_for(Session, "before_flush")
    def _canonicalise_fk_values(session, flush_context, instances):  # noqa: ARG001
        for obj in list(session.new) + list(session.dirty):
            for attr in FK_ATTRIBUTES:
                if not hasattr(obj, attr):
                    continue
                current = getattr(obj, attr)
                fixed = _canonical(current)
                if fixed != current:
                    try:
                        setattr(obj, attr, fixed)
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("Could not canonicalise %s.%s", type(obj).__name__, attr)

    _installed = True


def canonical_id(value: object) -> "str | None":
    """Return the 32-hex form for anything id-shaped, or None when unusable."""
    from uuid import UUID

    if value is None:
        return None
    if isinstance(value, UUID):
        return value.hex
    text = str(value).strip()
    if not text or "/" in text or "?" in text:
        return None
    try:
        return UUID(text).hex
    except (ValueError, AttributeError, TypeError):
        return None


def canonical_or_raw(value: object) -> "str | None":
    """Canonical when possible, otherwise the raw string (never lossy)."""
    canon = canonical_id(value)
    if canon is not None:
        return canon
    if value is None:
        return None
    text = str(value).strip()
    return text or None
