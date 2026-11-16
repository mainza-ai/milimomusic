"""
Bounded retries for GPU work (plan 2.2).

A transient OOM, a briefly-unavailable provider, or a full disk used to fail a job
permanently on the first attempt — the user had to recreate it by hand (and could
not reproduce the seed if the producer had rewritten the prompt).

`with_retries` retries only errors classified as transient, with exponential
backoff and jitter, and stops immediately on cancellation or a terminal error.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger("milimo.retry")

T = TypeVar("T")

# Substrings that indicate "try again", not "this will never work".
TRANSIENT_MARKERS = (
    "out of memory",
    "oom",
    "cuda error",
    "mps",  # MPS backend errors are usually allocation/stream hiccups
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "no space left",
    "resource temporarily unavailable",
    "try again",
)

# Substrings that will never succeed on retry.
TERMINAL_MARKERS = (
    "invalid api key",
    "unauthorized",
    "401",
    "403",
    "not found",
    "unsupported",
    "not implemented",
    "malformed",
    "invalid prompt",
    "cancelled",
)


def is_transient(exc: BaseException) -> bool:
    """Classify an error as worth retrying."""
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(exc, (NotImplementedError, ValueError, TypeError, KeyError)):
        return False
    if isinstance(exc, MemoryError):
        return True

    message = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in message for marker in TERMINAL_MARKERS):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any(marker in message for marker in TRANSIENT_MARKERS)


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    base_delay: float = 2.0,
    max_delay: float = 20.0,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    cancel_event=None,
) -> T:
    """Call `fn` until it succeeds, the attempts run out, or the error is terminal.

    `attempts` counts *total* tries (default 2 = one retry), matching the Job's
    `max_attempts` column so the DB records what actually happened.
    """
    last: Optional[BaseException] = None
    total = max(1, int(attempts))

    for attempt in range(1, total + 1):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise asyncio.CancelledError("Cancelled before attempt")
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not is_transient(exc) or attempt >= total:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter
            logger.warning(
                "Transient failure on attempt %d/%d (%s); retrying in %.1fs",
                attempt, total, exc, delay,
            )
            if on_retry:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:  # pragma: no cover - telemetry must not fail work
                    pass
            await asyncio.sleep(delay)

    assert last is not None
    raise last
