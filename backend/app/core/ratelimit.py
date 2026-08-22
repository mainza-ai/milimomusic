"""Lightweight sliding-window rate limiter for expensive routes.

No external dependency (slowapi not yet justified): an in-process per-IP
window is correct for the self-host single-user product model. If a
multi-instance deployment ever arrives, swap this module for a Redis-backed
implementation behind the same middleware signature.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from fastapi import Request
from starlette.responses import JSONResponse


# Route prefixes that cost real money / GPU seconds.
PROTECTED_PREFIXES = (
    "/agents/",
    "/generate/",
    "/transcribe/upload",
    "/mastering/",
    "/models/download",
    "/producer/",
)

_window: Deque[Tuple[float, str]] = deque()
_hits: Dict[str, Deque[float]] = defaultdict(deque)


def _limit() -> int:
    try:
        return max(1, int(os.environ.get("MILIMO_RATE_LIMIT_PER_MIN", "120")))
    except ValueError:
        return 120


def _is_protected(path: str) -> bool:
    return path.startswith(PROTECTED_PREFIXES)


async def enforce_rate_limit(request: Request, call_next):
    if not _is_protected(request.url.path) or request.method == "OPTIONS":
        return await call_next(request)

    now = time.monotonic()
    key = f"{request.client.host if request.client else 'unknown'}:{request.url.path.rsplit('/', 1)[0]}"
    q = _hits[key]
    window = 60.0
    while q and now - q[0] > window:
        q.popleft()

    limit = _limit()
    if len(q) >= limit:
        # Middleware sits OUTSIDE the exception handlers — return the envelope
        # directly instead of raising HTTPException.
        return JSONResponse(
            status_code=429,
            content={"error": {"code": "rate_limited",
                               "message": f"Rate limit exceeded ({limit}/min for this route group). Slow down."}},
        )
    q.append(now)
    # opportunistic global prune so the dict cannot grow unbounded
    if len(_hits) > 4096:
        stale = [k for k, v in _hits.items() if not v or now - v[-1] > window]
        for k in stale:
            _hits.pop(k, None)

    return await call_next(request)
