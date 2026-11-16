"""Lightweight sliding-window rate limiter for expensive routes.

No external dependency (slowapi not yet justified): an in-process per-client
window is correct for the self-host single-user product model. If a
multi-instance deployment ever arrives, swap this module for a Redis-backed
implementation behind the same middleware signature.

Two defects in the first version:

1. **It keyed on the request path's parent segment**, so `/jobs/<uuid>/voice-convert`
   got a fresh bucket per job id — the limiter was defeated by varying the id.
   Keys are now the *route template* (`/jobs/{job_id}/voice-convert`), which is
   what "this endpoint" actually means.
2. **The most expensive routes were not protected at all**: separation, voice
   conversion, video render, and training all cost more than the ones that were.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import Request
from starlette.responses import JSONResponse

# Route prefixes that cost real money / GPU seconds.
PROTECTED_PREFIXES = (
    "/agents/",
    "/generate/",
    "/transcribe/",
    "/mastering/",
    "/models/download",
    "/producer/",
    "/tracks/",          # separate, midi commit, extend, commit-vocal
    "/jobs/",            # voice-convert, inpaint, generate-cover
    "/videos/",          # render, render-advanced, keyframes, retake
    "/images/",          # diffusion generation
    "/api/training/",
    "/api/editor/",      # timeline render
    "/system/flush",
)

# Per-route costs, in "requests". A full song render is not 1x a subtitle nudge.
ROUTE_COSTS: Dict[str, int] = {
    "/generate/music": 4,
    "/generate/cover": 4,
    "/videos/render-advanced/{job_id}": 6,
    "/videos/render-lyric/{job_id}": 4,
    "/videos/keyframes/{job_id}": 4,
    "/jobs/{job_id}/voice-convert": 3,
    "/tracks/{job_id}/separate": 3,
    "/transcribe/upload": 3,
    "/mastering/match/{job_id}": 2,
    "/models/download": 2,
    "/api/editor/projects/{project_id}/compile": 4,
}

_hits: Dict[str, Deque[float]] = defaultdict(deque)

WINDOW_SECONDS = 60.0


def _limit() -> int:
    try:
        return max(1, int(os.environ.get("MILIMO_RATE_LIMIT_PER_MIN", "120")))
    except ValueError:
        return 120


def _is_protected(path: str) -> bool:
    return path.startswith(PROTECTED_PREFIXES)


def route_template(request: Request) -> str:
    """The matched route pattern, or the raw path when routing has not run.

    `/jobs/abc-123/voice-convert` → `/jobs/{job_id}/voice-convert`, so every job id
    shares one bucket instead of minting a new one.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    # `Starlette` populates scope["route"] during routing, which happens *after*
    # this middleware for pure ASGI middleware; base middleware runs inside it.
    return request.url.path


def _client_id(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    # Honour a single proxy hop when the operator declares one.
    if os.environ.get("MILIMO_TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            host = forwarded.split(",")[0].strip() or host
    return host


def cost_for(template: str) -> int:
    return ROUTE_COSTS.get(template, 1)


def check(request: Request) -> Optional[Tuple[int, str]]:
    """Record a hit. Returns (retry_after_seconds, message) when the caller is over."""
    template = route_template(request)
    now = time.monotonic()
    key = f"{_client_id(request)}:{template}"
    q = _hits[key]

    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()

    limit = _limit()
    if sum(1 for _ in q) >= limit:
        retry_after = int(WINDOW_SECONDS - (now - q[0])) + 1
        return retry_after, (
            f"Rate limit exceeded for {template} ({limit} units/min). "
            f"Retry in {retry_after}s."
        )

    for _ in range(cost_for(template)):
        q.append(now)

    if len(_hits) > 4096:
        stale = [k for k, v in _hits.items() if not v or now - v[-1] > WINDOW_SECONDS]
        for k in stale:
            _hits.pop(k, None)

    return None


async def enforce_rate_limit(request: Request, call_next):
    if not _is_protected(request.url.path) or request.method == "OPTIONS":
        return await call_next(request)

    blocked = check(request)
    if blocked:
        retry_after, message = blocked
        # Middleware sits OUTSIDE the exception handlers — return the envelope
        # directly instead of raising HTTPException.
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"error": {"code": "rate_limited", "message": message}},
        )
    return await call_next(request)


def reset() -> None:
    """Test hook: forget every recorded hit."""
    _hits.clear()
