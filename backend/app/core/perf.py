"""Response caching and optional compression (performance phase).

Two findings shaped this module, one of them from measurement and one from research
that overturned an initial assumption:

* **ETag/304 is worth it on localhost**, but for CPU and client work, not bandwidth:
  a polled endpoint that returns 304 stops paying for the JSON parse and the React
  re-render. The tag must derive from **input state**, never from the finished body —
  hashing the body means the expensive work already happened.
* **Compression is not worth it on loopback.** gzip's break-even link speed for a
  JSON response is roughly 400–860 Mbit/s; a loopback socket is far above that, so
  compressing spends CPU to make the response arrive *later*. It stays available for
  a LAN or remote client (`MILIMO_COMPRESS=1`) and never touches streamed responses,
  because gzipping `text/event-stream` breaks its framing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Iterable, Optional

from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("milimo.perf")

#: Prefixes whose responses must never be compressed: SSE framing and ranged media.
STREAMING_PREFIXES: tuple = ("/events", "/audio/", "/covers/")


def compression_enabled() -> bool:
    """Compression is opt-in: it only helps when the client is not on loopback."""
    return os.environ.get("MILIMO_COMPRESS", "").strip().lower() in ("1", "true", "yes", "on")


class SelectiveGZipMiddleware:
    """GZip everything that is not a stream (opt-in, see `compression_enabled`)."""

    def __init__(self, app, *, minimum_size: int = 1024,
                 exclude_prefixes: Iterable[str] = STREAMING_PREFIXES):
        self.app = app
        self._gzip = GZipMiddleware(app, minimum_size=minimum_size)
        self._exclude = tuple(exclude_prefixes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if path.startswith(self._exclude):
            await self.app(scope, receive, send)
            return
        await self._gzip(scope, receive, send)


def state_token(*parts: Any) -> str:
    """Cheap cache key from input state (never from the serialised body)."""
    raw = "|".join(str(part) for part in parts if part is not None)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


def cached_json(
    request: Optional[Request],
    payload: Any,
    *,
    token: Optional[str] = None,
    max_age: int = 30,
    status_code: int = 200,
) -> Response:
    """Return `payload` as JSON with an ETag, honouring `If-None-Match`.

    Pass a `token` describing the inputs for anything expensive. Without one the body
    is hashed as a correctness fallback — it works, but it has already paid the cost
    the 304 was meant to avoid.
    """
    if token is not None:
        etag = '"' + token + '"'
    else:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        etag = '"' + hashlib.sha1(body).hexdigest()[:32] + '"'

    headers = {
        "ETag": etag,
        "Cache-Control": f"private, max-age={max_age}",
    }
    if request is not None:
        inm = request.headers.get("if-none-match")
        if inm and etag in [part.strip() for part in inm.split(",")]:
            return Response(status_code=304, headers=headers)
    return JSONResponse(content=payload, status_code=status_code, headers=headers)
