"""Single-token authentication for the API surface.

Production stance (open-source, self-host product):
  * No `MILIMO_AUTH_TOKEN` set → only **loopback** clients are served. A
    non-loopback request is refused with an actionable 403 unless the operator
    explicitly opts out with `MILIMO_ALLOW_ANONYMOUS=1`.
  * Token set → protected routes require `Authorization: Bearer <token>`.
    `?auth=` is honoured **only** on `/events` (EventSource cannot set
    headers); everywhere else it is rejected so the token cannot leak into
    access logs, browser history or `Referer`.

Enforcement lives in :class:`SecurityGate`, a pure-ASGI middleware, because
FastAPI/Starlette `app.mount()` creates independent sub-applications that never
run route dependencies — `/stems` and `/generated_audio` used to be reachable
with a token configured. Provider media (/audio, /covers, /stems,
/generated_audio) remains readable from loopback without a header so that
`<audio>` / `<img>` tags work; a token-authenticated remote deployment gets
signed media tickets in a later phase.
"""
from __future__ import annotations

import hmac
import os
from typing import Iterable, Tuple

from fastapi import HTTPException
from starlette.requests import Request

# Paths that never require the API token.
#   /share  — a signed, expiring link IS the credential for that one track.
EXEMPT_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json", "/share")

# Static media served by app.mount(); readable from loopback without a header.
MEDIA_PREFIXES = ("/audio", "/covers", "/stems", "/generated_audio")

# The only place a query-string token is accepted.
QUERY_TOKEN_PATHS = ("/events",)

_LOOPBACK_EXACT = {"::1", "localhost", "testclient", "127.0.0.1"}


def get_auth_token() -> str:
    return (os.environ.get("MILIMO_AUTH_TOKEN") or "").strip()


def allow_anonymous() -> bool:
    return (os.environ.get("MILIMO_ALLOW_ANONYMOUS") or "").strip().lower() in {"1", "true", "yes", "on"}


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in _LOOPBACK_EXACT:
        return True
    return host.startswith("127.")


def _match_prefix(path: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _query_token(scope) -> str:
    from urllib.parse import parse_qs

    raw = scope.get("query_string") or b""
    if not raw:
        return ""
    try:
        values = parse_qs(raw.decode("latin-1")).get("auth") or []
    except (UnicodeDecodeError, ValueError):
        return ""
    return values[0].strip() if values else ""


def evaluate_request(method: str, path: str, headers: dict, query_token: str, client_host: str | None) -> Tuple[bool, str]:
    """Decide whether a request may proceed. Returns ``(allowed, reason)``.

    Pure function so it can be unit-tested without an ASGI server.
    """
    if _match_prefix(path, EXEMPT_PREFIXES):
        return True, "exempt"

    token = get_auth_token()
    loopback = is_loopback(client_host)

    if not token:
        if loopback or allow_anonymous():
            return True, "anonymous-allowed"
        return False, (
            "This Milimo Music instance has no MILIMO_AUTH_TOKEN configured and is "
            "reachable from a non-loopback address. Set MILIMO_AUTH_TOKEN to enable "
            "remote access, or set MILIMO_ALLOW_ANONYMOUS=1 to serve an open instance "
            "on a trusted network."
        )

    provided = ""
    auth_header = headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        provided = auth_header[7:].strip()
    elif query_token and _match_prefix(path, QUERY_TOKEN_PATHS):
        provided = query_token

    if provided and hmac.compare_digest(provided, token):
        return True, "token"

    # Local <audio>/<img> tags cannot send an Authorization header.
    if loopback and _match_prefix(path, MEDIA_PREFIXES):
        return True, "loopback-media"

    return False, (
        "Missing or invalid bearer token. Send 'Authorization: Bearer <MILIMO_AUTH_TOKEN>'."
    )


class SecurityGate:
    """Pure-ASGI gate applied to the whole surface, mounts included."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "") or ""
        client = scope.get("client") or ("", 0)
        client_host = client[0] if client else None
        headers = {"authorization": _header(scope, b"authorization")}

        allowed, reason = evaluate_request(
            method=scope.get("method", "GET"),
            path=path,
            headers=headers,
            query_token=_query_token(scope),
            client_host=client_host,
        )
        if allowed:
            return await self.app(scope, receive, send)

        from starlette.responses import JSONResponse

        is_authn = get_auth_token() != ""
        response = JSONResponse(
            status_code=401 if is_authn else 403,
            # Same envelope shape the route dependency produces, so clients have
            # exactly one error path whether the gate or a route rejected them.
            content={"detail": {"error": {
                "code": "unauthorized" if is_authn else "anonymous_blocked",
                "message": reason,
            }}},
        )
        return await response(scope, receive, send)


async def require_auth(request: "Request") -> None:
    """Route-level dependency kept for per-route documentation; the real
    enforcement is :class:`SecurityGate`, which also covers mounts."""
    path = request.url.path
    client_host = request.client.host if request.client else None
    allowed, reason = evaluate_request(
        method=request.method,
        path=path,
        headers={"authorization": request.headers.get("Authorization", "")},
        query_token=request.query_params.get("auth", ""),
        client_host=client_host,
    )
    if not allowed:
        raise HTTPException(
            status_code=401 if get_auth_token() else 403,
            detail={"error": {"code": "unauthorized", "message": reason}},
        )

