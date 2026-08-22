"""Optional single-token authentication for the API surface.

Production stance (open-source, self-host product):
  * No MILIMO_AUTH_TOKEN set  → API is open (localhost-first dev default).
  * Token set                 → every protected route requires
                                `Authorization: Bearer <token>`
                                (EventSource clients may use `?auth=<token>`
                                 because EventSource cannot set headers).
Static media (/audio, /covers) and health/docs stay reachable so browsers can
load artwork and operators can reach diagnostics.
"""
from __future__ import annotations

import os
from typing import List

from fastapi import HTTPException, Request


EXEMPT_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json", "/audio", "/covers")


def get_auth_token() -> str:
    return (os.environ.get("MILIMO_AUTH_TOKEN") or "").strip()


async def require_auth(request: Request) -> None:
    token = get_auth_token()
    if not token:
        return  # open mode (localhost-first default)

    path = request.url.path
    for prefix in EXEMPT_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return

    header = request.headers.get("Authorization", "")
    query_token = request.query_params.get("auth", "")
    if header == f"Bearer {token}" or (query_token and query_token == token):
        return

    raise HTTPException(
        status_code=401,
        detail={"error": {"code": "unauthorized",
                          "message": "Missing or invalid bearer token. Set Authorization: Bearer <MILIMO_AUTH_TOKEN>."}},
    )
