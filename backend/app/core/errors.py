"""
One error envelope.

The audit found three incompatible shapes on the wire:

    {"detail": "Job not found"}                     (HTTPException with a string)
    {"detail": {"error": {"code": ..., ...}}}       (HTTPException with a dict)
    {"detail": [{"loc": ..., "msg": ...}]}          (FastAPI validation errors)
    {"error": {"code": ..., "message": ...}}        (global handler)

A client had to branch on the shape before it could read the message.

`install_error_handlers()` makes every error emit **both** keys:

    {"detail": <the original payload, unchanged>,
     "error":  {"code": "...", "message": "...", "details": ...}}

Keeping `detail` means existing UI code keeps working, while `error.code` is now
always present for machine handling. New code should read `error`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("milimo.errors")

# Stable machine-readable code per status, used when a handler supplies only prose.
STATUS_CODES: Dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    501: "not_implemented",
    502: "upstream_error",
    503: "unavailable",
    507: "insufficient_storage",
}


class AppError(HTTPException):
    """Application error carrying a machine-readable code.

    Raise this instead of HTTPException when the client should branch on the
    failure kind rather than parse a message.
    """

    def __init__(self, status_code: int, code: str, message: str, details: Optional[Any] = None):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.details = details


def _cors_headers(request: Request) -> Dict[str, str]:
    """Mirror the app's CORS policy on error responses."""
    origin = request.headers.get("origin")
    if not origin:
        return {}
    import re

    if re.match(r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$", origin):
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    return {}


def _normalise(detail: Any, status_code: int) -> Dict[str, Any]:
    """Build `{code, message, details}` from whatever a handler raised."""
    fallback_code = STATUS_CODES.get(status_code, "error")

    if isinstance(detail, dict):
        nested = detail.get("error")
        if isinstance(nested, dict):
            return {
                "code": str(nested.get("code") or fallback_code),
                "message": str(nested.get("message") or nested.get("detail") or fallback_code),
                "details": nested.get("details"),
            }
        # FastAPI validation errors arrive as a list under "detail"; a plain dict
        # here is usually an ad-hoc payload — surface it as details.
        return {
            "code": str(detail.get("code") or fallback_code),
            "message": str(detail.get("message") or detail.get("detail") or fallback_code),
            "details": detail,
        }

    if isinstance(detail, list):
        first = detail[0] if detail else {}
        loc = ".".join(str(p) for p in (first.get("loc") or []) if p not in ("body", "query", "path"))
        message = str(first.get("msg") or "Invalid request")
        return {
            "code": fallback_code,
            "message": f"{loc}: {message}" if loc else message,
            "details": detail,
        }

    return {"code": fallback_code, "message": str(detail or fallback_code), "details": None}


def install_error_handlers(app: FastAPI) -> None:
    """Register the envelope handlers. Call once, after the app is created."""

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        payload = _normalise(exc.detail, exc.status_code)
        code = getattr(exc, "code", None) or payload["code"]
        body = {
            "detail": exc.detail,
            "error": {
                "code": code,
                "message": payload["message"],
                "details": getattr(exc, "details", None) or payload["details"],
            },
        }
        headers = dict(getattr(exc, "headers", None) or {})
        headers.update(_cors_headers(request))
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        payload = _normalise(exc.errors(), 422)
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "error": payload},
            headers=_cors_headers(request),
        )
