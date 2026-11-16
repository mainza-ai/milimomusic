"""
Observability primitives: request correlation, structured access logging, and a
rotating debug log.

The audit found the opposite of observability:

* `uvicorn.access` was explicitly silenced (`main.py`), so nothing recorded what the
  API served.
* no request id existed (`grep X-Request-ID` → zero hits), so a stack trace could
  not be tied to a request.
* `print()` was used in production paths for migration failures.
* `ai_debug.log` accumulated the **full lyrics and prompts** of every Co-Writer turn
  in the working directory, unbounded and unrotated, with no consent gate.

This module fixes all four without adding a dependency.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("milimo.access")

_request_id: ContextVar[str] = ContextVar("milimo_request_id", default="-")

# Paths whose access lines are pure noise (health polling, SSE heartbeats).
QUIET_PATHS = ("/health", "/events", "/favicon.ico")
# Paths whose bodies must never be logged.
SENSITIVE_PREFIXES = ("/config/llm", "/voice/profiles", "/auth")


def current_request_id() -> str:
    return _request_id.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, echo it, and emit one structured access line."""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        rid = incoming if incoming and len(incoming) <= 64 else uuid.uuid4().hex[:12]
        token = _request_id.set(rid)
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.exception(
                "req=%s %s %s -> 500 in %dms", rid, request.method, request.url.path, elapsed_ms
            )
            raise
        finally:
            _request_id.reset(token)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        response.headers["X-Request-ID"] = rid

        path = request.url.path
        if not any(path.startswith(p) for p in QUIET_PATHS):
            client = request.client.host if request.client else "-"
            logger.info(
                "req=%s %s %s -> %d in %dms client=%s",
                rid, request.method, path, response.status_code, elapsed_ms, client,
            )
        return response


def configure_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """One log configuration for the process.

    File handlers rotate: an unbounded `backend.log` on a long-running install
    eventually fills the same disk the media lives on.
    """
    root = logging.getLogger()
    if getattr(root, "_milimo_configured", False):
        return

    fmt = logging.Formatter(
        "%(asctime)s - [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, "backend.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError:
            pass

    root.setLevel(level)
    root._milimo_configured = True  # type: ignore[attr-defined]


def debug_enabled() -> bool:
    """Verbose prompt/lyric dumps are opt-in (they contain user content)."""
    return (os.environ.get("MILIMO_DEBUG_PROMPTS") or "").strip().lower() in {"1", "true", "yes", "on"}


def debug_log(message: str) -> None:
    """Write a verbose dump to the rotating debug log, only when opted in."""
    if not debug_enabled():
        return
    try:
        handler = logging.getLogger("milimo.debug")
        if not handler.handlers:
            from app.core.paths import get_data_dir

            log_path = os.path.join(str(get_data_dir()), "logs", "ai_debug.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            h = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            handler.addHandler(h)
            handler.setLevel(logging.DEBUG)
            handler.propagate = False
        handler.debug(message)
    except OSError:
        pass
