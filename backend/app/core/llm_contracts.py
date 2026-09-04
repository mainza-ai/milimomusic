"""Neutral LLM contracts shared by adapters, the agent runtime, and the API.

This module sits at the BOTTOM of the dependency graph (imports nothing from
`app.*`) so both `services/llm_service.py` (providers) and `agents/runtime`
(orchestration) can depend on it without circular imports.

It fixes two of the foundational gaps from the agent-foundation audit:
  G7 — untyped errors: every provider failure is classified into a typed
       exception so callers (retry policy, HTTP surface) can react by KIND
       instead of string-matching.
  G5 — usage capture: providers return a `LLMResult` envelope carrying token
       accounting alongside content, instead of dropping it on the floor.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Typed error hierarchy
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base class for all classified LLM failures."""


class LLMAuthError(LLMError):
    """Credentials missing, invalid, or forbidden. Do not retry this provider."""


class LLMQuotaError(LLMError):
    """Rate-limited or out of credit. Transient — failover/backoff applies."""


class LLMTimeoutError(LLMError):
    """The upstream call timed out. Transient."""


class LLMBadModelError(LLMError):
    """The requested model id is unknown/deprecated on this provider. Skip."""


class LLMParseError(LLMError):
    """The provider answered but the payload could not be parsed/validated."""


class LLMUpstreamError(LLMError):
    """Anything else from the provider (5xx, malformed envelopes, ...)."""


class AllProvidersFailedError(LLMError):
    """Every profile in the resilience chain failed. Carries the attempts."""

    def __init__(self, message: str, attempts: Optional[List[Dict]] = None):
        super().__init__(message)
        self.attempts = attempts or []


_STATUS_AUTH = {401, 403}
_STATUS_QUOTA = {402, 429}
_QUOTA_WORDS = ("quota", "rate limit", "too many requests", "insufficient")
_TIMEOUT_WORDS = ("timeout", "timed out", "read timed out", "connection reset")
_BADMODEL_WORDS = ("model_not_found", "no such model", "model not found",
                   "does not exist", "not a valid model", "deprecated model")


def classify_llm_error(provider: str, exc: BaseException) -> LLMError:
    """Map an arbitrary provider exception onto the typed hierarchy.

    Classification order matters: status codes first (they appear inside SDK
    messages), then explicit phrases, then transport-timeout types.
    """
    text = str(exc)
    low = text.lower()

    status: Optional[int] = None
    m = re.search(r"\b(4\d\d|5\d\d)\b", text)
    if m:
        try:
            status = int(m.group(1))
        except ValueError:
            status = None

    # Transport timeout types raise before any status exists.
    etype = type(exc).__name__.lower()
    if "timeout" in etype or isinstance(getattr(exc, "request", None), type(None)) and "timeout" in low:
        return LLMTimeoutError(f"{provider}: {text}")

    if status in _STATUS_AUTH or "invalid api key" in low or "unauthorized" in low or "forbidden" in low:
        return LLMAuthError(f"{provider}: {text}")
    if status in _STATUS_QUOTA or any(w in low for w in _QUOTA_WORDS):
        return LLMQuotaError(f"{provider}: {text}")
    if status == 404 or any(w in low for w in _BADMODEL_WORDS):
        return LLMBadModelError(f"{provider}: {text}")
    if status is None and ("timeout" in low or any(w in low for w in _TIMEOUT_WORDS)):
        return LLMTimeoutError(f"{provider}: {text}")
    if status is not None and status >= 500:
        return LLMUpstreamError(f"{provider}: {text}")

    # JSON/parse errors raised by our own extraction keep their identity.
    if isinstance(exc, (json.JSONDecodeError, KeyError)):
        return LLMParseError(f"{provider}: {text}")

    return LLMUpstreamError(f"{provider}: {text}")


# ---------------------------------------------------------------------------
# Result envelope with usage accounting
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    """One completed provider call — content PLUS the usage data the old layer dropped."""

    content: str
    provider: str
    model: str
    latency_ms: int = 0
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def tokens_in(self) -> int:
        return int(self.usage.get("prompt_tokens", 0) or 0)

    @property
    def tokens_out(self) -> int:
        return int(self.usage.get("completion_tokens", 0) or 0)


def extract_json_object(text: str) -> Dict:
    """Extract the outermost JSON object from arbitrary model output.

    Tolerates markdown fences, prose preambles, and reasoning residue. Raises
    json.JSONDecodeError (classified as LLMParseError by the caller) when no
    object exists.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def now_ms() -> int:
    return int(time.time() * 1000)
