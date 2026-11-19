"""Bounded conversation memory for agents (memory phase).

`RunContext.history` and the lyrics chat path both accept conversation history with no
budget: a long session grows until the prompt no longer fits the model's window, at
which point the request fails — or, worse, the provider silently truncates from the
top and the agent loses the instructions it was given first.

The fix is a **budgeted window**, not a smarter summariser: keep the most recent turns
that fit a character budget, always preserve a system/instruction message, and report
what was dropped so the caller can tell the user rather than pretend.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

#: Defaults are deliberately generous: local models commonly have 8–32 k windows, and
#: a chat turn is a few hundred characters.
DEFAULT_MAX_CHARS = 24_000
DEFAULT_MAX_MESSAGES = 24


def _limits() -> Tuple[int, int]:
    def _int(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, default)))
        except (TypeError, ValueError):
            return default

    return (
        _int("MILIMO_AGENT_HISTORY_MAX_CHARS", DEFAULT_MAX_CHARS),
        _int("MILIMO_AGENT_HISTORY_MAX_MESSAGES", DEFAULT_MAX_MESSAGES),
    )


def _content_length(message: Dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    return len(str(content or ""))


def trim_history(
    history: List[Dict[str, Any]] | None,
    *,
    max_chars: int | None = None,
    max_messages: int | None = None,
    keep_first: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return `(kept, report)` for a conversation window.

    Keeps the newest messages that fit both budgets. When `keep_first` is set and the
    first message is a system/instruction turn, it is always kept — dropping the agent
    instructions to save tokens is how a session starts behaving differently halfway
    through.

    `report` states what happened so the UI can say "earlier turns were dropped"
    instead of the model silently losing context.
    """
    messages = list(history or [])
    char_budget = DEFAULT_MAX_CHARS if max_chars is None else max(1, max_chars)
    msg_budget = DEFAULT_MAX_MESSAGES if max_messages is None else max(1, max_messages)

    if not messages:
        return [], {"kept": 0, "dropped": 0, "dropped_chars": 0, "truncated": False}

    pinned: List[Dict[str, Any]] = []
    body = messages
    if keep_first and len(messages) > 1 and str(messages[0].get("role", "")).lower() == "system":
        pinned = [messages[0]]
        body = messages[1:]

    kept_reversed: List[Dict[str, Any]] = []
    used_chars = sum(_content_length(m) for m in pinned)
    for message in reversed(body):
        cost = _content_length(message)
        if len(kept_reversed) >= max(0, msg_budget - len(pinned)):
            break
        if used_chars + cost > char_budget:
            break
        kept_reversed.append(message)
        used_chars += cost

    kept = pinned + list(reversed(kept_reversed))
    dropped = len(messages) - len(kept)
    dropped_chars = sum(_content_length(m) for m in messages) - used_chars
    return kept, {
        "kept": len(kept),
        "dropped": dropped,
        "dropped_chars": max(0, dropped_chars),
        "truncated": dropped > 0,
        "max_chars": char_budget,
        "max_messages": msg_budget,
    }


def capture_and_trim(
    history: List[Dict[str, Any]] | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """`trim_history` with the configured budgets (env-overridable)."""
    max_chars, max_messages = _limits()
    return trim_history(history, max_chars=max_chars, max_messages=max_messages)
