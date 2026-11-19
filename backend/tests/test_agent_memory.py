"""Budgeted conversation memory (memory phase).

Unbounded history is a memory *and* correctness problem: eventually the prompt no
longer fits, or the provider truncates from the top and the agent forgets its
instructions mid-session.
"""

from __future__ import annotations

import pytest

from app.core import agent_memory as am


def _msg(role, content):
    return {"role": role, "content": content}


def test_short_history_is_untouched():
    history = [_msg("user", "hi"), _msg("assistant", "hello")]
    kept, report = am.trim_history(history, max_chars=1000, max_messages=10)
    assert kept == history
    assert report["truncated"] is False


def test_the_message_budget_keeps_the_newest():
    history = [_msg("user", f"m{i}") for i in range(10)]
    kept, report = am.trim_history(history, max_chars=10_000, max_messages=3)
    assert [m["content"] for m in kept] == ["m7", "m8", "m9"]
    assert report["dropped"] == 7
    assert report["truncated"] is True


def test_the_character_budget_keeps_the_newest_that_fit():
    history = [_msg("user", "x" * 100) for _ in range(10)]
    kept, report = am.trim_history(history, max_chars=250, max_messages=100)
    assert len(kept) == 2, "two 100-char turns fit in 250"
    assert report["dropped"] == 8
    assert report["dropped_chars"] == 800


def test_the_system_prompt_survives_truncation():
    history = [_msg("system", "you are a careful songwriter")] + [
        _msg("user", f"turn {i}") for i in range(20)
    ]
    kept, report = am.trim_history(history, max_chars=200, max_messages=3)
    assert kept[0]["role"] == "system", "instructions must never be the thing dropped"
    assert len(kept) <= 3
    assert report["truncated"] is True


def test_an_empty_history_is_a_no_op():
    kept, report = am.trim_history(None)
    assert kept == []
    assert report["kept"] == 0 and report["truncated"] is False


def test_absurd_limits_are_clamped_not_crashed():
    kept, _ = am.trim_history([_msg("user", "a")], max_chars=0, max_messages=0)
    assert len(kept) <= 1


def test_budgets_are_configurable_by_env(monkeypatch):
    monkeypatch.setenv("MILIMO_AGENT_HISTORY_MAX_MESSAGES", "2")
    monkeypatch.setenv("MILIMO_AGENT_HISTORY_MAX_CHARS", "1000")
    history = [_msg("user", f"m{i}") for i in range(6)]
    kept, report = am.capture_and_trim(history)
    assert len(kept) == 2
    assert report["max_messages"] == 2


def test_non_string_content_counts_toward_the_budget():
    """Structured content is measured by its serialised size, not ignored."""
    one = {"role": "user", "content": {"text": "x" * 50}}
    per_message = len(str(one["content"]))
    history = [dict(one) for _ in range(4)]
    kept, report = am.trim_history(
        history, max_chars=per_message * 2 + 10, max_messages=10
    )
    assert len(kept) == 2
    assert report["truncated"] is True
