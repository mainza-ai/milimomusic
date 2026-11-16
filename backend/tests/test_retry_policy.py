"""Bounded retry classification (plan 2.2).

A transient OOM or 503 used to fail a job permanently on the first attempt, and a
bad API key used to burn the GPU-queue slot retrying something that can never work.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.retry import is_transient, with_retries


@pytest.mark.parametrize("exc", [
    RuntimeError("CUDA out of memory"),
    RuntimeError("MPS backend out of memory"),
    RuntimeError("503 Service Unavailable"),
    RuntimeError("Connection reset by peer"),
    RuntimeError("No space left on device"),
    TimeoutError("timed out"),
    ConnectionError("connection aborted"),
    MemoryError(),
])
def test_transient_errors_are_retryable(exc):
    assert is_transient(exc) is True


@pytest.mark.parametrize("exc", [
    RuntimeError("Invalid API key"),
    RuntimeError("401 Unauthorized"),
    RuntimeError("model not found"),
    NotImplementedError("not implemented"),
    ValueError("malformed request"),
    KeyError("prompt"),
])
def test_terminal_errors_are_not_retried(exc):
    assert is_transient(exc) is False


def test_cancellation_is_never_retried():
    assert is_transient(asyncio.CancelledError()) is False
    assert is_transient(KeyboardInterrupt()) is False


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    result = await with_retries(flaky, attempts=3, base_delay=0.01)
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_stops_after_attempts_are_exhausted():
    calls = {"n": 0}

    async def always_oom():
        calls["n"] += 1
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        await with_retries(always_oom, attempts=2, base_delay=0.01)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_terminal_error_is_not_retried():
    calls = {"n": 0}

    async def bad_key():
        calls["n"] += 1
        raise RuntimeError("Invalid API key")

    with pytest.raises(RuntimeError):
        await with_retries(bad_key, attempts=5, base_delay=0.01)
    assert calls["n"] == 1, "a terminal error must fail on the first attempt"


@pytest.mark.asyncio
async def test_on_retry_reports_each_attempt():
    seen = []

    async def flaky():
        if len(seen) < 2:
            raise RuntimeError("503 temporarily unavailable")
        return "done"

    await with_retries(flaky, attempts=3, base_delay=0.01,
                       on_retry=lambda attempt, exc, delay: seen.append((attempt, delay)))
    assert [a for a, _ in seen] == [1, 2]
    assert all(d > 0 for _, d in seen)


@pytest.mark.asyncio
async def test_cancel_event_aborts_before_retrying():
    class Event:
        def is_set(self):
            return True

    async def never_called():  # pragma: no cover - must not run
        raise AssertionError("should not be called")

    with pytest.raises(asyncio.CancelledError):
        await with_retries(never_called, attempts=3, cancel_event=Event())
