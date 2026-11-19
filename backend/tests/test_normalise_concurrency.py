"""Clip normalisation runs concurrently but preserves order (plan 9.10).

Clips are independent encodes; doing them one at a time was pure wall-clock waste.
The stitcher depends on input order, so concurrency must not reorder the result.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.services.video.video_orchestrator import video_orchestrator as orch


def _plan(n):
    return [type("C", (), {"duration": 1.0, "start_time": float(i)})() for i in range(n)]


def test_clips_normalise_concurrently_and_keep_their_order(monkeypatch, tmp_path):
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_normalise(src, dest, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.12)          # long enough for the pool to overlap
        with lock:
            state["active"] -= 1
        with open(dest, "wb") as fh:
            fh.write(b"x")
        return True

    monkeypatch.setattr("app.services.video.clip_normalise.normalise_clip", fake_normalise)
    monkeypatch.setattr("app.services.video.clip_normalise.probe_duration", lambda p: 1.0)
    monkeypatch.setenv("MILIMO_NORMALISE_CONCURRENCY", "3")

    clips = [str(tmp_path / f"clip_{i}.mp4") for i in range(5)]
    for c in clips:
        open(c, "wb").write(b"src")

    started = time.perf_counter()
    out = asyncio.run(orch._normalise_rendered_clips(
        rendered_clips=clips, plan_clips=_plan(5),
        width=320, height=180, fps=24, task_id="conc",
    ))
    elapsed = time.perf_counter() - started

    assert len(out) == 5
    # Order preserved: the stitcher depends on it.
    assert [f.split("_")[-1] for f in out] == ["000.mp4", "001.mp4", "002.mp4", "003.mp4", "004.mp4"]
    assert state["peak"] >= 2, "clips must overlap, not queue"
    assert elapsed < 0.12 * 5, "concurrency must beat serial wall time"


def test_concurrency_one_is_still_serial(monkeypatch, tmp_path):
    order = []

    def fake_normalise(src, dest, **kwargs):
        order.append(dest)
        with open(dest, "wb") as fh:
            fh.write(b"x")
        return True

    monkeypatch.setattr("app.services.video.clip_normalise.normalise_clip", fake_normalise)
    monkeypatch.setattr("app.services.video.clip_normalise.probe_duration", lambda p: 1.0)
    monkeypatch.setenv("MILIMO_NORMALISE_CONCURRENCY", "1")

    clips = [str(tmp_path / f"c_{i}.mp4") for i in range(3)]
    for c in clips:
        open(c, "wb").write(b"src")

    out = asyncio.run(orch._normalise_rendered_clips(
        rendered_clips=clips, plan_clips=_plan(3),
        width=160, height=120, fps=24, task_id="serial",
    ))
    assert len(out) == 3 and order == out


def test_a_failed_clip_keeps_the_raw_render(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.video.clip_normalise.normalise_clip",
                        lambda src, dest, **kw: False)
    monkeypatch.setattr("app.services.video.clip_normalise.probe_duration", lambda p: 1.0)

    clips = [str(tmp_path / "only.mp4")]
    open(clips[0], "wb").write(b"src")
    out = asyncio.run(orch._normalise_rendered_clips(
        rendered_clips=clips, plan_clips=_plan(1),
        width=160, height=120, fps=24, task_id="fail",
    ))
    assert out == clips, "a clip that cannot be normalised must not disappear"
