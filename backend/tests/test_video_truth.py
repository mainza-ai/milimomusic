"""Truth tests for the video pipeline (Phase 7A: plans 7.1, 7.2, 7.3, 7.6, 7.7).

Every test here exists because the pipeline previously *reported success* for
work it did not do: a jaw-warp animatic labelled neural lip sync, an encoder that
failed but returned a URL, a retake that dropped its own inputs, and two renders
racing for one output file.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app.services.video.lip_sync.live_portrait import LivePortraitProvider


# ── 7.1 lip-sync honesty ───────────────────────────────────────────────────

def test_liveportrait_is_unavailable_without_weights(monkeypatch):
    provider = LivePortraitProvider()
    monkeypatch.setattr(provider, "_get_weights_path", lambda: None)
    assert provider.is_available is False


def test_liveportrait_is_unavailable_without_an_audio_entry_point(monkeypatch, tmp_path):
    """Weights alone are not enough: upstream LivePortrait takes a driving *video*."""
    weights = tmp_path / "liveportrait"
    weights.mkdir()
    (weights / "inference.py").write_text("# video-driven only")
    provider = LivePortraitProvider()
    monkeypatch.setattr(provider, "_get_weights_path", lambda: str(weights))
    assert provider._audio_entry_point() is None
    assert provider.is_available is False


def test_liveportrait_entry_point_honours_the_env_override(monkeypatch, tmp_path):
    script = tmp_path / "audio_wrapper.py"
    script.write_text("# audio driven")
    provider = LivePortraitProvider()
    monkeypatch.setenv("MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT", str(script))
    assert provider._audio_entry_point() == str(script)

    monkeypatch.setenv("MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT", str(tmp_path / "missing.py"))
    assert provider._audio_entry_point() is None


def test_liveportrait_reports_the_jaw_warp_when_it_delegates(monkeypatch, tmp_path):
    """The bug in issue-class terms: provider returns True after delegating."""
    provider = LivePortraitProvider()
    monkeypatch.setattr(provider, "_get_weights_path", lambda: None)

    async def fake_fallback(**kwargs):
        return True

    monkeypatch.setattr(provider._fallback, "render_lip_sync", fake_fallback)

    ok = asyncio.run(provider.render_lip_sync(
        face_image_path=str(tmp_path / "face.png"),
        vocal_audio_path=str(tmp_path / "vocals.wav"),
        start_time=0.0, duration=1.0,
        out_path=str(tmp_path / "clip.mp4"),
    ))

    assert ok is True, "the fallback still has to produce a clip"
    assert provider.last_outcome["used_fallback"] is True
    assert provider.last_outcome["engine"] == "jaw_warp"
    assert "weights" in (provider.last_outcome["reason"] or "")


def test_task_status_exposes_the_engine_that_actually_ran():
    from app.services.video.types import VideoTaskStatusInfo

    info = VideoTaskStatusInfo(id="t", job_id="j")
    assert info.lip_sync_engine is None
    info.lip_sync_engine = "jaw_warp"
    assert info.to_dict()["lip_sync_engine"] == "jaw_warp"


# ── 7.2 render success verification ───────────────────────────────────────

def test_audio_reactive_render_raises_when_ffmpeg_fails(tmp_path, monkeypatch):
    """`/videos/render` used to return a URL for a file FFmpeg never wrote."""
    from app.services import video_service as vs

    class FakeProc:
        returncode = 1

        async def communicate(self):
            return b"", b"Unknown encoder"

    class FakeStream:
        def __init__(self, data): self._data = data
        def decode(self, *a, **kw): return self._data.decode(*a, **kw)

    async def fake_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    service = vs.VideoService()
    job = type("J", (), {
        "id": "job-1",
        "audio_path": str(tmp_path / "master.wav"),
        "cover_image_path": None,
        "character_image_path": None,
    })()
    (tmp_path / "master.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(service, "resolve_audio_path", lambda p: p)
    monkeypatch.setattr(service, "resolve_face_image", lambda job: None)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(service.render_audio_reactive_video(job=job, visual_style="neon-cyberpunk", resolution="720p"))
    assert "Unknown encoder" in str(exc.value) or "ffmpeg" in str(exc.value).lower()


# ── 7.3 retake honesty ────────────────────────────────────────────────────

def test_retake_request_accepts_camera_and_lighting():
    """The modal has always sent these; the API used to drop them."""
    from app.models import RetakeClipRequest

    req = RetakeClipRequest(prompt="a singer", camera="dolly zoom", lighting="neon rim")
    assert req.camera == "dolly zoom"
    assert req.lighting == "neon rim"


# ── 7.6 per-job render mutex ──────────────────────────────────────────────

def test_second_render_for_the_same_job_is_refused(client, monkeypatch):
    """Two renders used to write the same output file and race."""
    from app.services.video_service import video_service

    # A threading.Event, not asyncio.Event: the sync test client runs each request
    # on its own event loop, so an asyncio primitive would be bound to the wrong loop.
    import threading as _threading
    hold = _threading.Event()

    async def blocked_render(*args, **kwargs):
        # Keep the first render in flight so the job is genuinely occupied.
        await asyncio.get_running_loop().run_in_executor(None, hold.wait, 5.0)
        return ""

    monkeypatch.setattr(video_service, "render_advanced_music_video", blocked_render)
    monkeypatch.setattr(video_service, "render_lyric_music_video", blocked_render)

    # Create a completed job with audio. `Job.id` is a real UUID column, so it
    # must be a UUID object, not a string.
    import uuid as _uuid
    from sqlmodel import Session
    from app.main import engine
    from app.models import Job, JobStatus

    job_uuid = _uuid.uuid4()
    job_id = str(job_uuid)
    with Session(engine) as session:
        session.add(Job(id=job_uuid, status=JobStatus.COMPLETED, prompt="test",
                        title="t", audio_path="/audio/x.wav"))
        session.commit()

    first = client.post(f"/videos/render-advanced/{job_id}", json={})
    assert first.status_code == 202, first.text
    assert first.json()["cost_estimate"]["usd"] == 0.0  # local render

    second = client.post(f"/videos/render-lyric/{job_id}", json={})
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["error"]["code"] == "render_in_progress"

    # The sync test client closes each request's event loop, so the background task
    # cannot resume here to release itself — free the claim explicitly and assert the
    # release semantics directly (see test_job_render_claim_is_exclusive_per_task).
    hold.set()
    video_service.release_job_render(job_id)
    assert video_service.active_render_for_job(job_id) is None


# ── 7.7 cloud cost estimate ───────────────────────────────────────────────

def test_cloud_estimate_scales_with_duration_and_model():
    from app.services.video.pricing import estimate_cloud_cost

    cheap = estimate_cloud_cost("fal", 4.0, 5.0, overhead=1.0)     # 4 clips x 5 s
    pricey = estimate_cloud_cost("fal", 4.0, 10.0, overhead=1.0)
    assert cheap["seconds"] == 20.0
    assert pricey["usd"] > cheap["usd"]
    # Over-render headroom is applied by default.
    assert estimate_cloud_cost("fal", 4.0, 5.0)["seconds"] > 20.0
    assert cheap["currency"] == "USD"


def test_cloud_estimate_is_explicitly_unknown_for_unpriced_engines():
    from app.services.video.pricing import estimate_cloud_cost

    est = estimate_cloud_cost("replicate", 2.0, 5.0)
    assert est["usd"] is None
    assert "not priced" in est["note"].lower()


def test_job_render_claim_is_exclusive_per_task():
    """Claim/release semantics: one renderer per job, idempotent release."""
    from app.services.video_service import video_service

    video_service.release_job_render("job-claim-test")
    assert video_service.claim_job_render("job-claim-test", "task-a") is None
    assert video_service.claim_job_render("job-claim-test", "task-b") == "task-a"
    # Re-claiming with the same task id is a no-op, not a conflict.
    assert video_service.claim_job_render("job-claim-test", "task-a") is None
    # A different task cannot release someone else's claim.
    video_service.release_job_render("job-claim-test", "task-b")
    assert video_service.active_render_for_job("job-claim-test") == "task-a"
    video_service.release_job_render("job-claim-test", "task-a")
    assert video_service.active_render_for_job("job-claim-test") is None


def test_cloud_render_requires_cost_confirmation(client, monkeypatch):
    """A paid render above the threshold must not start without confirmation (7.7)."""
    import uuid as _uuid
    from sqlmodel import Session

    from app.main import engine
    from app.models import Job, JobStatus
    from app.services.video_service import video_service

    async def should_not_run(*args, **kwargs):  # pragma: no cover - must not start
        raise AssertionError("an unconfirmed paid render must not start")

    monkeypatch.setattr(video_service, "render_advanced_music_video", should_not_run)

    job_uuid = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_uuid, status=JobStatus.COMPLETED, prompt="p",
                        title="cost", audio_path="/audio/x.wav", duration_ms=180000))
        session.commit()

    try:
        r = client.post(
            f"/videos/render-advanced/{job_uuid}",
            json={"provider": "cloud_fal", "model_name": "wan_14b", "max_clip_duration": 5},
        )
        assert r.status_code == 409, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "cost_confirmation_required"
        assert err["estimate"]["usd"] > 5.0
        assert err["threshold_usd"] == 5.0
    finally:
        with Session(engine) as session:
            row = session.get(Job, job_uuid)
            if row:
                session.delete(row)
                session.commit()
