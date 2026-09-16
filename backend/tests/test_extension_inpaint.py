import pytest
import uuid
import asyncio
import httpx
import soundfile as sf
import numpy as np
from sqlmodel import Session, select

from app.main import app, engine
from app.models import Job, JobStatus, TrackExtendRequest, TrackInpaintRequest
from app.core.paths import get_generated_audio_dir


class SyncTestClient:
    def __init__(self, asgi_app):
        self.app = asgi_app
        self.transport = httpx.ASGITransport(app=asgi_app)
        
    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def get(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.get(url, **kwargs)
        return self._run(_call())

    def post(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.post(url, **kwargs)
        return self._run(_call())


def test_track_extend_request_schema():
    req = TrackExtendRequest(target_duration_sec=120.0)
    assert req.auto_generate_lyrics is False
    assert req.additional_lyrics is None
    assert req.crossfade_sec == 1.5


def test_track_inpaint_request_schema():
    req = TrackInpaintRequest(start_time=10.0, end_time=25.0)
    assert req.start_time == 10.0
    assert req.end_time == 25.0
    assert req.crossfade_sec == 1.0


def test_extension_lyrics_never_auto_added_by_default():
    client = SyncTestClient(app)
    gen_dir = get_generated_audio_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock parent audio file
    parent_id = uuid.uuid4()
    parent_wav = gen_dir / f"{parent_id}.wav"
    sr = 44100
    dummy_audio = np.zeros((sr * 30, 2), dtype=np.float32)
    sf.write(str(parent_wav), dummy_audio, sr)

    original_lyrics = "[Verse 1]\nHere is the original verse."

    with Session(engine) as session:
        parent_job = Job(
            id=str(parent_id),
            title="Test Parent Song",
            prompt="pop electronic upbeat",
            lyrics=original_lyrics,
            duration_ms=30000,
            audio_path=f"/audio/{parent_id}.wav",
            status=JobStatus.COMPLETED,
            tags="pop",
            seed=42,
        )
        session.add(parent_job)
        session.commit()

    # 1. Extend with default options (auto_generate_lyrics=False, additional_lyrics=None)
    res = client.post(
        f"/tracks/{parent_id}/extend",
        json={"target_duration_sec": 60.0}
    )
    assert res.status_code == 200, res.text
    child_id = res.json()["job_id"]

    with Session(engine) as session:
        child_job = session.get(Job, uuid.UUID(child_id) if "-" in child_id else child_id)
        assert child_job is not None
        # Must strictly match original lyrics without added verses or solo markers
        assert child_job.lyrics == original_lyrics
        assert child_job.is_extension is True

    # 2. Extend with explicit additional lyrics
    res2 = client.post(
        f"/tracks/{parent_id}/extend",
        json={
            "target_duration_sec": 60.0,
            "additional_lyrics": "[Chorus]\nHere is my custom continuation."
        }
    )
    assert res2.status_code == 200, res2.text
    child2_id = res2.json()["job_id"]

    with Session(engine) as session:
        child2_job = session.get(Job, uuid.UUID(child2_id) if "-" in child2_id else child2_id)
        assert child2_job is not None
        assert "[Chorus]\nHere is my custom continuation." in child2_job.lyrics
        assert "[Verse 1]" in child2_job.lyrics


def test_inpaint_track_endpoint():
    client = SyncTestClient(app)
    gen_dir = get_generated_audio_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    parent_id = uuid.uuid4()
    parent_wav = gen_dir / f"{parent_id}.wav"
    sr = 44100
    dummy_audio = np.zeros((sr * 20, 2), dtype=np.float32)
    sf.write(str(parent_wav), dummy_audio, sr)

    with Session(engine) as session:
        parent_job = Job(
            id=str(parent_id),
            title="Inpaint Parent Track",
            prompt="acoustic guitar melody",
            lyrics="[Verse]\nAcoustic melody playing.",
            duration_ms=20000,
            audio_path=f"/audio/{parent_id}.wav",
            status=JobStatus.COMPLETED,
            tags="acoustic, guitar",
            seed=123,
        )
        session.add(parent_job)
        session.commit()

    # Valid inpaint request
    res = client.post(
        f"/jobs/{parent_id}/inpaint",
        json={"start_time": 5.0, "end_time": 10.0, "crossfade_sec": 0.5}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "queued"
    assert "job_id" in data
    repair_id = data["job_id"]

    with Session(engine) as session:
        child = session.get(Job, uuid.UUID(repair_id) if "-" in repair_id else repair_id)
        assert child is not None
        assert child.is_repair is True
        assert child.parent_job_id == str(parent_id)
        assert child.title == "Inpaint Parent Track (Repaired)"

    # Invalid bounds checks: start >= end
    bad_res1 = client.post(
        f"/jobs/{parent_id}/inpaint",
        json={"start_time": 10.0, "end_time": 5.0}
    )
    assert bad_res1.status_code == 400

    # Invalid bounds checks: start >= duration
    bad_res2 = client.post(
        f"/jobs/{parent_id}/inpaint",
        json={"start_time": 30.0, "end_time": 40.0}
    )
    assert bad_res2.status_code == 400
