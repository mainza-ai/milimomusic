import json
import pytest
import httpx
import asyncio
from unittest.mock import patch, AsyncMock
from sqlmodel import Session, select
from app.main import app, engine, get_job_by_id
from app.models import Job


class SyncClient:
    def __init__(self, asgi_app):
        self.transport = httpx.ASGITransport(app=asgi_app)

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def post(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.post(url, **kwargs)
        return self._run(_call())

    def get(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.get(url, **kwargs)
        return self._run(_call())


@pytest.fixture
def client():
    return SyncClient(app)


@pytest.fixture
def sample_job():
    with Session(engine) as session:
        job = Job(
            prompt="Electronic synthwave beat",
            title="Vocal Test Track",
            duration_ms=60000,
            audio_path="/audio/test_master.wav",
            stems_json=json.dumps({"vocals": "/audio/test_vocals.wav", "drums": "/audio/test_drums.wav"}),
            status="completed"
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = str(job.id)
    
    yield job_id

    # Cleanup
    with Session(engine) as session:
        del_job = get_job_by_id(session, job_id)
        if del_job:
            session.delete(del_job)
            session.commit()


def test_voice_convert_route_f0_and_dry_wet(client, sample_job):
    with patch("app.services.voice_service.voice_service.convert_vocals", new_callable=AsyncMock) as mock_convert, \
         patch("app.services.voice_service.voice_service.remix_master_with_vocal") as mock_remix:
        
        mock_convert.return_value = "/audio/converted_stem.wav"
        mock_remix.return_value = "/audio/converted_master.wav"

        resp = client.post(f"/jobs/{sample_job}/voice-convert", json={
            "voice_profile_id": "default_aria",
            "pitch_shift": 2,
            "dry_wet": 50,  # 50% should normalize to 0.5
            "formant_preserve": True,
            "f0_method": "crepe"
        })

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["parent_job_id"] == sample_job
        assert data["audio_path"] == "/audio/converted_master.wav"

        # Verify f0_method and dry_wet were passed correctly
        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args.kwargs
        assert kwargs["f0_method"] == "crepe"
        assert kwargs["dry_wet"] == 0.5
        assert kwargs["pitch_shift"] == 2
        assert kwargs["formant_preserve"] is True


def test_commit_vocal_route(client, sample_job):
    resp = client.post(f"/tracks/{sample_job}/commit-vocal", json={
        "vocal_path": "/audio/new_take_mic.wav",
        "master_path": "/audio/new_remix_master.wav"
    })

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["id"] == sample_job
    assert data["audio_path"] == "/audio/new_remix_master.wav"
    stems = json.loads(data["stems_json"])
    assert stems["vocals"] == "/audio/new_take_mic.wav"
    assert stems["drums"] == "/audio/test_drums.wav"
