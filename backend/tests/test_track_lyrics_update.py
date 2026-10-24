"""Tests for updating and realigning track lyrics."""

import uuid
import json
import pytest
import httpx
from sqlmodel import Session

from app.main import app, engine, get_job_by_id
from app.models import Job, JobStatus


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_patch_job_lyrics(client):
    test_id = uuid.uuid4()
    with Session(engine) as s:
        job = Job(
            id=test_id,
            title="Lyrics Patch Test Track",
            prompt="upbeat pop",
            lyrics="[Verse 1]\nOriginal lyrics here.",
            status=JobStatus.COMPLETED,
        )
        s.add(job)
        s.commit()

    updated_lyrics = "[Verse 1]\nBrand new modified verse 1.\n\n[Chorus]\nCatchy new chorus!"
    resp = await client.patch(f"/jobs/{test_id}", json={"lyrics": updated_lyrics})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["lyrics"] == updated_lyrics

    # Verify persistence in database
    with Session(engine) as s:
        db_job = get_job_by_id(s, str(test_id))
        assert db_job is not None
        assert db_job.lyrics == updated_lyrics


@pytest.mark.asyncio
async def test_realign_lyrics_with_new_text(client):
    test_id = uuid.uuid4()
    with Session(engine) as s:
        job = Job(
            id=test_id,
            title="Lyrics Realign Test Track",
            prompt="rock anthem",
            lyrics="[Intro]\nGuitar riff.",
            duration_ms=60000,
            status=JobStatus.COMPLETED,
        )
        s.add(job)
        s.commit()

    realigned_lyrics = "[Verse 1]\nWe are singing the new lines.\n\n[Chorus]\nLoud and clear."
    resp = await client.post(f"/tracks/{test_id}/realign_lyrics", json={"lyrics": realigned_lyrics})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "realigned"
    assert "timed_lyrics" in data
    assert len(data["timed_lyrics"]) > 0

    # Verify lyrics and timed lyrics persisted
    with Session(engine) as s:
        db_job = get_job_by_id(s, str(test_id))
        assert db_job is not None
        assert db_job.lyrics == realigned_lyrics
        timed = json.loads(db_job.timed_lyrics_json)
        assert len(timed) > 0
        texts = [t.get("text", "") for t in timed]
        assert any("We are singing the new lines." in t for t in texts)
