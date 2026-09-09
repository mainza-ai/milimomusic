"""Tests for video_service and video API endpoints."""

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
import httpx
from sqlmodel import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app, engine
from app.models import Job, JobStatus
from app.services.video_service import video_service


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture()
def sample_job():
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        title="Cyber Drift",
        prompt="Synthesizer synthwave high energy electronic beat",
        lyrics="[Verse 1]\nDriving through the neon night\nElectric reflections in my sight",
        status=JobStatus.COMPLETED,
        duration_ms=60000,
        audio_path="generated_audio/test_cyber_drift.mp3",
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)

    yield job

    with Session(engine) as session:
        existing = session.get(Job, job_id)
        if existing:
            session.delete(existing)
            session.commit()


@pytest.mark.asyncio
async def test_generate_storyboard_scenes(sample_job):
    """Test storyboard scene generation produces required camera, lighting, and timeline fields."""
    scenes = await video_service.generate_storyboard(sample_job, visual_style="neon-cyberpunk")
    assert isinstance(scenes, list)
    assert len(scenes) >= 3

    for scene in scenes:
        assert "time" in scene
        assert "prompt" in scene
        assert "camera" in scene
        assert "lighting" in scene
        assert len(scene["prompt"]) > 0


@pytest.mark.asyncio
async def test_generate_storyboard_style_customization(sample_job):
    """Test storyboard generation customizes lighting and aesthetics per visual style."""
    retro_scenes = await video_service.generate_storyboard(sample_job, visual_style="retro-vhs")
    assert any("retro" in s["prompt"].lower() or "vhs" in s["lighting"].lower() or "analog" in s["prompt"].lower() for s in retro_scenes)


@pytest.mark.asyncio
async def test_storyboard_api_endpoint(client, sample_job):
    """Test POST /videos/storyboard/{job_id} endpoint returns 200 with scenes."""
    response = await client.post(
        f"/videos/storyboard/{sample_job.id}",
        json={"visual_style": "anime-cinematic"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["job_id"] == str(sample_job.id)
    assert data["visual_style"] == "anime-cinematic"
    assert len(data["scenes"]) >= 3


@pytest.mark.asyncio
async def test_storyboard_api_endpoint_404(client):
    """Test POST /videos/storyboard/{job_id} returns 404 for non-existent job."""
    bad_id = str(uuid.uuid4())
    response = await client.post(f"/videos/storyboard/{bad_id}", json={})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_render_audio_reactive_video_execution(sample_job, tmp_path):
    """Test render_audio_reactive_video handles audio resolution and mock ffmpeg execution."""
    audio_dir = Path("generated_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "test_cyber_drift.mp3"
    audio_file.write_bytes(b"\x00" * 1024)

    try:
        with patch("asyncio.create_subprocess_exec") as mock_proc:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"", b"")
            mock_proc.return_value = mock_process

            video_url = await video_service.render_audio_reactive_video(
                sample_job,
                visual_style="neon-cyberpunk",
                resolution="720p"
            )
            assert video_url.startswith("/audio/videos/")
            assert video_url.endswith("_reactive.mp4")
    finally:
        if audio_file.exists():
            audio_file.unlink()


@pytest.mark.asyncio
async def test_render_endpoint_validation(client):
    """Test POST /videos/render/{job_id} properly rejects missing jobs and jobs with no audio."""
    # 404 on unknown job
    bad_id = str(uuid.uuid4())
    response = await client.post(f"/videos/render/{bad_id}", json={})
    assert response.status_code == 404

    # 400 on job without audio
    no_audio_id = uuid.uuid4()
    job = Job(id=no_audio_id, title="No Audio", prompt="test", status=JobStatus.QUEUED, audio_path=None)
    with Session(engine) as session:
        session.add(job)
        session.commit()

    try:
        response = await client.post(f"/videos/render/{no_audio_id}", json={})
        assert response.status_code == 400
        assert "no completed audio" in response.json()["detail"].lower()
    finally:
        with Session(engine) as session:
            existing = session.get(Job, no_audio_id)
            if existing:
                session.delete(existing)
                session.commit()
