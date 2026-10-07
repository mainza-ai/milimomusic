import os
import sys
import uuid
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock
import pytest
import httpx
from sqlmodel import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app, engine
from app.models import Job, JobStatus, LyricVideoRequest
from app.services.video_service import video_service
from app.services.video.video_orchestrator import video_orchestrator
from app.services.video.subtitle_styles import detect_hardware_encoder


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def test_lyric_video_request_model():
    req = LyricVideoRequest()
    assert req.style_preset == "neon"
    assert req.aspect_ratio == "16:9"
    assert req.resolution == "720p"
    assert req.background_mode == "cover_art"
    assert req.burn_lyrics is True

    custom = LyricVideoRequest(
        style_preset="spotify",
        aspect_ratio="9:16",
        resolution="1080p",
        background_mode="spectrum",
        include_spectrum=True,
        font_family="Inter",
        font_size_override=48
    )
    assert custom.style_preset == "spotify"
    assert custom.aspect_ratio == "9:16"
    assert custom.include_spectrum is True
    assert custom.font_family == "Inter"


def test_detect_hardware_encoder():
    encoder, flags = detect_hardware_encoder()
    assert isinstance(encoder, str)
    assert encoder in ("h264_videotoolbox", "h264_nvenc", "libx264")
    assert isinstance(flags, list)


@pytest.mark.asyncio
async def test_render_lyric_video_endpoint_errors(client):
    # 404 on nonexistent job
    fake_id = uuid.uuid4()
    res = await client.post(f"/videos/render-lyric/{fake_id}", json={})
    assert res.status_code == 404

    # 400 on job without completed audio
    no_audio_id = uuid.uuid4()
    job = Job(id=no_audio_id, title="No Audio", prompt="test", status=JobStatus.QUEUED, audio_path=None)
    with Session(engine) as session:
        session.add(job)
        session.commit()

    try:
        res = await client.post(f"/videos/render-lyric/{no_audio_id}", json={})
        assert res.status_code == 400
        assert "no completed audio" in res.json()["detail"].lower()
    finally:
        with Session(engine) as session:
            existing = session.get(Job, no_audio_id)
            if existing:
                session.delete(existing)
                session.commit()


@pytest.mark.asyncio
async def test_render_lyric_video_endpoint_success(client, tmp_path):
    # Create fake audio file
    fake_audio = tmp_path / "test_track.wav"
    fake_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        title="Nightfall",
        prompt="Synthwave pop",
        status=JobStatus.COMPLETED,
        audio_path=str(fake_audio),
        lyrics="In the middle of the night\nNeon shadows fade away",
        timed_lyrics_json=json.dumps([
            {
                "text": "In the middle of the night",
                "start": 1.0,
                "end": 3.5,
                "words": [
                    {"word": "In", "start": 1.0, "end": 1.3},
                    {"word": "the", "start": 1.35, "end": 1.6},
                    {"word": "middle", "start": 1.65, "end": 2.2},
                    {"word": "of", "start": 2.25, "end": 2.5},
                    {"word": "the", "start": 2.55, "end": 2.8},
                    {"word": "night", "start": 2.85, "end": 3.4}
                ]
            }
        ])
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()

    try:
        with patch.object(video_orchestrator, "render_lyric_music_video", new_callable=AsyncMock) as mock_render:
            mock_render.return_value = f"/audio/videos/{job_id}_lyric.mp4"

            res = await client.post(f"/videos/render-lyric/{job_id}", json={
                "style_preset": "neon",
                "aspect_ratio": "16:9",
                "background_mode": "spectrum"
            })
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "queued"
            assert "task_id" in data
            assert data["job_id"] == str(job_id)

            # Let background task fire
            await asyncio.sleep(0.05)
            assert mock_render.called
    finally:
        with Session(engine) as session:
            existing = session.get(Job, job_id)
            if existing:
                session.delete(existing)
                session.commit()


@pytest.mark.asyncio
async def test_lyric_video_cancellation(client):
    task_id = f"lyric_cancel_{uuid.uuid4().hex[:8]}"
    cancel_ev = asyncio.Event()

    with video_orchestrator._lock:
        video_orchestrator._video_cancels[task_id] = cancel_ev

    # Cancel via API
    res = await client.post(f"/videos/tasks/{task_id}/cancel")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert cancel_ev.is_set()


@pytest.mark.asyncio
async def test_lyric_video_cancellation_kills_proc(client):
    from unittest.mock import MagicMock
    from app.services.video.types import VideoTaskStatusInfo
    task_id = f"proc_cancel_{uuid.uuid4().hex[:8]}"
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()

    with video_orchestrator._lock:
        video_orchestrator._active_procs[task_id] = mock_proc
        video_orchestrator._tasks[task_id] = VideoTaskStatusInfo(
            id=task_id,
            job_id=str(uuid.uuid4()),
            status="processing",
            step="Encoding",
            progress=50
        )

    res = await client.post(f"/videos/tasks/{task_id}/cancel")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    mock_proc.kill.assert_called_once()
    assert video_orchestrator.get_task(task_id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_render_lyric_music_video_real_pipeline(tmp_path):
    import math
    import wave
    import struct
    from PIL import Image
    from app.services.video.video_orchestrator import VIDEO_DIR

    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        data = bytearray()
        for i in range(44100):
            sample = int(10000 * math.sin(2 * math.pi * 440 * i / 44100))
            data.extend(struct.pack("<h", sample))
        wf.writeframes(data)

    cover_path = tmp_path / "cover.png"
    img = Image.new("RGB", (320, 240), color=(30, 20, 40))
    img.save(str(cover_path))

    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        title="Integration Track",
        prompt="Synthwave",
        status=JobStatus.COMPLETED,
        audio_path=str(wav_path),
        cover_image_path=str(cover_path),
        lyrics="Hello beautiful world",
        timed_lyrics_json=json.dumps([
            {
                "text": "Hello beautiful world",
                "start": 0.1,
                "end": 0.9,
                "words": [
                    {"word": "Hello", "start": 0.1, "end": 0.3},
                    {"word": "beautiful", "start": 0.35, "end": 0.65},
                    {"word": "world", "start": 0.7, "end": 0.9}
                ]
            }
        ])
    )

    task_id = f"test_render_{uuid.uuid4().hex[:8]}"
    rendered_url = await video_orchestrator.render_lyric_music_video(
        job=job,
        task_id=task_id,
        config={
            "style_preset": "neon",
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "background_mode": "cover_art",
            "include_spectrum": False,
            "burn_lyrics": True
        }
    )

    assert rendered_url is not None
    assert str(job_id) in rendered_url
    rendered_filename = os.path.basename(rendered_url.split("?")[0])
    rendered_disk_path = os.path.join(VIDEO_DIR, rendered_filename)
    assert os.path.isfile(rendered_disk_path)
    assert os.path.getsize(rendered_disk_path) > 1000

    try:
        os.remove(rendered_disk_path)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_render_lyric_music_video_aspect_ratios(tmp_path):
    """Verify fast-path rendering succeeds across all supported aspect ratios (9:16, 1:1, 21:9)."""
    import wave
    import struct
    import math
    from PIL import Image
    from app.services.video.video_orchestrator import VIDEO_DIR

    wav_path = tmp_path / "ar_test.wav"
    sample_rate = 22050
    duration_s = 1.0
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = bytearray()
        for i in range(int(sample_rate * duration_s)):
            sample = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sample_rate))
            data.extend(struct.pack("<h", sample))
        wf.writeframes(data)

    cover_path = tmp_path / "ar_cover.png"
    img = Image.new("RGB", (400, 400), color=(50, 80, 120))
    img.save(str(cover_path))

    aspect_ratios = ["9:16", "1:1", "21:9"]
    for ar in aspect_ratios:
        job_id = uuid.uuid4()
        job = Job(
            id=job_id,
            title=f"Aspect Ratio {ar} Test",
            prompt="Electronic",
            status=JobStatus.COMPLETED,
            audio_path=str(wav_path),
            cover_image_path=str(cover_path),
            lyrics="Beat drops now",
            timed_lyrics_json=json.dumps([
                {
                    "text": "Beat drops now",
                    "start": 0.1,
                    "end": 0.9,
                    "words": [
                        {"word": "Beat", "start": 0.1, "end": 0.3},
                        {"word": "drops", "start": 0.35, "end": 0.65},
                        {"word": "now", "start": 0.7, "end": 0.9}
                    ]
                }
            ])
        )

        task_id = f"test_ar_{ar.replace(':', '_')}_{uuid.uuid4().hex[:6]}"
        rendered_url = await video_orchestrator.render_lyric_music_video(
            job=job,
            task_id=task_id,
            config={
                "style_preset": "neon",
                "aspect_ratio": ar,
                "resolution": "720p",
                "background_mode": "cover_art",
                "burn_lyrics": True
            }
        )

        assert rendered_url is not None
        rendered_filename = os.path.basename(rendered_url.split("?")[0])
        rendered_disk_path = os.path.join(VIDEO_DIR, rendered_filename)
        assert os.path.isfile(rendered_disk_path)
        assert os.path.getsize(rendered_disk_path) > 1000

        try:
            os.remove(rendered_disk_path)
        except OSError:
            pass

