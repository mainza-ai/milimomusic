"""
Production Features & Deployment Verification Test Suite for Milimo Music v2.
Tests:
1. Playlists SQLite persistence & REST endpoints (Create, List, Update, Add/Delete Tracks)
2. Studio Profile SQLite persistence & REST endpoints (Get, Update)
3. Multi-Modal Model Manager (Audio, Image, Video catalogs, Active switcher, Auto-install check)
4. Video Studio & Storyboard generation
5. Docker LLM Host Gateway Normalization
6. Unified Single-Process Frontend Serving & SPA routing
"""

import os
import sys
from pathlib import Path
import pytest
import httpx
import asyncio
from uuid import uuid4

# Ensure backend & muscriptor paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "muscriptor"))

from app.main import app, create_db_and_tables
from app.services.model_manager import model_manager
from app.services.llm_service import _normalize_llm_url
from app.services.video_service import VideoService


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

    def put(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.put(url, **kwargs)
        return self._run(_call())

    def delete(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.delete(url, **kwargs)
        return self._run(_call())


@pytest.fixture(autouse=True)
def init_db():
    create_db_and_tables()


@pytest.fixture
def client():
    return SyncClient(app)


def test_playlists_full_lifecycle(client):
    """Test full CRUD on SQLite backed Playlists."""
    # 1. Create Playlist
    create_payload = {
        "name": "Production Test Playlist",
        "description": "Created during automated verification",
        "cover_color": "from-teal-500 to-cyan-500"
    }
    res = client.post("/playlists", json=create_payload)
    assert res.status_code == 200, res.text
    p = res.json()
    p_id = p["id"]
    assert p["name"] == create_payload["name"]

    # 2. Get Playlist
    res = client.get(f"/playlists/{p_id}")
    assert res.status_code == 200
    p_detail = res.json()
    assert p_detail["name"] == "Production Test Playlist"
    assert p_detail["tracks"] == []

    # 3. Add Tracks
    track1_id = str(uuid4())
    track2_id = str(uuid4())
    res = client.post(f"/playlists/{p_id}/tracks", json={"job_id": track1_id, "position": 0})
    assert res.status_code == 200

    res = client.post(f"/playlists/{p_id}/tracks", json={"job_id": track2_id, "position": 1})
    assert res.status_code == 200

    res = client.get(f"/playlists/{p_id}")
    p_data = res.json()
    assert len(p_data["song_ids"]) == 2
    assert p_data["track_count"] == 2

    # 4. Update Playlist
    res = client.put(f"/playlists/{p_id}", json={"name": "Updated Playlist Title"})
    assert res.status_code == 200
    assert res.json()["name"] == "Updated Playlist Title"

    # 5. Remove Track
    res = client.delete(f"/playlists/{p_id}/tracks/{track1_id}")
    assert res.status_code == 200

    res = client.get(f"/playlists/{p_id}")
    p_data = res.json()
    assert len(p_data["song_ids"]) == 1
    assert p_data["track_count"] == 1
    assert p_data["song_ids"][0] == track2_id

    # 6. Delete Playlist
    res = client.delete(f"/playlists/{p_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "deleted"

    # 7. Verify 404
    res = client.get(f"/playlists/{p_id}")
    assert res.status_code == 404


def test_studio_profile_endpoints(client):
    """Test Studio User Profile retrieval and update."""
    res = client.get("/profile/studio")
    assert res.status_code == 200
    prof = res.json()
    assert "artist_name" in prof
    assert "bio" in prof

    # Update Profile
    update_data = {
        "artist_name": "Mainza Studio Pro",
        "bio": "AI Music Architecture & Production",
        "preferences": {
            "default_model_id": "minimax_music3_bf16",
            "target_lufs": -14.0,
            "daw_theme": "dark"
        }
    }
    res = client.put("/profile/studio", json=update_data)
    assert res.status_code == 200
    saved = res.json()
    assert saved["artist_name"] == "Mainza Studio Pro"
    assert saved["preferences"]["target_lufs"] == -14.0


def test_model_manager_multi_modal_catalog(client):
    """Test multi-modal model catalog and active switcher."""
    res = client.get("/models/tree")
    assert res.status_code == 200
    models = res.json()["models"]
    assert len(models) >= 19

    categories = {m["category"] for m in models}
    assert "audio" in categories
    assert "image" in categories
    assert "video" in categories

    # Verify Black Forest Labs FLUX.2 and FLUX.1 models in image category
    image_model_ids = {m["id"] for m in models if m["category"] == "image"}
    assert "flux_2_klein_4b" in image_model_ids
    assert "flux_2_klein_9b" in image_model_ids
    assert "flux_2_dev" in image_model_ids
    assert "flux_2_klein_4b_mlx" in image_model_ids
    assert "flux_1_schnell" in image_model_ids

    # Active model endpoint
    res = client.get("/models/active")
    assert res.status_code == 200
    active = res.json()
    assert "active_model" in active
    assert "id" in active["active_model"]

    # Model switch endpoint
    res = client.post("/models/select", json={"model_id": "minimax_music3_mxfp4"})
    assert res.status_code == 200
    assert res.json()["active_model"]["id"] == "minimax_music3_mxfp4"

    # Re-check active
    res = client.get("/models/active")
    assert res.json()["active_model"]["id"] == "minimax_music3_mxfp4"

    # Auto-install check policy test
    res = client.get("/models/auto-install-check")
    assert res.status_code == 200
    check = res.json()
    assert "needs_download" in check
    # Audio models policy: never download image or video on boot
    if check["needs_download"]:
        assert "minimax-music3" in check.get("recommended_repo_id", "").lower()


def test_cover_image_generation_with_flux(client):
    """Test cover image generation endpoint with FLUX.2 multi-modal model selection."""
    payload = {
        "prompt": "Neon synthwave skyline with glowing cyan towers",
        "style": "cinematic neon",
        "aspect_ratio": "1:1",
        "model_id": "flux_2_klein_4b"
    }
    res = client.post("/generate/cover-image", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "url" in data
    assert data["url"].startswith("/covers/")
    assert data["model_id"] == "flux_2_klein_4b"
    assert "FLUX.2" in data.get("model_name", "")


def test_huggingface_search_and_custom_model(client):
    """Test dynamic Hugging Face Hub search and custom model registration."""
    # Search HF models
    res = client.get("/models/search?q=flux&limit=3")
    assert res.status_code == 200
    search_data = res.json()
    assert "models" in search_data
    assert len(search_data["models"]) > 0
    first = search_data["models"][0]
    assert "repo_id" in first
    assert "downloads" in first
    assert "likes" in first
    assert "category" in first

    # Register custom model
    dummy_repo = "test-community/custom-synth-music"
    entry = model_manager.register_custom_model(dummy_repo, {
        "name": "Custom Synth Music Engine",
        "category": "audio",
        "size_gb": 4.2,
        "architecture": "Music Transformer",
        "license": "MIT"
    })
    assert entry["repo_id"] == dummy_repo

    # Check that it appears in model tree
    tree = client.get("/models/tree").json()["models"]
    matched = next((m for m in tree if m.get("repo_id") == dummy_repo), None)
    assert matched is not None
    assert matched["name"] == "Custom Synth Music Engine"

    # Clean up custom model
    del_res = client.delete(f"/models/custom/{entry['id']}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"


def test_video_pipeline_planning_and_duration_constraints(client):
    """Test video scene segmentation respecting duration constraints and vocal detection."""
    from sqlmodel import Session
    from app.main import engine
    from app.models import Job

    # Create dummy completed job in DB
    job_id = uuid4()
    with Session(engine) as session:
        job = Job(
            id=job_id,
            prompt="Neon city night pulse",
            tags="synthwave, cyberpunk",
            lyrics="[Verse 1]\nNeon shadows on the street\nElectric pulse beneath my feet\n[Chorus]\nCan you feel the signal glow",
            duration_ms=60000,
            audio_path="/audio/sample.mp3",
            status="completed"
        )
        session.add(job)
        session.commit()

    # Plan video with Wan2.1 5.0s clip constraint
    plan_res = client.post(f"/videos/plan/{job_id}", json={
        "max_clip_duration": 5.0,
        "bpm": 120.0,
        "visual_style": "neon-cyberpunk",
        "model_name": "Wan2.1 T2V (5.0s clips)"
    })
    assert plan_res.status_code == 200
    plan = plan_res.json()
    assert plan["status"] == "ok"
    assert plan["total_clips"] == 15
    assert len(plan["clips"]) == 15
    first_clip = plan["clips"][0]
    assert first_clip["duration"] <= 5.0
    assert first_clip["scene_type"] in ["VOCAL_PERFORMANCE", "CINEMATIC_BROLL"]
    assert "prompt" in first_clip
    assert "camera" in first_clip

    # Trigger advanced render
    render_res = client.post(f"/videos/render-advanced/{job_id}", json={
        "visual_style": "neon-cyberpunk",
        "max_clip_duration": 5.0,
        "enable_lip_sync": True,
        "burn_lyrics": True
    })
    assert render_res.status_code == 200
    task_data = render_res.json()
    assert "task_id" in task_data
    task_id = task_data["task_id"]

    # Check task status endpoint
    status_res = client.get(f"/videos/tasks/{task_id}")
    assert status_res.status_code == 200
    task_status = status_res.json()
    assert "progress" in task_status
    assert "step" in task_status


def test_video_model_duration_constraints_and_clamping(client):
    """Test model max duration resolution and auto-clamping defaults."""
    from sqlmodel import Session
    from app.main import engine
    from app.models import Job

    # 1. Test VideoService static lookup
    assert VideoService.get_model_max_duration("wan2.1") == 5.0
    assert VideoService.get_model_max_duration("Wan2.1 T2V (5.0s clips)") == 5.0
    assert VideoService.get_model_max_duration("cogvideox") == 10.0
    assert VideoService.get_model_max_duration("hailuo_h3") == 15.0
    assert VideoService.get_model_max_duration("MiniMax Hailuo H3 (15.0s clips)") == 15.0
    assert VideoService.get_model_max_duration("hunyuan") == 15.0
    assert VideoService.get_model_max_duration("audioreactive") == 120.0
    assert VideoService.get_model_max_duration("unknown_model") == 5.0

    # 2. Test auto-clamping in planning API
    job_id = uuid4()
    with Session(engine) as session:
        job = Job(
            id=job_id,
            prompt="High energy electronic dance anthem",
            tags="edm, festival",
            duration_ms=40000,
            audio_path="/audio/edm.mp3",
            status="completed"
        )
        session.add(job)
        session.commit()

    # Omitted max_clip_duration should default to Wan 2.1 max duration (5.0s)
    res = client.post(f"/videos/plan/{job_id}", json={
        "model_name": "Wan2.1 T2V (5.0s clips)"
    })
    assert res.status_code == 200
    plan = res.json()
    assert plan["model_max_duration"] == 5.0
    assert plan["max_clip_duration"] == 5.0
    for clip in plan["clips"]:
        assert clip["duration"] <= 5.0

    # Omitted max_clip_duration with Hailuo H3 should default to 15.0s
    res_hailuo = client.post(f"/videos/plan/{job_id}", json={
        "model_name": "hailuo_h3"
    })
    assert res_hailuo.status_code == 200
    plan_hailuo = res_hailuo.json()
    assert plan_hailuo["model_max_duration"] == 15.0
    assert plan_hailuo["max_clip_duration"] == 15.0

    # User requesting 30.0s for Hailuo H3 should be clamped to 15.0s
    res_clamped = client.post(f"/videos/plan/{job_id}", json={
        "model_name": "hailuo_h3",
        "max_clip_duration": 30.0
    })
    assert res_clamped.status_code == 200
    plan_clamped = res_clamped.json()
    assert plan_clamped["max_clip_duration"] == 15.0


def test_docker_llm_url_normalization(monkeypatch):
    """Test Docker container host gateway rewriting for local LLMs."""
    raw_ollama = "http://localhost:11434"
    raw_lmstudio = "http://127.0.0.1:1234/v1"
    raw_cloud = "https://integrate.api.nvidia.com/v1"

    # Outside Docker: URLs unchanged
    monkeypatch.delenv("MILIMO_IN_DOCKER", raising=False)
    assert _normalize_llm_url(raw_ollama) == raw_ollama
    assert _normalize_llm_url(raw_lmstudio) == raw_lmstudio
    assert _normalize_llm_url(raw_cloud) == raw_cloud

    # Inside Docker: localhost/127.0.0.1 rewritten to host.docker.internal
    monkeypatch.setenv("MILIMO_IN_DOCKER", "1")
    assert _normalize_llm_url(raw_ollama) == "http://host.docker.internal:11434"
    assert _normalize_llm_url(raw_lmstudio) == "http://host.docker.internal:1234/v1"
    assert _normalize_llm_url(raw_cloud) == raw_cloud  # Cloud untouched


def test_unified_frontend_serving(client):
    """Test single-process FastAPI serving frontend index.html with SPA fallback."""
    dist_index = Path(__file__).parent.parent.parent / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        res = client.get("/")
        assert res.status_code == 200
        assert "<!doctype html>" in res.text.lower() or "<html" in res.text.lower()

        # SPA history client route fallback
        res = client.get("/library")
        assert res.status_code == 200
        assert "<!doctype html>" in res.text.lower() or "<html" in res.text.lower()


def test_minimax_provider_sampling_parameters():
    """Verify MiniMaxProvider supports sampling controls (temperature, cfg_scale, top_k)."""
    from app.providers.minimax_provider import MiniMaxProvider
    provider = MiniMaxProvider()
    assert hasattr(provider, "generate")
    # Verify signature accepts sampling params
    import inspect
    sig = inspect.signature(provider.generate)
    assert "temperature" in sig.parameters
    assert "cfg_scale" in sig.parameters
    assert "top_k" in sig.parameters


def test_huggingface_audio_provider_registration():
    """Verify HuggingFaceAudioProvider dynamically registers via ProviderRegistry."""
    from app.providers.registry import ProviderRegistry
    from app.providers.hf_audio_provider import HuggingFaceAudioProvider

    hf_provider = ProviderRegistry.get_provider("hf:facebook/musicgen-small")
    assert isinstance(hf_provider, HuggingFaceAudioProvider)
    assert hf_provider.model_id == "facebook/musicgen-small"
    assert hf_provider.name == "hf:facebook/musicgen-small"


def test_image_service_raster_png_generation():
    """Verify ImageService generates true studio-grade raster PNGs."""
    import tempfile
    from PIL import Image
    from app.services.image_service import image_service

    cover_res = image_service.generate_cover(
        prompt="Cyberpunk synthesizer in Tokyo rain",
        visual_style="neon-cyberpunk",
        aspect_ratio="1:1"
    )
    assert cover_res is not None
    cover_path = cover_res.get("file_path") or cover_res.get("dest_path")
    assert cover_path is not None
    assert os.path.isfile(cover_path)
    assert cover_path.endswith(".png")

    with open(cover_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "File must have valid PNG magic bytes"

    img = Image.open(cover_path)
    assert img.size == (1024, 1024)
    assert img.mode in ("RGB", "RGBA")


@pytest.mark.asyncio
async def test_voice_service_acoustic_formant_equalization():
    """Verify VoiceService processes audio through acoustic formant EQ chains."""
    import tempfile
    import numpy as np
    import scipy.io.wavfile as wav
    from app.services.voice_service import voice_service

    sr = 44100
    t = np.linspace(0, 1.0, sr)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav.write(tf.name, sr, samples)
        input_audio = tf.name

    try:
        url_aria = await voice_service.convert_vocals(input_audio, profile="aria")
        aria_local = url_aria.replace("/audio/", "generated_audio/")
        assert os.path.isfile(aria_local)
        assert os.path.getsize(aria_local) > 0

        url_marcus = await voice_service.convert_vocals(input_audio, profile="marcus")
        marcus_local = url_marcus.replace("/audio/", "generated_audio/")
        assert os.path.isfile(marcus_local)
        assert os.path.getsize(marcus_local) > 0
    finally:
        if os.path.isfile(input_audio):
            os.remove(input_audio)


@pytest.mark.asyncio
async def test_video_service_lip_sync_and_procedural_broll():
    """Verify VideoService renders viseme lip-sync and dynamic procedural B-roll clips."""
    import tempfile
    import cv2
    import numpy as np
    import scipy.io.wavfile as wav

    video_srv = VideoService()

    # Create dummy portrait
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (40, 30, 50)
    # Face circle & lips
    cv2.circle(img, (320, 180), 90, (180, 190, 210), -1)
    cv2.ellipse(img, (320, 220), (25, 8), 0, 0, 360, (50, 40, 120), -1)

    # Create dummy vocal audio with tone
    sr = 44100
    t = np.linspace(0, 1.0, sr)
    audio = (np.sin(2 * np.pi * 330 * t) * 16000).astype(np.int16)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "face.png")
        cv2.imwrite(img_path, img)

        audio_path = os.path.join(tmpdir, "vocal.wav")
        wav.write(audio_path, sr, audio)

        lip_clip = os.path.join(tmpdir, "lip_clip.mp4")
        await video_srv.render_lip_sync_clip(
            face_image_path=img_path,
            vocal_audio_path=audio_path,
            start_time=0.0,
            duration=1.0,
            out_path=lip_clip,
            width=640,
            height=360
        )
        assert os.path.isfile(lip_clip)
        assert os.path.getsize(lip_clip) > 1000, "Lip-sync MP4 clip must be generated"

        broll_clip = os.path.join(tmpdir, "broll_clip.mp4")
        await video_srv.render_broll_clip(
            style="neon-cyberpunk",
            duration=1.0,
            out_path=broll_clip,
            width=640,
            height=360
        )
        assert os.path.isfile(broll_clip)
        assert os.path.getsize(broll_clip) > 1000, "Procedural B-roll MP4 clip must be generated"


@pytest.mark.asyncio
async def test_voice_service_dataset_ingestion_and_f0_analysis():
    """Verify voice profile dataset ingestion extracts acoustic properties (F0, spectral centroid) and generates audio preview."""
    import tempfile
    import numpy as np
    import scipy.io.wavfile as wav
    from app.services.voice_service import voice_service

    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # 261.63 Hz (Middle C)
    samples = (np.sin(2 * np.pi * 261.63 * t) * 16000).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav.write(tf.name, sr, samples)
        wav_path = tf.name

    try:
        with open(wav_path, "rb") as f:
            dataset_bytes = f.read()

        profile = voice_service.create_profile(
            name="Test Acoustic Ingestion",
            description="Profile with synthetic C4 vocal tone",
            consent_confirmed=True,
            f0_method="rmvpe",
            dataset_bytes=dataset_bytes,
            dataset_filename="c4_tone.wav"
        )

        assert profile["name"] == "Test Acoustic Ingestion"
        assert profile["acoustic_features"] is not None
        assert "median_f0_hz" in profile["acoustic_features"]
        assert "spectral_centroid_hz" in profile["acoustic_features"]
        assert profile["sample_audio_path"] is not None

        # Verify preview was generated on disk
        preview_local = profile["sample_audio_path"].replace("/audio/", "generated_audio/")
        assert os.path.isfile(preview_local)
        assert os.path.getsize(preview_local) > 0

        # Verify profile can be fetched
        fetched = voice_service.get_profile(profile["id"])
        assert fetched is not None
        assert fetched["id"] == profile["id"]

        # Clean up
        voice_service.delete_profile(profile["id"])
        assert voice_service.get_profile(profile["id"]) is None
    finally:
        if os.path.isfile(wav_path):
            os.remove(wav_path)


@pytest.mark.asyncio
async def test_voice_service_dry_wet_and_pitch_shifting():
    """Verify VoiceService processes pitch shifting, formant preservation, and dry/wet blending."""
    import tempfile
    import numpy as np
    import scipy.io.wavfile as wav
    from app.services.voice_service import voice_service

    sr = 44100
    t = np.linspace(0, 1.0, sr)
    samples = (np.sin(2 * np.pi * 330.0 * t) * 16000).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav.write(tf.name, sr, samples)
        input_audio = tf.name

    try:
        # Convert with pitch shift + 50% dry/wet
        out_url = await voice_service.convert_vocals(
            vocal_stem_path=input_audio,
            profile_id="default_aria",
            pitch_shift=2,
            dry_wet=0.5,
            formant_preserve=True
        )
        local_path = out_url.replace("/audio/", "generated_audio/")
        assert os.path.isfile(local_path)
        assert os.path.getsize(local_path) > 1000
    finally:
        if os.path.isfile(input_audio):
            os.remove(input_audio)


def test_voice_service_remix_master_with_stems():
    """Verify remix_master_with_vocal sums non-vocal stems with converted vocals into full stereo master."""
    import tempfile
    import numpy as np
    import scipy.io.wavfile as wav
    from app.services.voice_service import voice_service

    sr = 44100
    t = np.linspace(0, 1.0, sr)
    vocal_samples = (np.sin(2 * np.pi * 440.0 * t) * 12000).astype(np.int16)
    drums_samples = (np.sin(2 * np.pi * 100.0 * t) * 14000).astype(np.int16)
    bass_samples = (np.sin(2 * np.pi * 60.0 * t) * 14000).astype(np.int16)

    temp_files = []
    try:
        def make_temp_wav(data):
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.write(tf.name, sr, data)
            temp_files.append(tf.name)
            return tf.name

        vocal_file = make_temp_wav(vocal_samples)
        drums_file = make_temp_wav(drums_samples)
        bass_file = make_temp_wav(bass_samples)

        stems = {
            "vocals": vocal_file,
            "drums": drums_file,
            "bass": bass_file
        }

        remix_url = voice_service.remix_master_with_vocal(
            original_audio_path=vocal_file,
            converted_vocal_path=vocal_file,
            stems_dict=stems,
            output_filename="test_remix_run.wav"
        )
        local_remix = remix_url.replace("/audio/", "generated_audio/")
        assert os.path.isfile(local_remix)
        assert os.path.getsize(local_remix) > 1000
    finally:
        for f in temp_files:
            if os.path.isfile(f):
                os.remove(f)


@pytest.mark.asyncio
async def test_voice_profile_endpoints_json_and_multipart():
    """Verify /voice/profiles handles both JSON and multipart form data, and voice-convert remixes master track."""
    import json
    import httpx
    from app.main import app, engine
    from app.models import Job
    from sqlmodel import Session
    import tempfile
    import numpy as np
    import scipy.io.wavfile as wav

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Test JSON creation
        res_json = await client.post("/voice/profiles", json={
            "name": "Endpoint JSON Profile",
            "description": "Created via JSON API",
            "consent_confirmed": True,
            "f0_method": "rmvpe"
        })
        assert res_json.status_code == 200
        profile_data = res_json.json()["profile"]
        p_id = profile_data["id"]

        # 2. Test Multipart creation with file upload
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        samples = (np.sin(2 * np.pi * 300 * t) * 16000).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav.write(tf.name, sr, samples)
            temp_wav = tf.name

        try:
            with open(temp_wav, "rb") as f:
                res_form = await client.post(
                    "/voice/profiles",
                    data={
                        "name": "Endpoint Multipart Profile",
                        "description": "Created via Multipart Form",
                        "consent_confirmed": "true",
                        "f0_method": "crepe"
                    },
                    files={"audio_file": ("vocal_sample.wav", f, "audio/wav")}
                )
            assert res_form.status_code == 200
            form_profile = res_form.json()["profile"]
            assert form_profile["name"] == "Endpoint Multipart Profile"
            assert form_profile["acoustic_features"] is not None

            # Clean up form profile
            await client.delete(f"/voice/profiles/{form_profile['id']}")
        finally:
            if os.path.isfile(temp_wav):
                os.remove(temp_wav)

        # 3. Test Voice Conversion route on a Job
        # Create a dummy completed job with audio
        dummy_wav_dest = "generated_audio/test_vc_source.wav"
        wav.write(dummy_wav_dest, sr, samples)

        with Session(engine) as session:
            job = Job(
                prompt="Original prompt",
                title="Original Track",
                audio_path="/audio/test_vc_source.wav",
                stems_json=json.dumps({"vocals": "/audio/test_vc_source.wav"}),
                status="completed"
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            test_job_id = str(job.id)

        vc_res = await client.post(f"/jobs/{test_job_id}/voice-convert", json={
            "voice_profile_id": p_id,
            "pitch_shift": 1,
            "dry_wet": 0.8,
            "formant_preserve": True
        })
        assert vc_res.status_code == 200
        converted_job = vc_res.json()
        assert converted_job["parent_job_id"] == test_job_id
        assert converted_job["voice_profile_id"] == p_id
        assert converted_job["audio_path"] is not None
        # Verify remixed audio file exists
        remixed_local = converted_job["audio_path"].replace("/audio/", "generated_audio/")
        assert os.path.isfile(remixed_local)

        # Clean up json profile
        await client.delete(f"/voice/profiles/{p_id}")


