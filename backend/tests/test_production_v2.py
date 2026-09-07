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
    assert VideoService.get_model_max_duration("cogvideox") == 6.0
    assert VideoService.get_model_max_duration("hailuo_h3") == 8.0
    assert VideoService.get_model_max_duration("MiniMax Hailuo H3 (8.0s clips)") == 8.0
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

    # Omitted max_clip_duration with Hailuo should default to 8.0s
    res_hailuo = client.post(f"/videos/plan/{job_id}", json={
        "model_name": "hailuo_h3"
    })
    assert res_hailuo.status_code == 200
    plan_hailuo = res_hailuo.json()
    assert plan_hailuo["model_max_duration"] == 8.0
    assert plan_hailuo["max_clip_duration"] == 8.0

    # User requesting 20.0s for Wan 2.1 should be clamped to 5.0s
    res_clamped = client.post(f"/videos/plan/{job_id}", json={
        "model_name": "wan2.1",
        "max_clip_duration": 20.0
    })
    assert res_clamped.status_code == 200
    plan_clamped = res_clamped.json()
    assert plan_clamped["max_clip_duration"] == 5.0


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
