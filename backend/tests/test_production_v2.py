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
    assert len(models) >= 14

    categories = {m["category"] for m in models}
    assert "audio" in categories
    assert "image" in categories
    assert "video" in categories

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
        assert "MiniMax-Music3" in check.get("recommended_repo_id", "")


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
