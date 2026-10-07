"""
Integration tests for Standalone Image Studio & Visual Asset Gallery.
Verifies end-to-end local asset generation, gallery cataloging, cover assignment,
favorite toggles, and asset lifecycle management.
"""

import asyncio
import uuid
import httpx
import pytest
from sqlmodel import Session

from app.main import app, engine
from app.models import Job, JobStatus, VisualAsset
from app.services.image_service import image_service


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

    def patch(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.patch(url, **kwargs)
        return self._run(_call())

    def delete(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.delete(url, **kwargs)
        return self._run(_call())


@pytest.fixture
def client():
    return SyncTestClient(app)


def test_visual_asset_studio_lifecycle(client, monkeypatch):
    # Mock heavy neural diffusion generation to ensure ultra-fast unit test execution
    test_cover_url = "/covers/ai_cover_mock123.png"
    monkeypatch.setattr(
        image_service,
        "generate_cover",
        lambda **kwargs: {
            "url": test_cover_url,
            "file_path": "/tmp/ai_cover_mock123.png",
            "dest_path": "/tmp/ai_cover_mock123.png",
            "prompt": kwargs.get("prompt", ""),
            "style": kwargs.get("style", ""),
            "model_id": "custom_aitrader_flux2_klein_9b_mlx_4bit",
            "model_name": "FLUX.2 Klein 9B (MLX 4-bit)",
            "is_installed": True,
            "engine": "mlx",
            "format": "png",
        }
    )

    with Session(engine) as session:
        # Create a test track job
        test_job = Job(
            id=uuid.uuid4(),
            prompt="High energy electronic dance anthem",
            status=JobStatus.COMPLETED,
            title="Neon Horizon",
        )
        session.add(test_job)
        session.commit()
        session.refresh(test_job)
        job_id = str(test_job.id)

    # 1. Generate a new visual asset via Image Studio API
    gen_payload = {
        "prompt": "Futuristic neon city skyline with holographic synthwave sunset",
        "title": "Neon Horizon Cover",
        "style": "neon cyberpunk synthwave",
        "aspect_ratio": "1:1",
        "asset_type": "album_cover",
        "linked_job_id": job_id,
    }
    resp = client.post("/images/generate", json=gen_payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert data["image_url"] == test_cover_url
    asset = data["asset"]
    asset_id = asset["id"]
    assert asset["title"] == "Neon Horizon Cover"
    assert asset["asset_type"] == "album_cover"
    assert asset["aspect_ratio"] == "1:1"

    # 2. Verify asset is listed in the gallery
    gallery_resp = client.get("/images/gallery")
    assert gallery_resp.status_code == 200
    gallery_data = gallery_resp.json()
    asset_ids = [a["id"] for a in gallery_data["assets"]]
    assert asset_id in asset_ids

    # 3. Test filtering gallery by type
    type_resp = client.get("/images/gallery?asset_type=album_cover")
    assert type_resp.status_code == 200
    assert any(a["id"] == asset_id for a in type_resp.json()["assets"])

    # 4. Patch asset (Toggle Favorite & update title)
    patch_resp = client.patch(
        f"/images/assets/{asset_id}",
        json={"title": "Neon Horizon Master Cover", "is_favorite": True},
    )
    assert patch_resp.status_code == 200
    patched_data = patch_resp.json()
    assert patched_data["asset"]["is_favorite"] is True
    assert patched_data["asset"]["title"] == "Neon Horizon Master Cover"

    fav_resp = client.get("/images/gallery?favorite_only=true")
    assert fav_resp.status_code == 200
    assert any(a["id"] == asset_id for a in fav_resp.json()["assets"])

    # 5. Set asset as job cover
    set_cover_resp = client.post(f"/images/assets/{asset_id}/set-cover/{job_id}")
    assert set_cover_resp.status_code == 200
    cover_result = set_cover_resp.json()
    assert cover_result["success"] is True
    assert cover_result["job_id"] == job_id
    assert cover_result["cover_image_path"] == test_cover_url

    # Verify Job in DB now points to this cover
    with Session(engine) as session:
        refreshed_job = session.get(Job, uuid.UUID(job_id))
        assert refreshed_job is not None
        assert refreshed_job.cover_image_path == test_cover_url

    # 6. Delete asset from gallery
    del_resp = client.delete(f"/images/assets/{asset_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["success"] is True
    assert del_data["deleted_id"] == asset_id

    # Verify asset is no longer in gallery
    post_del_resp = client.get("/images/gallery")
    assert post_del_resp.status_code == 200
    assert all(a["id"] != asset_id for a in post_del_resp.json()["assets"])

    # Clean up test job
    with Session(engine) as session:
        j = session.get(Job, uuid.UUID(job_id))
        if j:
            session.delete(j)
            session.commit()
