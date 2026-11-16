import os
import sys
import uuid
from pathlib import Path
import pytest
import httpx
from sqlmodel import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app, engine
from app.models import Job

@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")

@pytest.mark.asyncio
async def test_audio_file_serving(client):
    """Verify a real file in the canonical audio dir is served (200/206)."""
    from app.core.paths import get_generated_audio_dir
    probe = get_generated_audio_dir() / ".dual_mount_probe.wav"
    probe.write_bytes(os.urandom(128 * 1024))
    try:
        response = await client.get("/audio/.dual_mount_probe.wav")
        assert response.status_code in [200, 206], f"Failed to serve audio: {response.status_code}"
        assert len(response.content) > 0
    finally:
        try:
            probe.unlink()
        except OSError:
            pass

@pytest.mark.asyncio
async def test_cover_image_serving(client, monkeypatch):
    """Verify generated cover artwork is served from /covers/ without 404.

    Diffusion is stubbed so the test proves the *serving* contract, not model output.
    (With real FLUX weights installed, the unstubbed call diffuses for minutes.)
    """
    from app.services import image_service as isvc

    def _no_diffusion(**kwargs):
        # The real method reports unavailability in its return value, not by raising.
        return {"engine": "none", "diffusion_error": "stubbed for static-serving test"}

    monkeypatch.setattr(isvc.image_service, "_render_diffusion_image", _no_diffusion, raising=False)
    result = isvc.image_service.generate_cover(
        prompt="dual mount cover probe",
        style="test style",
        model_id="nonexistent-ci-model",  # forces raster fallback
    )
    assert result.get("url", "").startswith("/covers/")
    response = await client.get(result["url"])
    assert response.status_code == 200, f"Failed to serve cover: {response.status_code}"
    assert response.headers.get("content-type", "").startswith("image/")
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_generate_cover_endpoint(client, monkeypatch):
    """Verify that POST /jobs/{job_id}/generate-cover generates artwork and updates the job.

    Diffusion is stubbed: this asserts the endpoint contract (a `/covers/` path that is
    immediately fetchable), not model quality. With FLUX weights on the machine the real
    call takes minutes and used to hang the suite.
    """
    from app.services import image_service as isvc

    def _no_diffusion(**kwargs):
        return {"engine": "none", "diffusion_error": "stubbed for endpoint contract test"}

    monkeypatch.setattr(isvc.image_service, "_render_diffusion_image", _no_diffusion, raising=False)

    test_job = Job(
        id=uuid.uuid4(),
        title="Test Cover Track",
        prompt="A cosmic synthwave journey across nebulae",
        status="completed",
        audio_path="/audio/test.wav"
    )
    with Session(engine) as session:
        session.add(test_job)
        session.commit()
        session.refresh(test_job)

    response = await client.post(f"/jobs/{test_job.id}/generate-cover", json={
        "style": "cinematic"
    })
    assert response.status_code == 200, f"Failed to generate cover: {response.text}"
    data = response.json()
    assert "cover_image_path" in data
    assert data["cover_image_path"] is not None
    assert "/covers/" in data["cover_image_path"]
    
    # Check that the cover is immediately fetchable via GET
    cover_path = data["cover_image_path"]
    if cover_path.startswith("http"):
        cover_url = "/" + cover_path.split("/", 3)[-1]
    else:
        cover_url = cover_path
    
    cover_res = await client.get(cover_url)
    assert cover_res.status_code == 200, f"Generated cover URL {cover_url} returned {cover_res.status_code}"
