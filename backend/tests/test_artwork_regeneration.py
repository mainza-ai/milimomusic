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
async def test_artwork_regeneration_uses_active_model(client):
    """Verify that regenerating artwork via POST /jobs/{job_id}/generate-cover works without MLX stream errors."""
    pytest.importorskip("mflux", reason="Requires mflux MLX diffusion (Apple Silicon local test)")
    test_job = Job(
        id=uuid.uuid4(),
        title="Regeneration Neural Track",
        prompt="Cyberpunk night alley with neon reflections and glowing synths",
        tags="synthwave, electronic, cyberpunk",
        status="completed",
        audio_path="/audio/test.wav"
    )
    with Session(engine) as session:
        session.add(test_job)
        session.commit()
        session.refresh(test_job)

    # Trigger generation
    response = await client.post(f"/jobs/{test_job.id}/generate-cover", json={})
    assert response.status_code == 200, f"Expected 200, got: {response.text}"
    data = response.json()
    assert data.get("cover_image_path") is not None
    cover_path = data["cover_image_path"]
    assert cover_path.startswith("/covers/")

    # Verify image file exists and is authentic (> 100 KB)
    local_cover_file = os.path.join("data", "covers", os.path.basename(cover_path))
    assert os.path.exists(local_cover_file), f"Cover file {local_cover_file} does not exist"
    assert os.path.getsize(local_cover_file) > 100000, f"Expected neural cover file > 100KB, got {os.path.getsize(local_cover_file)}"

@pytest.mark.asyncio
async def test_non_uuid_job_id_does_not_crash(client):
    """Verify that non-UUID strings like models return 404 cleanly instead of crashing with 500 StatementError."""
    response = await client.get("/jobs/models")
    assert response.status_code in [404, 422], f"Expected 404 or 422, got {response.status_code}"
