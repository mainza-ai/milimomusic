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
    """Verify that the remixed master audio file for 6b5f64a5-7768-42f6-8bfb-6da1a6dbd961 is served (200/206)."""
    response = await client.get("/audio/6b5f64a5-7768-42f6-8bfb-6da1a6dbd961_remixed_master.wav")
    assert response.status_code in [200, 206], f"Failed to serve audio: {response.status_code}"
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_cover_image_serving(client):
    """Verify that cover artwork is served from /covers/ without 404."""
    response = await client.get("/covers/ai_cover_5c4e826616.png")
    assert response.status_code == 200, f"Failed to serve cover: {response.status_code}"
    assert response.headers.get("content-type", "").startswith("image/")
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_generate_cover_endpoint(client):
    """Verify that POST /jobs/{job_id}/generate-cover generates artwork and updates the job."""
    test_job = Job(
        id=str(uuid.uuid4()),
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
