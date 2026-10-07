import io
import os
import sys
import uuid
from pathlib import Path
import pytest
import httpx
from PIL import Image
from sqlmodel import Session, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app, engine
from app.models import Job, VisualAsset


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def create_test_image_bytes(format="PNG", size=(64, 64), color="blue") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_job_cover_png_success(client):
    """Verify that uploading a PNG image sets cover_image_path on the job and catalogs VisualAsset."""
    test_job = Job(
        id=uuid.uuid4(),
        title="Custom Upload Album",
        prompt="Acoustic guitar in sunrise",
        tags="folk, acoustic",
        status="completed",
        audio_path="/audio/test_upload.wav"
    )
    with Session(engine) as session:
        session.add(test_job)
        session.commit()
        session.refresh(test_job)

    png_bytes = create_test_image_bytes("PNG", (128, 128), "cyan")
    files = {"file": ("my_artwork.png", png_bytes, "image/png")}

    response = await client.post(f"/jobs/{test_job.id}/upload-cover", files=files)
    assert response.status_code == 200, f"Expected 200, got: {response.text}"
    data = response.json()
    assert data.get("cover_image_path") is not None
    assert data["cover_image_path"].startswith("/covers/")
    assert data["cover_image_path"].endswith(".png")

    # Verify database persistence
    with Session(engine) as session:
        updated = session.get(Job, test_job.id)
        assert updated is not None
        assert updated.cover_image_path == data["cover_image_path"]

        # Verify VisualAsset cataloging
        asset = session.exec(
            select(VisualAsset).where(VisualAsset.linked_job_id == str(test_job.id))
        ).first()
        assert asset is not None
        assert asset.image_url == data["cover_image_path"]
        assert asset.style == "user_upload"


@pytest.mark.asyncio
async def test_upload_job_cover_jpeg_success(client):
    """Verify that uploading a JPEG image sets cover_image_path on the job."""
    test_job = Job(
        id=uuid.uuid4(),
        title="JPEG Cover Album",
        prompt="Electric synthwave sunset",
        tags="synthwave",
        status="completed",
        audio_path="/audio/test_jpeg.wav"
    )
    with Session(engine) as session:
        session.add(test_job)
        session.commit()
        session.refresh(test_job)

    jpg_bytes = create_test_image_bytes("JPEG", (100, 100), "orange")
    files = {"file": ("album_cover.jpg", jpg_bytes, "image/jpeg")}

    response = await client.post(f"/jobs/{test_job.id}/upload-cover", files=files)
    assert response.status_code == 200, f"Expected 200, got: {response.text}"
    data = response.json()
    assert data["cover_image_path"].startswith("/covers/")
    assert data["cover_image_path"].endswith(".jpg")


@pytest.mark.asyncio
async def test_upload_job_cover_404_for_missing_job(client):
    """Verify that uploading cover to a non-existent job returns 404."""
    random_id = str(uuid.uuid4())
    png_bytes = create_test_image_bytes("PNG")
    files = {"file": ("cover.png", png_bytes, "image/png")}

    response = await client.post(f"/jobs/{random_id}/upload-cover", files=files)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_job_cover_rejects_svg_and_non_images(client):
    """Verify that SVG (XSS vector) and executable files are rejected with 400 Bad Request."""
    job_id = uuid.uuid4()
    test_job = Job(
        id=job_id,
        title="Security Test Album",
        prompt="Test",
        status="completed"
    )
    with Session(engine) as session:
        session.add(test_job)
        session.commit()

    # SVG rejection
    svg_payload = b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"
    files = {"file": ("malicious.svg", svg_payload, "image/svg+xml")}
    response = await client.post(f"/jobs/{job_id}/upload-cover", files=files)
    assert response.status_code == 400
    assert "bad_type" in response.text

    # EXE / binary rejection
    files = {"file": ("script.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")}
    response = await client.post(f"/jobs/{job_id}/upload-cover", files=files)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_standalone_upload_image_endpoint(client):
    """Verify that POST /upload/image works and returns public /covers/ URL."""
    png_bytes = create_test_image_bytes("PNG", (80, 80), "purple")
    files = {"file": ("standalone_cover.png", png_bytes, "image/png")}

    response = await client.post("/upload/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert data["url"].startswith("/covers/")
    assert data["url"].endswith(".png")
