"""Regression tests for media-serving correctness (Phase 2).

- /audio honors single byte ranges (206) and rejects bad ones (416).
- CORS defaults cover 127.0.0.1 dev origins.
- /download_track sniffs media type from extension and 404s on missing files.
"""

import os
import sys
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.core.paths import get_generated_audio_dir


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture()
def probe_wav():
    path = get_generated_audio_dir() / ".test_range_probe.wav"
    path.write_bytes(os.urandom(256 * 1024))
    try:
        yield "/audio/.test_range_probe.wav"
    finally:
        try:
            path.unlink()
        except OSError:
            pass


@pytest.mark.asyncio
async def test_audio_full_response_has_accept_ranges(client, probe_wav):
    r = await client.get(probe_wav)
    assert r.status_code == 200
    assert r.headers.get("accept-ranges") == "bytes"
    assert len(r.content) == 256 * 1024


@pytest.mark.asyncio
async def test_audio_single_range_206(client, probe_wav):
    r = await client.get(probe_wav, headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206, r.status_code
    assert r.headers["content-range"] == f"bytes 0-1023/{256 * 1024}"
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 1024


@pytest.mark.asyncio
async def test_audio_open_ended_and_suffix_ranges(client, probe_wav):
    size = 256 * 1024
    r = await client.get(probe_wav, headers={"Range": f"bytes={size - 100}-"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes {size - 100}-{size - 1}/{size}"
    assert len(r.content) == 100

    r = await client.get(probe_wav, headers={"Range": "bytes=-50"})
    assert r.status_code == 206
    assert len(r.content) == 50


@pytest.mark.asyncio
async def test_audio_unsatisfiable_range_416(client, probe_wav):
    r = await client.get(probe_wav, headers={"Range": "bytes=999999999-"})
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{256 * 1024}"


@pytest.mark.asyncio
async def test_audio_missing_file_404(client):
    r = await client.get("/audio/does-not-exist-xyz.wav")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_cors_allows_loopback_origins(client):
    for origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
        r = await client.get("/health", headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") == origin, origin


@pytest.mark.asyncio
async def test_download_track_sniffs_type_and_404s(client, monkeypatch, tmp_path):
    import app.main as main_module

    # Real file so FileResponse streams it; CWD-relative mount target.
    song_path = Path(__file__).parent.parent / "generated_audio" / ".test_dl_song.wav"
    song_path.write_bytes(os.urandom(4096))
    try:
        class FakeJob:
            title = "My Song"
            audio_path = "/audio/.test_dl_song.wav"

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(main_module, "Session", lambda engine: FakeSession())
        monkeypatch.setattr(main_module, "get_job_by_id", lambda s, jid: FakeJob() if jid == "real" else None)

        r = await client.get("/download_track/real")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "audio/wav"
        assert "My_Song.wav" in r.headers.get("content-disposition", "")
        assert len(r.content) == 4096

        r = await client.get("/download_track/ghost")
        assert r.status_code == 404
    finally:
        try:
            song_path.unlink()
        except OSError:
            pass
