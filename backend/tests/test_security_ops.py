"""Phase-1 security & ops tests — hermetic, env-driven, no network."""
from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest

from app.main import app, reconcile_orphan_jobs, _delete_job_artifacts, engine
from app.models import Job, JobStatus
from sqlmodel import Session


class SyncTestClient:
    """httpx 0.27+-compatible sync client over ASGI transport."""

    def __init__(self, asgi_app):
        self.transport = httpx.ASGITransport(app=asgi_app)

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
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


@pytest.fixture()
def client():
    return SyncTestClient(app)


# ---------------------------------------------------------------------------
# Bearer-token auth
# ---------------------------------------------------------------------------
class TestAuth:
    def test_open_when_token_unset(self, client, monkeypatch):
        monkeypatch.delenv("MILIMO_AUTH_TOKEN", raising=False)
        assert client.get("/profiles").status_code == 200

    def test_401_without_header_when_token_set(self, client, monkeypatch):
        monkeypatch.setenv("MILIMO_AUTH_TOKEN", "sekrit")
        r = client.get("/profiles")
        assert r.status_code == 401
        assert r.json()["detail"]["error"]["code"] == "unauthorized"

    def test_200_with_bearer_when_token_set(self, client, monkeypatch):
        monkeypatch.setenv("MILIMO_AUTH_TOKEN", "sekrit")
        r = client.get("/profiles", headers={"Authorization": "Bearer sekrit"})
        assert r.status_code == 200

    def test_query_param_token_for_eventsource_style_clients(self, client, monkeypatch):
        monkeypatch.setenv("MILIMO_AUTH_TOKEN", "sekrit")
        r = client.get("/profiles?auth=sekrit")
        assert r.status_code == 200

    def test_health_exempt_even_with_token(self, client, monkeypatch):
        monkeypatch.setenv("MILIMO_AUTH_TOKEN", "sekrit")
        assert client.get("/health").status_code == 200

    def test_wrong_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("MILIMO_AUTH_TOKEN", "sekrit")
        r = client.get("/profiles", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Upload hardening
# ---------------------------------------------------------------------------
class TestUploadHardening:
    def test_bad_extension_rejected(self, client):
        r = client.post("/upload/image",
                        files={"file": ("evil.svg", b"<svg onload='x'></svg>")})
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "bad_type"

    def test_svg_content_sniff_rejected(self, client):
        # claims .png but content is SVG text — magic-byte sniff must reject
        r = client.post("/upload/image",
                        files={"file": ("fake.png", b"<svg xmlns='x'></svg>")})
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "bad_content"

    def test_valid_png_accepted(self, client):
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080600000"
            "01f15c4890000000d49444154789c626001000000ffff030000060505"
            "7de24db10000000049454e44ae426082"
        )
        r = client.post("/upload/image", files={"file": ("ok.png", png)})
        assert r.status_code == 200
        assert r.json()["filename"].endswith(".png")

    def test_transcribe_rejects_non_audio(self, client):
        r = client.post("/transcribe/upload",
                        files={"file": ("notes.txt", b"definitely not audio")})
        assert r.status_code == 400

    def test_oversize_audio_rejected(self, client, monkeypatch):
        monkeypatch.setenv("MAX_AUDIO_UPLOAD_MB", "1")
        # 1.5 MB of WAV-magic-prefixed junk exceeds the 1 MB cap
        body = b"RIFF" + b"\x00" * (int(1.5 * 1024 * 1024))
        r = client.post("/transcribe/upload", files={"file": ("big.wav", body)})
        assert r.status_code == 413
        assert r.json()["detail"]["error"]["code"] == "payload_too_large"

    def test_randomized_filename_no_traversal(self, client):
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080600000"
            "01f15c4890000000d49444154789c626001000000ffff030000060505"
            "7de24db10000000049454e44ae426082"
        )
        r = client.post("/upload/image",
                        files={"file": ("../../evil_name.png", png)})
        assert r.status_code == 200
        fname = r.json()["filename"]
        assert "/" not in fname and ".." not in fname


# ---------------------------------------------------------------------------
# Boot reconciliation
# ---------------------------------------------------------------------------
def test_reconcile_orphan_jobs():
    with Session(engine) as session:
        zombie = Job(prompt="reconcile-test", status=JobStatus.PROCESSING)
        session.add(zombie)
        session.commit()
        zombie_id = zombie.id

    reconcile_orphan_jobs()

    with Session(engine) as session:
        refreshed = session.get(Job, zombie_id)
        assert refreshed.status == JobStatus.FAILED
        assert "restart" in (refreshed.error_msg or "").lower()
        session.delete(refreshed)
        session.commit()


# ---------------------------------------------------------------------------
# Cascade artifact deletion
# ---------------------------------------------------------------------------
def test_cascade_delete_removes_all_artifacts(tmp_path, monkeypatch):
    jid = str(uuid.uuid4())
    paths = [
        f"generated_audio/stems/{jid}_vocals.wav",
        f"generated_audio/stems/{jid}_Acoustic Piano.wav",   # instrument part
        f"generated_audio/mastered/{jid}_mastered.wav",
        f"generated_tokens/{jid}.json",
        f"generated_audio/{jid}.mid",
    ]
    try:
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b"x")

        removed = _delete_job_artifacts(jid, f"/audio/{jid}.mp3")

        assert removed >= len(paths) - 0  # every seeded artifact swept
        for p in paths:
            assert not os.path.exists(p), f"{p} survived cascade delete"
    finally:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)


# ---------------------------------------------------------------------------
# Rate limiting (unit-level: window math through the live middleware)
# ---------------------------------------------------------------------------
def test_rate_limit_enforced_on_protected_prefix(client, monkeypatch):
    from app.core import ratelimit
    monkeypatch.setenv("MILIMO_RATE_LIMIT_PER_MIN", "3")
    ratelimit._hits.clear()  # isolate window state

    codes = []
    for _ in range(6):
        r = client.get("/agents")
        codes.append(r.status_code)

    # /agents (listing) is NOT in protected prefixes — use a protected one.
    codes = []
    ratelimit._hits.clear()
    for i in range(6):
        payload = {"input": {}}
        r = client.post("/agents/experiencer/run", json=payload)
        codes.append(r.status_code)
    assert 429 in codes, f"expected a 429 among {codes}"
