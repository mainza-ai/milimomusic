"""Regression tests for the file-read and auth findings (plan tasks 0.1–0.3).

The audit found a three-request, unauthenticated read of any file the process could
open: `POST /tracks/{id}/commit-vocal` accepted any `master_path`, stored it as
`job.audio_path`, and `GET /download_track/{id}` served it. These tests pin the fix
so it cannot come back.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.media_paths import (
    ALL_MEDIA_EXTS,
    MediaPathError,
    is_allowed_media_path,
    media_roots,
    resolve_media_reference,
    safe_media_name,
)


class _SyncClient:
    """Minimal sync wrapper over the ASGI app (the suite has no shared fixture)."""

    def __init__(self, asgi_app):
        import asyncio

        import httpx

        self._asyncio = asyncio
        self.transport = httpx.ASGITransport(app=asgi_app)

    def _run(self, coro):
        loop = self._asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def request(self, method, url, **kwargs):
        import httpx

        async def _go():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as c:
                return await c.request(method, url, **kwargs)

        return self._run(_go())

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)


@pytest.fixture
def client():
    from app.main import app

    return _SyncClient(app)


# ── media_paths unit behaviour ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "bad",
    [
        "/etc/passwd",
        "/etc/hosts",
        "../../.env",
        "../../../etc/passwd",
        "..\\..\\.env",
        "/tmp/whatever.wav",
    ],
)
def test_outside_media_roots_is_refused(bad):
    with pytest.raises(MediaPathError):
        resolve_media_reference(bad, allow_exts=ALL_MEDIA_EXTS, field="master_path")


def test_nul_byte_is_refused():
    with pytest.raises(MediaPathError):
        resolve_media_reference("/audio/x.wav\x00.png", field="p")


def test_repo_secrets_are_not_media(tmp_path):
    """The two files that made the exploit worth doing."""
    repo = Path(__file__).resolve().parents[2]
    for secret in (repo / "backend" / "llm_config.json", repo / ".env"):
        if not secret.exists():
            continue
        assert not is_allowed_media_path(secret), f"{secret} must not be servable"


def test_public_url_resolves_into_media_roots(tmp_path):
    root = media_roots()[0]
    name = f"_probe_{uuid.uuid4().hex[:8]}.wav"
    target = root / name
    target.write_bytes(b"RIFF....WAVE")
    try:
        resolved = resolve_media_reference(f"/audio/{name}", allow_exts=ALL_MEDIA_EXTS, field="audio_path")
        assert resolved == str(target.resolve())
        assert is_allowed_media_path(resolved)
        # absolute form of the same legit path is also accepted
        assert resolve_media_reference(str(target), allow_exts=ALL_MEDIA_EXTS, field="audio_path")
    finally:
        target.unlink(missing_ok=True)


def test_wrong_extension_is_rejected():
    root = media_roots()[0]
    name = f"_probe_{uuid.uuid4().hex[:8]}.json"
    target = root / name
    target.write_text("{}")
    try:
        with pytest.raises(MediaPathError):
            resolve_media_reference(str(target), allow_exts=ALL_MEDIA_EXTS, field="master_path")
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.parametrize("raw", ["../../.env", "a/b/../../../etc/passwd", "x\x00y"])
def test_safe_media_name_strips_paths(raw):
    with pytest.raises(ValueError):
        safe_media_name(raw)


def test_safe_media_name_returns_bare_filename():
    assert safe_media_name("/audio/clip.mp4") == "clip.mp4"
    assert safe_media_name("clip.mp4?t=123") == "clip.mp4"


# ── endpoint behaviour ──────────────────────────────────────────────────────

def _make_job(session_factory, engine):
    from app.models import Job

    job_id = uuid.uuid4()
    with session_factory(engine) as session:
        session.add(Job(
            id=job_id,
            prompt="security probe",
            status="COMPLETED",
            audio_path="/audio/_probe_missing.wav",
        ))
        session.commit()
    return job_id


def test_commit_vocal_rejects_arbitrary_master_path(client, tmp_path):
    """The exploit chain's first request: `master_path` must not be an arbitrary file."""
    from sqlmodel import Session

    from app.main import engine

    job_id = _make_job(Session, engine)
    res = client.post(
        f"/tracks/{job_id}/commit-vocal",
        json={"vocal_path": "/audio/x.wav", "master_path": "/etc/passwd"},
    )
    assert res.status_code in (400, 404), res.text
    assert "passwd" not in res.text


def test_commit_vocal_rejects_traversal(client):
    from sqlmodel import Session

    from app.main import engine

    job_id = _make_job(Session, engine)
    res = client.post(
        f"/tracks/{job_id}/commit-vocal",
        json={"vocal_path": "../../.env"},
    )
    assert res.status_code == 400, res.text


def test_download_track_refuses_paths_outside_media_roots(client):
    """Second half of the chain: even a poisoned row must not serve a host file."""
    from sqlmodel import Session

    from app.main import engine
    from app.models import Job

    job_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(
            id=job_id,
            prompt="poisoned row",
            status="COMPLETED",
            audio_path=str(Path("/etc/passwd")),
        ))
        session.commit()

    res = client.get(f"/download_track/{job_id}")
    assert res.status_code == 404
    assert b"root:" not in res.content


def test_thumbnail_rejects_traversal(client):
    res = client.get("/api/v1/thumbnail/..%2F..%2F.env")
    assert res.status_code in (400, 404)
    assert b"OPENAI" not in res.content


def test_gallery_route_rejects_arbitrary_path(client):
    res = client.post("/api/gallery/route", json={"media_path": "/etc/hosts", "target_slot": "references"})
    assert res.status_code in (400, 404)


# ── auth gate semantics (pure function) ────────────────────────────────────

def test_auth_gate_blocks_remote_when_no_token(monkeypatch):
    from app.core.security import evaluate_request

    monkeypatch.delenv("MILIMO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MILIMO_ALLOW_ANONYMOUS", raising=False)

    assert evaluate_request("GET", "/history", {}, "", "127.0.0.1")[0] is True
    assert evaluate_request("GET", "/history", {}, "", "203.0.113.9")[0] is False
    # mounts must not escape the gate
    assert evaluate_request("GET", "/stems/x.wav", {}, "", "203.0.113.9")[0] is False
    # health stays reachable for operators
    assert evaluate_request("GET", "/health", {}, "", "203.0.113.9")[0] is True


def test_auth_gate_allows_explicit_anonymous_override(monkeypatch):
    from app.core.security import evaluate_request

    monkeypatch.delenv("MILIMO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("MILIMO_ALLOW_ANONYMOUS", "1")
    assert evaluate_request("GET", "/history", {}, "", "203.0.113.9")[0] is True


def test_auth_gate_query_token_only_on_events(monkeypatch):
    from app.core.security import evaluate_request

    monkeypatch.setenv("MILIMO_AUTH_TOKEN", "s3cret")
    assert evaluate_request("GET", "/events", {}, "s3cret", "203.0.113.9")[0] is True
    assert evaluate_request("GET", "/history", {}, "s3cret", "203.0.113.9")[0] is False


# ── share links (decision D2) ──────────────────────────────────────────────

def test_share_token_roundtrip():
    from app.core.sharing import mint_share_token, verify_share_token

    share = mint_share_token("job-123", ttl_hours=1)
    payload = verify_share_token(share["token"])
    assert payload is not None
    assert payload["j"] == "job-123"
    assert payload["k"] == "track"


def test_share_token_rejects_tampering():
    from app.core.sharing import mint_share_token, verify_share_token

    token = mint_share_token("job-123", ttl_hours=1)["token"]
    body, _, sig = token.partition(".")
    # Swap the payload but keep the signature: must be rejected.
    forged = base64.urlsafe_b64encode(b'{"j":"job-999","k":"track","e":9999999999}').decode().rstrip("=")
    assert verify_share_token(f"{forged}.{sig}") is None
    # …and a truncated signature.
    assert verify_share_token(f"{body}.{sig[:-4]}") is None


def test_share_token_expires(monkeypatch):
    from app.core import sharing

    token = sharing.mint_share_token("job-123", ttl_hours=0.001)["token"]
    assert sharing.verify_share_token(token) is not None
    real_time = time.time()
    monkeypatch.setattr(sharing.time, "time", lambda: real_time + 3600)
    assert sharing.verify_share_token(token) is None


def test_share_routes_are_public_and_validated(client):
    """`/share/...` needs no API token (the link is the credential) but a bad one 403s."""
    res = client.get("/share/not-a-real-token")
    assert res.status_code == 403
    assert res.json()["error"]["code"] == "invalid_share_token"


def test_share_link_serves_a_track_without_auth(client):
    import uuid as _uuid
    from sqlmodel import Session

    from app.core.sharing import mint_share_token
    from app.main import engine
    from app.models import Job

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="share probe", title="Shareable",
                        status="completed", lyrics="hello"))
        session.commit()

    token = mint_share_token(str(job_id), ttl_hours=1)["token"]
    res = client.get(f"/share/{token}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["title"] == "Shareable"
    assert body["lyrics"] == "hello"
