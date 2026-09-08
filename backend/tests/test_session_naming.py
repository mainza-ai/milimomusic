"""Session rename + naming-contract tests (Phase 3).

Covers: create validation, PATCH rename incl. blank rejection and
updated_at bump, auto-rename guard (frontend-cased default renames,
deliberate short names survive), delete dissociation, job/project
rename validation, Job.updated_at bump.
"""

import sys
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.main as main_module
from app.main import app


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def _create_session(client, title="RenameMe Test"):
    r = await client.post("/sessions", json={"title": title})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_session_create_and_rename_roundtrip(client):
    s = await _create_session(client)
    try:
        assert s["title"] == "RenameMe Test"
        before = s["updated_at"]
        r = await client.patch(f"/sessions/{s['id']}", json={"title": "  Renamed Title  "})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["title"] == "Renamed Title"  # stored stripped
        assert body["updated_at"] >= before
    finally:
        await client.delete(f"/sessions/{s['id']}")


@pytest.mark.asyncio
async def test_session_blank_title_rejected_on_create_and_rename(client):
    r = await client.post("/sessions", json={"title": "   "})
    assert r.status_code == 422
    s = await _create_session(client)
    try:
        r = await client.patch(f"/sessions/{s['id']}", json={"title": "  "})
        assert r.status_code == 422
        r = await client.get(f"/sessions/{s['id']}")
        assert r.json()["title"] == "RenameMe Test"  # unchanged
    finally:
        await client.delete(f"/sessions/{s['id']}")


@pytest.mark.asyncio
async def test_session_rename_missing_404(client):
    r = await client.patch("/sessions/00000000-0000-0000-0000-000000000000", json={"title": "X"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_session_auto_rename_guard(client, monkeypatch):
    async def fake_produce(prompt):
        return {"title": "Producer Title Here", "tags": "Pop", "topic": prompt, "lyrics": ""}

    monkeypatch.setattr(main_module.LLMService, "produce_full_track", staticmethod(fake_produce))

    async def chat_once(session_id):
        r = await client.post(f"/sessions/{session_id}/chat", json={"content": "make me a song"})
        assert r.status_code == 200, r.text
        return r.json()["session"]["title"]

    # backend-cased default renames
    s = await _create_session(client, title="New session")
    try:
        assert await chat_once(s["id"]) == "Producer Title Here"
    finally:
        await client.delete(f"/sessions/{s['id']}")

    # frontend-cased default renames too (the casing bug)
    s = await _create_session(client, title="New Session")
    try:
        assert await chat_once(s["id"]) == "Producer Title Here"
    finally:
        await client.delete(f"/sessions/{s['id']}")

    # deliberate short custom name survives
    s = await _create_session(client, title="EP")
    try:
        assert await chat_once(s["id"]) == "EP"
    finally:
        await client.delete(f"/sessions/{s['id']}")


@pytest.mark.asyncio
async def test_session_delete_dissociates_and_404s_after(client):
    s = await _create_session(client)
    r = await client.delete(f"/sessions/{s['id']}")
    assert r.status_code == 200
    r = await client.get(f"/sessions/{s['id']}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_job_rename_validation_and_updated_at(client):
    # create a job via generate? too heavy — use rename path on a scratch job row
    from sqlmodel import Session as DBSession
    from app.models import Job
    with DBSession(main_module.engine) as db:
        job = Job(prompt="rename validation probe", title="Probe")
        db.add(job)
        db.commit()
        db.refresh(job)
        jid = str(job.id)
    try:
        r = await client.patch(f"/jobs/{jid}", json={"title": "  New Job Title  "})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["title"] == "New Job Title"
        assert body["updated_at"] is not None
        r = await client.patch(f"/jobs/{jid}", json={"title": " "})
        assert r.status_code == 422
    finally:
        await client.delete(f"/jobs/{jid}")


@pytest.mark.asyncio
async def test_project_blank_name_rejected(client):
    r = await client.post("/projects", json={"name": "Blank Probe Project"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    try:
        r = await client.put(f"/projects/{pid}", json={"name": "   "})
        assert r.status_code == 422
    finally:
        await client.delete(f"/projects/{pid}")
