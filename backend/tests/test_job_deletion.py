"""Tests for Job Deletion & UUID Integrity (Phase 3).

Covers:
- Deletion of jobs with 32-char hex IDs
- Deletion of jobs with 36-char hyphenated UUID IDs
- Nullification of foreign references in session, sessionmessage, playlisttrack, and child jobs
- Verification of filesystem artifact cleanup (audio, stems, midi)
- 404 response on non-existent job ID
"""

import os
import json
import uuid
from pathlib import Path
import pytest
import httpx
from sqlmodel import Session, select, text

import app.main as main_module
from app.main import app, engine, get_job_by_id
from app.models import Job, JobStatus, Session as DbSession, SessionMessage, Playlist, PlaylistTrack


@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_delete_job_32_hex_id(client):
    """Verify deleting a job stored with a 32-character hex ID."""
    test_u = uuid.uuid4()
    test_hex = test_u.hex
    audio_path = f"/audio/{test_hex}.mp3"
    disk_file = Path(f"generated_audio/{test_hex}.mp3")
    disk_file.parent.mkdir(parents=True, exist_ok=True)
    disk_file.write_text("mock audio")

    stem_file = Path(f"generated_audio/stems/{test_hex}_vocals.wav")
    stem_file.parent.mkdir(parents=True, exist_ok=True)
    stem_file.write_text("mock stem")

    with Session(engine) as s:
        job = Job(
            id=test_u,
            title="Hex Delete Test",
            prompt="Testing hex delete",
            status=JobStatus.COMPLETED,
            audio_path=audio_path,
        )
        s.add(job)
        s.commit()

    try:
        # Verify it exists
        with Session(engine) as s:
            job = get_job_by_id(s, test_hex)
            assert job is not None
            assert job.title == "Hex Delete Test"

        # Delete via API
        r = await client.delete(f"/jobs/{test_hex}")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "deleted"

        # Verify DB removal
        with Session(engine) as s:
            assert get_job_by_id(s, test_hex) is None

        # Verify disk removal
        assert not disk_file.exists()
        assert not stem_file.exists()
    finally:
        if disk_file.exists():
            disk_file.unlink()
        if stem_file.exists():
            stem_file.unlink()


@pytest.mark.asyncio
async def test_delete_job_36_hyphenated_uuid(client):
    """Verify deleting a job stored with a 36-character hyphenated UUID string in SQLite."""
    test_u = uuid.uuid4()
    test_uuid = str(test_u)
    test_hex = test_u.hex
    audio_path = f"/audio/{test_uuid}.mp3"
    disk_file = Path(f"generated_audio/{test_uuid}.mp3")
    disk_file.parent.mkdir(parents=True, exist_ok=True)
    disk_file.write_text("mock audio")

    with Session(engine) as s:
        job = Job(
            id=test_u,
            title="Hyphen Delete Test",
            prompt="Testing hyphen delete",
            status=JobStatus.COMPLETED,
            audio_path=audio_path,
        )
        s.add(job)
        s.commit()
        # Explicitly update SQLite row to have hyphenated string representation
        s.exec(text("UPDATE job SET id = :uuid WHERE id = :hex").params(uuid=test_uuid, hex=test_hex))
        s.commit()

    try:
        # Verify it exists and can be retrieved via both hyphen and hex
        with Session(engine) as s:
            job = get_job_by_id(s, test_uuid)
            assert job is not None
            assert job.title == "Hyphen Delete Test"

            job_hex = get_job_by_id(s, test_hex)
            assert job_hex is not None

        # Delete via API using hyphenated UUID
        r = await client.delete(f"/jobs/{test_uuid}")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "deleted"

        # Verify DB removal
        with Session(engine) as s:
            assert get_job_by_id(s, test_uuid) is None

        # Verify disk removal
        assert not disk_file.exists()
    finally:
        if disk_file.exists():
            disk_file.unlink()


@pytest.mark.asyncio
async def test_delete_job_cascades_foreign_references(client):
    """Verify relational cascade nullification when deleting a job."""
    test_u = uuid.uuid4()
    test_id = test_u.hex
    session_id = uuid.uuid4()
    playlist_id = f"pl_{uuid.uuid4().hex[:12]}"
    child_u = uuid.uuid4()

    with Session(engine) as s:
        # Insert test job
        job = Job(
            id=test_u,
            title="Cascade Parent",
            prompt="Prompt",
            status=JobStatus.COMPLETED,
        )
        s.add(job)
        # Insert child job pointing to test job
        child = Job(
            id=child_u,
            title="Cascade Child",
            prompt="Prompt",
            status=JobStatus.COMPLETED,
            parent_job_id=test_id,
        )
        s.add(child)
        # Insert session pointing to test job
        sess = DbSession(
            id=session_id,
            title="Cascade Session",
            active_job_id=test_id,
        )
        s.add(sess)
        # Insert session message pointing to test job
        msg_id = uuid.uuid4()
        msg = SessionMessage(
            id=msg_id,
            session_id=session_id,
            role="assistant",
            content="done",
            generated_job_id=test_id,
        )
        s.add(msg)
        # Insert playlist and track
        pl = Playlist(id=playlist_id, name="Test Playlist")
        s.add(pl)
        pt_id = f"pt_{uuid.uuid4().hex[:12]}"
        pt = PlaylistTrack(id=pt_id, playlist_id=playlist_id, job_id=test_id, position=0)
        s.add(pt)
        s.commit()

    try:
        # Delete parent job
        r = await client.delete(f"/jobs/{test_id}")
        assert r.status_code == 200, r.text

        # Verify cascades
        with Session(engine) as s:
            # Session active_job_id nullified
            sess_db = s.exec(select(DbSession).where(DbSession.id == session_id)).one()
            assert sess_db.active_job_id is None

            # Session message generated_job_id nullified
            msg_db = s.exec(select(SessionMessage).where(SessionMessage.id == msg_id)).one()
            assert msg_db.generated_job_id is None

            # Playlist track removed
            pt_db = s.exec(select(PlaylistTrack).where(PlaylistTrack.job_id == test_id)).all()
            assert len(pt_db) == 0

            # Child job parent_job_id nullified
            child_db = s.exec(select(Job).where(Job.id == child_u)).one()
            assert child_db.parent_job_id is None
    finally:
        with Session(engine) as s:
            s.exec(text("DELETE FROM job WHERE id = :id").params(id=child_u.hex))
            s.exec(text("DELETE FROM sessionmessage WHERE id = :id").params(id=msg_id.hex))
            s.exec(text("DELETE FROM session WHERE id = :id").params(id=session_id.hex))
            s.exec(text("DELETE FROM playlisttrack WHERE id = :id").params(id=pt_id))
            s.exec(text("DELETE FROM playlist WHERE id = :id").params(id=playlist_id))
            s.commit()


@pytest.mark.asyncio
async def test_delete_nonexistent_job_returns_404(client):
    """Verify 404 is returned when attempting to delete a non-existent job."""
    r = await client.delete(f"/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    data = r.json()
    assert "not found" in data["detail"].lower()
