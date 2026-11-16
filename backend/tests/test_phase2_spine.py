"""Phase 2 spine tests: queue integrity, recovery policy, schema authority.

These pin the behaviours that made "did my job survive a crash?" unanswerable:
a leaked fd per queue update, non-atomic task updates, three divergent recovery
policies, and a schema with no version.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session

from app.core import recovery
from app.core.task_queue import DurableTaskQueue
from app.models import Job, JobStatus


@pytest.fixture()
def queue(tmp_path):
    return DurableTaskQueue(db_path=str(tmp_path / "tasks.db"))


def test_queue_does_not_leak_connections(queue):
    """`with sqlite3.connect(...)` commits but never closes — one fd per update."""
    task = queue.enqueue_task("t-leak", "video", payload={"a": 1})
    assert task is not None

    before = _open_fd_count()
    for i in range(60):
        queue.update_task("t-leak", progress=i)
        queue.get_task("t-leak")
    after = _open_fd_count()

    # A leak of 120 fds would be obvious; allow a little noise from the runtime.
    assert after - before < 15, f"file descriptors grew from {before} to {after}"


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:  # pragma: no cover - non-POSIX
        return 0


def test_enqueue_is_idempotent(queue):
    queue.enqueue_task("dup", "video", payload={"v": 1})
    queue.update_task("dup", progress=42, status="running")
    again = queue.enqueue_task("dup", "video", payload={"v": 2})

    assert again is not None
    assert again.progress == 42, "a duplicate enqueue must not reset progress"
    assert again.status == "running"
    assert again.payload.get("v") == 1, "existing payload must survive"


def test_update_task_preserves_payload_merge(queue):
    queue.enqueue_task("p", "video", payload={"a": 1})
    queue.update_task("p", payload_updates={"b": 2})
    task = queue.get_task("p")
    assert task.payload == {"a": 1, "b": 2}


def test_fail_task_sets_terminal_state(queue):
    queue.enqueue_task("f", "video", payload={})
    task = queue.fail_task("f", "boom")
    assert task.status == "error"
    assert task.error == "boom"


# ── recovery policy ─────────────────────────────────────────────────────────

def test_queued_jobs_survive_restart():
    """The old reconciler failed QUEUED jobs, destroying work nobody had started."""
    from app.main import engine

    job_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="queued probe", status=JobStatus.QUEUED))
        session.commit()

    recovery.recover_jobs(engine)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.QUEUED, "queued work must be preserved for re-dispatch"
        session.delete(job)
        session.commit()


def test_in_flight_jobs_are_marked_interrupted():
    from app.main import engine

    job_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="running probe", status=JobStatus.PROCESSING))
        session.commit()

    recovery.recover_jobs(engine)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.INTERRUPTED
        assert "cannot be resumed" in (job.error_msg or "")
        session.delete(job)
        session.commit()


def test_stalled_job_sweeper_marks_wedged_jobs_failed():
    """A wedged model call must not hold the GPU lane forever."""
    from app.main import engine

    job_id = uuid.uuid4()
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    with Session(engine) as session:
        session.add(Job(
            id=job_id,
            prompt="stalled probe",
            status=JobStatus.PROCESSING,
            last_heartbeat_at=old,
            updated_at=old,
        ))
        session.commit()

    stalled = recovery.sweep_stalled_jobs(engine, max_stall_seconds=60)
    assert str(job_id) in stalled

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == JobStatus.FAILED
        assert "Stalled" in (job.error_msg or "")
        session.delete(job)
        session.commit()


def test_fresh_heartbeat_is_not_swept():
    from app.main import engine

    job_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(
            id=job_id, prompt="alive probe", status=JobStatus.PROCESSING,
            last_heartbeat_at=datetime.now(timezone.utc),
        ))
        session.commit()

    assert str(job_id) not in recovery.sweep_stalled_jobs(engine, max_stall_seconds=3600)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        session.delete(job)
        session.commit()


# ── schema authority ────────────────────────────────────────────────────────

def test_alembic_is_the_schema_authority():
    """The DB must be versioned, and the hot-path indexes must exist."""
    from app.main import engine

    with Session(engine) as session:
        version = session.exec(_text("SELECT version_num FROM alembic_version")).first()
        indexes = [r[0] for r in session.exec(_text(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )).all()]

    assert version, "alembic_version is empty: the database is outside migration control"
    for required in ("ix_job_status_created", "ix_job_created_at", "ix_job_session_id"):
        assert required in indexes, f"missing index {required}"


def _text(sql):
    from sqlalchemy import text

    return text(sql)


def test_new_job_columns_exist():
    from app.main import engine

    with Session(engine) as session:
        cols = {r[1] for r in session.exec(_text("PRAGMA table_info(job)")).all()}
    for required in ("provenance_json", "quality", "last_heartbeat_at", "attempts", "max_attempts"):
        assert required in cols, f"job.{required} missing — migration 0003 did not apply"


def test_duplicate_playlist_membership_is_rejected():
    """A unique index now backs the rule instead of handler-side luck."""
    from app.main import engine

    with Session(engine) as session:
        idx = {r[0] for r in session.exec(_text(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )).all()}
    assert "ux_playlist_track" in idx
    assert "ux_agent_assignment_role" in idx


def test_owner_id_is_present_and_stamped():
    """D2: every job row is owned, so a hosted mode is a migration not a rewrite."""
    from app.main import engine
    from app.models import Job
    import uuid as _uuid

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="owner probe", status=JobStatus.QUEUED))
        session.commit()
        job = session.get(Job, job_id)
        assert job.owner_id == "local"
        session.delete(job)
        session.commit()


def test_scoped_helper_is_a_noop_for_the_local_owner():
    from sqlmodel import select

    from app.core.sharing import owner_id, scoped
    from app.main import engine
    from app.models import Job

    assert owner_id() == "local"
    with Session(engine) as session:
        statement = scoped(select(Job), Job)
        # The statement must still execute (no schema assumptions).
        session.exec(statement).all()


# ── soft delete (plan 2.11) ────────────────────────────────────────────────

def test_delete_is_soft_by_default(client):
    """A mistaken DELETE must be recoverable; ?purge=true is the explicit erase."""
    import uuid as _uuid

    from app.main import engine
    from app.models import Job

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="soft delete probe", status=JobStatus.COMPLETED))
        session.commit()

    res = client.request("DELETE", f"/jobs/{job_id}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["soft"] is True

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job is not None, "the row must survive a soft delete"
        assert job.deleted_at is not None

    # It must disappear from history.
    history = client.get("/history?limit=200").json()
    assert all(str(j.get("id")).replace("-", "") != job_id.hex for j in history)


def test_purge_removes_the_row(client):
    import uuid as _uuid

    from app.main import engine
    from app.models import Job

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="purge probe", status=JobStatus.COMPLETED))
        session.commit()

    res = client.request("DELETE", f"/jobs/{job_id}?purge=true")
    assert res.status_code == 200, res.text

    with Session(engine) as session:
        assert session.get(Job, job_id) is None
