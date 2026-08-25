"""Lifecycle integrity: terminal-state guards + single-instance lock."""
import os
import threading
import time
import uuid

import pytest


@pytest.fixture()
def test_db_engine(tmp_path):
    from sqlmodel import SQLModel, create_engine
    import app.models  # register tables on shared metadata
    eng = create_engine(f"sqlite:///{tmp_path}/guard.db")
    SQLModel.metadata.create_all(eng)
    return eng


def test_abort_raises_on_failed_job(test_db_engine):
    from app.orchestration.pipeline import _abort_if_terminal
    from app.models import Job, JobStatus
    from sqlmodel import Session
    import asyncio

    jid = uuid.uuid4()
    with Session(test_db_engine) as session:
        session.add(Job(id=jid, title="t", prompt="p", status=JobStatus.FAILED))
        session.commit()

    with pytest.raises(asyncio.CancelledError):
        _abort_if_terminal(test_db_engine, jid, stage="test")


def test_abort_passes_on_processing_job(test_db_engine):
    from app.orchestration.pipeline import _abort_if_terminal
    from app.models import Job, JobStatus
    from sqlmodel import Session

    jid = uuid.uuid4()
    with Session(test_db_engine) as session:
        session.add(Job(id=jid, title="t", prompt="p", status=JobStatus.PROCESSING))
        session.commit()

    _abort_if_terminal(test_db_engine, jid, stage="test")  # must NOT raise


def test_abort_honors_cancel_event():
    import asyncio
    from app.orchestration.pipeline import _abort_if_terminal
    ev = threading.Event()
    ev.set()
    with pytest.raises(asyncio.CancelledError):
        _abort_if_terminal(None, uuid.uuid4(), cancel_event=ev)


def test_instance_lock_refuses_second_holder(tmp_path, monkeypatch):
    import app.core.instance_lock as il
    monkeypatch.setattr(il, "LOCK_PATH", tmp_path / ".milimo.lock")
    assert il.acquire_instance_lock() is True
    # Same pid re-acquire (simulating second boot while we live) must refuse.
    assert il.acquire_instance_lock() is False
    il.release_instance_lock()
    assert not (tmp_path / ".milimo.lock").exists()


def test_instance_lock_steals_stale_dead_pid(tmp_path, monkeypatch):
    import app.core.instance_lock as il
    lock = tmp_path / ".milimo.lock"
    dead_pid = 999999
    monkeypatch.setattr(il, "LOCK_PATH", lock)
    lock.write_text(f"{dead_pid}:{time.time() - 120}")  # dead + beyond grace
    assert il.acquire_instance_lock() is True
    il.release_instance_lock()
