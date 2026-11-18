"""Render takes: versioning, selection and deletion (plan 7.4)."""

from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlmodel import Session

from app.main import engine
from app.models import Job, JobStatus
from app.services.video import takes as takes_service


@pytest.fixture()
def job_id():
    job_uuid = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_uuid, status=JobStatus.COMPLETED, prompt="t", title="Take test",
                        audio_path="/audio/x.wav"))
        session.commit()
    yield str(job_uuid)
    with Session(engine) as session:
        job = session.get(Job, job_uuid)
        if job:
            session.delete(job)
            session.commit()


def _take(session, job_id, name, *, finished=True):
    take = takes_service.create_take(
        session, job_id=job_id, task_id=name, kind="advanced", config={"visual_style": name}
    )
    if finished:
        path = f"/audio/videos/{name}_master_mv.mp4"
        takes_service.finish_take(
            session, take, status="completed", video_path=path, file_path=f"/tmp/{name}.mp4",
            qc={"ok": True},
        )
    return take


def test_two_takes_coexist_and_the_newest_selection_wins(job_id):
    with Session(engine) as session:
        first = _take(session, job_id, "take-one")
        second = _take(session, job_id, "take-two")
        takes_service.select_take(session, job_id, str(second.id))

        listed = takes_service.list_takes(session, job_id)
        assert len(listed) == 2
        selected = [t for t in listed if t["is_selected"]]
        assert len(selected) == 1, "exactly one take may be selected"
        assert selected[0]["id"] == str(second.id)

        job = session.get(Job, uuid.UUID(job_id))
        assert job.video_path == "/audio/videos/take-two_master_mv.mp4"
        assert json.loads(job.video_config_json)["visual_style"] == "take-two"

        # Switching back must not lose the other take.
        takes_service.select_take(session, job_id, str(first.id))
        session.refresh(job)
        assert job.video_path == "/audio/videos/take-one_master_mv.mp4"
        assert len(takes_service.list_takes(session, job_id)) == 2


def test_deleting_the_selected_take_falls_back_to_another(job_id):
    with Session(engine) as session:
        first = _take(session, job_id, "keep-me")
        second = _take(session, job_id, "delete-me")
        takes_service.select_take(session, job_id, str(second.id))

        result = takes_service.delete_take(session, job_id, str(second.id))
        assert result["deleted"] is True

        remaining = takes_service.list_takes(session, job_id)
        assert [t["id"] for t in remaining] == [str(first.id)]
        job = session.get(Job, uuid.UUID(job_id))
        assert job.video_path == "/audio/videos/keep-me_master_mv.mp4"


def test_deleting_the_last_take_clears_the_job_pointer(job_id):
    with Session(engine) as session:
        only = _take(session, job_id, "only")
        takes_service.select_take(session, job_id, str(only.id))

        takes_service.delete_take(session, job_id, str(only.id))
        job = session.get(Job, uuid.UUID(job_id))
        assert job.video_path is None
        assert takes_service.list_takes(session, job_id) == []


def test_an_incomplete_take_cannot_be_selected(job_id):
    with Session(engine) as session:
        pending = _take(session, job_id, "pending", finished=False)
        assert takes_service.select_take(session, job_id, str(pending.id)) is None
        job = session.get(Job, uuid.UUID(job_id))
        assert job.video_path is None


def test_takes_are_scoped_to_their_job(job_id):
    """A take id from another job must not be selectable or deletable."""
    other = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=other, status=JobStatus.COMPLETED, prompt="o", title="Other"))
        session.commit()
        foreign = _take(session, str(other), "foreign")
        try:
            assert takes_service.select_take(session, job_id, str(foreign.id)) is None
            assert takes_service.delete_take(session, job_id, str(foreign.id))["deleted"] is False
        finally:
            session.delete(session.get(Job, other))
            session.commit()


def test_clip_workspace_persists_plan_and_clips(tmp_path, job_id):
    """Plan 7.11 groundwork: a take keeps its per-scene clips and plan."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake-mp4")
    # Redirect the workspace into tmp_path so the test never touches real data.
    import app.services.video.takes as t
    original = t.clip_workspace
    t.clip_workspace = lambda render_id: str(tmp_path / f"take-{render_id}")
    try:
        workspace = takes_service.save_clip_plan("abc123", {"style": "neon", "clips": []}, [str(src)])
        assert workspace and os.path.isdir(workspace)
        loaded = takes_service.load_clip_plan("abc123")
        assert loaded["style"] == "neon"
        assert len(loaded["clips"]) == 1
        assert os.path.isfile(loaded["clips"][0])
        assert loaded["clips"][0].endswith("clip_000.mp4")
    finally:
        t.clip_workspace = original


def test_interrupted_renders_are_failed_on_boot(job_id):
    """A killed server used to leave a permanent `rendering` row (7.28)."""
    with Session(engine) as session:
        stale = _take(session, job_id, "interrupted", finished=False)
        done = _take(session, job_id, "finished")

        marked = takes_service.reconcile_stale_renders(session)
        assert marked >= 1

        session.refresh(stale)
        session.refresh(done)
        assert stale.status == "failed"
        assert "restarted" in (stale.error or "").lower()
        # Completed takes are untouched.
        assert done.status == "completed"
        assert done.error is None
