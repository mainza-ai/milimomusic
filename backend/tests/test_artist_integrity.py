"""Artist-domain integrity tests (Phase E).

Covers: release status state machine, orchestrator cursor resolution
(retry deduplication), concurrent-produce guard, and the profile-delete
cascade policy (active-run block + job detachment).
"""
import json
import sys
import uuid
from pathlib import Path

import pytest
from sqlmodel import Session, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import AgentAssignment, AgentRun, ArtistProfile, Job, JobStatus, Release
from app.release_state import (
    ACTIVE_RUN_STATUSES,
    album_run_release_id,
    can_transition,
    resolve_track_rows,
    transition_release,
)


# ---------------------------------------------------------------- state machine

def test_release_transitions_valid():
    assert can_transition("planned", "in_progress")
    assert can_transition("in_progress", "completed")
    assert can_transition("completed", "in_progress")  # reopen for re-production


def test_release_transitions_invalid():
    assert not can_transition("completed", "planned")
    assert not can_transition("in_progress", "in_progress")


def test_transition_rejects_unknown_target():
    r = Release(title="t", profile_id="p", status="planned")
    with pytest.raises(ValueError):
        transition_release(r, "shipped")


def test_transition_noop_is_idempotent():
    r = Release(title="t", profile_id="p", status="planned")
    transition_release(r, "planned")
    assert r.status == "planned"


# ---------------------------------------------------------------- cursor resolution

class _Run:
    def __init__(self, release_id, state):
        self.input_json = json.dumps({"release_id": release_id})
        self.state_json = json.dumps(state)


class _Job:
    def __init__(self, jid):
        self.id = jid
        self.title = jid


def test_album_run_release_id_parses_input():
    run = _Run("rel-1", {})
    assert album_run_release_id(run) == "rel-1"


def test_resolve_non_orchestrated_passthrough():
    jobs = [_Job("a"), _Job("b")]
    out = resolve_track_rows(jobs, [], "rel-1")
    assert [(j.id, slot) for j, slot in out] == [("a", None), ("b", None)]


def test_resolve_cursor_hides_superseded_retries():
    # Slot 0 succeeded twice (retry 'a2' superseded 'a1'); slot 1 failed once.
    rows = [_Job("a1"), _Job("a2"), _Job("f1")]
    runs = [_Run("rel-1", {
        "slot_jobs": {"0": "a2"},
        "failed_jobs": {"0": ["a1"], "1": ["f1"]},
    })]
    out = resolve_track_rows(rows, runs, "rel-1")
    assert [(j.id, slot) for j, slot in out] == [("a2", 0), ("f1", 1)]


def test_resolve_failed_attempt_hidden_once_retry_wins():
    rows = [_Job("f1"), _Job("w1")]
    runs = [_Run("rel-1", {
        "slot_jobs": {"3": "w1"},
        "failed_jobs": {"3": ["f1"]},
    })]
    out = resolve_track_rows(rows, runs, "rel-1")
    assert [(j.id, slot) for j, slot in out] == [("w1", 3)]


def test_resolve_legacy_job_ids_positional_mapping():
    rows = [_Job("j0"), _Job("j1")]
    runs = [_Run("rel-1", {"job_ids": ["j0", "j1"]})]
    out = resolve_track_rows(rows, runs, "rel-1")
    assert [(j.id, slot) for j, slot in out] == [("j0", 0), ("j1", 1)]


def test_resolve_ignores_other_releases_and_bad_json():
    rows = [_Job("a")]
    runs = [
        _Run("other", {"slot_jobs": {"0": "zzz"}}),
        _Run("rel-1", {"slot_jobs": {"0": "a"}}),
    ]
    runs[1].state_json = "{not json"  # corrupt cursor tolerated
    out = resolve_track_rows(rows, runs, "rel-1")
    # corrupt cursor → no slot info → passthrough
    assert [(j.id, slot) for j, slot in out] == [("a", None)]


def test_resolve_cursor_job_missing_from_rows_is_skipped():
    rows = [_Job("a")]
    runs = [_Run("rel-1", {"slot_jobs": {"0": "ghost", "1": "a"}})]
    out = resolve_track_rows(rows, runs, "rel-1")
    assert [(j.id, slot) for j, slot in out] == [("a", 1)]


# ---------------------------------------------------------------- orchestrator wiring

@pytest.fixture
def test_db():
    from sqlmodel import SQLModel, create_engine
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.mark.asyncio
async def test_orchestrator_writes_slot_cursor_and_release_status(test_db):
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.agents.orchestrator import RunRegistry
    from app.agents.orchestrator.album import AlbumOrchestrator

    registry = RunRegistry()
    events = []

    def publish(evt, data):
        events.append((evt, data))

    orchestrator = AlbumOrchestrator(registry, publish)

    with Session(test_db) as session:
        profile = ArtistProfile(name="Cursor Artist", bio="b")
        session.add(profile)
        session.commit()
        session.refresh(profile)
        release = Release(title="Cursor Album", profile_id=str(profile.id), status="planned")
        session.add(release)
        session.commit()
        session.refresh(release)
        vision = {
            "journey_title": "Cursor Album",
            "concept_statement": "c",
            "song_seeds": [
                {"working_title": "S1", "mood": "m", "energy": 0.5},
                {"working_title": "S2", "mood": "m", "energy": 0.5},
            ],
        }
        run = AgentRun(
            agent_name="album_orchestrator", status="pending",
            input_json=json.dumps({"release_id": str(release.id)}),
            state_json=json.dumps({"vision": vision}),
            budget_json=json.dumps({"caps": {}}),
            profile_id=str(profile.id),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id, release_id = run.id, release.id

    mock_job = MagicMock()
    mock_job.id = "job-new"
    mock_job.status = JobStatus.COMPLETED

    with patch("app.agents.orchestrator.album.create_track_from_seed", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_job
        await orchestrator.execute(
            parent_run_id=run_id, release_id=release_id,
            autopilot=True, engine=test_db,
        )

    with Session(test_db) as session:
        run = session.get(AgentRun, run_id)
        assert run.status == "succeeded"
        state = json.loads(run.state_json)
        assert state["slot_jobs"] == {"0": "job-new", "1": "job-new"}
        release = session.get(Release, release_id)
        assert release.status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_records_failed_job_in_slot(test_db):
    from unittest.mock import AsyncMock, patch

    from app.agents.orchestrator import RunRegistry
    from app.agents.orchestrator.album import AlbumOrchestrator
    from app.agents.orchestrator.bridge import TrackProductionError

    registry = RunRegistry()
    orchestrator = AlbumOrchestrator(registry, lambda *a, **k: None)

    with Session(test_db) as session:
        profile = ArtistProfile(name="Fail Artist", bio="b")
        session.add(profile)
        session.commit()
        session.refresh(profile)
        release = Release(title="Fail Album", profile_id=str(profile.id))
        session.add(release)
        session.commit()
        session.refresh(release)
        vision = {
            "journey_title": "Fail Album", "concept_statement": "c",
            "song_seeds": [{"working_title": "S1", "mood": "m", "energy": 0.5}],
        }
        run = AgentRun(
            agent_name="album_orchestrator", status="pending",
            input_json=json.dumps({"release_id": str(release.id)}),
            state_json=json.dumps({"vision": vision}),
            budget_json=json.dumps({"caps": {}}),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id, release_id = run.id, release.id

    with patch("app.agents.orchestrator.album.create_track_from_seed", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = TrackProductionError("did not complete", job_id="job-failed")
        await orchestrator.execute(
            parent_run_id=run_id, release_id=release_id,
            autopilot=True, engine=test_db,
        )

    with Session(test_db) as session:
        run = session.get(AgentRun, run_id)
        assert run.status == "failed"
        state = json.loads(run.state_json)
        assert state["failed_jobs"] == {"0": ["job-failed"]}
        release = session.get(Release, release_id)
        assert release.status == "in_progress"  # honest partial state


# ---------------------------------------------------------------- API: produce guard + delete cascade

from fastapi.testclient import TestClient  # noqa: E402
import asyncio  # noqa: E402
import httpx  # noqa: E402

from app.main import app  # noqa: E402


class SyncClient:
    def __init__(self, asgi):
        self.transport = httpx.ASGITransport(app=asgi)

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def _call(self, method, url, **kw):
        async def _c():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await getattr(c, method)(url, **kw)
        return self._run(_c())

    def get(self, url, **kw): return self._call("get", url, **kw)
    def post(self, url, **kw): return self._call("post", url, **kw)
    def delete(self, url, **kw): return self._call("delete", url, **kw)


@pytest.fixture
def client():
    return SyncClient(app)


@pytest.fixture
def artist_fixture(test_db=None):
    """Create a profile + release through the real app engine; yield ids."""
    from app.main import engine as app_engine

    with Session(app_engine) as session:
        profile = ArtistProfile(name=f"ITL-{uuid.uuid4().hex[:8]}", bio="integration")
        session.add(profile)
        session.commit()
        session.refresh(profile)
        release = Release(title="ITL Release", profile_id=str(profile.id))
        session.add(release)
        session.commit()
        session.refresh(release)
        pid, rid = str(profile.id), str(release.id)
    yield {"profile_id": pid, "release_id": rid}
    # cleanup
    with Session(app_engine) as session:
        for run in session.exec(select(AgentRun)).all():
            if album_run_release_id(run) == rid or str(run.profile_id or "") == pid:
                session.delete(run)
        for j in session.exec(select(Job).where(Job.release_id == rid)).all():
            session.delete(j)
        rel = session.get(Release, uuid.UUID(rid))
        if rel:
            session.delete(rel)
        prof = session.get(ArtistProfile, uuid.UUID(pid))
        if prof:
            for a in session.exec(select(AgentAssignment).where(AgentAssignment.profile_id == pid)).all():
                session.delete(a)
            session.delete(prof)
        session.commit()


def test_produce_guard_returns_409_on_active_run(client, artist_fixture):
    from app.main import engine as app_engine

    rid = artist_fixture["release_id"]
    pid = artist_fixture["profile_id"]
    with Session(app_engine) as session:
        session.add(AgentRun(
            agent_name="album_orchestrator", status="running",
            input_json=json.dumps({"release_id": rid}),
        ))
        session.commit()

    resp = client.post(f"/releases/{rid}/produce", json={"autopilot": False})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "run_active"


def test_delete_profile_blocks_on_active_run(client, artist_fixture):
    from app.main import engine as app_engine

    rid = artist_fixture["release_id"]
    pid = artist_fixture["profile_id"]
    with Session(app_engine) as session:
        session.add(AgentRun(
            agent_name="album_orchestrator", status="awaiting_approval",
            input_json=json.dumps({"release_id": rid}),
        ))
        session.commit()

    resp = client.delete(f"/profiles/{pid}")
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "active_run"
    # profile still present
    assert client.get(f"/profiles/{pid}").status_code == 200


def test_delete_profile_cascades_and_detaches_jobs(client, artist_fixture):
    from app.main import engine as app_engine

    rid = artist_fixture["release_id"]
    pid = artist_fixture["profile_id"]
    with Session(app_engine) as session:
        session.add(Job(
            title="Detached Track", prompt="p", lyrics="l", release_id=rid,
            artist_profile_id=pid, status=JobStatus.COMPLETED,
        ))
        session.add(AgentAssignment(profile_id=pid, role="producer", agent_name="songwriter"))
        session.commit()

    resp = client.delete(f"/profiles/{pid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["releases_deleted"] == 1
    assert body["jobs_detached"] == 1

    with Session(app_engine) as session:
        assert session.get(Release, uuid.UUID(rid)) is None
        assert session.get(ArtistProfile, uuid.UUID(pid)) is None
        jobs = session.exec(select(Job).where(Job.title == "Detached Track")).all()
        assert len(jobs) == 1
        assert jobs[0].release_id is None
        assert jobs[0].artist_profile_id is None
        session.delete(jobs[0])  # test cleanup
        session.commit()


def test_tracklist_reports_lifecycle_status(client, artist_fixture):
    rid = artist_fixture["release_id"]
    resp = client.get(f"/releases/{rid}/tracks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "planned"
    assert data["rollup"] == "pending"
    assert data["tracks"] == []
