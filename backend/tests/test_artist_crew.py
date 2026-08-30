"""Crew agent tests (Wave 3 / 3A): stylist + critic in the album pipeline.

Covers: registry presence, the bounded revise path (max ONE revision round),
graceful degradation when crew agents fail, review persistence into the album
cursor, the tracklist review join, and produce-endpoint crew-flag plumbing.
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import AgentRun, ArtistProfile, Job, JobStatus, Release


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_stylist_and_critic_registered():
    from app.agents.registry import AGENTS, list_agents

    assert {"stylist", "critic"} <= set(AGENTS.keys())
    names = {a["name"] for a in list_agents()}
    assert {"stylist", "critic"} <= names


def _mk_draft(title="Song", lyrics=None, tags=None):
    from app.agents.songwriter.schemas import SongDraft

    return SongDraft(
        title=title,
        lyrics=lyrics or "[Verse 1]\nRiding on the late train home again\nThe copper dust is on my coat\n[Chorus]\nSing the rails to sleep tonight",
        style_tags=tags or ["indie folk", "warm", "fingerpicked guitar"],
    )


@pytest.fixture
def release_with_vision(test_db):
    with Session(test_db) as session:
        profile = ArtistProfile(name="Crew Artist", bio="b", voice_profile_id=None)
        session.add(profile); session.commit(); session.refresh(profile)
        release = Release(title="Crew Album", profile_id=str(profile.id), vision_json=json.dumps({
            "journey_title": "Crew Album", "concept_statement": "c",
            "song_seeds": [{"working_title": "S1", "mood": "m", "energy": 0.5}],
        }))
        session.add(release); session.commit(); session.refresh(release)
        yield test_db, release.id, profile


def _run_create(test_db, release_id, profile, monkeypatch, critic_side_effects, stylist=None):
    """Run create_track_from_seed with mocked LLM crew + generation."""
    from app.agents.critic.schemas import Critique
    from app.agents.stylist.schemas import StylingChoice
    from app.agents.orchestrator import bridge as bridge_mod

    def _critic(verdict, score=0.5, notes="n"):
        class _R:
            class output:
                pass
        r = _R()
        r.output = Critique(verdict=verdict, score=score, notes=notes, contradictions=[])
        return r

    class _StylistR:
        output = StylingChoice(style_tags=["synthwave", "warm", "guitar"], rationale="r")

    if stylist is None:
        stylist = AsyncMock(return_value=_StylistR())

    writer = AsyncMock(side_effect=[
        _mk_draft(tags=["folk"]),
        _mk_draft(title="Song v2", tags=["folk"]),
    ])

    review_sink: dict = {}

    async def _fake_generate(job_id, req, eng):
        with Session(eng) as s:
            job = s.get(Job, job_id)
            job.status = JobStatus.COMPLETED
            s.add(job); s.commit()

    with patch.object(bridge_mod.SONGWRITER_AGENT, "run", writer), \
         patch.object(bridge_mod.STYLIST_AGENT, "run", stylist), \
         patch.object(bridge_mod.CRITIC_AGENT, "run", AsyncMock(side_effect=critic_side_effects)), \
         patch("app.services.llm_service.LLMService.rewrite_caption", AsyncMock(return_value={})), \
         patch("app.services.music_service.music_service.generate_task", _fake_generate):
        from app.agents.orchestrator.bridge import create_track_from_seed

        job = asyncio.get_event_loop().run_until_complete(create_track_from_seed(
            seed={"working_title": "S1", "mood": "m", "energy": 0.5, "target_duration_s": 180},
            album_context={"album_title": "T", "album_concept": "C", "artist_name": "A"},
            artist_profile_id=str(profile.id), release_id=str(release_id),
            project_id=None, provider_name="minimax_music3",
            crew_flags={"stylist": True, "critic": True}, review_sink=review_sink,
            engine=test_db,
        ))
    return job, writer, review_sink, stylist


def test_critic_revise_triggers_exactly_one_revision(test_db, release_with_vision):
    from app.agents.critic.schemas import Critique

    test_db, release_id, profile = release_with_vision
    revise = Critique(verdict="revise", score=0.3, notes="hook is missing", contradictions=[])
    final_pass = Critique(verdict="pass", score=0.8, notes="good now", contradictions=[])
    job, writer, review_sink, stylist = _run_create(
        test_db, release_id, profile, None,
        critic_side_effects=[_R(revise), _R(final_pass)],
    )
    # Bounded: exactly ONE revision round → writer called twice
    assert writer.await_count == 2
    revised_seed = writer.await_args_list[1].kwargs.get("seed") or writer.await_args_list[1].args[0]
    assert revised_seed["revision_notes"] == "hook is missing"
    assert review_sink["review"]["verdict"] == "pass"
    assert job.status == JobStatus.COMPLETED


def _R(critique):
    class _R:
        output = critique
    return _R()


def test_critic_failure_does_not_kill_track(test_db, release_with_vision):
    test_db, release_id, profile = release_with_vision

    async def _boom(*a, **k):
        raise RuntimeError("critic LLM down")

    job, writer, review_sink, stylist = _run_create(
        test_db, release_id, profile, None,
        critic_side_effects=[_boom()],
    )
    assert job is not None and job.status == JobStatus.COMPLETED
    assert review_sink["review"]["verdict"] == "unavailable"


def test_stylist_refines_tags_and_failure_is_survivable(test_db, release_with_vision):
    test_db, release_id, profile = release_with_vision
    from app.agents.critic.schemas import Critique

    job, writer, review_sink, stylist = _run_create(
        test_db, release_id, profile, None,
        critic_side_effects=[Critique(verdict="pass", score=0.9, notes="ok", contradictions=[])],
    )
    assert job is not None
    assert "synthwave" in job.tags  # stylist's refinement won
    assert stylist.await_count == 1

    # stylist failure path: tags fall back to the songwriter's
    async def _stylist_boom(*a, **k):
        raise RuntimeError("no taste model")
    job2, _, _, _ = _run_create(
        test_db, release_id, profile, None,
        critic_side_effects=[Critique(verdict="pass", score=0.9, notes="ok", contradictions=[])],
        stylist=AsyncMock(side_effect=_stylist_boom),
    )
    assert "folk" in job2.tags


# ---------------------------------------------------------------- API-level fixtures

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.main import engine as app_engine  # noqa: E402


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
def artist_fixture():
    from app.release_state import album_run_release_id

    with Session(app_engine) as session:
        profile = ArtistProfile(name=f"CRW-{uuid.uuid4().hex[:8]}", bio="integration")
        session.add(profile)
        session.commit()
        session.refresh(profile)
        release = Release(title="CRW Release", profile_id=str(profile.id))
        session.add(release)
        session.commit()
        session.refresh(release)
        pid, rid = str(profile.id), str(release.id)
    yield {"profile_id": pid, "release_id": rid}
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
            session.delete(prof)
        session.commit()


def test_produce_endpoint_persists_crew_flags(client, artist_fixture, monkeypatch):
    from app.main import engine as app_engine

    rid = artist_fixture["release_id"]

    def _no_spawn(coro):
        coro.close()
        return None

    monkeypatch.setattr("app.main.spawn_background_task", _no_spawn)
    out = client.post(f"/releases/{rid}/produce", json={
        "autopilot": False, "crew": {"stylist": True, "critic": True},
    })
    assert out.status_code == 200
    with Session(app_engine) as session:
        run = session.exec(select(AgentRun).where(AgentRun.agent_name == "album_orchestrator")).all()[-1]
        cfg = json.loads(run.input_json)
        assert cfg["crew"] == {"stylist": True, "critic": True}
        session.delete(run)
        session.commit()


def test_tracklist_joins_critic_review(client, artist_fixture):
    from app.main import engine as app_engine

    rid = artist_fixture["release_id"]
    with Session(app_engine) as session:
        job = Job(title="Reviewed Track", prompt="p", lyrics="l", release_id=rid, status=JobStatus.COMPLETED)
        session.add(job); session.commit(); session.refresh(job)
        album_run = AgentRun(agent_name="album_orchestrator", status="succeeded",
                             release_id=rid,
                             input_json=json.dumps({"release_id": rid}),
                             state_json=json.dumps({
                                 "completed_seeds": [0], "slot_jobs": {"0": str(job.id)},
                                 "reviews": {"0": {"verdict": "concern", "score": 0.55,
                                                    "notes": "second verse drifts", "contradictions": []}},
                             }))
        session.add(album_run); session.commit()
        job_id = str(job.id)

    data = client.get(f"/releases/{rid}/tracks").json()
    row = next(t for t in data["tracks"] if t["id"] == job_id)
    assert row["review"]["verdict"] == "concern"
    assert row["review"]["notes"] == "second verse drifts"


# ---------------------------------------------------------------- 3D: run observability

def test_run_stats_aggregation(client, artist_fixture):
    from datetime import datetime, timedelta, timezone as tz
    from app.main import engine as app_engine

    pid = artist_fixture["profile_id"]
    old = datetime.now(tz.utc) - timedelta(days=5)
    with Session(app_engine) as session:
        session.add(AgentRun(agent_name="experiencer", status="succeeded", profile_id=pid,
                             tokens_in=100, tokens_out=200, latency_ms=4000))
        session.add(AgentRun(agent_name="experiencer", status="failed", profile_id=pid,
                             tokens_in=50, tokens_out=0, latency_ms=12000))
        session.add(AgentRun(agent_name="world_builder", status="succeeded", profile_id=pid,
                             tokens_in=80, tokens_out=300, latency_ms=6000))
        session.commit()

    data = client.get(f"/agents/runs/stats?profile_id={pid}").json()
    assert data["total"] == 3
    assert data["statuses"]["succeeded"] == 2
    assert data["statuses"]["failed"] == 1
    assert data["success_rate"] == 0.667
    assert data["latency_ms"]["p50"] == 6000
    assert data["latency_ms"]["p95"] == 12000
    assert data["tokens_out"] == 500
    assert data["by_agent"]["experiencer"]["count"] == 2
    assert data["by_agent"]["world_builder"]["tokens_out"] == 300

    # window filter drops nothing here (all within 30d), but must not error
    data_w = client.get(f"/agents/runs/stats?profile_id={pid}&window_days=1").json()
    assert "total" in data_w
