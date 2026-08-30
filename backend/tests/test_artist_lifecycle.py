"""Artist-domain lifecycle tests (Phase F): release lifecycle API, crew
override chain, lore grounding, with_stats aggregate, pagination, and the
single-seed retry flow."""
import asyncio
import json
import sys
import uuid
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import (
    AgentAssignment, AgentRun, ArtistProfile, Job, JobStatus, Release,
)
from app.release_state import album_run_release_id

from app.agents.runtime.overrides import (
    load_artist_lore,
    resolve_artist_grounding,
    resolve_chain_head,
)
from app.agents.runtime.policy import ModelProfile, ResiliencePolicy


# ---------------------------------------------------------------- override chain

@pytest.fixture
def mem_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def _mk_profile(session, **kw):
    p = ArtistProfile(name=f"OVR-{uuid.uuid4().hex[:6]}", bio="b", **kw)
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


def test_assignment_override_beats_profile_default(mem_db):
    with Session(mem_db) as s:
        p = _mk_profile(s, default_provider="deepseek", default_model="prof-model")
        s.add(AgentAssignment(profile_id=str(p.id), role="producer", agent_name="songwriter",
                              model_provider="openai", model="assign-model"))
        s.commit()
        head = resolve_chain_head(s, str(p.id), "songwriter")
        assert head == ModelProfile(provider="openai", model="assign-model")


def test_profile_default_used_when_no_assignment(mem_db):
    with Session(mem_db) as s:
        p = _mk_profile(s, default_provider="deepseek", default_model="prof-model")
        head = resolve_chain_head(s, str(p.id), "experiencer")
        assert head == ModelProfile(provider="deepseek", model="prof-model")


def test_no_overrides_returns_none(mem_db):
    with Session(mem_db) as s:
        p = _mk_profile(s)
        assert resolve_chain_head(s, str(p.id), "songwriter") is None
        assert resolve_chain_head(s, None, "songwriter") is None
        assert resolve_chain_head(s, str(uuid.uuid4()), "songwriter") is None


def test_model_only_override_rides_active_provider(mem_db, monkeypatch):
    class _FakeCM:
        def get_config(self):
            return {"provider": "nvidia", "nvidia": {}}

    monkeypatch.setattr("app.services.config_manager.ConfigManager", _FakeCM)
    with Session(mem_db) as s:
        p = _mk_profile(s)
        s.add(AgentAssignment(profile_id=str(p.id), role="producer", agent_name="songwriter",
                              model="pinned-model"))
        s.commit()
        head = resolve_chain_head(s, str(p.id), "songwriter")
        assert head == ModelProfile(provider="nvidia", model="pinned-model")


def test_chain_head_rides_first_without_duplicate_provider(monkeypatch):
    monkeypatch.setattr(
        "app.agents.runtime.policy.ConfigManager_safe_config",
        lambda: {"provider": "deepseek", "deepseek": {"model": "global-model"}},
    )
    policy = ResiliencePolicy(chain_head=ModelProfile(provider="deepseek", model="override-model"))
    chain = policy.resolve_chain()
    assert chain[0].provider == "deepseek"
    assert chain[0].model == "override-model"
    assert [p.provider for p in chain].count("deepseek") == 1


# ---------------------------------------------------------------- lore grounding

def test_load_artist_lore_structured_and_raw(mem_db):
    with Session(mem_db) as s:
        p = _mk_profile(s)
        assert load_artist_lore(s, str(p.id)) == ""
        p.lore_json = json.dumps({"hometown": "Lusaka", "era": "1970s"})
        s.add(p); s.commit()
        lore = load_artist_lore(s, str(p.id))
        assert "Lusaka" in lore
        p.lore_json = "plain text lore"
        s.add(p); s.commit()
        assert load_artist_lore(s, str(p.id)) == "plain text lore"
        p.lore_json = "{corrupt"
        s.add(p); s.commit()
        assert load_artist_lore(s, str(p.id)) == "{corrupt"


def test_grounding_round_trip(mem_db):
    with Session(mem_db) as s:
        p = _mk_profile(s, default_provider="deepseek", default_model="m")
        p.lore_json = json.dumps({"origin": "the copperbelt"})
        s.add(p); s.commit()
        head, lore = resolve_artist_grounding(s, str(p.id), "experiencer")
        assert head.model == "m"
        assert "copperbelt" in lore


def test_experiencer_prompt_includes_lore_block():
    from app.agents.experiencer.agent import EXPERIENCER_AGENT
    from app.agents.experiencer.schemas import AlbumBrief
    from app.agents.runtime.context import RunContext

    brief = AlbumBrief(album_title="Test", album_concept="Concept", track_target=2)
    ctx = RunContext(agent_name="experiencer", artist_lore='{"origin": "copperbelt"}')
    msgs = EXPERIENCER_AGENT.build_messages(brief, ctx)
    assert any("ARTIST WORLD LORE" in m["content"] for m in msgs)
    ctx_none = RunContext(agent_name="experiencer")
    msgs_none = EXPERIENCER_AGENT.build_messages(brief, ctx_none)
    assert not any("ARTIST WORLD LORE" in m["content"] for m in msgs_none)


# ---------------------------------------------------------------- API: lifecycle

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
    def patch(self, url, **kw): return self._call("patch", url, **kw)
    def put(self, url, **kw): return self._call("put", url, **kw)
    def delete(self, url, **kw): return self._call("delete", url, **kw)


@pytest.fixture
def client():
    return SyncClient(app)


@pytest.fixture
def artist_fixture():
    with Session(app_engine) as session:
        profile = ArtistProfile(name=f"LFC-{uuid.uuid4().hex[:8]}", bio="integration")
        session.add(profile)
        session.commit()
        session.refresh(profile)
        release = Release(title="LFC Release", profile_id=str(profile.id))
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
            for a in session.exec(select(AgentAssignment).where(AgentAssignment.profile_id == pid)).all():
                session.delete(a)
            session.delete(prof)
        session.commit()


def test_get_release_returns_counts_and_active_flag(client, artist_fixture):
    rid = artist_fixture["release_id"]
    with Session(app_engine) as session:
        session.add(Job(title="t", prompt="p", lyrics="l", release_id=rid, status=JobStatus.COMPLETED))
        session.commit()
    data = client.get(f"/releases/{rid}").json()
    assert data["track_total"] == 1
    assert data["active_run"] is False
    assert data["release"]["title"] == "LFC Release"


def test_patch_release_rename_and_transition(client, artist_fixture):
    rid = artist_fixture["release_id"]
    out = client.patch(f"/releases/{rid}", json={"title": "Renamed"})
    assert out.status_code == 200
    assert out.json()["title"] == "Renamed"

    out = client.patch(f"/releases/{rid}", json={"status": "in_progress"})
    assert out.status_code == 200
    assert out.json()["status"] == "in_progress"

    out = client.patch(f"/releases/{rid}", json={"status": "completed"})
    assert out.status_code == 200
    assert out.json()["status"] == "completed"


def test_patch_release_rejects_bad_transition_and_empty_title(client, artist_fixture):
    rid = artist_fixture["release_id"]
    out = client.patch(f"/releases/{rid}", json={"status": "shipped"})
    assert out.status_code == 422
    assert out.json()["detail"]["error"]["code"] == "invalid_transition"

    out = client.patch(f"/releases/{rid}", json={"title": "   "})
    assert out.status_code == 422


def test_patch_profile_lore_round_trip(client, artist_fixture):
    pid = artist_fixture["profile_id"]
    lore = json.dumps({"origin": "copperbelt", "era": "1970s"})
    out = client.patch(f"/profiles/{pid}", json={"lore_json": lore})
    assert out.status_code == 200
    detail = client.get(f"/profiles/{pid}").json()
    assert detail["profile"]["lore_json"] == lore


def test_delete_release_detaches_jobs_and_blocks_on_active_run(client, artist_fixture):
    rid = artist_fixture["release_id"]
    track_title = f"Detach {uuid.uuid4().hex[:6]}"
    with Session(app_engine) as session:
        session.add(Job(title=track_title, prompt="p", lyrics="l", release_id=rid, status=JobStatus.COMPLETED))
        session.add(AgentRun(agent_name="album_orchestrator", status="running",
                             release_id=rid,
                             input_json=json.dumps({"release_id": rid})))
        session.commit()

    out = client.delete(f"/releases/{rid}")
    assert out.status_code == 409

    with Session(app_engine) as session:
        for run in session.exec(select(AgentRun).where(AgentRun.agent_name == "album_orchestrator")).all():
            if album_run_release_id(run) == rid:
                session.delete(run)
        session.commit()

    out = client.delete(f"/releases/{rid}")
    assert out.status_code == 200
    assert out.json()["jobs_detached"] == 1
    with Session(app_engine) as session:
        job = session.exec(select(Job).where(Job.title == track_title)).first()
        assert job is not None  # detached, not deleted
        assert job.release_id is None
        session.delete(job)
        session.commit()


# ---------------------------------------------------------------- API: stats + pagination

def test_profiles_with_stats_and_total(client, artist_fixture):
    pid = artist_fixture["profile_id"]
    with Session(app_engine) as session:
        session.add(AgentAssignment(profile_id=pid, role="producer", agent_name="songwriter"))
        session.commit()
    data = client.get("/profiles?with_stats=1&limit=200").json()
    assert "total" in data
    stats = data.get("stats", {}).get(pid)
    assert stats is not None
    assert stats["crew_count"] == 1
    assert stats["release_count"] == 1
    assert stats["last_activity"]


def test_agent_runs_pagination_shape(client):
    data = client.get("/agents/runs?limit=2&offset=0").json()
    assert "total" in data
    assert isinstance(data["runs"], list)
    assert len(data["runs"]) <= 2


def test_profile_releases_pagination_shape(client, artist_fixture):
    rid = artist_fixture["release_id"]
    data = client.get(f"/profiles/{artist_fixture['profile_id']}/releases?limit=5").json()
    assert "total" in data
    assert any(str(r["id"]) == rid for r in data["releases"])


# ---------------------------------------------------------------- project scoping (H28)

def test_assignment_provider_validation(client, artist_fixture):
    pid = artist_fixture["profile_id"]
    out = client.put(f"/profiles/{pid}/assignments", json={
        "assignments": [{"role": "producer", "agent_name": "experiencer", "model_provider": "skynet"}],
    })
    assert out.status_code == 422
    assert "Unknown provider" in out.json()["detail"]["error"]["message"]

    out = client.put(f"/profiles/{pid}/assignments", json={
        "assignments": [{"role": "producer", "agent_name": "experiencer", "model_provider": "deepseek", "model": "m"}],
    })
    assert out.status_code == 200
    assert out.json()["assignments"][0]["model_provider"] == "deepseek"


def test_profile_create_validates_project_exists(client):
    out = client.post("/profiles", json={"name": "Scoped Artist", "project_id": str(uuid.uuid4())})
    assert out.status_code == 404
    assert out.json()["detail"]["error"]["code"] == "not_found"

    out = client.post("/profiles", json={"name": "Scoped Artist", "project_id": "not-a-uuid"})
    assert out.status_code == 422

    from app.models import Project
    with Session(app_engine) as session:
        project = Project(name=f"PRJ-{uuid.uuid4().hex[:6]}")
        session.add(project)
        session.commit()
        session.refresh(project)
        project_id = str(project.id)
    try:
        out = client.post("/profiles", json={"name": "Scoped Artist", "project_id": project_id})
        assert out.status_code == 200
        pid = out.json()["id"]
        client.delete(f"/profiles/{pid}")
    finally:
        with Session(app_engine) as session:
            project = session.get(Project, uuid.UUID(project_id))
            if project:
                session.delete(project)
                session.commit()


# ---------------------------------------------------------------- single-seed retry

def test_retry_endpoint_rejects_non_failed_job(client, artist_fixture):
    rid = artist_fixture["release_id"]
    out = client.post(f"/releases/{rid}/tracks/{uuid.uuid4()}/retry")
    assert out.status_code == 422
    assert out.json()["detail"]["error"]["code"] == "not_retryable"


def test_retry_endpoint_rejects_when_run_active(client, artist_fixture):
    rid = artist_fixture["release_id"]
    with Session(app_engine) as session:
        session.add(AgentRun(agent_name="album_orchestrator", status="running",
                             release_id=rid,
                             input_json=json.dumps({"release_id": rid})))
        session.commit()
    out = client.post(f"/releases/{rid}/tracks/{uuid.uuid4()}/retry")
    assert out.status_code == 409


@pytest.mark.asyncio
async def test_retry_success_promotes_cursor_winner():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    from unittest.mock import AsyncMock, MagicMock, patch

    from app.agents.orchestrator import RunRegistry
    from app.agents.orchestrator.album import AlbumOrchestrator, retry_single_seed

    orchestrator = AlbumOrchestrator(RunRegistry(), lambda *a, **k: None)

    with Session(engine) as session:
        profile = ArtistProfile(name="Retry Artist", bio="b")
        session.add(profile); session.commit(); session.refresh(profile)
        release = Release(title="Retry Album", profile_id=str(profile.id), vision_json=json.dumps({
            "journey_title": "Retry Album", "concept_statement": "c",
            "song_seeds": [{"working_title": "S1", "mood": "m", "energy": 0.5}],
        }))
        session.add(release); session.commit(); session.refresh(release)
        album_run = AgentRun(agent_name="album_orchestrator", status="failed",
                             release_id=str(release.id),
                             input_json=json.dumps({"release_id": str(release.id)}),
                             state_json=json.dumps({
                                 "completed_seeds": [],
                                 "slot_jobs": {},
                                 "failed_jobs": {"0": ["old-failed-job"]},
                             }))
        session.add(album_run); session.commit(); session.refresh(album_run)
        retry_run = AgentRun(agent_name="track_retry", status="queued",
                             profile_id=str(profile.id),
                             release_id=str(release.id),
                             input_json=json.dumps({
                                 "release_id": str(release.id),
                                 "job_id": "old-failed-job",
                                 "seed_slot": 0,
                                 "album_run_id": str(album_run.id),
                             }))
        session.add(retry_run); session.commit(); session.refresh(retry_run)
        run_id, release_id, album_run_id = retry_run.id, release.id, album_run.id

    mock_job = MagicMock()
    mock_job.id = "new-winner-job"
    mock_job.title = "S1"

    with patch("app.agents.orchestrator.album.create_track_from_seed", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_job
        await retry_single_seed(
            parent_run_id=run_id, release_id=release_id,
            engine=engine, orchestrator=orchestrator)

    with Session(engine) as session:
        run = session.get(AgentRun, run_id)
        assert run.status == "succeeded"
        album = session.get(AgentRun, album_run_id)
        state = json.loads(album.state_json)
        assert state["slot_jobs"] == {"0": "new-winner-job"}
        assert 0 in state["completed_seeds"]
        assert "old-failed-job" not in state["failed_jobs"]["0"]


@pytest.mark.asyncio
async def test_retry_failure_pins_new_failed_job():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    from unittest.mock import AsyncMock, patch

    from app.agents.orchestrator import RunRegistry
    from app.agents.orchestrator.album import AlbumOrchestrator, retry_single_seed
    from app.agents.orchestrator.bridge import TrackProductionError

    orchestrator = AlbumOrchestrator(RunRegistry(), lambda *a, **k: None)

    with Session(engine) as session:
        profile = ArtistProfile(name="Retry Fail Artist", bio="b")
        session.add(profile); session.commit(); session.refresh(profile)
        release = Release(title="Retry Fail Album", profile_id=str(profile.id), vision_json=json.dumps({
            "journey_title": "t", "concept_statement": "c",
            "song_seeds": [{"working_title": "S1", "mood": "m", "energy": 0.5}],
        }))
        session.add(release); session.commit(); session.refresh(release)
        album_run = AgentRun(agent_name="album_orchestrator", status="failed",
                             release_id=str(release.id),
                             input_json=json.dumps({"release_id": str(release.id)}),
                             state_json=json.dumps({"failed_jobs": {"0": ["old-job"]}}))
        session.add(album_run); session.commit(); session.refresh(album_run)
        retry_run = AgentRun(agent_name="track_retry", status="queued",
                             release_id=str(release.id),
                             input_json=json.dumps({
                                 "release_id": str(release.id),
                                 "job_id": "old-job",
                                 "seed_slot": 0,
                                 "album_run_id": str(album_run.id),
                             }))
        session.add(retry_run); session.commit(); session.refresh(retry_run)
        run_id, release_id, album_run_id = retry_run.id, release.id, album_run.id

    with patch("app.agents.orchestrator.album.create_track_from_seed", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = TrackProductionError("did not complete", job_id="second-failed-job")
        await retry_single_seed(
            parent_run_id=run_id, release_id=release_id,
            engine=engine, orchestrator=orchestrator)

    with Session(engine) as session:
        run = session.get(AgentRun, run_id)
        assert run.status == "failed"
        album = session.get(AgentRun, album_run_id)
        state = json.loads(album.state_json)
        assert state["failed_jobs"]["0"] == ["old-job", "second-failed-job"]


# ---------------------------------------------------------------- lore -> songwriter (A3)

def test_songwriter_prompt_includes_album_context_lore():
    from app.agents.songwriter.agent import SONGWRITER_AGENT
    from app.agents.runtime.context import RunContext

    seed = {"working_title": "S1", "mood": "m", "story_seed": "s", "energy": 0.5}
    ctx_with = {"album_title": "T", "album_concept": "C", "artist_name": "A",
                "artist_lore": '{"origin": "copperbelt"}'}
    msgs = SONGWRITER_AGENT.build_messages(seed, ctx_with, RunContext(agent_name="songwriter"))
    assert any("ARTIST WORLD LORE" in m["content"] for m in msgs)

    ctx_without = {"album_title": "T", "album_concept": "C", "artist_name": "A"}
    msgs_without = SONGWRITER_AGENT.build_messages(seed, ctx_without, RunContext(agent_name="songwriter"))
    assert not any("ARTIST WORLD LORE" in m["content"] for m in msgs_without)


# ---------------------------------------------------------------- world-builder (A2)

def test_lore_generate_route_persists_canon(client, artist_fixture, monkeypatch):
    from app.agents.world_builder.agent import WORLD_BUILDER_AGENT
    from app.agents.world_builder.schemas import WorldLore

    pid = artist_fixture["profile_id"]

    lore = WorldLore(
        origin_story="Raised on the copperbelt, shaped by mine-town radio.",
        era_setting="1970s Zambia",
        appearance="Wide-brim hat, worn acoustic guitar, dust-colored coat.",
        musical_dna=["fingerpicked guitar", "warm baritone", "railway rhythms"],
        influences=["Zamrock", "1970s highlife"],
        lore_facts=["Never performed outside Zambia before 1975."],
        avoid_contradictions=["Do not describe the artist as city-bred."],
        signature="Songs are places you can return to.",
    )

    class _Outcome:
        attempts = []

    class _Result:
        output = lore
        outcome = _Outcome()
        attempts = []
        tokens_in = 10
        tokens_out = 20
        latency_ms = 5

    async def _fake_run(brief, ctx, policy=None):
        # Grounding: existing lore should be in the prompt context
        assert brief.artist_name
        return _Result()

    monkeypatch.setattr(WORLD_BUILDER_AGENT, "run", _fake_run)

    resp = client.post(f"/profiles/{pid}/lore/generate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["lore"]["era_setting"] == "1970s Zambia"

    detail = client.get(f"/profiles/{pid}").json()
    assert "copperbelt" in detail["profile"]["lore_json"]

    # ledger row exists and succeeded
    runs = client.get(f"/agents/runs?profile_id={pid}").json()["runs"]
    assert any(r["agent_name"] == "world_builder" and r["status"] == "succeeded" for r in runs)


def test_world_builder_registered_and_listed():
    from app.agents.registry import AGENTS, list_agents
    assert "world_builder" in AGENTS
    assert any(a["name"] == "world_builder" for a in list_agents())


def test_world_builder_schema_rejects_empty_name():
    from app.agents.world_builder.schemas import WorldBuilderBrief
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        WorldBuilderBrief(artist_name="")


# ---------------------------------------------------------------- B2: track ordering

def test_track_order_round_trip_and_validation(client, artist_fixture):
    from app.models import Release as _Release
    rid = artist_fixture["release_id"]
    from app.main import engine as app_engine
    with Session(app_engine) as session:
        j1 = Job(title="o1", prompt="p", lyrics="l", release_id=rid, status=JobStatus.COMPLETED)
        j2 = Job(title="o2", prompt="p", lyrics="l", release_id=rid, status=JobStatus.COMPLETED)
        session.add(j1); session.add(j2); session.commit(); session.refresh(j1); session.refresh(j2)
        ids = [str(j2.id), str(j1.id)]

    # unknown id → 422
    out = client.patch(f"/releases/{rid}/track-order", json={"job_ids": [str(uuid.uuid4())]})
    assert out.status_code == 422

    out = client.patch(f"/releases/{rid}/track-order", json={"job_ids": ids})
    assert out.status_code == 200

    data = client.get(f"/releases/{rid}/tracks").json()
    assert [t["id"] for t in data["tracks"]] == ids

    with Session(app_engine) as session:
        for j in session.exec(select(Job).where(Job.release_id == rid)).all():
            session.delete(j)
        session.commit()


def test_track_order_requires_array(client, artist_fixture):
    rid = artist_fixture["release_id"]
    out = client.patch(f"/releases/{rid}/track-order", json={"job_ids": "nope"})
    assert out.status_code == 422


# ---------------------------------------------------------------- C3: ledger retention

def test_prune_agent_runs_spares_cursors_and_active():
    from datetime import datetime, timedelta, timezone as tz
    from app.main import prune_agent_runs, engine as app_engine

    old = datetime.now(tz.utc) - timedelta(days=90)
    with Session(app_engine) as session:
        old_leaf = AgentRun(agent_name="experiencer", status="succeeded", created_at=old)
        old_album = AgentRun(agent_name="album_orchestrator", status="failed",
                             created_at=old, input_json=json.dumps({"release_id": "keep-cursor"}))
        fresh = AgentRun(agent_name="experiencer", status="succeeded")
        active = AgentRun(agent_name="experiencer", status="running", created_at=old)
        for r in (old_leaf, old_album, fresh, active):
            session.add(r)
        session.commit()
        session.refresh(old_leaf); session.refresh(old_album); session.refresh(fresh); session.refresh(active)
        kept_ids = {str(fresh.id), str(active.id), str(old_album.id)}

    deleted = prune_agent_runs(30)
    assert deleted >= 1

    with Session(app_engine) as session:
        assert session.get(AgentRun, old_leaf.id) is None          # pruned
        assert session.get(AgentRun, old_album.id) is not None     # cursor: never pruned
        assert session.get(AgentRun, fresh.id) is not None         # fresh: kept
        assert session.get(AgentRun, active.id) is not None        # active: kept
        # cleanup
        for rid in (old_album.id, fresh.id, active.id):
            r = session.get(AgentRun, rid)
            if r:
                session.delete(r)
        session.commit()
