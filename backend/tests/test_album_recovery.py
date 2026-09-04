"""Tests for AlbumOrchestrator cursor resumption and recovery."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlmodel import SQLModel, create_engine, Session
from app.models import AgentRun, Release, Job, JobStatus, ArtistProfile
from app.agents.orchestrator import RunRegistry
from app.agents.orchestrator.album import AlbumOrchestrator


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.mark.asyncio
async def test_album_orchestrator_skips_completed_seeds(test_db):
    registry = RunRegistry()
    events = []
    def publish(evt, data):
        events.append((evt, data))

    orchestrator = AlbumOrchestrator(registry, publish)

    # Setup database with ArtistProfile, Release and AgentRun having completed_seeds = [0, 1]
    with Session(test_db) as session:
        profile = ArtistProfile(name="Test Artist", bio="Test Bio")
        session.add(profile)
        session.commit()
        session.refresh(profile)

        release = Release(title="Test Album", profile_id=str(profile.id))
        session.add(release)
        session.commit()
        session.refresh(release)

        vision = {
            "journey_title": "Test Album",
            "concept_statement": "A journey through sound",
            "song_seeds": [
                {"working_title": "Track 1", "mood": "Energetic", "energy": 0.8},
                {"working_title": "Track 2", "mood": "Calm", "energy": 0.3},
                {"working_title": "Track 3", "mood": "Epic", "energy": 0.9},
            ]
        }
        state = {
            "vision": vision,
            "completed_seeds": [0, 1],
            "track_job_ids": ["job-1", "job-2"],
        }
        run = AgentRun(
            agent_name="orchestrator",
            status="pending",
            state_json=json.dumps(state),
            budget_json=json.dumps({"caps": {}}),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id
        release_id = release.id

    # Mock create_track_from_seed so we verify only seed index 2 is invoked
    mock_job = MagicMock()
    mock_job.id = "job-3"
    mock_job.status = JobStatus.COMPLETED

    with patch("app.agents.orchestrator.album.create_track_from_seed", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_job

        await orchestrator.execute(
            parent_run_id=run_id,
            release_id=release_id,
            autopilot=True,
            engine=test_db,
        )

        # Assert create_track_from_seed was called exactly ONCE (for seed 2, skipping 0 and 1)
        assert mock_create.call_count == 1
        called_seed = mock_create.call_args.kwargs["seed"]
        assert called_seed["working_title"] == "Track 3"

    # Verify run state in database
    with Session(test_db) as session:
        final_run = session.get(AgentRun, run_id)
        assert final_run.status == "succeeded"
        final_state = json.loads(final_run.state_json)
        assert 0 in final_state["completed_seeds"]
        assert 1 in final_state["completed_seeds"]
        assert 2 in final_state["completed_seeds"]
        assert len(final_state["completed_seeds"]) == 3
