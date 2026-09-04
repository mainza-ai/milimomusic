"""Experiencer-for-release adapter: runs the ExperiencerAgent against a Release
and persists the vision onto release.vision_json (auto-persist on success)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from sqlmodel import Session

from app.agents.experiencer.agent import EXPERIENCER_AGENT
from app.agents.experiencer.schemas import AlbumBrief
from app.agents.runtime.context import RunContext
from app.agents.runtime.overrides import resolve_artist_grounding
from app.agents.runtime.policy import ResiliencePolicy
from app.models import AgentRun, ArtistProfile, Release

logger = logging.getLogger("milimo.agents.album")

DEFAULT_TRACK_TARGET = 5


async def run_experiencer_for_release(
    *,
    release: Release,
    session: Session,
    engine,
    parent_run_id: str | None = None,
) -> Dict[str, Any]:
    profile = session.get(ArtistProfile, release.profile_id)
    name = profile.name if profile else ""
    bio = (profile.bio if profile else "") or ""
    tags = (profile.tags if profile else "") or ""

    brief = AlbumBrief(
        album_title=release.title,
        album_concept=release.description or "",
        artist_name=name,
        artist_bio=bio,
        tags=tags,
        track_target=DEFAULT_TRACK_TARGET,
        extra_direction="",
    )
    child = AgentRun(
        agent_name="experiencer",
        status="running",
        input_json=brief.model_dump_json(),
        parent_run_id=parent_run_id,
        profile_id=str(release.profile_id),
    )
    session.add(child)
    session.commit()
    session.refresh(child)

    # Crew overrides + artist lore grounding for THIS artist's experiencer.
    chain_head, artist_lore = resolve_artist_grounding(
        session, str(release.profile_id), "experiencer")
    ctx = RunContext(
        agent_name="experiencer", run_id=str(child.id),
        artist_profile_id=str(release.profile_id),
        artist_lore=artist_lore or None,
    )
    policy = ResiliencePolicy(chain_head=chain_head)
    try:
        result = await EXPERIENCER_AGENT.run(brief, ctx, policy)
        vision = result.vision
        child.status = "succeeded"
        child.output_json = vision.model_dump_json()
        child.finished_at = datetime.now(timezone.utc)
        session.add(child)

        # Auto-persist: the Release now carries its imagined journey.
        release.vision_json = vision.model_dump_json()
        release.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add(release)
        session.commit()

        logger.info(f"[album] Vision '{vision.journey_title}' persisted to release "
                    f"{release.id} ({len(vision.song_seeds)} seeds).")
        return vision.model_dump()
    except Exception as exc:
        child.status = "failed"
        child.error_type = type(exc).__name__
        child.error_message = str(exc)[:2000]
        child.finished_at = datetime.now(timezone.utc)
        session.add(child)
        session.commit()
        raise
