"""Seed→Track bridge: the creative hop from an ExperiencerVision.SongSeed to a
fully-generated, fully-transcribed Job attached to a Release.

Chain (verified against live systems):
  songwriter draft → sanitize_lyrics → genre-first tag ordering → rewrite_caption
  → GenerationRequest (EXPLICIT duration + seed) → Job(release_id, artist_profile_id)
  → await music_service.generate_task (GPU lock serializes automatically)
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from sqlmodel import Session

from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import ResiliencePolicy
from app.agents.songwriter.agent import SONGWRITER_AGENT
from app.agents.songwriter.schemas import SongDraft
from app.models import GenerationRequest, Job
from app.services.lyrics_graph import sanitize_lyrics

logger = logging.getLogger("milimo.agents.bridge")


class TrackProductionError(RuntimeError):
    """A created Job failed to complete generation; carries the Job row id so
    the orchestrator can pin the failure to its album seed slot."""

    def __init__(self, message: str, job_id: str):
        super().__init__(message)
        self.job_id = job_id

# MiniMax destructures tags positionally: [0]=genre, [1]=tempo/mood, [2:]=instruments.
# Unsorted/exotic tags yield nonsense captions — ordering IS validation here.
KNOWN_GENRES = {
    "pop", "rock", "indie folk", "folk", "indie rock", "synthwave", "alt r&b",
    "r&b", "hip hop", "trap", "country", "jazz", "blues", "electronic", "edm",
    "house", "techno", "ambient", "shoegaze", "dream pop", "punk", "metal",
    "soul", "funk", "reggae", "latin", "afrobeats", "classical", "lo-fi",
}


def order_tags_genre_first(tags: List[str]) -> List[str]:
    """Return tags ordered [primary-genre, texture..., instruments...], max 6."""

    def is_genre(tag: str) -> bool:
        return tag.strip().lower() in KNOWN_GENRES

    genres = [t for t in tags if is_genre(t)]
    rest = [t for t in tags if not is_genre(t)]
    ordered = (genres + rest)[:6]
    return ordered or ["Pop"]


def energy_to_duration_s(energy: float, base_s: int = 120, span_s: int = 120) -> int:
    """Energy-scaled track length: 0.0→base_s, 1.0→base_s+span_s (120–240s)."""
    clamped = min(1.0, max(0.0, float(energy)))
    return int(base_s + clamped * span_s)


def build_steering_prose(seed: dict) -> str:
    """Synthesize seed fields with no direct parameter destination into the
    user_message steering string consumed by downstream caption/producer steps."""
    parts = [
        f"Working title: {seed.get('working_title', '')}",
        f"Mood: {seed.get('mood', '')}",
    ]
    energy = float(seed.get("energy", 0.5))
    if energy >= 0.75:
        parts.append("Energy: high and driving")
    elif energy <= 0.25:
        parts.append("Energy: sparse and restrained")
    else:
        parts.append("Energy: moderate")
    hint = seed.get("placement_hint")
    if hint and hint != "mid-album":
        parts.append(f"Arc placement: {hint}")
    return ". ".join(p for p in parts if p.strip(": "))


async def create_track_from_seed(
    *,
    seed: dict,
    album_context: Dict[str, Any],
    artist_profile_id,
    release_id,
    project_id: Optional[str],
    provider_name: str = "minimax_music3",
    ctx: Optional[RunContext] = None,
    policy: Optional[ResiliencePolicy] = None,
    engine=None,
    progress_cb=None,
) -> Job:
    """Run ONE seed through the full creative pipeline. Raises on failure;
    callers (album runner) own ledger/cursor updates."""
    eng = engine
    if eng is None:
        from app.main import engine as _eng  # late bind avoids circular import
        eng = _eng
    step = progress_cb or (lambda *a, **k: None)

    # 1) The Songwriter writes.
    step("songwriter", "Writing lyrics…")
    policy = policy or ResiliencePolicy()
    draft: SongDraft = await SONGWRITER_AGENT.run(seed, album_context, ctx, policy)

    # 2) Sanitize + validate (reuse the exact production utilities).
    clean_lyrics = sanitize_lyrics(draft.lyrics)
    if len(clean_lyrics) < 30:
        raise ValueError(
            f"Songwriter produced unusable lyrics for '{draft.title}' "
            f"({len(clean_lyrics)} chars after sanitize)."
        )
    tags_list = order_tags_genre_first(draft.style_tags)
    tags_str = ", ".join(tags_list)

    # 3) Professional structured caption (never raises; falls back honestly).
    from app.services.llm_service import LLMService
    step("caption", "Crafting studio caption…")
    caption_result = await LLMService.rewrite_caption(
        concept=f"{draft.title} — {build_steering_prose(seed)}",
        lyrics=clean_lyrics,
        tags=tags_str,
    )
    structured_caption = caption_result.get("structured_caption") or {}

    # 4) Explicit duration + seed (defaults are traps: 30s / randomized).
    import os as _os
    _cap_s = int(_os.environ.get("MILIMO_MAX_DURATION_S", "240"))
    duration_ms = min(energy_to_duration_s(float(seed.get("energy", 0.5))), _cap_s) * 1000
    seed_val = random.randint(0, 2**32 - 1)

    # Rich prompt prevents producer-enhancement from rewriting the
    # songwriter's curated tags/prompt after the fact (friction F6).
    rich_prompt = (
        f"{draft.title}. {build_steering_prose(seed)}. "
        f"Style: {tags_str}. Album: {album_context.get('album_title', '')}"
    )
    req = GenerationRequest(
        prompt=rich_prompt,
        lyrics=clean_lyrics,
        title=draft.title,
        duration_ms=duration_ms,
        tags=tags_str,
        seed=seed_val,
        model_provider=provider_name,
        structured_caption=structured_caption or None,
        project_id=project_id,
    )

    job = Job(
        title=req.title,
        prompt=req.prompt,
        lyrics=req.lyrics,
        duration_ms=req.duration_ms,
        tags=req.tags,
        seed=req.seed,
        model_provider=req.model_provider,
        llm_model=req.llm_model,
        project_id=req.project_id,
        temperature=req.temperature,
        cfg_scale=req.cfg_scale,
        topk=req.topk,
        artist_profile_id=str(artist_profile_id),
        release_id=str(release_id),
        voice_profile_id=None,
    )

    from app.services.music_service import music_service

    with Session(eng) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
        job_id_str = str(job.id)

    logger.info(f"[bridge] Track '{req.title}' queued as job {job_id_str} "
                f"({duration_ms // 1000}s, tags='{tags_str}')")
    step("generation", f"Generating audio for '{req.title}'…")

    # 5) Block on real generation+pipeline (GPU lock serializes album children).
    await music_service.generate_task(job_id, req, eng)

    with Session(eng) as session:
        final = session.get(Job, job_id)
        session.refresh(final)
        # Pipeline catches generation errors internally (marks FAILED, returns).
        # Truth must propagate to the album ledger — never count a failed track.
        if str(final.status).lower() not in ("completed", "jobstatus.completed"):
            raise TrackProductionError(
                f"Track '{req.title}' did not complete (status={final.status}"
                f"{': ' + str(final.error_msg)[:120] if getattr(final, 'error_msg', None) else ''})",
                job_id=job_id_str,
            )
        return final
