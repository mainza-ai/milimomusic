"""Songwriter output contracts."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SongDraft(BaseModel):
    """The songwriter's finished work for ONE seed: real lyrics + real tags."""

    title: str = Field(default="", description="Final song title (may refine the working_title).")
    lyrics: str = Field(
        ...,
        description="Complete lyric sheet with section headers like [Verse 1], [Chorus]. "
        "No preamble, no explanations — only the song.",
    )
    style_tags: List[str] = Field(
        ...,
        description="2-6 concrete style/genre tags. FIRST tag must be the primary genre "
        "(e.g. 'Indie Folk', 'Synthwave'); later tags instruments/production texture.",
    )
    lyrical_rationale: str = Field(
        default="",
        description="One or two sentences: how the lyric serves the seed's moment in the journey.",
    )
    shortfall: Optional[str] = Field(
        default=None,
        description="Honest note if the draft could not fully honor the seed/vision constraints.",
    )
