"""World-Builder contracts: brief in, WorldLore out."""
from typing import List

from pydantic import BaseModel, Field


class WorldBuilderBrief(BaseModel):
    """What the World-Builder needs to imagine an artist's world."""

    artist_name: str = Field(..., min_length=1)
    artist_bio: str = ""
    tags: str = ""
    extra_direction: str = ""


class WorldLore(BaseModel):
    """The artist's canonical world document. Persisted to ArtistProfile.lore_json."""

    origin_story: str = Field(default="", description="Where they came from and how they became an artist — a tight paragraph.")
    era_setting: str = Field(default="", description="The time/place texture this artist lives in.")
    appearance: str = Field(default="", description="Visual identity brief — usable directly for cover-art generation.")
    musical_dna: List[str] = Field(default_factory=list, description="3-6 unmistakable musical traits.")
    influences: List[str] = Field(default_factory=list, description="Scenes, eras, or artists that shaped them.")
    lore_facts: List[str] = Field(default_factory=list, description="Binding canon facts the whole crew must respect.")
    avoid_contradictions: List[str] = Field(default_factory=list, description="Tempting-but-wrong details to never invent.")
    signature: str = Field(default="", description="One-line artist signature or motto.")
