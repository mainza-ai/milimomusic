"""Experiencer schemas — the typed contract between the brief and the vision.

Field descriptions double as prompt documentation: the persona prompt embeds
the JSON shape, and these descriptions are what the model is asked to honor.
Validation is deliberately forgiving (extra keys ignored, coercions allowed)
per the house pattern in lyrics_schemas.py.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class AlbumBrief(BaseModel):
    """What the caller hands the Experiencer: the album concept to expand."""

    album_title: str = Field(..., min_length=1, description="Working title of the album.")
    album_concept: str = Field(
        ...,
        min_length=3,
        description="The core concept/premise of the album — a sentence to a paragraph.",
    )
    artist_name: str = Field(default="", description="Artist profile name, if one exists.")
    artist_bio: str = Field(default="", description="Artist bio/lore for identity grounding.")
    tags: str = Field(default="", description="Comma-separated style/genre hints (optional).")
    track_target: int = Field(
        default=10, ge=1, le=30,
        description="How many song seeds to imagine.",
    )
    extra_direction: str = Field(
        default="",
        description="Free-form steering from the user (mood, references, constraints).",
    )


# ---------------------------------------------------------------------------
# Output — the Creative Vision artifact
# ---------------------------------------------------------------------------
class EmotionalBeat(BaseModel):
    """One phase of the imagined life journey across the album's running order."""

    position: int = Field(..., ge=0, description="1-based or 0-based ordering along the arc.")
    label: str = Field(..., description="Short name for this emotional phase, e.g. 'Departure'.")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Emotional intensity 0..1.")
    description: str = Field(default="", description="One or two sentences on what is felt here.")


class SongSeed(BaseModel):
    """A single imagined song: not lyrics, not a caption — an EXPERIENCE seed.

    Downstream agents (songwriter, style curator) turn each seed into a real
    track; the Experiencer's job is to make every seed feel lived-in.
    """

    working_title: str = Field(..., description="Evocative placeholder title.")
    mood: str = Field(..., description="Dominant emotional color, e.g. 'ache under neon light'.")
    story_seed: str = Field(
        ...,
        min_length=20,
        description="The specific human moment this song lives in — who, where, what is at stake.",
    )
    suggested_style_tags: List[str] = Field(
        default_factory=list,
        description="2-6 style/genre tags consistent with the artist and moment.",
    )
    energy: float = Field(0.5, ge=0.0, le=1.0)
    placement_hint: str = Field(
        default="anywhere",
        description="opener | early | mid | late | closer | anywhere",
    )


class ExperiencerVision(BaseModel):
    """The full creative-direction artifact produced from one brief."""

    journey_title: str = Field(..., description="Name for the imagined life journey/arc.")
    concept_statement: str = Field(
        ..., min_length=30,
        description="Two-to-four sentences expanding the raw concept into a world worth living in.",
    )
    life_journey_narrative: str = Field(
        ..., min_length=100,
        description="Multi-paragraph narrative imagining the experiences behind the album — scenes, places, turning points.",
    )
    emotional_arc: List[EmotionalBeat] = Field(
        ..., min_length=1, description="Ordered phases of the journey.",
    )
    song_seeds: List[SongSeed] = Field(
        ..., min_length=1, description="One seed per intended track (aim for track_target).",
    )
    recurring_motifs: List[str] = Field(
        default_factory=list,
        description="Images/phrases/symbols that can recur across songs for cohesion.",
    )
    listener_experience_notes: str = Field(
        default="",
        description="What someone should FEEL listening start to finish.",
    )
