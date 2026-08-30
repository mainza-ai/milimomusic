"""Stylist contracts: draft + seed in, a tight tag list out."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class StylistBrief(BaseModel):
    """The draft + context the Stylist curates from."""

    seed: Dict[str, Any] = Field(..., description="The experience seed this song must serve.")
    draft: Dict[str, Any] = Field(..., description="The songwriter's draft: title, lyrics, style_tags, rationale.")
    artist_name: str = ""
    album_title: str = ""
    artist_lore: str = ""


class StylingChoice(BaseModel):
    """Final tags for generation. `order_tags_genre_first` runs after this."""

    style_tags: List[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="2-6 tags. FIRST tag is the primary genre; later tags texture/instruments.",
    )
    rationale: str = Field(default="", description="One sentence: why these tags serve the seed.")
