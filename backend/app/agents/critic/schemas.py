"""Critic contracts: seed + draft in, a bounded-verdict review out."""
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class CriticBrief(BaseModel):
    """The draft under review, plus the seed it must serve."""

    seed: Dict[str, Any] = Field(..., description="The experience seed this song must serve.")
    draft: Dict[str, Any] = Field(..., description="The songwriter's draft: title, lyrics, style_tags, rationale.")
    revision_context: str = Field(
        default="",
        description="When reviewing a revision, what changed since the last review.",
    )


class Critique(BaseModel):
    """Pre-generation review. `revise` triggers exactly ONE bounded revision."""

    verdict: Literal["pass", "revise", "concern"]
    score: float = Field(..., ge=0, le=1, description="Overall fit/quality, 0-1.")
    notes: str = Field(..., description="Actionable, specific. On 'revise' these ARE the fix list.")
    contradictions: List[str] = Field(default_factory=list, description="Lore/world contradictions found, if any.")
