"""
Performer Role Ownership & Visible-Cast Scoping for Milimo Music Director Mode v2.

Solves the 'flapping lips' problem during instrumental solos, enforces 'mouth_movement: closed'
when vocals are inactive, and scopes performance prompts strictly to visible performers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("milimo.video.music_performance")


@dataclass
class PerformerRole:
    """Designation for an individual performer or band member."""

    role_id: str
    name: str
    instrument: Optional[str] = None
    is_vocalist: bool = False
    visual_appearance: Optional[str] = None


class MusicPerformanceDirector:
    """Generates strictly scoped visual performance instructions per clip."""

    @classmethod
    def format_performer_prompt_instructions(
        cls,
        visible_cast: Optional[List[str]] = None,
        is_vocal_active: bool = True,
        section_label: str = "Verse",
        is_instrumental_solo: bool = False,
        shot_intent: str = "performance",  # "performance" | "narrative" | "scenery" | "choreography"
    ) -> str:
        """
        Produce scoped performer and mouth movement instructions for diffusion prompts.

        Guarantees:
        1. When vocal is inactive or during solos, enforces 'mouth_movement: closed'.
        2. Narrative and scenery shots are stripped of vocal and musician boilerplate.
        3. Visible cast scoping: only references performers present in the current camera frame.
        """
        instructions: List[str] = []

        # 1. Scenery and Narrative shots: strip musician and vocal boilerplate completely
        if shot_intent in ["scenery", "landscape", "atmospheric"]:
            return "cinematic environment establishing shot, atmospheric background scenery, no active performers speaking or singing."

        if shot_intent == "narrative":
            return "cinematic storytelling scene, natural dramatic character movement, subtle expressions, no musical lip-sync."

        # 2. Vocal state enforcement
        if not is_vocal_active or is_instrumental_solo or section_label.lower() in ["intro", "outro", "solo"]:
            instructions.append(
                "mouth_movement: closed. Performer does not sing or move mouth. "
                "Facial expression focused, intense or emotive without vocalization."
            )
            if is_instrumental_solo:
                instructions.append(
                    "Dynamic close-up focus on instrument playing technique, hands and fingers performing on strings/keys/drums."
                )
        else:
            instructions.append(
                "Dynamic singing performance, natural vocal delivery matching song cadence, emotive facial engagement."
            )

        # 3. Visible Cast Scoping
        if visible_cast:
            cast_list = ", ".join(visible_cast)
            instructions.append(f"Visible performers in shot: {cast_list}. Focus direction exclusively on them.")
        else:
            # Default to lead performer if cast not specified
            instructions.append("Center framed lead musical performer.")

        return " ".join(instructions)
