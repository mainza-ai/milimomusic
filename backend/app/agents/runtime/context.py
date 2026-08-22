"""Per-run execution context for agents.

A RunContext is everything an agent needs to know about WHO it is working for
and WHERE its output should attach — the seam that will later carry Artist
Profile identity (see wiki/concepts/artist-profiles-vision.md). Today it also
finishes the audit's G4 gap at the surface level: session/project linkage
finally has a first-class carrier instead of dead `chat_history` kwargs.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunContext:
    agent_name: str
    run_id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)

    # Attachment points (all optional today; the album vision fills them in)
    session_id: Optional[str] = None      # Ask-Producer chat thread
    project_id: Optional[str] = None       # owning project folder
    artist_profile_id: Optional[str] = None  # future: identity + memory scope

    # Conversation memory window (oldest → newest). Empty for one-shot runs.
    history: List[Dict[str, str]] = field(default_factory=list)

    # Free-form context the caller wants visible to the model.
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_public(self) -> Dict:
        return {
            "run_id": self.run_id,
            "agent": self.agent_name,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "artist_profile_id": self.artist_profile_id,
        }
