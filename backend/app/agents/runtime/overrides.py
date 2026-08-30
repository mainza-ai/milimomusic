"""Crew override resolution — makes per-artist model assignments REAL.

Resolution order (the override chain):
    1. AgentAssignment override for (profile, agent)   → strongest
    2. ArtistProfile default_provider/default_model    → artist-wide
    3. None                                            → global config chain

A model pinned without a provider rides the globally-active provider. The
resolved head becomes ResiliencePolicy.chain_head: it is attempted FIRST and
the global failover chain remains intact behind it. The winning attempt
(provider + model) is recorded in the run ledger's attempts_json — overrides
are always observable after the fact.
"""
from __future__ import annotations

import json
import logging
import uuid as uuidlib
from typing import Optional, Tuple

from sqlmodel import Session, select

from app.agents.runtime.policy import ModelProfile

logger = logging.getLogger("milimo.agents.overrides")


def _clean(value: Optional[str]) -> Optional[str]:
    v = (value or "").strip()
    return v or None


def resolve_chain_head(session: Session, profile_id: Optional[str], agent_name: str) -> Optional[ModelProfile]:
    """Return the override chain head for this artist+agent, or None."""
    if not profile_id:
        return None
    from app.models import AgentAssignment, ArtistProfile

    try:
        pid = uuidlib.UUID(str(profile_id))
    except (ValueError, AttributeError, TypeError):
        return None
    profile = session.get(ArtistProfile, pid)
    if profile is None:
        return None

    provider: Optional[str] = None
    model: Optional[str] = None

    assignment = session.exec(
        select(AgentAssignment).where(
            AgentAssignment.profile_id == str(pid),
            AgentAssignment.agent_name == agent_name,
        )
    ).first()
    if assignment is not None:
        provider = _clean(assignment.model_provider)
        model = _clean(assignment.model)

    if provider is None and model is None:
        provider = _clean(getattr(profile, "default_provider", None))
        model = _clean(getattr(profile, "default_model", None))

    if provider is None and model is None:
        return None
    if provider is None:
        # Model pinned without a provider: ride the globally-active provider.
        from app.services.config_manager import ConfigManager
        provider = _clean(ConfigManager().get_config().get("provider"))
        if provider is None:
            logger.warning("crew override: model '%s' set without provider; no active provider — ignoring", model)
            return None

    return ModelProfile(provider=provider, model=model)


def load_artist_lore(session: Session, profile_id: Optional[str]) -> str:
    """Load the artist's lore document as prompt-ready text ('' if none).

    lore_json may hold a structured doc (World Builder domain) or freeform
    text; both are passed through honestly. Corrupt JSON degrades to raw text.
    """
    if not profile_id:
        return ""
    from app.models import ArtistProfile

    try:
        pid = uuidlib.UUID(str(profile_id))
    except (ValueError, AttributeError, TypeError):
        return ""
    profile = session.get(ArtistProfile, pid)
    if profile is None:
        return ""
    raw = (getattr(profile, "lore_json", "") or "").strip()
    if not raw or raw == "{}":
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        if isinstance(parsed, str):
            return parsed
    except Exception:
        pass
    return raw


def resolve_artist_grounding(session: Session, profile_id: Optional[str], agent_name: str) -> Tuple[Optional[ModelProfile], str]:
    """One round-trip helper: (chain_head, lore_text) for a profile-scoped run."""
    return resolve_chain_head(session, profile_id, agent_name), load_artist_lore(session, profile_id)
