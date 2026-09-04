"""Stylist agent — the crew's tag curator, run pre-generation.

When enabled (per-run crew flag), the Stylist reviews the songwriter's draft
against the seed and the artist's world, returning a tightened 2–6 tag list.
`order_tags_genre_first` still runs afterwards as the deterministic guard, so
MiniMax's positional destructuring is always safe regardless of LLM output.
"""
from app.agents.stylist.agent import STYLIST_AGENT

__all__ = ["STYLIST_AGENT"]
