"""World-Builder agent — keeps the artist's canonical world document fresh.

Given an artist identity (name/bio/tags), it imagines and structures the world
that artist lives in: origin, era, appearance, musical DNA, and the facts every
other crew member must stay consistent with. Output persists to the artist's
``lore_json`` and grounds both the Experiencer (vision) and the Songwriter
(lyrics) through their prompt blocks.
"""
from app.agents.world_builder.agent import WORLD_BUILDER_AGENT

__all__ = ["WORLD_BUILDER_AGENT"]
