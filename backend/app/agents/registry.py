"""Agent registry — name → definition.

The seam both the API surface and the future album orchestrator read from;
adding an agent means adding ONE entry here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Type

from pydantic import BaseModel

from app.agents.experiencer.agent import EXPERIENCER_AGENT
from app.agents.experiencer.schemas import AlbumBrief
from app.agents.world_builder.agent import WORLD_BUILDER_AGENT
from app.agents.world_builder.schemas import WorldBuilderBrief
from app.agents.stylist.agent import STYLIST_AGENT
from app.agents.stylist.schemas import StylistBrief
from app.agents.critic.agent import CRITIC_AGENT
from app.agents.critic.schemas import CriticBrief


@dataclass
class AgentEntry:
    agent: object                 # ExperiencerAgent today; AgentDefinition protocol later
    input_schema: Type[BaseModel]
    description: str


AGENTS: Dict[str, AgentEntry] = {
    EXPERIENCER_AGENT.name: AgentEntry(
        agent=EXPERIENCER_AGENT,
        input_schema=AlbumBrief,
        description=EXPERIENCER_AGENT.description,
    ),
    WORLD_BUILDER_AGENT.name: AgentEntry(
        agent=WORLD_BUILDER_AGENT,
        input_schema=WorldBuilderBrief,
        description=WORLD_BUILDER_AGENT.description,
    ),
    STYLIST_AGENT.name: AgentEntry(
        agent=STYLIST_AGENT,
        input_schema=StylistBrief,
        description=STYLIST_AGENT.description,
    ),
    CRITIC_AGENT.name: AgentEntry(
        agent=CRITIC_AGENT,
        input_schema=CriticBrief,
        description=CRITIC_AGENT.description,
    ),
}


def get_agent(name: str) -> AgentEntry:
    return AGENTS[name]


def list_agents() -> List[Dict]:
    out = []
    for name, entry in AGENTS.items():
        out.append({
            "name": name,
            "display_name": getattr(entry.agent, "display_name", name),
            "description": entry.description,
            "input_schema": entry.input_schema.model_json_schema(),
        })
    return out
