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
