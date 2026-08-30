"""Critic agent — the crew's pre-generation quality gate.

Reviews the songwriter's draft against the seed BEFORE the expensive generation
step. A `revise` verdict triggers exactly ONE songwriter revision round (fed by
the critic's notes), then a single re-review — never a loop. Verdicts persist in
the album run cursor and surface on the tracklist.
"""
from app.agents.critic.agent import CRITIC_AGENT

__all__ = ["CRITIC_AGENT"]
