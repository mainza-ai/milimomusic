"""The World-Builder agent implementation — persona + contract, no provider code.

Execution goes through ResiliencePolicy exactly like the other agents (provider
failover, typed errors, usage capture), so per-artist model overrides and the
global failover chain apply here too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import PolicyOutcome, ResiliencePolicy
from app.agents.runtime.usage import RunUsage
from app.agents.world_builder.persona import WORLD_BUILDER_SYSTEM
from app.agents.world_builder.schemas import WorldBuilderBrief, WorldLore


@dataclass
class WorldBuilderResult:
    lore: WorldLore
    outcome: PolicyOutcome
    usage: RunUsage

    # ── Registry-generic surface (route layer persists via these) ──────────
    @property
    def output(self) -> WorldLore:
        return self.lore

    @property
    def tokens_in(self) -> int:
        return self.usage.total_tokens_in

    @property
    def tokens_out(self) -> int:
        return self.usage.total_tokens_out

    @property
    def latency_ms(self) -> int:
        return self.usage.total_latency_ms

    @property
    def attempts(self):
        return self.outcome.attempts


class WorldBuilderAgent:
    name = "world_builder"
    display_name = "The World Builder"
    description = (
        "Imagines and maintains the artist's canonical world document — origin, "
        "era, appearance, musical DNA, and binding lore facts every other agent "
        "must stay consistent with."
    )

    def build_messages(self, brief: WorldBuilderBrief, ctx: Optional[RunContext] = None) -> List[dict]:
        identity = (
            f"ARTIST IDENTITY\n"
            f"- Name: {brief.artist_name}\n"
            f"- Bio: {brief.artist_bio or '(none provided)'}\n"
            f"- Style tags: {brief.tags or '(none)'}\n"
        )
        direction = f"\nUSER STEERING (honor explicitly): {brief.extra_direction}\n" if brief.extra_direction else ""
        existing = ""
        if ctx is not None and getattr(ctx, "artist_lore", None):
            existing = (
                "\nEXISTING LORE (revise and extend it — keep everything still true)\n"
                f"{ctx.artist_lore}\n"
            )
        user = (
            f"{identity}{existing}{direction}\n"
            "Imagine this artist's world and return the JSON lore document. "
            "Make it specific enough that a stranger could recognize their world."
        )
        messages = [{"role": "system", "content": WORLD_BUILDER_SYSTEM}]
        for m in (ctx.history if ctx else []):
            content = m.get("content", "")
            if content:
                messages.append({"role": m.get("role", "user"), "content": content})
        messages.append({"role": "user", "content": user})
        return messages

    async def run(
        self,
        brief: WorldBuilderBrief,
        ctx: RunContext,
        policy: ResiliencePolicy | None = None,
    ) -> WorldBuilderResult:
        policy = policy or ResiliencePolicy()
        messages = self.build_messages(brief, ctx)
        outcome = await policy.run_structured(
            messages,
            WorldLore,
            temperature=0.9,  # world-building wants imagination headroom
        )
        lore = outcome.structured
        assert isinstance(lore, WorldLore)
        usage = RunUsage()
        usage.attempts.extend(outcome.attempts)
        return WorldBuilderResult(lore=lore, outcome=outcome, usage=usage)


WORLD_BUILDER_AGENT = WorldBuilderAgent()
