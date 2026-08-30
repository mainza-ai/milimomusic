"""The Stylist agent — persona + contract, no provider code.

Runs through ResiliencePolicy like every other agent (failover, typed errors,
usage capture), so per-artist model overrides apply here too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import PolicyOutcome, ResiliencePolicy
from app.agents.runtime.usage import RunUsage
from app.agents.stylist.persona import STYLIST_SYSTEM
from app.agents.stylist.schemas import StylistBrief, StylingChoice


@dataclass
class StylistResult:
    choice: StylingChoice
    outcome: PolicyOutcome
    usage: RunUsage

    # ── Registry-generic surface ────────────────────────────────────────────
    @property
    def output(self) -> StylingChoice:
        return self.choice

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


class StylistAgent:
    name = "stylist"
    display_name = "The Stylist"
    description = (
        "Curates the final 2-6 style tags for one song draft against its seed "
        "and the artist's world — genre first, every tag audible."
    )

    def build_messages(self, brief: StylistBrief, ctx: Optional[RunContext] = None) -> List[dict]:
        seed = brief.seed or {}
        draft = brief.draft or {}
        lore_block = f"\nARTIST WORLD LORE (stay consistent)\n{brief.artist_lore}\n" if brief.artist_lore else ""
        user = (
            f"SONG CONTEXT\n"
            f"- Artist: {brief.artist_name or '(unnamed)'}\n"
            f"- Album: {brief.album_title or '(untitled)'}\n"
            f"- Arc position: {seed.get('placement_hint', 'mid-album')}\n"
            f"- Mood: {seed.get('mood', '')} · Energy: {seed.get('energy', 0.5)}\n"
            f"- Seed story: {seed.get('story_seed', '')}\n"
            f"- Suggested tags: {', '.join(seed.get('suggested_style_tags', [])) or '(none)'}\n"
            f"{lore_block}\n"
            f"DRAFT\n"
            f"- Title: {draft.get('title', '')}\n"
            f"- Lyric excerpt (first 600 chars): {str(draft.get('lyrics', ''))[:600]}\n"
            f"- Songwriter tags: {', '.join(draft.get('style_tags', [])) or '(none)'}\n\n"
            "Choose the final tags. Return JSON matching the schema exactly."
        )
        messages = [{"role": "system", "content": STYLIST_SYSTEM}]
        for m in (ctx.history if ctx else []):
            content = m.get("content", "")
            if content:
                messages.append({"role": m.get("role", "user"), "content": content})
        messages.append({"role": "user", "content": user})
        return messages

    async def run(
        self,
        brief: StylistBrief,
        ctx: RunContext,
        policy: ResiliencePolicy | None = None,
    ) -> StylistResult:
        policy = policy or ResiliencePolicy()
        messages = self.build_messages(brief, ctx)
        outcome = await policy.run_structured(messages, StylingChoice, temperature=0.4)
        choice = outcome.structured
        assert isinstance(choice, StylingChoice)
        usage = RunUsage()
        usage.attempts.extend(outcome.attempts)
        return StylistResult(choice=choice, outcome=outcome, usage=usage)


STYLIST_AGENT = StylistAgent()
