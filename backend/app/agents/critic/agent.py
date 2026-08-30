"""The Critic agent — persona + contract, no provider code.

Runs through ResiliencePolicy like every other agent (failover, typed errors,
usage capture), so per-artist model overrides apply here too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.agents.critic.persona import CRITIC_SYSTEM
from app.agents.critic.schemas import CriticBrief, Critique
from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import PolicyOutcome, ResiliencePolicy
from app.agents.runtime.usage import RunUsage


@dataclass
class CriticResult:
    review: Critique
    outcome: PolicyOutcome
    usage: RunUsage

    # ── Registry-generic surface ────────────────────────────────────────────
    @property
    def output(self) -> Critique:
        return self.review

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


class CriticAgent:
    name = "critic"
    display_name = "The Critic"
    description = (
        "Pre-generation quality gate: reviews the song draft against its seed "
        "and the artist's canon — pass / revise / concern, with specific notes."
    )

    def build_messages(self, brief: CriticBrief, ctx: Optional[RunContext] = None) -> List[dict]:
        seed = brief.seed or {}
        draft = brief.draft or {}
        revision_block = f"\nREVIEW CONTEXT: {brief.revision_context}\n" if brief.revision_context else ""
        user = (
            f"THE SEED (this song must serve it)\n"
            f"- Working title: {seed.get('working_title', '')}\n"
            f"- Mood: {seed.get('mood', '')} · Energy: {seed.get('energy', 0.5)}\n"
            f"- Story: {seed.get('story_seed', '')}\n"
            f"- Arc position: {seed.get('placement_hint', 'mid-album')}\n"
            f"{revision_block}\n"
            f"THE DRAFT\n"
            f"- Title: {draft.get('title', '')}\n"
            f"- Lyric sheet:\n{str(draft.get('lyrics', ''))[:2400]}\n"
            f"- Songwriter's own rationale: {draft.get('lyrical_rationale', '')}\n\n"
            "Review the draft. Return JSON matching the schema exactly."
        )
        messages = [{"role": "system", "content": CRITIC_SYSTEM}]
        for m in (ctx.history if ctx else []):
            content = m.get("content", "")
            if content:
                messages.append({"role": m.get("role", "user"), "content": content})
        messages.append({"role": "user", "content": user})
        return messages

    async def run(
        self,
        brief: CriticBrief,
        ctx: RunContext,
        policy: ResiliencePolicy | None = None,
    ) -> CriticResult:
        policy = policy or ResiliencePolicy()
        messages = self.build_messages(brief, ctx)
        outcome = await policy.run_structured(messages, Critique, temperature=0.3)
        review = outcome.structured
        assert isinstance(review, Critique)
        usage = RunUsage()
        usage.attempts.extend(outcome.attempts)
        return CriticResult(review=review, outcome=outcome, usage=usage)


CRITIC_AGENT = CriticAgent()
