"""The Experiencer agent — imagination engine of the artist's crew.

Given an album brief it imagines the lived experience behind the concept and
returns a structured ExperiencerVision: journey, arc, and per-song seeds that
downstream agents (songwriter, style curator) turn into real tracks.

Execution goes through ResiliencePolicy (provider failover + typed errors +
usage capture). No provider code lives here — this file is persona + contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.agents.experiencer.persona import EXPERIENCER_SYSTEM
from app.agents.experiencer.schemas import AlbumBrief, ExperiencerVision
from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import PolicyOutcome, ResiliencePolicy
from app.agents.runtime.usage import RunUsage


@dataclass
class ExperiencerResult:
    vision: ExperiencerVision
    outcome: PolicyOutcome
    usage: RunUsage
    shortfall_notes: str = ""   # set when fewer seeds arrived than requested

    # ── Registry-generic surface (route layer persists via these) ──────────
    @property
    def output(self) -> ExperiencerVision:
        return self.vision

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


class ExperiencerAgent:
    name = "experiencer"
    display_name = "The Experiencer"
    description = (
        "Imagines the lived human journey inside an album concept and returns "
        "an emotional arc plus one experience-grounded seed per song."
    )

    def build_messages(self, brief: AlbumBrief, ctx: Optional[RunContext] = None) -> List[dict]:
        artist_block = ""
        if brief.artist_name or brief.artist_bio:
            artist_block = f"\nARTIST IDENTITY\n- Name: {brief.artist_name or '(unnamed)'}\n- Bio/lore: {brief.artist_bio or '(none provided)'}\n"
        lore_block = ""
        if ctx is not None and getattr(ctx, "artist_lore", None):
            lore_block = (
                "\nARTIST WORLD LORE (canonical history — stay consistent with it)\n"
                f"{ctx.artist_lore}\n"
            )
        tags_block = f"\nSTYLE HINTS (serve the moments; never let them flatten them): {brief.tags}\n" if brief.tags else ""
        direction_block = f"\nUSER STEERING (honor explicitly): {brief.extra_direction}\n" if brief.extra_direction else ""

        user = (
            f"ALBUM BRIEF\n"
            f"- Title: {brief.album_title}\n"
            f"- Concept: {brief.album_concept}\n"
            f"{artist_block}{lore_block}{tags_block}{direction_block}"
            f"- Target track count: {brief.track_target}\n\n"
            "Walk through this concept as a lived journey and return the JSON vision. "
            f"Produce exactly {brief.track_target} song_seeds placed along the emotional_arc."
        )
        history = ctx.history if ctx else []
        messages = [{"role": "system", "content": EXPERIENCER_SYSTEM}]
        # Conversation memory window (G4): oldest → newest, then the live ask.
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                messages.append({"role": "assistant" if role == "producer" else role, "content": content})
        messages.append({"role": "user", "content": user})
        return messages

    async def run(
        self,
        brief: AlbumBrief,
        ctx: RunContext,
        policy: ResiliencePolicy | None = None,
    ) -> ExperiencerResult:
        policy = policy or ResiliencePolicy()
        messages = self.build_messages(brief, ctx)
        outcome = await policy.run_structured(
            messages,
            ExperiencerVision,
            temperature=0.9,  # imagination wants headroom above default sampling
        )
        vision = outcome.structured
        assert isinstance(vision, ExperiencerVision)

        shortfall = ""
        if len(vision.song_seeds) < brief.track_target:
            shortfall = (
                f"Requested {brief.track_target} song seeds but the model returned "
                f"{len(vision.song_seeds)}. The arc is intact; additional seeds can be "
                f"imagined in a follow-up run scoped to specific arc positions."
            )

        usage = RunUsage()
        usage.attempts.extend(outcome.attempts)
        return ExperiencerResult(vision=vision, outcome=outcome, usage=usage, shortfall_notes=shortfall)


EXPERIENCER_AGENT = ExperiencerAgent()
