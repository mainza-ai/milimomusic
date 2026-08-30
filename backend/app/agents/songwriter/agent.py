"""The Songwriter agent: one seed → one finished song draft."""
from __future__ import annotations

from typing import List, Optional

from app.agents.runtime.context import RunContext
from app.agents.runtime.policy import ResiliencePolicy
from app.agents.songwriter.persona import SONGWRITER_SYSTEM
from app.agents.songwriter.schemas import SongDraft


class SongSeedInput:
    """Duck-typed view of an ExperiencerVision.SongSeed (avoids import cycle)."""


class SongwriterAgent:
    name = "songwriter"
    display_name = "The Songwriter"
    description = (
        "Turns one experience-seed into a finished song draft: complete lyric "
        "sheet, final title, and genre-first style tags grounded in the album vision."
    )

    def build_messages(
        self,
        seed: dict,
        album_context: dict,
        ctx: Optional[RunContext] = None,
    ) -> List[dict]:
        placement = seed.get("placement_hint", "mid-album")
        duration_s = seed.get("target_duration_s") or 180
        lore_block = ""
        if album_context.get("artist_lore"):
            lore_block = (
                "\nARTIST WORLD LORE (canonical history — stay consistent with it)\n"
                f"{album_context['artist_lore']}\n"
            )
        user = (
            f"ALBUM CONTEXT\n"
            f"- Album: {album_context.get('album_title', '(untitled)')}\n"
            f"- Concept: {album_context.get('album_concept', '')}\n"
            f"- Artist: {album_context.get('artist_name', '(unnamed)')}\n"
            f"- Arc position: {placement}\n"
            f"{lore_block}\n"
            f"THE SEED (write THIS song)\n"
            f"- Working title: {seed.get('working_title', '')}\n"
            f"- Mood: {seed.get('mood', '')}\n"
            f"- Story: {seed.get('story_seed', '')}\n"
            f"- Energy: {seed.get('energy', 0.5)} (0=still, 1=explosive)\n"
            f"- Suggested tags: {', '.join(seed.get('suggested_style_tags', []))}\n"
            f"- Target duration: ~{duration_s}s — size the lyric sheet accordingly.\n\n"
            "Write the song. Return JSON matching the schema exactly."
        )
        messages = [{"role": "system", "content": SONGWRITER_SYSTEM}]
        for m in (ctx.history if ctx else []):
            content = m.get("content", "")
            if content:
                messages.append({"role": m.get("role", "user"), "content": content})
        messages.append({"role": "user", "content": user})
        return messages

    async def run(
        self,
        seed: dict,
        album_context: dict,
        ctx: RunContext,
        policy: ResiliencePolicy | None = None,
    ) -> SongDraft:
        policy = policy or ResiliencePolicy()
        messages = self.build_messages(seed, album_context, ctx)
        outcome = await policy.run_structured(
            messages,
            SongDraft,
            temperature=0.85,
        )
        # Usage flows to the ledger via the route's accounting of outcome.attempts;
        # the draft itself is what the bridge consumes.
        draft = outcome.structured
        assert isinstance(draft, SongDraft)
        return draft


SONGWRITER_AGENT = SongwriterAgent()
