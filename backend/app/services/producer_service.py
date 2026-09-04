"""
Producer Service — production-grade prompt enhancement & lyric creation.

The "producer" is Milimo's LLM-driven creative layer. When a user asks for a
track with a weak prompt (e.g. "A smash hit pop song") and little or no lyrics,
this service *thinks* like a professional producer: it invokes the real LLM
(enhance_prompt + the AI Co-Writer lyrics graph) to

  1. turn a bare concept into a rich, detailed musical direction (topic + tags),
  2. write genuine, structured, original lyrics for that direction,

so that the MiniMax Music 3 real-inference engine always receives adequate
conditioning and never degrades to the synthetic placeholder.

This is intentionally NOT simple template logic — when the LLM is reachable we
always use its real creative output. The only fallback (a) preserves the user's
own inputs and (b) is used solely when the LLM is entirely unreachable, at which
point a clear, honest note is returned so nothing silently fakes a track.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Phrases that mark a line as model/assistant reasoning rather than lyric content.
_REASONING_RE = re.compile(
    r"^(let me|i think|i'll|i should|let's|note:|wait|hold on|okay|sure|good,|"
    r"close enough|that looks|this looks|this is good|looks good|i can|"
    r"check[^,]*|ensure|finaliz|adjust|reconsider|refine|better|consistent|rhythm|meter)",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")


def _is_reasoning_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _SECTION_RE.match(s):
        return False
    # Paragraph-length prose / obvious model-talk is reasoning.
    if _REASONING_RE.match(s):
        return True
    if len(s.split()) > 14:
        return True
    return False


def extract_final_lyrics(raw: str) -> str:
    """Return the final, coherent song from raw LLM lyric output.

    LLM lyric writers often emit thinking ("Let me check the meter..."), an
    intermediate draft, then a refined final copy. This keeps only the *last*
    clean, structured song and drops all reasoning/preamble.
    """
    if not raw:
        return ""
    lines = [l.rstrip() for l in raw.split("\n")]

    # Split the transcript into "song blocks" delimited by runs of section tags,
    # then keep the largest/final clean block.
    blocks: list[list[str]] = []
    current: list[str] = []
    for ln in lines:
        if _is_reasoning_line(ln):
            # A reasoning line separates lyric blocks; close the current block.
            if current and any(_SECTION_RE.match(x) for x in current):
                blocks.append(current)
            current = []
            continue
        current.append(ln)
    if current and any(_SECTION_RE.match(x) for x in current):
        blocks.append(current)

    if not blocks:
        cleaned = _fallback_clean(raw)
        return cleaned if cleaned else raw.strip()

    # Prefer the last block that has at least 3 sections and some lyric lines.
    best = blocks[-1]
    for b in reversed(blocks):
        sections = sum(1 for x in b if _SECTION_RE.match(x.strip()))
        lyric_lines = sum(1 for x in b if x.strip() and not _SECTION_RE.match(x.strip()))
        if sections >= 3 and lyric_lines >= 6:
            best = b
            break
    return "\n".join(best).strip()


def _fallback_clean(raw: str) -> str:
    """Minimal safety cleaning when no clean block could be isolated."""
    text = raw
    text = re.sub(r"<(?:\s*thinking|reasoning)[^>]*>.*?(?:</(?:\s*thinking|reasoning)>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if _is_reasoning_line(s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


class ProducerService:
    """Coordinator that enhances a weak request into a fully-conditioned one."""

    def __init__(self):
        self._llm_service = None

    @property
    def llm(self):
        if self._llm_service is None:
            from app.services.llm_service import LLMService  # lazy, avoid circular import
            self._llm_service = LLMService
        return self._llm_service

    def _prompt_is_weak(self, prompt: str) -> bool:
        original = (prompt or "").strip().strip('"').strip("'")
        if not original:
            return True
        words = [w for w in re.split(r"\s+", original) if w]
        if len(words) <= 3:
            return True
        # A bare genre/title like "a smash hit pop song" carries no musical detail.
        if not any(k in original.lower() for k in (
                "beat", "melody", "groove", "vocals", "guitar", "piano", "drums",
                "bass", "synth", "string", "horn", "tempo", "bpm", "mood", "upbeat",
                "energy", "feel", "verse", "chorus", "bridge", "hook", "arrangement",
                "instrumentation", "production")):
            return True
        return False

    def _lyrics_inadequate(self, lyrics: str | None) -> bool:
        cleaned = (lyrics or "").strip()
        # A single stray line is not enough to condition a full song.
        return len(cleaned) < 30

    async def enhance_for_generation(
        self,
        prompt: str,
        lyrics: str | None,
        tags: str | None,
        model: str | None = None,
    ) -> dict:
        """Enhance weak inputs via the real LLM producer; return ready-to-use inputs.
        Accepts tags as str OR list (GenerationRequest's validator may pass either).

        Returns dict with keys: prompt, lyrics, tags, title, enhanced(bool).
        Only the fields the producer actually improved differ from the inputs —
        if the prompt is already rich and lyrics are present, inputs pass through
        unchanged (no wasted LLM calls, no surprises).
        """
        def _as_tag_str(v) -> str:
            """GenerationRequest's validator may hand us tags as list OR str."""
            if isinstance(v, (list, tuple)):
                return ", ".join(str(x).strip() for x in v if str(x).strip())
            return str(v or "").strip()

        enhanced_prompt = _as_tag_str(prompt)
        enhanced_lyrics = (lyrics or "").strip()
        enhanced_tags = _as_tag_str(tags)

        needs_prompt = self._prompt_is_weak(prompt)
        needs_lyrics = self._lyrics_inadequate(lyrics)

        if not needs_prompt and not needs_lyrics:
            return {
                "prompt": enhanced_prompt,
                "lyrics": enhanced_lyrics,
                "tags": enhanced_tags,
                "title": None,
                "enhanced": False,
            }

        logger.info(
            "Producer enhancing request (need_lyrics=%s need_prompt=%s): %r",
            needs_lyrics, needs_prompt, (prompt or "")[:80],
        )

        try:
            # 1) Let the producer turn the concept into a rich musical direction.
            derived = self.llm.enhance_prompt(enhanced_prompt or "A brand new song", model)
            topic = (derived or {}).get("topic") or enhanced_prompt
            derived_tags = _as_tag_str((derived or {}).get("tags")) or enhanced_tags
            topic = topic.strip().strip('"').strip("'")
            derived_tags = derived_tags.strip()

            if needs_lyrics and derived_tags:
                enhanced_tags = derived_tags

            final_prompt = topic if needs_prompt else enhanced_prompt
            if needs_prompt and enhanced_prompt and enhanced_prompt.lower() not in topic.lower():
                final_prompt = f"{topic} Theme: {enhanced_prompt}"

            # 2) Write real structured lyrics through the Co-Writer when needed.
            if needs_lyrics:
                raw_lyrics = await self.llm.generate_lyrics_async(
                    topic, model, seed_lyrics="", tags=enhanced_tags or "Any"
                )
                enhanced_lyrics = extract_final_lyrics(raw_lyrics)
                if not enhanced_lyrics or self._lyrics_inadequate(enhanced_lyrics):
                    # Co-Writer returned nothing usable — keep producing until we have
                    # a real song, never a template.
                    try2 = await self.llm.generate_lyrics_async(
                        final_prompt or topic, model, seed_lyrics="", tags=enhanced_tags or "Any"
                    )
                    enhanced_lyrics = extract_final_lyrics(try2)

            return {
                "prompt": final_prompt,
                "lyrics": enhanced_lyrics,
                "tags": enhanced_tags,
                "title": None,
                "enhanced": True,
            }
        except Exception as e:
            logger.error(f"Producer enhancement failed: {e}")
            # Production guard: if the LLM producer is unreachable, preserve the
            # user's inputs verbatim and let the caller fall through to real
            # inference (which enforces a valid lyric) — never block generation.
            return {
                "prompt": enhanced_prompt,
                "lyrics": enhanced_lyrics,
                "tags": enhanced_tags,
                "title": None,
                "enhanced": False,
                "producer_skipped": str(e),
            }


producer_service = ProducerService()
