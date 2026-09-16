"""Expressive Vocal Performance Tokens & Multi-Singer Casting Engine.

Provides parsing, normalization, and conversion of expressive speech/singing tokens
([breath], [whisper], [pause Nms], [falsetto], [belt]) and section-level multi-singer
casting directives ([Verse 1 | Lead: Artist], [Chorus | Duet]) for AI Co-Writer,
lyric conditioning, MuScriptor notation, and DAW track routing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import logging

logger = logging.getLogger("milimo.lyrics.performance_tokens")

# Performance Token Definitions
EXPRESSIVE_TOKENS = {
    "[breath]": {"type": "acoustic_inhalation", "duration_ms": 200, "description": "Audible inhalation before singing"},
    "[whisper]": {"type": "vocal_aspiration", "formant_shift": 1.15, "description": "Intimate unvoiced vocal delivery"},
    "[pause]": {"type": "silence", "default_ms": 300, "description": "Rhythmic rest / hesitation"},
    "[laughter]": {"type": "vocalization", "description": "Subtle vocal chuckle / laugh"},
    "[sigh]": {"type": "vocalization", "description": "Exhalation sigh"},
    "[gasp]": {"type": "vocalization", "description": "Sudden intake of breath"},
    "[falsetto]": {"type": "vocal_register", "register": "head", "description": "High head-voice register"},
    "[belt]": {"type": "vocal_intensity", "intensity": "forte", "description": "Full-power chest voice resonance"},
    "[vibrato]": {"type": "modulation", "rate_hz": 5.5, "description": "Expressive pitch vibrato modulation"},
}


@dataclass
class SectionCasting:
    section_name: str
    vocal_role: str
    singer_name: Optional[str] = None
    lines: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


def parse_song_sections_with_casting(lyrics_text: str) -> List[SectionCasting]:
    """Parse lyrics into sections with extracted vocal casting and expressive tokens.

    Supports syntax:
      [Verse 1 | Lead: Marcus]
      [Chorus | Duet: Marcus + Aria]
      [breath] Into the open sky [whisper] so quiet...
    """
    if not lyrics_text:
        return []

    section_header_pattern = re.compile(
        r"^\[\s*(Intro|Verse\s*\d*|Chorus\s*\d*|Bridge|Hook|Outro|Drop|Pre-Chorus)(?:\s*\|\s*([^\]]+))?\s*\]",
        re.IGNORECASE,
    )
    token_pattern = re.compile(r"\[(breath|whisper|pause(?:\s*\d+ms)?|laughter|sigh|gasp|falsetto|belt|vibrato)\]", re.IGNORECASE)

    sections: List[SectionCasting] = []
    current_section = SectionCasting(section_name="Intro", vocal_role="Lead", singer_name="Lead Vocal")

    for raw_line in lyrics_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        header_match = section_header_pattern.match(line)
        if header_match:
            # Finalize previous section if it has lines
            if current_section.lines:
                sections.append(current_section)

            sec_name = header_match.group(1).strip().title()
            meta = header_match.group(2) or "Lead"
            role = "Lead"
            singer = meta.strip()

            if ":" in meta:
                parts = meta.split(":", 1)
                role = parts[0].strip()
                singer = parts[1].strip()

            current_section = SectionCasting(section_name=sec_name, vocal_role=role, singer_name=singer)
        else:
            # Extract tokens in this line
            found_tokens = [m.lower() for m in token_pattern.findall(line)]
            if found_tokens:
                current_section.tokens.extend(found_tokens)
            current_section.lines.append(line)

    if current_section.lines:
        sections.append(current_section)

    return sections


def clean_lyrics_for_notation(lyrics_text: str) -> str:
    """Strip bracketed performance tokens and casting headers for clean sheet music lyrics."""
    if not lyrics_text:
        return ""
    # Strip performance tokens like [breath], [whisper], [pause 300ms]
    cleaned = re.sub(r"\[(breath|whisper|pause(?:\s*\d+ms)?|laughter|sigh|gasp|falsetto|belt|vibrato)\]", "", lyrics_text, flags=re.IGNORECASE)
    # Strip section casting headers like [Verse 1 | Lead: Maya]
    cleaned = re.sub(r"\[\s*(?:Intro|Verse|Chorus|Bridge|Hook|Outro|Drop|Pre-Chorus)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    # Collapse multiple spaces and blank lines
    lines = [re.sub(r"\s+", " ", l).strip() for l in cleaned.split("\n")]
    return "\n".join(l for l in lines if l)


def format_prompt_for_generator(lyrics_text: str, provider: str = "minimax") -> str:
    """Format performance tokens into generator-compatible conditioning."""
    if not lyrics_text:
        return ""
    if provider.lower() == "minimax":
        # MiniMax Music 3 interprets [whisper], [breath], [pause] effectively in lyric streams
        return lyrics_text
    elif "mulacover" in provider.lower():
        # Canonical bracket formatting for MuLaCover
        return lyrics_text
    return lyrics_text
