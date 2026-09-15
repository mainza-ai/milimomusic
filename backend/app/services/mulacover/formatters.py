"""
Formatting and normalization utilities for MuLaCover inputs.
Translates Milimo Music freeform prompts and AI Co-Writer lyrics into
MuLaCover's strict symbolic and token conditioning contracts.
"""

import re
from typing import Optional, Dict, Any, List


SUPPORTED_SECTIONS = {
    "intro": "[Intro]",
    "verse": "[Verse]",
    "chorus": "[Chorus]",
    "interlude": "[Interlude]",
    "bridge": "[Bridge]",
    "outro": "[Outro]",
}


def format_style_tags(
    prompt: Optional[str] = None,
    tags: Optional[str] = None,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    instrument: Optional[str] = None,
    topic: Optional[str] = None,
) -> str:
    """Format style conditioning into MuLaCover's exact key-value syntax:
    `topic:[...]; genre:[...]; instrument:[...]; mood:[...]`.

    Handles:
    1. Already formatted strings: preserves or merges missing fields.
    2. Freeform comma-separated tag strings (e.g., 'synthwave, energetic, drums, neon city').
    3. Dedicated keyword arguments.
    """
    raw_combined = f"{prompt or ''} {tags or ''}".strip()

    # Check if raw text already has key-value pairs
    existing_pairs: Dict[str, str] = {}
    pattern = re.compile(r"(topic|genre|instrument|mood):\s*\[([^\]]*)\]", re.IGNORECASE)
    for match in pattern.finditer(raw_combined):
        key = match.group(1).lower()
        val = match.group(2).strip()
        if val:
            existing_pairs[key] = val

    # Override or fill with explicit arguments if provided
    final_topic = topic or existing_pairs.get("topic")
    final_genre = genre or existing_pairs.get("genre")
    final_instrument = instrument or existing_pairs.get("instrument")
    final_mood = mood or existing_pairs.get("mood")

    # If any fields are still missing, perform heuristic categorization from freeform text
    if not (final_topic and final_genre and final_instrument and final_mood) and raw_combined:
        clean_text = pattern.sub("", raw_combined)
        # Strip XML/HTML tags
        clean_text = re.sub(r"<[^>]+>", " ", clean_text)
        tokens = [t.strip() for t in re.split(r"[,;]+", clean_text) if t.strip()]

        known_genres = {
            "pop", "rock", "metal", "hip hop", "rap", "r&b", "soul", "funk", "jazz",
            "blues", "country", "folk", "electronic", "edm", "house", "techno",
            "ambient", "synthwave", "lofi", "lo-fi", "classical", "cinematic", "disco",
            "reggae", "latin", "afrobeats", "trap", "punk", "indie", "alternative"
        }
        known_moods = {
            "happy", "sad", "melancholic", "energetic", "dark", "uplifting", "chill",
            "relaxing", "intense", "romantic", "nostalgic", "hopeful", "dreamy",
            "aggressive", "peaceful", "euphoric", "somber", "brooding", "epic"
        }
        known_instruments = {
            "acoustic guitar", "electric guitar", "guitar", "piano", "synthesizer",
            "synth", "strings", "violin", "cello", "drums", "bass", "brass",
            "horns", "saxophone", "flute", "percussion", "organ", "808", "vocal"
        }

        found_genres: List[str] = []
        found_moods: List[str] = []
        found_instruments: List[str] = []
        remaining_topics: List[str] = []

        for token in tokens:
            low = token.lower()
            if any(g in low for g in known_genres):
                found_genres.append(token)
            elif any(m in low for m in known_moods):
                found_moods.append(token)
            elif any(i in low for i in known_instruments):
                found_instruments.append(token)
            else:
                remaining_topics.append(token)

        if not final_genre and found_genres:
            final_genre = ", ".join(found_genres[:2])
        if not final_mood and found_moods:
            final_mood = ", ".join(found_moods[:2])
        if not final_instrument and found_instruments:
            final_instrument = ", ".join(found_instruments[:3])
        if not final_topic and remaining_topics:
            final_topic = ", ".join(remaining_topics[:3])

    # Fallback sensible defaults for missing fields to satisfy tokenizer requirements
    final_topic = final_topic or "music"
    final_genre = final_genre or "pop"
    final_instrument = final_instrument or "piano, synthesizer, drums"
    final_mood = final_mood or "hopeful"

    return f"topic:[{final_topic}]; genre:[{final_genre}]; instrument:[{final_instrument}]; mood:[{final_mood}]"


def sanitize_lyrics_for_mulacover(lyrics: Optional[str]) -> str:
    """Format multiline lyrics strictly according to MuLaCover's rules:
    1. Section markers on their own lines (e.g. `[Intro]`, `[Verse]`, `[Chorus]`).
    2. One lyric line per line.
    3. Exactly one blank line between sections.
    4. Strips timestamps (LRC formats), Co-Writer reasoning tags, and markdown.
    """
    if not lyrics or not lyrics.strip():
        return "[Intro]\n\n[Verse]\nInstrumental melody\n\n[Outro]"

    # Strip Co-Writer thinking or reasoning tags
    text = re.sub(r"<think>.*?</think>", "", lyrics, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```[a-zA-Z]*\n?|```", "", text)

    # Strip timestamp annotations like [00:12.34] or [01:23]
    text = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", "", text)

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []

    # Map sections like [Verse 1], [Chorus 2], (Bridge) to canonical headers
    section_regex = re.compile(
        r"^[\[\(]\s*(intro|verse|chorus|interlude|bridge|outro)[\s\d_:-]*[\]\)]",
        re.IGNORECASE
    )

    has_any_section = False
    for line in lines:
        if not line:
            continue
        match = section_regex.match(line)
        if match:
            canonical = SUPPORTED_SECTIONS[match.group(1).lower()]
            cleaned_lines.append(canonical)
            has_any_section = True
        else:
            cleaned_lines.append(line)

    if not has_any_section and cleaned_lines:
        # Wrap unstructured lines in default section structure
        half = len(cleaned_lines) // 2
        cleaned_lines = (
            ["[Verse]"] + cleaned_lines[:half] +
            ["[Chorus]"] + cleaned_lines[half:] +
            ["[Outro]"]
        )

    # Reconstruct with strict spacing: sections separated by blank line, no blank lines within a section
    formatted_sections: List[List[str]] = []
    current_section: List[str] = []

    for line in cleaned_lines:
        if line.startswith("[") and line.endswith("]"):
            if current_section:
                formatted_sections.append(current_section)
                current_section = []
            current_section.append(line)
        else:
            if not current_section:
                current_section.append("[Verse]")
            current_section.append(line)

    if current_section:
        formatted_sections.append(current_section)

    return "\n\n".join("\n".join(sec) for sec in formatted_sections)


def parse_mulacover_tags(tag_str: str) -> Dict[str, str]:
    """Parse MuLaCover key-value string into a dictionary."""
    result: Dict[str, str] = {}
    pattern = re.compile(r"(topic|genre|instrument|mood):\s*\[([^\]]*)\]", re.IGNORECASE)
    for match in pattern.finditer(tag_str or ""):
        result[match.group(1).lower()] = match.group(2).strip()
    return result
