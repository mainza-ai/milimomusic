from app.services.lyrics.performance_tokens import (
    parse_song_sections_with_casting,
    clean_lyrics_for_notation,
    format_prompt_for_generator,
)


def test_performance_tokens_and_casting():
    lyrics = """
[Verse 1 | Lead: Maya]
[breath] Standing in the rain
Waiting for the morning light [whisper] so quietly

[Chorus | Duet: Maya + Marcus]
[belt] We will rise above the storm
[pause 300ms] Every shadow turns to dawn
    """

    sections = parse_song_sections_with_casting(lyrics)
    assert len(sections) == 2

    sec1 = sections[0]
    assert sec1.section_name == "Verse 1"
    assert sec1.vocal_role == "Lead"
    assert sec1.singer_name == "Maya"
    assert "breath" in sec1.tokens
    assert "whisper" in sec1.tokens

    sec2 = sections[1]
    assert sec2.section_name == "Chorus"
    assert sec2.vocal_role == "Duet"
    assert "Marcus" in (sec2.singer_name or "")
    assert "belt" in sec2.tokens


def test_clean_lyrics_for_notation():
    lyrics = "[Verse 1 | Lead: Maya]\n[breath] Hello world [whisper] so softly"
    cleaned = clean_lyrics_for_notation(lyrics)
    assert "[breath]" not in cleaned
    assert "[whisper]" not in cleaned
    assert "[Verse 1" not in cleaned
    assert "Hello world so softly" in cleaned
