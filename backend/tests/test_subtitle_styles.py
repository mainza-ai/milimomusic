"""Tests for subtitle_styles.py — Advanced SubStation Alpha (.ass) generation and presets."""

import pytest
from app.services.video.subtitle_styles import (
    generate_karaoke_ass_script,
    resolve_preset,
    format_ass_timestamp,
    SUBTITLE_PRESETS,
)


def test_format_ass_timestamp():
    """Verify timestamp formatting into H:MM:SS.cs."""
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(12.34) == "0:00:12.34"
    assert format_ass_timestamp(75.5) == "0:01:15.50"
    assert format_ass_timestamp(3661.05) == "1:01:01.05"


def test_resolve_preset_defaults_and_aliases():
    """Verify preset resolution honors exact IDs and user-friendly aliases."""
    preset_neon = resolve_preset("neon")
    assert preset_neon.id == "neon"

    preset_alias = resolve_preset("karaoke")
    assert preset_alias.id == "neon"

    preset_spotify = resolve_preset("spotify")
    assert preset_spotify.id == "spotify"

    preset_vertical = resolve_preset("tiktok")
    assert preset_vertical.id == "social_vertical"
    assert preset_vertical.alignment == 5  # Middle Center

    # Fallback for unknown
    assert resolve_preset("unknown_style_xyz").id == "neon"


def test_word_level_karaoke_ass_generation():
    """Verify word-level \\kf tags are generated correctly from timed word sequences."""
    timed_lines = [
        {
            "start": 10.0,
            "end": 14.0,
            "text": "Never gonna give you up",
            "is_section": False,
            "words": [
                {"word": "Never", "start": 10.0, "end": 10.5},
                {"word": "gonna", "start": 10.55, "end": 11.2},
                {"word": "give", "start": 11.25, "end": 11.8},
                {"word": "you", "start": 11.85, "end": 12.4},
                {"word": "up", "start": 12.45, "end": 13.5},
            ],
        }
    ]

    ass = generate_karaoke_ass_script(
        timed_lines=timed_lines,
        width=1920,
        height=1080,
        subtitle_style="neon",
        aspect_ratio="16:9",
    )

    assert "[Script Info]" in ass
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass
    assert "[V4+ Styles]" in ass
    assert "Style: Default," in ass
    assert "[Events]" in ass

    # Verify that dialogue line contains word-level \kf tags instead of a single line tag
    assert "Dialogue: 0,0:00:10.00,0:00:14.00,Default,,0,0,0,," in ass
    assert r"{\kf50}Never" in ass
    assert r"{\kf65}gonna" in ass
    assert r"{\kf55}give" in ass
    assert r"{\kf55}you" in ass
    assert r"{\kf105}up" in ass


def test_missing_words_generates_syllable_breakdown():
    """Verify lines without explicit word timestamps generate proportional syllable karaoke tags."""
    timed_lines = [
        {
            "start": 5.0,
            "end": 9.0,
            "text": "Electric city lights shining",
            "is_section": False,
            "words": [],  # Empty words list
        }
    ]

    ass = generate_karaoke_ass_script(
        timed_lines=timed_lines,
        width=1280,
        height=720,
        subtitle_style="neon",
    )

    assert "Dialogue: 0,0:00:05.00,0:00:09.00,Default,,0,0,0,," in ass
    # Should still contain word-level tags
    assert r"\kf" in ass
    assert "Electric" in ass
    assert "shining" in ass


def test_vertical_safe_zone_alignment():
    """Verify 9:16 vertical aspect ratio applies middle-center alignment and safe margins."""
    timed_lines = [
        {"start": 1.0, "end": 3.0, "text": "Vertical video format", "is_section": False}
    ]

    ass_16_9 = generate_karaoke_ass_script(
        timed_lines=timed_lines,
        width=1280,
        height=720,
        subtitle_style="neon",
        aspect_ratio="16:9",
    )

    ass_9_16 = generate_karaoke_ass_script(
        timed_lines=timed_lines,
        width=720,
        height=1280,
        subtitle_style="neon",
        aspect_ratio="9:16",
    )

    # 16:9 uses bottom alignment (2)
    assert ",2," in ass_16_9.split("Style: Default,")[1]

    # 9:16 shifts alignment to center (5) to stay clear of TikTok/Reels UI buttons
    assert ",5," in ass_9_16.split("Style: Default,")[1]


def test_section_header_chapter_cue():
    """Verify [Chorus] section tags are formatted as top-screen chapter cue banners."""
    timed_lines = [
        {"start": 0.0, "end": 2.0, "text": "[Chorus]", "is_section": True},
        {"start": 2.0, "end": 6.0, "text": "Singing in the rain", "is_section": False},
    ]

    ass = generate_karaoke_ass_script(
        timed_lines=timed_lines,
        width=1280,
        height=720,
    )

    assert "SectionHeader" in ass
    assert r"{\fad(250,250)}— CHORUS —" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,SectionHeader" in ass
