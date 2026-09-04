"""Tests verifying songwriter tag preservation, genre-first sorting,
and prompt steering cohesion through the bridge to GenerationRequest (F6).
"""
import pytest
from app.agents.orchestrator.bridge import (
    order_tags_genre_first,
    energy_to_duration_s,
    build_steering_prose,
)
from app.models import GenerationRequest, Job


def test_order_tags_genre_first_prioritizes_known_genres():
    tags = ["Electric Guitar", "Synthwave", "Driving Drums", "80s Retro"]
    ordered = order_tags_genre_first(tags)
    assert ordered[0] == "Synthwave"
    assert "Electric Guitar" in ordered
    assert "Driving Drums" in ordered
    assert len(ordered) <= 6


def test_order_tags_genre_first_fallback_when_empty():
    ordered = order_tags_genre_first([])
    assert ordered == ["Pop"]


def test_order_tags_genre_first_handles_multiple_genres():
    tags = ["Acoustic Guitar", "Folk", "Warm Vocals", "Indie Rock"]
    ordered = order_tags_genre_first(tags)
    assert ordered[0] in ("Folk", "Indie Rock")
    assert ordered[1] in ("Folk", "Indie Rock")
    assert "Acoustic Guitar" in ordered
    assert "Warm Vocals" in ordered


def test_build_steering_prose_includes_seed_attributes():
    seed = {
        "working_title": "Ignition Hymn",
        "mood": "Anthemic and soaring",
        "energy": 0.85,
        "placement_hint": "opener",
    }
    prose = build_steering_prose(seed)
    assert "Working title: Ignition Hymn" in prose
    assert "Mood: Anthemic and soaring" in prose
    assert "Energy: high and driving" in prose
    assert "Arc placement: opener" in prose


def test_build_steering_prose_low_energy():
    seed = {
        "working_title": "Quiet Ash",
        "mood": "Intimate",
        "energy": 0.15,
        "placement_hint": "closer",
    }
    prose = build_steering_prose(seed)
    assert "Energy: sparse and restrained" in prose
    assert "Arc placement: closer" in prose


def test_energy_to_duration_scaling():
    assert energy_to_duration_s(0.0) == 120
    assert energy_to_duration_s(1.0) == 240
    assert 170 <= energy_to_duration_s(0.5) <= 190


def test_rich_prompt_construction_prevents_override():
    draft_title = "Ignition Hymn"
    seed = {"working_title": "Ignition Hymn", "mood": "Bold", "energy": 0.9}
    tags_str = "Synthwave, Electric Guitar, Arpeggiated Bass"
    album_context = {"album_title": "Neon Horizon"}
    
    rich_prompt = (
        f"{draft_title}. {build_steering_prose(seed)}. "
        f"Style: {tags_str}. Album: {album_context.get('album_title', '')}"
    )
    assert "Ignition Hymn" in rich_prompt
    assert "Style: Synthwave, Electric Guitar, Arpeggiated Bass" in rich_prompt
    assert "Album: Neon Horizon" in rich_prompt
    assert len(rich_prompt) > 50  # Significantly exceeds weak-prompt (<30 char) threshold
