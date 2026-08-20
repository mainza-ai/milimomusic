"""
Milimo Music v2 Core Unit Tests.
Tests Provider Abstraction, MiniMax Music 3, MuScriptor, Stem Separator, Voice Service, Model Tree.
"""

import os
import pytest
import asyncio
from app.providers.registry import provider_registry
from app.providers.minimax_provider import MiniMaxMusic3Provider
from app.transcription.muscriptor_provider import muscriptor_provider
from app.transcription.stem_separator import stem_separator
from app.transcription.karaoke import lyric_sync_engine
from app.services.voice_service import voice_service
from app.services.model_manager import model_manager


def test_provider_registry_defaults():
    capabilities = provider_registry.list_capabilities()
    assert len(capabilities) >= 2
    
    # Check default active provider is MiniMax Music 3
    active_caps = provider_registry.get_active_capabilities()
    assert active_caps.provider_id == "minimax_music3"
    assert active_caps.supports_structured_caption is True
    assert active_caps.max_duration_sec == 300

    # Check HeartMuLa provider is registered
    heartmula_provider = provider_registry.get_provider("heartmula")
    assert heartmula_provider.get_capabilities().provider_id == "heartmula"
    assert heartmula_provider.get_capabilities().supports_lora is True


def test_minimax_structured_caption_parsing():
    raw_prompt = "A high-energy cyberpunk chase scene"
    tags = "Synthwave, 128 BPM, Dark, Driving Bass"
    
    parsed = MiniMaxMusic3Provider.parse_structured_caption(raw_prompt, tags)
    assert "global_metadata" in parsed
    assert "vocal_details" in parsed
    assert "arrangement" in parsed
    assert "Synthwave" in parsed["global_metadata"]

    formatted = MiniMaxMusic3Provider.format_full_caption(parsed, raw_prompt)
    assert "[Global Metadata]" in formatted
    assert "[Vocal Details]" in formatted
    assert "[Arrangement]" in formatted


def test_minimax_section_tag_sanitization():
    raw_lyrics = "Intro:\nHello world\nVerse 1:\nWalking down the road\nChorus:\nShining bright\nOutro:\nGoodbye"
    sanitized = MiniMaxMusic3Provider.sanitize_section_tags(raw_lyrics)
    assert "[Intro]" in sanitized
    assert "[Verse 1]" in sanitized
    assert "[Chorus]" in sanitized
    assert "[Outro]" in sanitized


@pytest.mark.asyncio
async def test_muscriptor_transcription():
    result = await muscriptor_provider.transcribe(
        audio_file_path="generated_audio/test.wav",
        job_id="test_job_123"
    )
    assert result.midi_path.endswith(".mid")
    assert result.musicxml_path.endswith(".musicxml")
    assert len(result.notes) > 0
    assert "bpm" in result.beat_grid
    assert os.path.exists("generated_audio/test_job_123.mid")
    assert os.path.exists("generated_audio/test_job_123.musicxml")


@pytest.mark.asyncio
async def test_stem_separation():
    result = await stem_separator.separate(
        audio_path="/audio/test.wav",
        job_id="test_job_stem"
    )
    assert result.vocals_path.endswith("_vocals.wav")
    assert result.drums_path.endswith("_drums.wav")
    assert result.bass_path.endswith("_bass.wav")
    assert result.other_path.endswith("_other.wav")
    assert result.instrumental_path.endswith("_instrumental.wav")


def test_voice_profile_management():
    # Test consent enforcement
    with pytest.raises(ValueError):
        voice_service.create_profile(
            name="Unauthorized Voice",
            description="Test",
            consent_confirmed=False
        )

    # Test valid profile creation
    profile = voice_service.create_profile(
        name="Test Singer",
        description="Acoustic Folk Vocals",
        consent_confirmed=True,
        f0_method="rmvpe"
    )
    assert profile["name"] == "Test Singer"
    assert profile["consent_confirmed"] is True

    # Test list and delete
    profiles = voice_service.list_profiles()
    assert any(p["id"] == profile["id"] for p in profiles)
    voice_service.delete_profile(profile["id"])


def test_model_tree_and_hardware():
    tree = model_manager.get_model_tree()
    assert len(tree) >= 2
    minimax_entry = next((m for m in tree if "minimax" in m["id"]), None)
    assert minimax_entry is not None
    assert minimax_entry["is_default"] is True
    assert minimax_entry["size_gb"] > 0

    hw = model_manager.detect_hardware()
    assert hw.hardware_tier is not None
    assert hw.can_run_minimax_full is True


def test_lyric_sync_and_lrc():
    lyrics = "[Intro]\nHello night\n[Verse 1]\nStars in the sky\n[Chorus]\nWe are alive"
    timed_lines = lyric_sync_engine.align_lyrics(lyrics, duration_sec=30.0)
    assert len(timed_lines) >= 3

    lrc = lyric_sync_engine.generate_lrc(timed_lines)
    assert "[00:" in lrc
    assert "Hello night" in lrc

    srt = lyric_sync_engine.generate_srt(timed_lines)
    assert "-->" in srt
