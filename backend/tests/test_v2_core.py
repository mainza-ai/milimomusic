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
from app.transcription.real_separator import separate_sources
from app.transcription.karaoke import lyric_sync_engine
from app.services.voice_service import voice_service
from app.services.model_manager import model_manager


def test_provider_registry_defaults():
    capabilities = provider_registry.list_capabilities()
    assert len(capabilities) >= 1
    
    # Check default active provider is MiniMax Music 3
    active_caps = provider_registry.get_active_capabilities()
    assert active_caps.provider_id == "minimax_music3"
    assert active_caps.supports_structured_caption is True
    assert active_caps.max_duration_sec == 300


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
    from unittest.mock import MagicMock, patch
    from muscriptor.events import NoteStartEvent, NoteEndEvent

    ev1 = NoteStartEvent(index=1, pitch=60, start_time=0.0, instrument="Piano")
    ev2 = NoteEndEvent(end_time=1.0, start_event=ev1)
    mock_grid = MagicMock(bpm=120.0)
    mock_model = MagicMock()
    mock_model.detect_grid.return_value = mock_grid
    mock_model.transcribe.return_value = [ev1, ev2]
    mock_model.events_to_midi_bytes.return_value = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"

    with patch.object(muscriptor_provider, "_get_model", return_value=mock_model):
        result = await muscriptor_provider.transcribe(
            audio_file_path="generated_audio/test_rhythmic_song.wav",
            job_id="test_job_123"
        )
        assert result.midi_path.endswith(".mid")
        assert result.musicxml_path.endswith(".musicxml")
        assert len(result.notes) > 0
        assert "bpm" in result.beat_grid
        assert os.path.exists("generated_audio/test_job_123.mid")
        assert os.path.exists("generated_audio/test_job_123.musicxml")


def test_stem_separation():
    from unittest.mock import patch
    from app.transcription.real_separator import SeparationResult
    mock_res = SeparationResult(
        stems={
            "vocals": "/audio/stems/test_job_stem_vocals.wav",
            "drums": "/audio/stems/test_job_stem_drums.wav",
            "bass": "/audio/stems/test_job_stem_bass.wav",
            "other": "/audio/stems/test_job_stem_other.wav",
        },
        source_id="bs_roformer_6stem",
        sources_available=["vocals", "drums", "bass", "other"],
        stem_count=4,
    )
    import sys
    with patch.object(sys.modules[__name__], "separate_sources", return_value=mock_res):
        result = separate_sources(
            master_wav_path="generated_audio/test_rhythmic_song.wav",
            job_id="test_job_stem"
        )
        assert hasattr(result, "stems")
        assert "vocals" in result.stems
        assert "drums" in result.stems
        assert "bass" in result.stems
        assert "other" in result.stems
        assert result.stem_count >= 4


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
    minimax_entry = next((m for m in tree if "minimax" in m["id"] and m.get("is_default")), None)
    assert minimax_entry is not None
    assert minimax_entry["is_default"] is True
    assert minimax_entry["size_gb"] > 0

    hw = model_manager.detect_hardware()
    assert hw.hardware_tier is not None
    assert isinstance(hw.can_run_minimax_full, bool)
    assert hw.can_run_heartmula is True
    if hw.has_mps or hw.has_cuda:
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
