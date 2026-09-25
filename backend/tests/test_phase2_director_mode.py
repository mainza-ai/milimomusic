"""
Unit and integration tests for Phase 2: Director Mode v2, Audio Analysis,
Music Timing, Performer Scoping, and Prompt Enhancement.
"""

import numpy as np
import pytest

from app.services.video.audio_analysis import AudioAnalysisResult, AudioSignalAnalyzer
from app.services.video.convrot_layout import ConvRotLayoutParser, ConvRotQuantConfig
from app.services.video.director_music_timing import DirectorMusicTiming, PlannedMusicClip
from app.services.video.music_performance import MusicPerformanceDirector
from app.services.video.prompt_enhancer import (
    PERFORMANCE_AUDIO_GUIDANCE,
    PromptFidelityReport,
    VideoPromptEnhancer,
)


def test_audio_analysis_synthetic_fallback():
    # Test fallback analysis on a non-existent or synthetic file
    res = AudioSignalAnalyzer._fallback_analysis("dummy.wav")
    assert isinstance(res, AudioAnalysisResult)
    assert res.duration_sec == 30.0
    assert res.tempo_bpm == 120.0
    assert len(res.beats) > 0
    assert len(res.downbeats) > 0
    assert len(res.sections) > 0


def test_director_music_timing_pacing_and_lattice():
    analysis = AudioAnalysisResult(
        duration_sec=32.0,
        tempo_bpm=128.0,
        beats=[round(i * (60.0 / 128.0), 3) for i in range(64)],
        downbeats=[round(i * (60.0 / 128.0) * 4, 3) for i in range(16)],
        sections=[
            {"label": "Intro", "start": 0.0, "end": 8.0, "energy": 0.3},
            {"label": "Verse", "start": 8.0, "end": 20.0, "energy": 0.5},
            {"label": "Chorus", "start": 20.0, "end": 32.0, "energy": 0.8},
        ],
        vocal_intervals=[(8.0, 20.0), (22.0, 31.0)],
        percussion_cues=[{"timestamp": 8.0, "intensity": 2.5, "type": "transient_drop"}],
    )

    # Test balanced pacing (0)
    clips_balanced = DirectorMusicTiming.plan_capped_music_clips(
        analysis, model_name="wan2.1", pacing_bias=0
    )
    assert len(clips_balanced) >= 4
    # Check that clips span the whole duration seamlessly
    assert clips_balanced[0].start_time == 0.0
    assert clips_balanced[-1].end_time == 32.0
    for i in range(len(clips_balanced) - 1):
        assert abs(clips_balanced[i].end_time - clips_balanced[i + 1].start_time) < 0.001

    # Test slow pacing (-2) yields fewer clips
    clips_slow = DirectorMusicTiming.plan_capped_music_clips(
        analysis, model_name="wan2.1", pacing_bias=-2
    )
    assert len(clips_slow) <= len(clips_balanced)

    # Test rapid montage (+2) yields more clips
    clips_fast = DirectorMusicTiming.plan_capped_music_clips(
        analysis, model_name="wan2.1", pacing_bias=2
    )
    assert len(clips_fast) >= len(clips_balanced)

    # Test model lattice snapping for H3
    h3_render_dur, h3_trim = DirectorMusicTiming.resolve_model_lattice(3.5, "minimax_h3")
    assert h3_render_dur >= 3.5
    assert h3_trim >= 0.0


def test_music_performance_scoping():
    # Vocal active performance
    p_vocal = MusicPerformanceDirector.format_performer_prompt_instructions(
        visible_cast=["Lead Singer Alex"],
        is_vocal_active=True,
        section_label="Chorus",
        shot_intent="performance",
    )
    assert "singing performance" in p_vocal
    assert "Alex" in p_vocal

    # Non-vocal instrumental break enforces mouth_movement: closed
    p_closed = MusicPerformanceDirector.format_performer_prompt_instructions(
        visible_cast=["Lead Guitarist Maya"],
        is_vocal_active=False,
        section_label="Verse",
        is_instrumental_solo=True,
        shot_intent="performance",
    )
    assert "mouth_movement: closed" in p_closed
    assert "instrument playing technique" in p_closed

    # Scenery shot strips musician boilerplate
    p_scenery = MusicPerformanceDirector.format_performer_prompt_instructions(
        shot_intent="scenery",
    )
    assert "establishing shot" in p_scenery
    assert "no active performers" in p_scenery


def test_prompt_enhancer_vocal_bypass_and_repair():
    enhancer = VideoPromptEnhancer(max_fidelity_retries=1, auto_continue_on_fail=False)

    # Test Two-Tier Vocal Bypass prompt building
    base_prompt = "Cyberpunk street in rain. <d>Singer: Hello world</d> Walk forward."
    built = enhancer.build_music_video_prompt(
        base_visual_prompt=base_prompt,
        section_label="Chorus",
        performer_instructions="Alex center framed.",
        camera_motion="Forward dolly zoom.",
        lighting_design="Neon cyan rim.",
        is_music_timeline=True,
    )
    assert PERFORMANCE_AUDIO_GUIDANCE in built
    # Dialogue must be stripped
    assert "<d>" not in built
    assert "Hello world" not in built

    # Test repair retries in Interactive mode (auto_continue=False)
    call_count = 0

    def bad_generator():
        nonlocal call_count
        call_count += 1
        return "<d>Illegal speech</d>"

    report = enhancer.enhance_with_repair_retries(bad_generator, is_music_timeline=True)
    # Should attempt initial + 1 retry = 2 calls
    assert call_count == 2
    assert not report.is_valid
    assert not report.auto_continued

    # Test auto_continue in Batch mode
    enhancer_batch = VideoPromptEnhancer(max_fidelity_retries=1, auto_continue_on_fail=True)
    report_batch = enhancer_batch.enhance_with_repair_retries(bad_generator, is_music_timeline=True)
    assert report_batch.auto_continued
    assert report_batch.is_valid

    # Test localized card repair
    cards = ["Card 0 ok", "Card 1 bad", "Card 2 ok"]
    repaired_cards = VideoPromptEnhancer.repair_localized_card(1, cards, lambda idx: "Card 1 fixed")
    assert repaired_cards == ["Card 0 ok", "Card 1 fixed", "Card 2 ok"]


def test_convrot_quant_parsing():
    mock_meta = {
        "comfy_quant": '{"format": "convrot_int8", "group_size": 256, "grouped_qkv": true}'
    }
    cfg = ConvRotLayoutParser.parse_quant_descriptor(mock_meta)
    assert isinstance(cfg, ConvRotQuantConfig)
    assert cfg.quant_format == "convrot_int8"
    assert cfg.group_size == 256
    assert cfg.turbo_preset_steps == 4
