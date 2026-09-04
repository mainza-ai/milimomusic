"""
Provider provenance tests (production-grade behavior):
- GeneratedAudioResult fallback fields have sane defaults.
- generate() reports used_fallback_synth=True + an honest reason when the real
  inference path is unavailable (never silent).
- structured_caption: caller-provided structured caption sections are honored and
  missing sections are filled from the auto-constructed caption.
"""

import os
import pytest

import app.providers.minimax_provider as mp
from app.providers.base import GeneratedAudioResult
from app.providers.minimax_provider import MiniMaxMusic3Provider

# Strong prompt/lyrics so the producer enhancement passes through untouched
# (no LLM calls in tests) and generation only exercises the provider itself.
STRONG_PROMPT = ("A nostalgic synthwave night drive with driving drums, warm bass "
                 "and evocative vocals, verse and chorus structure")
STRONG_LYRICS = ("[Verse 1]\nStars over the freeway\n[Chorus]\nWe drive through the night "
                 "and never look back")


@pytest.fixture
def no_mlx(monkeypatch):
    """Force the procedural fallback path deterministically (no 28GB model load)."""
    monkeypatch.setattr(mp, "_MLX_AUDIO_AVAILABLE", False)
    monkeypatch.setattr(mp, "_MLX_IMPORT_ERROR", "test: mlx-audio not installed")
    return monkeypatch


def _cleanup(job_id: str):
    for suffix in (".mp3", ".wav"):
        p = os.path.join("generated_audio", f"{job_id}{suffix}")
        if os.path.exists(p):
            os.remove(p)


def test_generated_audio_result_fallback_defaults():
    result = GeneratedAudioResult(audio_path="/audio/x.wav", duration_sec=10.0)
    assert result.used_fallback_synth is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_generate_reports_fallback_when_mlx_unavailable(no_mlx):
    result = await MiniMaxMusic3Provider().generate(
        job_id="test_fallback_001",
        prompt=STRONG_PROMPT,
        lyrics=STRONG_LYRICS,
        duration_ms=3000,
        tags="Synthwave, 128 BPM",
    )
    try:
        assert result.used_fallback_synth is True
        assert result.fallback_reason is not None
        assert "mlx-audio" in result.fallback_reason.lower()
        assert result.metadata.get("real_inference") is False
        assert result.duration_sec == 3.0
    finally:
        _cleanup("test_fallback_001")


@pytest.mark.asyncio
async def test_generate_honors_provided_structured_caption(no_mlx):
    provided = {
        "global_metadata": "Basic Attributes: Genre Synthwave, tempo 118. Custom user caption.",
        "vocal_details": "Vocal Gender & Timbre: Singer B (Male), gravelly baritone.",
    }
    result = await MiniMaxMusic3Provider().generate(
        job_id="test_caption_001",
        prompt=STRONG_PROMPT,
        lyrics=STRONG_LYRICS,
        duration_ms=3000,
        tags="Synthwave, 118 BPM",
        structured_caption=provided,
    )
    try:
        cap = result.structured_caption
        # Provided sections win verbatim...
        assert "Custom user caption" in cap["global_metadata"]
        assert "gravelly baritone" in cap["vocal_details"]
        # ...and the missing Arrangement section is auto-filled so the model
        # always receives a complete three-heading caption.
        assert bool((cap.get("arrangement") or "").strip())
        assert "Synthwave" in cap["arrangement"]
    finally:
        _cleanup("test_caption_001")


@pytest.mark.asyncio
async def test_generate_falls_back_to_auto_caption_when_none_provided(no_mlx):
    result = await MiniMaxMusic3Provider().generate(
        job_id="test_caption_002",
        prompt=STRONG_PROMPT,
        lyrics=STRONG_LYRICS,
        duration_ms=3000,
        tags="Synthwave, 128 BPM, Dark",
    )
    try:
        cap = result.structured_caption
        assert bool(cap.get("global_metadata"))
        assert "Vocal Gender" in (cap.get("vocal_details") or "")
        assert "Instrument Lifecycle" in (cap.get("arrangement") or "")
    finally:
        _cleanup("test_caption_002")
