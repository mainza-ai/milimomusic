"""
Comprehensive unit & integration tests for MuLaCover integration in Milimo Music.
Tests formatters, SymbolicHub, Provider registry, and API endpoints.
"""
import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
import pretty_midi

# Add backend and mulacover/src directory to sys.path
import sys
backend_dir = Path(__file__).parent.parent
mulacover_dir = backend_dir.parent / "mulacover" / "src"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(mulacover_dir) not in sys.path:
    sys.path.insert(0, str(mulacover_dir))

import asyncio
import httpx
from app.services.mulacover.formatters import (
    format_style_tags,
    sanitize_lyrics_for_mulacover,
    parse_mulacover_tags,
)
from app.services.mulacover.symbolic_hub import SymbolicHub
from app.providers.registry import get_provider, list_providers
from app.models import CoverGenerationRequest, LeadSheetExtractRequest
from app.main import app


class SyncTestClient:
    def __init__(self, asgi_app):
        self.app = asgi_app
        self.transport = httpx.ASGITransport(app=asgi_app)
        
    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def get(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.get(url, **kwargs)
        return self._run(_call())

    def post(self, url, **kwargs):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as c:
                return await c.post(url, **kwargs)
        return self._run(_call())


def test_format_style_tags():
    """Verify structured tags are formatted to canonical topic:[...]; genre:[...]; etc."""
    raw = "pop, synthwave, 120bpm, energetic, synthesizer, drums"
    formatted = format_style_tags(prompt=raw, genre="Synthwave")
    assert "genre:[Synthwave]" in formatted
    assert "mood:" in formatted
    assert "instrument:" in formatted
    
    # Check already canonical string
    canonical = "topic:[cyberpunk city]; genre:[synthwave]; instrument:[analog synth, 808]; mood:[nostalgic]"
    res = format_style_tags(prompt=canonical)
    assert res == canonical


def test_parse_mulacover_tags():
    """Verify parsing canonical tags into dictionary components."""
    tag_str = "topic:[cyberpunk city]; genre:[synthwave]; instrument:[analog synth]; mood:[energetic]"
    parsed = parse_mulacover_tags(tag_str)
    assert parsed["topic"] == "cyberpunk city"
    assert parsed["genre"] == "synthwave"
    assert parsed["instrument"] == "analog synth"
    assert parsed["mood"] == "energetic"


def test_sanitize_lyrics_for_mulacover():
    """Verify bracketed tags, verse indicators, and blank lines are cleaned cleanly."""
    raw_lyrics = """
    [Verse 1]
    Riding down the midnight highway
    (Instrumental Solo)
    [Chorus]
    Neon lights flashing in the rain
    <fade out>
    [Outro]
    """
    cleaned = sanitize_lyrics_for_mulacover(raw_lyrics)
    assert "[Verse]" in cleaned
    assert "[Chorus]" in cleaned
    assert "[Outro]" in cleaned
    assert "Riding down the midnight highway" in cleaned
    assert "Neon lights flashing in the rain" in cleaned


def test_symbolic_hub_midi_condition_and_export():
    """Verify loading MIDI into SymbolicCondition and exporting lead sheet back to disk."""
    from mulacover.symbolic import SymbolicCondition
    hub = SymbolicHub()
    
    # Create temporary melody and chord MIDIs
    pm_mel = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_mel = pretty_midi.Instrument(program=0)
    inst_mel.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst_mel.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
    pm_mel.instruments.append(inst_mel)
    
    pm_chord = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst_ch = pretty_midi.Instrument(program=0)
    inst_ch.notes.append(pretty_midi.Note(velocity=80, pitch=48, start=0.0, end=1.0))
    inst_ch.notes.append(pretty_midi.Note(velocity=80, pitch=52, start=0.0, end=1.0))
    inst_ch.notes.append(pretty_midi.Note(velocity=80, pitch=55, start=0.0, end=1.0))
    pm_chord.instruments.append(inst_ch)
    
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f_mel, \
         tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f_ch, \
         tempfile.TemporaryDirectory() as out_dir:
        
        pm_mel.write(f_mel.name)
        pm_chord.write(f_ch.name)
        
        try:
            cond = hub.create_condition_from_midi(melody_path=f_mel.name, chord_path=f_ch.name)
            assert cond is not None
            exported = hub.export_lead_sheet(cond, out_dir)
            assert "melody" in exported
            assert os.path.exists(exported["melody"])
        finally:
            if os.path.exists(f_mel.name):
                os.remove(f_mel.name)
            if os.path.exists(f_ch.name):
                os.remove(f_ch.name)


def test_mulacover_provider_registration():
    """Verify mulacover is registered in generation provider registry."""
    providers = list_providers()
    assert "mulacover" in providers
    provider = get_provider("mulacover")
    assert provider is not None
    caps = provider.get_capabilities()
    assert caps.provider_id == "mulacover"
    assert caps.supports_section_tags is True
    assert caps.default_sample_rate == 48000


def test_models_capabilities_endpoint():
    """Verify /models/capabilities includes mulacover and audio checkpoints."""
    client = SyncTestClient(app)
    response = client.get("/models/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "capabilities" in data
    provider_ids = [c["provider_id"] for c in data["capabilities"]]
    assert "mulacover" in provider_ids
    assert "minimax_music3" in provider_ids


def test_transcribe_lead_sheet_endpoint_validation():
    """Verify /transcribe/lead-sheet handles missing files gracefully."""
    client = SyncTestClient(app)
    # Post with non-existent audio path
    payload = {
        "audio_path": "/tmp/non_existent_audio_file_12345.wav",
        "bpm": 120.0
    }
    response = client.post("/transcribe/lead-sheet", json=payload)
    assert response.status_code == 404


def test_transcribe_lead_sheet_audio_path_resolution(tmp_path, monkeypatch):
    """Verify /transcribe/lead-sheet resolves /audio/ URLs and relative paths."""
    from mulacover.symbolic import SymbolicCondition
    from app.services.mulacover.symbolic_hub import symbolic_hub

    # Create dummy audio file in generated_audio
    test_audio = Path("generated_audio") / "test_resolve_lead_sheet.wav"
    test_audio.parent.mkdir(parents=True, exist_ok=True)
    
    # Write a tiny valid wav file (440Hz sine wave, 0.25s)
    import numpy as np
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(test_audio), sig, sr)

    try:
        # Mock neural transcription so we test the endpoint without running heavy weights
        import torch
        dummy_cond = SymbolicCondition(
            melody=torch.tensor([[0, 60, 4]], dtype=torch.int64),
            chords=torch.tensor([[0, 0, 4]], dtype=torch.int64),
            drums=torch.empty((0, 3), dtype=torch.int64),
            bpm=120.0
        )
        async def mock_neural(*args, **kwargs):
            return dummy_cond

        monkeypatch.setattr(symbolic_hub, "transcribe_milimo_neural", mock_neural)

        client = SyncTestClient(app)
        
        # 1. Test resolution with /audio/ prefix
        res1 = client.post("/transcribe/lead-sheet", json={
            "audio_path": "/audio/test_resolve_lead_sheet.wav",
            "bpm": 120.0
        })
        assert res1.status_code == 200, res1.text
        assert res1.json()["bpm"] == 120.0

        # 2. Test resolution with full localhost URL
        res2 = client.post("/transcribe/lead-sheet", json={
            "audio_path": "http://localhost:5173/audio/test_resolve_lead_sheet.wav",
            "bpm": 120.0
        })
        assert res2.status_code == 200, res2.text
    finally:
        if test_audio.exists():
            test_audio.unlink()


def test_generate_cover_endpoint_validation(monkeypatch):
    """Verify /generate/cover validates missing symbolic inputs and enqueues valid requests."""
    client = SyncTestClient(app)
    
    # 1. Reject when missing both audio reference and symbolic MIDI
    invalid_payload = {
        "title": "Neon Dreams (Remix)",
        "lyrics": "[Verse]\\nNight city lights",
        "tags": "genre:synthwave; mood:energetic; instrument:synthesizer",
        "cover_mode": "lead_sheet",
        "bpm": 128.0,
        "duration_ms": 60000
    }
    bad_res = client.post("/generate/cover", json=invalid_payload)
    assert bad_res.status_code == 400
    detail = bad_res.json()["detail"]
    code = detail.get("error", {}).get("code") or detail.get("code")
    assert code == "missing_symbolic_input"

    # 2. Enqueue valid request with ref_audio_path (mock installed state for CI environments)
    monkeypatch.setattr("app.services.mulacover.bundle_downloader.is_mulacover_installed", lambda *args, **kwargs: True)
    valid_payload = {
        "title": "Neon Dreams (Remix)",
        "ref_audio_path": "/audio/test_reference.wav",
        "lyrics": "[Verse]\\nNight city lights",
        "tags": "genre:synthwave; mood:energetic; instrument:synthesizer",
        "cover_mode": "audio_reference",
        "bpm": 128.0,
        "duration_ms": 60000
    }
    response = client.post("/generate/cover", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_mulacover_bundle_downloader():
    """Verify bundle downloader functions and model manager resolution."""
    from app.services.mulacover.bundle_downloader import is_mulacover_installed, resolve_mulacover_dir, get_bundle_manifest
    from app.services.model_manager import model_manager

    base_dir = resolve_mulacover_dir()
    manifest = get_bundle_manifest(base_dir)
    assert len(manifest) >= 4

    tree = model_manager.get_model_tree()
    mula_entry = next((m for m in tree if m["id"] == "mulacover"), None)
    assert mula_entry is not None
    assert mula_entry["is_installed"] == is_mulacover_installed(base_dir)
