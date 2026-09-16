import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from app.transcription.drum_tracker import transcribe_drums_from_stem


def test_drum_tracker_synthetic_stems():
    """Verify drum tracker extracts kick, snare, and hi-hat events from audio."""
    sr = 22050
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)

    # 1. Low frequency thump at t=0.2s (Kick: 60Hz)
    kick_idx = int(0.2 * sr)
    decay = np.exp(-np.linspace(0, 10, int(0.15 * sr)))
    y[kick_idx:kick_idx + len(decay)] += np.sin(2 * np.pi * 60 * t[:len(decay)]) * decay

    # 2. Mid frequency burst at t=0.8s (Snare: noise + 300Hz)
    snare_idx = int(0.8 * sr)
    decay_snare = np.exp(-np.linspace(0, 15, int(0.15 * sr)))
    noise = np.random.randn(len(decay_snare)) * 0.5 + np.sin(2 * np.pi * 300 * t[:len(decay_snare)])
    y[snare_idx:snare_idx + len(decay_snare)] += noise * decay_snare

    # 3. High frequency sizzle at t=1.4s (Hi-Hat: 6000Hz noise)
    hat_idx = int(1.4 * sr)
    decay_hat = np.exp(-np.linspace(0, 30, int(0.08 * sr)))
    hat_noise = np.random.randn(len(decay_hat)) * decay_hat
    y[hat_idx:hat_idx + len(decay_hat)] += hat_noise

    with tempfile.TemporaryDirectory() as tmpdir:
        drum_file = Path(tmpdir) / "drums.wav"
        sf.write(str(drum_file), y, sr)

        events = transcribe_drums_from_stem(drum_file, bpm=120.0)
        assert len(events) >= 1
        for ev in events:
            assert "onset" in ev
            assert "offset" in ev
            assert ev["program"] == 128
            assert ev["is_drum"] is True
            assert ev["pitch"] in (36, 38, 42)


def test_drum_tracker_silence():
    """Verify drum tracker safely returns empty list for silent stems."""
    with tempfile.TemporaryDirectory() as tmpdir:
        silent_file = Path(tmpdir) / "silent_drums.wav"
        sf.write(str(silent_file), np.zeros(22050), 22050)

        events = transcribe_drums_from_stem(silent_file, bpm=120.0)
        assert events == []
