"""Regression tests for faithful track-extension continuity.

Covers the CROSS 107.7-vs-112 BPM divergence: the timed remainder slice is
transcription-normalized text and must NEVER steer the KV replay conditioning
(it is not a substring of the stored parent lyrics). Also covers song-state
restore validation and post-hoc replay verification.
"""
import numpy as np
import pytest
import soundfile as sf

from app.providers.minimax_provider import (
    resolve_extension_lyrics,
    decide_extension_lyrics,
    validate_song_state,
    verify_extension_replay,
    quantize_crossfade_to_beats,
    repair_beat_grid,
    _dominant_tempo_bpm,
)
from app.providers.minimax_local_hooks import conditioning_fingerprint


PARENT_LYRICS = """[Intro]
Haunting acoustic oud riff

[Chorus]
Crossed the ocean, crossed the border, crossed the line for you

[Verse]
Gold teeth glinting when I smile through the pain

[Bridge]
When the jury read the verdict, who is by your side?
They say love is gentle, nah, love is homicide.
"""

# What split_remaining_lyrics actually returns for the bar above: the timed
# line header carries transcription context ("[Bridge - ...]") that is NOT a
# substring of the stored lyrics. This exact shape caused the CROSS steering.
REMAINDER_SLICE = """[Bridge - Stripped oud picking and clinking bangles]
When the jury read the verdict, who is by your side?
They say love is gentle — nah, love is homicide.
"""


def test_resolve_verbatim_when_no_explicit_lyrics():
    eff, steered = resolve_extension_lyrics(PARENT_LYRICS, "")
    assert eff == PARENT_LYRICS
    assert steered is False


def test_resolve_verbatim_when_explicit_is_blank():
    eff, steered = resolve_extension_lyrics(PARENT_LYRICS, "   \n ")
    assert eff == PARENT_LYRICS
    assert steered is False


def test_resolve_verbatim_when_explicit_contained():
    eff, steered = resolve_extension_lyrics(
        PARENT_LYRICS, "Crossed the ocean, crossed the border"
    )
    assert eff == PARENT_LYRICS
    assert steered is False


def test_remainder_slice_would_steer_if_passed_as_lyrics():
    """Lock in WHY the route must never pass the remainder as lyrics.

    The transcription-normalized remainder is not a substring of the stored
    parent lyrics, so the replay rule would treat it as new steering text.
    The route therefore routes it via tail_lyrics (fallback-only) instead.
    """
    assert REMAINDER_SLICE.strip() not in PARENT_LYRICS  # guard: fixture is faithful
    eff, steered = resolve_extension_lyrics(PARENT_LYRICS, REMAINDER_SLICE)
    assert steered is True
    assert eff != PARENT_LYRICS


def test_decide_default_never_steers():
    steer, full = decide_extension_lyrics(PARENT_LYRICS, None, None)
    assert steer == ""
    assert full == PARENT_LYRICS
    eff, steered = resolve_extension_lyrics(PARENT_LYRICS, steer)
    assert steered is False and eff == PARENT_LYRICS


def test_decide_remainder_has_no_path_to_steer():
    """decide_extension_lyrics takes no remainder input by construction."""
    import inspect
    params = inspect.signature(decide_extension_lyrics).parameters
    assert "remainder" not in params and "tail_lyrics" not in params


def test_decide_additional_lyrics_steer_with_verse_tag():
    steer, full = decide_extension_lyrics(PARENT_LYRICS, "a brand new closing line", None)
    assert steer == "[Verse]\na brand new closing line"
    assert full == f"{PARENT_LYRICS}\n\n[Verse]\na brand new closing line"
    _, steered = resolve_extension_lyrics(PARENT_LYRICS, steer)
    assert steered is True


def test_decide_additional_keeps_existing_tag():
    steer, full = decide_extension_lyrics(PARENT_LYRICS, "[Outro]\nwe ride out", None)
    assert steer == "[Outro]\nwe ride out"
    assert full.endswith("[Outro]\nwe ride out")


def test_decide_auto_lyrics_used_verbatim():
    steer, full = decide_extension_lyrics(PARENT_LYRICS, None, "[Chorus]\nmachine-written end")
    assert steer == "[Chorus]\nmachine-written end"
    assert full.endswith("[Chorus]\nmachine-written end")


def test_genuinely_new_lyrics_steer_and_flag():
    new = "[Outro]\nBrand new closing line never sung before"
    assert new not in PARENT_LYRICS
    eff, steered = resolve_extension_lyrics(PARENT_LYRICS, new)
    assert steered is True
    assert eff == f"{PARENT_LYRICS}\n\n{new}".strip()
    assert new in eff


def test_resolve_empty_base_with_new_lyrics():
    eff, steered = resolve_extension_lyrics("", "[Verse]\nHello")
    assert eff == "[Verse]\nHello"
    assert steered is True


def _fake_state(seed=7, caption="cap", lyrics="lyr", snapshot="/snap", frames=10):
    return {
        "codes": [np.zeros(4, dtype=np.float32) for _ in range(frames)],
        "seed": seed,
        "conditioning": conditioning_fingerprint(caption, lyrics),
        "snapshot_path": snapshot,
        "frame_rate": 25.0,
        "frames": frames,
    }


def test_validate_state_accepts_exact_match():
    st = _fake_state()
    assert validate_song_state(st, 7, "cap", "lyr", "/snap") is st


def test_validate_state_rejects_seed_mismatch():
    assert validate_song_state(_fake_state(), 999, "cap", "lyr", "/snap") is None


def test_validate_state_rejects_conditioning_mismatch():
    assert validate_song_state(_fake_state(), 7, "cap", "OTHER", "/snap") is None
    assert validate_song_state(_fake_state(), 7, "OTHER", "lyr", "/snap") is None


def test_validate_state_rejects_snapshot_mismatch():
    assert validate_song_state(_fake_state(), 7, "cap", "lyr", "/other") is None


def test_validate_state_rejects_empty_or_none():
    assert validate_song_state(None, 7, "cap", "lyr", "/snap") is None
    st = _fake_state(frames=0)
    st["codes"] = []
    assert validate_song_state(st, 7, "cap", "lyr", "/snap") is None


def _click_track(bpm, seconds=12.0, sr=44100):
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    y[::period] = 1.0
    # short exponential click decay so onset detection sees clean attacks
    decay = np.exp(-np.arange(64) / 8.0).astype(np.float32)
    for start in range(0, n, period):
        end = min(n, start + 64)
        y[start:end] = np.maximum(y[start:end], decay[: end - start])
    stereo = np.stack([y, y], axis=1)
    return stereo, sr


def test_dominant_tempo_on_click_track():
    y, sr = _click_track(120.0)
    bpm = _dominant_tempo_bpm(y.mean(axis=1), sr)
    assert bpm is not None
    assert any(abs(bpm - g) < 2.0 for g in (60.0, 120.0, 240.0))


def test_verify_identical_audio_passes(tmp_path):
    y, sr = _click_track(112.0)
    p1 = str(tmp_path / "a.wav")
    p2 = str(tmp_path / "b.wav")
    sf.write(p1, y, sr)
    sf.write(p2, y, sr)
    out = verify_extension_replay(p2, p1, cut_sec=10.0)
    assert out["replay_verified"] is True
    assert out["corr"] > 0.99


def test_verify_different_tempo_fails(tmp_path):
    y_parent, sr = _click_track(112.0)
    y_render, _ = _click_track(96.0)
    pp = str(tmp_path / "parent.wav")
    rp = str(tmp_path / "render.wav")
    sf.write(pp, y_parent, sr)
    sf.write(rp, y_render, sr)
    out = verify_extension_replay(rp, pp, cut_sec=10.0)
    assert out["replay_verified"] is False


def test_verify_missing_file_returns_none(tmp_path):
    out = verify_extension_replay(str(tmp_path / "nope.wav"), str(tmp_path / "alsono.wav"), 10.0)
    assert out["replay_verified"] is None


def test_quantize_crossfade_to_whole_beats():
    grid = {"bpm": 120.0, "beats_per_bar": 4, "first_downbeat": 0.0}
    assert quantize_crossfade_to_beats(1.5, grid) == pytest.approx(1.5)  # 3 beats
    assert quantize_crossfade_to_beats(1.6, grid) == pytest.approx(1.5)  # snaps down
    assert quantize_crossfade_to_beats(1.4, grid) == pytest.approx(1.5)  # snaps up
    assert quantize_crossfade_to_beats(0.1, grid) == pytest.approx(0.5)  # min 1 beat


def test_quantize_crossfade_clamps_to_bar_and_bad_grid():
    grid = {"bpm": 120.0, "beats_per_bar": 4, "first_downbeat": 0.0}
    assert quantize_crossfade_to_beats(10.0, grid) == pytest.approx(2.0)  # max 1 bar
    assert quantize_crossfade_to_beats(1.5, {}) == 1.5
    assert quantize_crossfade_to_beats(1.5, None) == 1.5


def test_repair_beat_grid_leaves_good_grid_alone(tmp_path):
    y, sr = _click_track(112.0)
    p = str(tmp_path / "c.wav")
    sf.write(p, y, sr)
    grid = {"bpm": 112.3, "beats_per_bar": 4, "first_downbeat": 0.5}
    out, repaired = repair_beat_grid(p, grid)
    assert repaired is False
    assert out == grid


def test_repair_beat_grid_fixes_fallback_grid(tmp_path):
    y, sr = _click_track(100.0, seconds=20.0)
    p = str(tmp_path / "c100.wav")
    sf.write(p, y, sr)
    out, repaired = repair_beat_grid(p, {"bpm": 120.0, "beats_per_bar": 4, "first_downbeat": 0.0})
    assert repaired is True
    assert abs(out["bpm"] - 100.0) / 100.0 < 0.03
    assert out["first_downbeat"] >= 0.0
