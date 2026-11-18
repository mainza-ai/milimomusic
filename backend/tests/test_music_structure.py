"""Musical analysis truth tests (plans 7.8, 7.9).

Downbeats used to be `beats[::4]`; sections used to be 16 s RMS windows. Both are
now derived from evidence: low-band beat accents for metre, and the song's own
generation tags for structure.
"""

from __future__ import annotations

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from app.services.video.audio_analysis import (  # noqa: E402
    AudioSignalAnalyzer,
    _normalise_tag,
)


def _kick_track(bpm: float, bars: int, meter: int, sr: int = 22050):
    """Synthesise a track whose bar downbeat is loud and the rest are quiet.

    Only the low band carries the accent, exactly like a real kick pattern.
    """
    beat = 60.0 / bpm
    total = int(sr * beat * meter * bars)
    y = np.zeros(total, dtype=np.float32)
    kick_len = int(sr * 0.12)
    t = np.arange(kick_len) / sr
    env = np.exp(-t * 30)
    tone = np.sin(2 * np.pi * 55 * t) * env

    for b in range(bars * meter):
        start = int(b * beat * sr)
        amp = 0.9 if b % meter == 0 else 0.15
        end = min(total, start + kick_len)
        y[start:end] += (tone[: end - start] * amp).astype(np.float32)
    return y, sr


def test_metre_is_three_for_a_waltz():
    y, sr = _kick_track(bpm=120, bars=16, meter=3)
    tempo, beats, downbeats, meter = AudioSignalAnalyzer._extract_beats_and_downbeats(y, sr)
    assert meter == 3, f"3/4 track estimated as {meter}/4"
    assert len(downbeats) < len(beats)
    # Downbeats must land on the accent pattern, not every fourth beat.
    gaps = np.diff(downbeats)
    if len(gaps) > 1:
        median_gap = float(np.median(gaps))
        assert abs(median_gap - 3 * (60.0 / tempo)) < 0.15


def test_metre_is_four_for_a_four_four_track():
    y, sr = _kick_track(bpm=100, bars=12, meter=4)
    _, beats, downbeats, meter = AudioSignalAnalyzer._extract_beats_and_downbeats(y, sr)
    assert meter == 4
    assert downbeats and downbeats[0] == pytest.approx(beats[0], abs=1e-6) or downbeats


def test_too_few_beats_falls_back_to_four_four_without_raising():
    meter, phase = AudioSignalAnalyzer._estimate_meter(np.zeros(1000, dtype=np.float32), 22050, [0.1, 0.2])
    assert (meter, phase) == (4, 0)


# ── structure from generation tags ───────────────────────────────────────

def _timed(tag_lines):
    """timed_lyrics payload with section markers, as the aligner emits it."""
    return [
        {"text": f"[{tag}]", "start": start, "end": start + 0.1, "is_section": True, "words": []}
        for tag, start in tag_lines
    ]


def test_sections_come_from_the_generation_tags():
    timed = _timed([("Intro", 0.0), ("Verse", 8.0), ("Chorus", 32.0), ("Bridge", 56.0), ("Outro", 72.0)])
    y = np.zeros(22050 * 90, dtype=np.float32)
    sections, source = AudioSignalAnalyzer._extract_sections(y, 22050, 90.0, timed, lyrics=None)

    assert source == "lyrics_tags"
    assert [s["label"] for s in sections] == ["Intro", "Verse", "Chorus", "Bridge", "Outro"]
    assert sections[1]["start"] == 8.0
    assert sections[2]["end"] == 56.0
    assert all(s.get("source") == "lyrics_tags" for s in sections)


def test_tag_structure_survives_a_leading_gap():
    """A song that starts singing at 4 s gets a real Intro section first."""
    timed = _timed([("Verse", 4.0), ("Chorus", 20.0)])
    y = np.zeros(22050 * 40, dtype=np.float32)
    sections, source = AudioSignalAnalyzer._extract_sections(y, 22050, 40.0, timed, lyrics="[Verse]\n")
    assert source == "lyrics_tags"
    assert sections[0]["label"] == "Intro"
    assert sections[0]["start"] == 0.0 and sections[0]["end"] == 4.0


def test_pre_chorus_and_hook_tags_are_recognised():
    timed = _timed([("Verse", 0.0), ("Pre-Chorus", 10.0), ("Hook", 20.0), ("Chorus", 30.0)])
    y = np.zeros(22050 * 40, dtype=np.float32)
    sections, source = AudioSignalAnalyzer._extract_sections(y, 22050, 40.0, timed)
    assert source == "lyrics_tags"
    assert [s["label"] for s in sections] == ["Verse", "Pre-Chorus", "Hook", "Chorus"]


def test_unknown_tags_are_ignored_not_invented():
    timed = _timed([("Verse", 0.0), ("Kazoo Solo", 10.0), ("Chorus", 20.0)])
    y = np.zeros(22050 * 30, dtype=np.float32)
    sections, source = AudioSignalAnalyzer._extract_sections(y, 22050, 30.0, timed)
    assert source == "lyrics_tags"
    assert "Kazoo Solo" not in [s["label"] for s in sections]


def test_without_section_markers_structure_falls_back_to_energy():
    y = np.zeros(22050 * 30, dtype=np.float32)
    sections, source = AudioSignalAnalyzer._extract_sections(y, 22050, 30.0, [{"text": "hello", "start": 1.0, "end": 2.0}])
    assert source == "energy_windows"
    assert sections


def test_tag_normalisation():
    assert _normalise_tag("[Pre-Chorus]") == "pre-chorus"
    assert _normalise_tag("(HOOK)") == "hook"
    assert _normalise_tag("  [Verse 2]  ") == "verse-2"
