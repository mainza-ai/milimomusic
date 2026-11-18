"""Articulation gating for the procedural avatar (plan 8.13).

The jaw-warp used to open on loudness: a loud guitar moved the mouth, a quiet
sustained vowel did not. These tests pin the timing maths and the fallback
behaviour (no timings ⇒ unchanged envelope path).
"""

from __future__ import annotations

import pytest

from app.services.video import visemes


def _line(words, is_section=False):
    return {
        "text": " ".join(w["word"] for w in words) if words else "[Verse]",
        "start": words[0]["start"] if words else 0.0,
        "end": words[-1]["end"] if words else 0.0,
        "words": words,
        "is_section": is_section,
    }


def test_word_intervals_use_word_level_timings():
    timed = [_line([
        {"word": "hold", "start": 1.0, "end": 1.4},
        {"word": "on", "start": 1.5, "end": 1.7},
    ])]
    intervals = visemes.word_intervals(timed, start_time=0.0, duration=5.0)
    assert intervals == [(1.0, 1.4), (1.5, 1.7)]


def test_intervals_are_clipped_to_the_clip_window():
    timed = [_line([{"word": "long", "start": 2.0, "end": 6.0}])]
    intervals = visemes.word_intervals(timed, start_time=4.0, duration=1.0)
    assert intervals == [(4.0, 5.0)]


def test_section_markers_are_not_sung():
    timed = [
        _line([], is_section=True),
        _line([{"word": "sing", "start": 2.0, "end": 2.5}]),
    ]
    intervals = visemes.word_intervals(timed, start_time=0.0, duration=5.0)
    assert intervals == [(2.0, 2.5)]


def test_lines_without_word_timings_fall_back_to_line_spans():
    timed = [{"text": "a line", "start": 1.0, "end": 2.0, "words": []}]
    assert visemes.word_intervals(timed, start_time=0.0, duration=5.0) == [(1.0, 2.0)]


def test_adjacent_words_merge_into_one_interval():
    timed = [_line([
        {"word": "one", "start": 1.0, "end": 1.5},
        {"word": "two", "start": 1.51, "end": 2.0},
    ])]
    assert visemes.word_intervals(timed, start_time=0.0, duration=5.0) == [(1.0, 2.0)]


def test_gate_is_open_on_words_and_closed_in_gaps():
    weights = visemes.articulation_weights(
        total_frames=100, fps=25.0, start_time=0.0,
        intervals=[(1.0, 2.0)],
    )
    # Frame 25 ≈ 1.0 s (start of the word), frame 50 ≈ 2.0 s (just past its end).
    assert weights[10] == 0.0, "before the word the mouth must be shut"
    assert max(weights[30:45]) > 0.4, "inside the word the mouth must move"
    assert weights[60] == 0.0, "after the word the mouth must shut again"


def test_gate_ramps_rather_than_clicking():
    weights = visemes.articulation_weights(
        total_frames=60, fps=25.0, start_time=0.0, intervals=[(1.0, 2.0)],
    )
    # Just after the word starts the gate is still ramping up.
    at_start = weights[25]          # t = 1.00 s
    ramped = weights[27]            # t = 1.08 s
    assert at_start < ramped, "the gate must ramp in, not jump"
    assert 0.0 <= min(weights) and max(weights) <= 1.0


def test_no_timings_means_no_gating():
    """Unknown timing must degrade to the old envelope path, not mute the singer."""
    weights = visemes.articulation_weights(
        total_frames=10, fps=25.0, start_time=0.0, intervals=[],
    )
    assert weights == [1.0] * 10


def test_gate_speech_multiplies_and_clamps():
    assert visemes.gate_speech([1.0, 1.0, 1.0], [1.0, 0.0, 0.5]) == [1.0, 0.0, 0.5]
    assert visemes.gate_speech([1.5], [0.8]) == [pytest.approx(1.0)]


def test_gate_speech_tolerates_length_mismatch():
    out = visemes.gate_speech([1.0], [1.0, 1.0])
    assert len(out) == 2
    assert out[0] == 1.0
