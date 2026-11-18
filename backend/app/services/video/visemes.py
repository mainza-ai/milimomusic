"""Viseme timing for the procedural jaw-warp (plan 8.13).

The always-available avatar drove the mouth from a loudness envelope, so it opened
on a loud guitar and stayed shut on a quiet sustained vowel. The karaoke engine
already produces MMS forced-alignment word timings for the subtitles; using them to
gate articulation is the cheapest real quality win in the lip-sync stack — no
weights, no new dependency, works on any laptop.

The gate is deliberately simple and testable: 1 while a syllable is being sung,
0 in the gaps, with short attack/release ramps so the jaw does not click. Loudness
still modulates *how far* the jaw opens; the gate decides *whether* it moves.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Ramps are in seconds; ~60 ms is short enough to read as articulation and long
#: enough to hide a frame-level jump at 25 fps.
ATTACK_S = 0.06
RELEASE_S = 0.08
#: Syllable rate used to shape openness inside a word. Sung English sits around
#: 4–6 syllables/second; this is a modulation, not a transcription.
SYLLABLE_RATE_HZ = 4.5


def word_intervals(
    timed_lines: Optional[Iterable[Dict[str, Any]]],
    *,
    start_time: float,
    duration: float,
) -> List[Tuple[float, float]]:
    """Absolute (start, end) intervals of sung words overlapping a clip window.

    Accepts the `timed_lyrics` payload the aligner emits: lines with optional
    `words: [{word, start, end}]`. Section markers are skipped — they are not sung.
    Falls back to line spans when a line carries no word timings.
    """
    window_start = float(start_time)
    window_end = float(start_time) + float(duration)
    intervals: List[Tuple[float, float]] = []

    for line in timed_lines or []:
        if line.get("is_section"):
            continue
        words = line.get("words") or []
        spans: List[Tuple[float, float]] = []
        for word in words:
            try:
                w_start = float(word.get("start"))
                w_end = float(word.get("end"))
            except (TypeError, ValueError):
                continue
            if w_end > w_start:
                spans.append((w_start, w_end))
        if not spans:
            try:
                l_start = float(line.get("start"))
                l_end = float(line.get("end"))
            except (TypeError, ValueError):
                continue
            if l_end > l_start:
                spans.append((l_start, l_end))
        for w_start, w_end in spans:
            if w_end <= window_start or w_start >= window_end:
                continue
            intervals.append((max(w_start, window_start), min(w_end, window_end)))

    intervals.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.02:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _ramp(distance: float, length: float) -> float:
    if length <= 0:
        return 1.0 if distance <= 0 else 0.0
    return max(0.0, min(1.0, distance / length))


def articulation_weights(
    *,
    total_frames: int,
    fps: float,
    start_time: float,
    intervals: Sequence[Tuple[float, float]],
    attack_s: float = ATTACK_S,
    release_s: float = RELEASE_S,
) -> List[float]:
    """Per-frame mouth gate in [0, 1].

    Inside a sung interval the value ramps to 1 and oscillates with a syllable-like
    modulation; outside it ramps to 0. An empty interval list means "no timing
    known" and the caller should not gate at all — this function returns all 1.0 so
    the behaviour degrades to the old envelope-only path rather than muting the
    performer.
    """
    if not intervals:
        return [1.0] * max(0, total_frames)

    fps = float(fps) if fps and fps > 0 else 25.0
    weights: List[float] = []
    for frame in range(max(0, total_frames)):
        t = float(start_time) + (frame / fps)
        weight = 0.0
        for w_start, w_end in intervals:
            if w_start <= t <= w_end:
                # Ramp in from the word start, out towards its end.
                weight = min(
                    _ramp(t - w_start, attack_s),
                    _ramp(w_end - t, release_s),
                )
                if weight > 0:
                    modulation = 0.72 + 0.28 * math.sin(2 * math.pi * SYLLABLE_RATE_HZ * (t - w_start))
                    weight *= modulation
                break
        weights.append(max(0.0, min(1.0, weight)))
    return weights


def gate_speech(
    envelope: Sequence[float],
    weights: Sequence[float],
) -> List[float]:
    """Multiply a loudness envelope by the articulation gate."""
    if not weights:
        return list(envelope)
    length = max(len(envelope), len(weights))
    gated: List[float] = []
    for index in range(length):
        env = float(envelope[index]) if index < len(envelope) else 0.0
        gate = float(weights[index]) if index < len(weights) else 0.0
        gated.append(max(0.0, min(1.0, env * gate)))
    return gated
