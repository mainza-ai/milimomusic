"""Splice-alignment regression tests for track extension.

Covers the "extension restarts the song" bug: the old length heuristic
(len(data_e) > cut + 10s) misclassified a SHORT tail on a full-timeline
render as a delta snippet and appended the regen from 0:00, so e.g.
cut 1:49 / target 1:58 produced intro material instead of the continuation.
The slicer now takes an explicit extension_is_full_timeline flag.
"""
import os
import sys
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.providers.minimax_provider import concatenate_and_crossfade_audio

SR = 8000
HEAD_VAL = 0.25   # parent / regen-head material
TAIL_VAL = 0.75   # genuinely new continuation material


def _wav(path, seconds, head_until=None):
    n = int(seconds * SR)
    data = np.ones((n, 2), dtype=np.float32) * HEAD_VAL
    if head_until is not None:
        data[int(head_until * SR):] = TAIL_VAL
    sf.write(str(path), data, SR)
    return str(path)


@pytest.fixture()
def workdir(tmp_path):
    return tmp_path


def _tail_mean(path, t0, t1):
    got, _ = sf.read(str(path), dtype="float32")
    return float(got[int(t0 * SR):int(t1 * SR), 0].mean())


def test_full_timeline_short_tail_slices_at_cut(workdir):
    """User scenario: cut 109s, target 118s — tail must be regen[cut:], not regen[0:]."""
    parent = _wav(workdir / "p.wav", 109.5)
    regen = _wav(workdir / "e.wav", 118.0, head_until=109.0)
    out = str(workdir / "out.wav")
    total = concatenate_and_crossfade_audio(
        parent, regen, out, 1.5, 109.0, None, 118.0, True)
    assert abs(total - 118.0) < 0.5
    assert abs(_tail_mean(out, 115, 117) - TAIL_VAL) < 0.08
    assert abs(_tail_mean(out, 5, 10) - HEAD_VAL) < 0.08


def test_full_timeline_long_tail_still_slices_at_cut(workdir):
    """Previously-working shape (cut 60s, target 120s) keeps working."""
    parent = _wav(workdir / "p.wav", 60.5)
    regen = _wav(workdir / "e.wav", 120.0, head_until=60.0)
    out = str(workdir / "out.wav")
    total = concatenate_and_crossfade_audio(
        parent, regen, out, 1.5, 60.0, None, 120.0, True)
    assert abs(total - 120.0) < 0.5
    assert abs(_tail_mean(out, 100, 110) - TAIL_VAL) < 0.08
    assert abs(_tail_mean(out, 5, 10) - HEAD_VAL) < 0.08


def test_delta_snippet_appends_from_zero(workdir):
    """Fresh-segment fallback renders only the delta: append from index 0."""
    parent = _wav(workdir / "p.wav", 109.5)
    delta = _wav(workdir / "d.wav", 10.5, head_until=0.0)  # all-new material
    out = str(workdir / "out.wav")
    concatenate_and_crossfade_audio(
        parent, delta, out, 1.5, 109.0, None, 118.0, False)
    got, _ = sf.read(out, dtype="float32")
    # Combined ~= parent + delta (trimmed/padded to target); the appended
    # section must be the delta's content from its own start.
    assert abs(float(got[int(112 * SR):int(114 * SR), 0].mean()) - TAIL_VAL) < 0.12


def test_output_trims_to_exact_target(workdir):
    """Overshoot is trimmed to the exact requested target duration."""
    parent = _wav(workdir / "p.wav", 60.5)
    regen = _wav(workdir / "e.wav", 130.0, head_until=60.0)
    out = str(workdir / "out.wav")
    total = concatenate_and_crossfade_audio(
        parent, regen, out, 1.5, 60.0, None, 120.0, True)
    assert abs(total - 120.0) < 0.5
