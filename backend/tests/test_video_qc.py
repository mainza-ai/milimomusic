"""QC gate: sub-second faults must be caught (plan 7.14).

FFmpeg's `blackdetect`/`freezedetect`/`silencedetect` default to a 2 s minimum.
A stitched music video rarely produces a 2 s black gap — it produces 0.2 s ones at
a failed cut. These tests plant faults of that size and require the gate to see
them.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.services.video import qc

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _make(path, *, duration=3.0, black_window=None, silent_window=None, audio=True, static=False):
    """Synthesise a small clip, optionally with a planted fault."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if static:
        cmd += ["-f", "lavfi", "-i", f"color=c=gray:s=160x120:d={duration}:r=25"]
    else:
        cmd += ["-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={duration}"]

    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}:sample_rate=44100"]

    vf = []
    if black_window:
        start, end = black_window
        vf.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:"
            f"enable='between(t,{start},{end})'"
        )
    if vf:
        cmd += ["-vf", ",".join(vf)]

    if audio and silent_window:
        start, end = silent_window
        cmd += ["-af", f"volume=0:enable='between(t,{start},{end})'"]

    if not audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    assert os.path.isfile(path) and os.path.getsize(path) > 0
    return str(path)


def test_a_200ms_black_gap_is_detected(tmp_path):
    """0.2 s is invisible to FFmpeg's 2 s default and would ship silently."""
    clip = _make(tmp_path / "black.mp4", black_window=(1.0, 1.2))
    gaps = qc.detect_black(clip)
    assert gaps, "blackdetect missed a 200 ms black gap"
    assert 0.1 <= max(g["duration"] for g in gaps) <= 0.4


def test_a_200ms_audio_dropout_is_detected(tmp_path):
    clip = _make(tmp_path / "silent.mp4", silent_window=(1.0, 1.2))
    starts = qc.detect_silence(clip)
    assert starts, "silencedetect missed a 200 ms dropout"
    assert any(abs(s["start"] - 1.0) < 0.25 for s in starts)


def test_a_static_clip_reports_a_freeze(tmp_path):
    clip = _make(tmp_path / "freeze.mp4", duration=2.0, static=True)
    freezes = qc.detect_freeze(clip)
    assert freezes, "freezedetect missed a static clip"
    # A freeze that runs to EOF reports a start with no duration.
    assert freezes[0]["start"] is not None


def test_missing_audio_is_fatal(tmp_path):
    clip = _make(tmp_path / "noaudio.mp4", audio=False)
    report = qc.run_qc(clip)
    assert report["ok"] is False
    assert "missing_audio" in report["fatal"]


def test_duration_mismatch_is_a_warning_not_a_failure(tmp_path):
    clip = _make(tmp_path / "short.mp4", duration=3.0)
    report = qc.run_qc(clip, expected_duration=8.0)
    assert report["ok"] is True
    assert any("differs from song" in w for w in report["warnings"])


def test_report_is_json_serialisable_and_declares_sync_limits(tmp_path):
    import json

    clip = _make(tmp_path / "ok.mp4", duration=2.0)
    report = qc.run_qc(clip, expected_duration=2.0, expected_width=160, expected_height=120)
    json.dumps(report)  # must not raise
    assert report["ok"] is True
    assert report["fatal"] == []
    drift = report["checks"]["av_drift"]
    assert "perceptual lip sync requires human review" in drift["note"].lower()
    # Every detector must run at sub-second thresholds, not the 2 s defaults.
    assert report["thresholds"]["black_min_s"] < 2.0
    assert report["thresholds"]["freeze_min_s"] < 2.0


def test_missing_file_is_fatal_not_an_exception():
    report = qc.run_qc("/nonexistent/file.mp4")
    assert report["ok"] is False
    assert "missing_output" in report["fatal"]
