"""Clip duration/geometry normalisation (plan 4.1 / 4.2).

The audit's P0: diffusion emits frame counts on a lattice and rounds *down*, so a
3.7 s plan renders as 3.56 s; nothing consumed the trim the planner computed, a
40-clip video finished seconds before the song, and every later cut drifted off
its beat. Mixed model frame rates/resolutions could also not be concatenated.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from app.services.video.clip_normalise import (
    DURATION_TOLERANCE_S,
    normalise_clip,
    probe_duration,
    verify_assembly,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_clip(path: str, seconds: float, size: str = "320x240", fps: int = 25) -> str:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            path,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe required")
def test_probe_duration_reads_seconds(tmp_path):
    src = _make_clip(str(tmp_path / "a.mp4"), 2.0)
    assert abs((probe_duration(src) or 0) - 2.0) < 0.1


def test_probe_duration_returns_none_for_missing():
    assert probe_duration("/nope/does-not-exist.mp4") is None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe required")
def test_normalise_pads_a_short_clip_to_target(tmp_path):
    """The lattice-loss case: a clip renders short and must be padded, not accepted."""
    src = _make_clip(str(tmp_path / "short.mp4"), 1.4)
    dest = str(tmp_path / "norm.mp4")

    ok = normalise_clip(src, dest, duration_s=2.0, width=640, height=360, fps=16)
    assert ok
    assert abs((probe_duration(dest) or 0) - 2.0) <= DURATION_TOLERANCE_S


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe required")
def test_normalise_trims_a_long_clip(tmp_path):
    src = _make_clip(str(tmp_path / "long.mp4"), 3.0)
    dest = str(tmp_path / "norm.mp4")

    ok = normalise_clip(src, dest, duration_s=1.5, width=640, height=360, fps=16)
    assert ok
    assert abs((probe_duration(dest) or 0) - 1.5) <= DURATION_TOLERANCE_S


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe required")
def test_normalise_unifies_geometry_across_sources(tmp_path):
    """A 320x240@25 clip and a 640x360@16 clip must come out identical in shape."""
    a = normalise_clip(
        _make_clip(str(tmp_path / "a.mp4"), 1.0, size="320x240", fps=25),
        str(tmp_path / "a_norm.mp4"),
        duration_s=1.0, width=640, height=360, fps=16,
    )
    b = normalise_clip(
        _make_clip(str(tmp_path / "b.mp4"), 1.0, size="640x360", fps=16),
        str(tmp_path / "b_norm.mp4"),
        duration_s=1.0, width=640, height=360, fps=16,
    )
    assert a and b

    def _dims(path):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()

    assert _dims(str(tmp_path / "a_norm.mp4")) == _dims(str(tmp_path / "b_norm.mp4"))


def test_normalise_refuses_missing_input(tmp_path):
    assert normalise_clip(str(tmp_path / "nope.mp4"), str(tmp_path / "o.mp4"),
                          duration_s=1.0, width=64, height=64, fps=16) is False


def test_verify_assembly_reports_drift():
    report = verify_assembly(59.4, 60.0)
    assert report is not None
    assert report["ok"] is False
    assert abs(report["drift_s"] + 0.6) < 1e-6

    ok_report = verify_assembly(60.0, 60.0)
    assert ok_report["ok"] is True


def test_verify_assembly_is_safe_without_inputs():
    assert verify_assembly(None, 60.0) is None
    assert verify_assembly(60.0, None) is None


def test_transition_overlap_matches_stitcher_math():
    """Normalisation must reserve exactly what the xfade stitcher will consume."""
    from app.services.video.video_orchestrator import video_orchestrator

    class Clip:
        def __init__(self, d):
            self.duration = d

    clips = [Clip(4.0), Clip(4.0), Clip(4.0)]
    # 25% of the shortest clip, capped at 0.5s
    assert video_orchestrator._transition_overlap("crossfade", clips) == 0.5
    # hard cuts reserve nothing
    assert video_orchestrator._transition_overlap("beat_cut", clips) == 0.0
    # a single clip has no transition
    assert video_orchestrator._transition_overlap("crossfade", [Clip(4.0)]) == 0.0
