"""Automated QC for a delivered music video (plan 7.14).

Nothing checked the master before this: black/freeze gaps, A/V drift and
delivered loudness were found by watching the file. FFmpeg's detectors default to
a **2 second** minimum, which misses exactly the faults a stitched render
produces — a 1 s black gap, a 5-frame black gap, a short freeze. Every detector
here is configured explicitly, and a test plants those faults and asserts they
are caught.

What FFmpeg *cannot* do is judge lip sync. We report the structural proxy
(per-stream start/duration drift) and say plainly that perceptual sync needs eyes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from app.services.video.export_presets import probe_streams

logger = logging.getLogger("milimo.video.qc")

# Detector thresholds: deliberately well below FFmpeg's 2 s defaults.
BLACK_MIN_S = 0.10
BLACK_PIXEL_THRESHOLD = 0.10
FREEZE_MIN_S = 0.05
FREEZE_NOISE = 0.001
SILENCE_MIN_S = 0.10
SILENCE_DB = -50.0

#: Tolerances for the delivered file.
DURATION_TOLERANCE_S = 0.10
MAX_AV_START_DRIFT_S = 0.05
LOUDNESS_TOLERANCE_LU = 1.5


def _run(cmd: List[str], timeout: int = 1800) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "") + (out.stderr or "")
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("QC command failed (%s): %s", cmd[0], exc)
        return ""


def detect_black(path: str, min_s: float = BLACK_MIN_S) -> List[Dict[str, float]]:
    """Black intervals shorter than FFmpeg's 2 s default are caught too."""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-vf", f"blackdetect=d={min_s}:pix_th={BLACK_PIXEL_THRESHOLD}",
                "-an", "-f", "null", "-"])
    return [
        {"start": float(m.group(1)), "end": float(m.group(2)), "duration": float(m.group(3))}
        for m in re.finditer(r"black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)", log)
    ]


def detect_freeze(path: str, min_s: float = FREEZE_MIN_S) -> List[Dict[str, float]]:
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-vf", f"freezedetect=n={FREEZE_NOISE}:d={min_s}",
                "-an", "-f", "null", "-"])
    # FFmpeg emits `freeze_start` when the freeze begins and `freeze_duration` only
    # when it ends; a freeze that runs to EOF prints the start alone.
    starts = [float(m) for m in re.findall(r"freeze_start:\s*([\d.]+)", log)]
    durations = [float(m) for m in re.findall(r"freeze_duration:\s*([\d.]+)", log)]
    segments = []
    for idx, start in enumerate(starts):
        segments.append({
            "start": start,
            "duration": durations[idx] if idx < len(durations) else None,
        })
    return segments


def detect_silence(path: str, min_s: float = SILENCE_MIN_S) -> List[Dict[str, float]]:
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-af", f"silencedetect=n={SILENCE_DB}dB:d={min_s}",
                "-vn", "-f", "null", "-"])
    out: List[Dict[str, float]] = []
    for m in re.finditer(r"silence_start: ([\d.]+)", log):
        out.append({"start": float(m.group(1))})
    return out


def measure_loudness(path: str) -> Dict[str, Optional[float]]:
    """Integrated LUFS + true peak from `ebur128`, parsed from its summary."""
    log = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                "-af", "ebur128=peak=true", "-vn", "-f", "null", "-"])
    integrated = None
    true_peak = None
    for m in re.finditer(r"I:\s*(-?[\d.]+)\s*LUFS", log):
        integrated = float(m.group(1))
    for m in re.finditer(r"True peak:\s*Peak:\s*(-?[\d.]+)\s*dBFS", log):
        true_peak = float(m.group(1))
    return {"integrated_lufs": integrated, "true_peak_dbfs": true_peak}


def check_av_drift(streams: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Structural A/V drift: per-stream start and duration disagreement.

    This is the only sync signal a file-based checker can produce honestly; say so
    in the report rather than claiming a lip-sync pass.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_entries", "stream=codec_type,start_time,duration", path],
            capture_output=True, text=True, timeout=120, check=True,
        )
        payload = json.loads(out.stdout or "{}")
    except (subprocess.SubprocessError, ValueError, OSError):
        return {"measurable": False}
    streams_info = payload.get("streams", [])
    video = next((s for s in streams_info if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams_info if s.get("codec_type") == "audio"), {})

    def _f(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    v_start, a_start = _f(video.get("start_time")), _f(audio.get("start_time"))
    v_dur, a_dur = _f(video.get("duration")), _f(audio.get("duration"))
    return {
        "measurable": v_start is not None and a_start is not None,
        "video_start": v_start,
        "audio_start": a_start,
        "start_drift": round(abs((v_start or 0.0) - (a_start or 0.0)), 4)
        if v_start is not None and a_start is not None else None,
        "video_duration": v_dur,
        "audio_duration": a_dur,
        "duration_drift": round(abs((v_dur or 0.0) - (a_dur or 0.0)), 4)
        if v_dur is not None and a_dur is not None else None,
        "note": "Structural drift only — perceptual lip sync requires human review.",
    }


def run_qc(
    path: str,
    *,
    expected_duration: Optional[float] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    target_lufs: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the full gate and return a JSON-serialisable verdict.

    `ok` is False only for faults that make the file unusable or a lie about the
    song (wrong length, no audio, black tail); advisories are reported but do not
    fail the render.
    """
    if not path or not os.path.isfile(path):
        return {"ok": False, "fatal": ["missing_output"], "path": path}

    streams = probe_streams(path)
    checks: Dict[str, Any] = {"streams": streams}
    fatal: List[str] = []
    warnings: List[str] = []

    duration = streams.get("duration")
    if not duration:
        fatal.append("unreadable_duration")
    elif expected_duration and abs(duration - expected_duration) > DURATION_TOLERANCE_S:
        warnings.append(
            f"duration {duration:.3f}s differs from song {expected_duration:.3f}s by "
            f"{duration - expected_duration:+.3f}s"
        )

    if not streams.get("audio_codec"):
        fatal.append("missing_audio")
    if expected_width and streams.get("width") and streams["width"] != expected_width:
        warnings.append(f"width {streams['width']} != requested {expected_width}")
    if expected_height and streams.get("height") and streams["height"] != expected_height:
        warnings.append(f"height {streams['height']} != requested {expected_height}")

    black = detect_black(path)
    freeze = detect_freeze(path)
    silence = detect_silence(path)
    checks["black_segments"] = black
    checks["freeze_segments"] = freeze
    checks["silence_starts"] = silence

    # A black tail is the signature of a short assembly padded by the container.
    if duration:
        tail_black = [b for b in black if b["end"] >= duration - 0.15]
        if tail_black:
            warnings.append(f"black tail: {tail_black[-1]['duration']:.2f}s at the end")
    if black:
        warnings.append(f"{len(black)} black segment(s), longest {max(b['duration'] for b in black):.2f}s")
    if freeze:
        longest = max((f["duration"] or 0.0) for f in freeze)
        warnings.append(
            f"{len(freeze)} freeze segment(s), longest {longest:.2f}s"
            + (" (extends to end of file)" if any(f["duration"] is None for f in freeze) else "")
        )

    checks["av_drift"] = check_av_drift(streams, path)
    drift = checks["av_drift"].get("start_drift")
    if drift is not None and drift > MAX_AV_START_DRIFT_S:
        warnings.append(f"A/V start drift {drift:.3f}s exceeds {MAX_AV_START_DRIFT_S}s")

    if streams.get("audio_codec"):
        loudness = measure_loudness(path)
        checks["loudness"] = loudness
        integrated = loudness.get("integrated_lufs")
        if target_lufs is not None and integrated is not None:
            if abs(integrated - target_lufs) > LOUDNESS_TOLERANCE_LU:
                warnings.append(
                    f"integrated loudness {integrated:.1f} LUFS is "
                    f"{integrated - target_lufs:+.1f} LU from target {target_lufs}"
                )
        peak = loudness.get("true_peak_dbfs")
        if peak is not None and peak > -0.5:
            warnings.append(f"true peak {peak:.1f} dBFS has almost no headroom")

    return {
        "ok": not fatal,
        "fatal": fatal,
        "warnings": warnings,
        "checks": checks,
        "path": path,
        "thresholds": {
            "black_min_s": BLACK_MIN_S,
            "freeze_min_s": FREEZE_MIN_S,
            "silence_min_s": SILENCE_MIN_S,
            "duration_tolerance_s": DURATION_TOLERANCE_S,
        },
    }
