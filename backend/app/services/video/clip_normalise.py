"""
Clip normalisation and duration repair for video assembly (plan 4.1 / 4.2).

Two defects made assembled videos drift away from the song:

1. **Lattice loss.** Diffusion models only emit frame counts on a discrete lattice
   (`min_frames + k * frame_step`), and the generators round *down*. A 3.7 s plan
   renders as 3.56 s. The planner already computed the trim that would fix this
   (`director_music_timing.music_output_trim`) but nothing consumed it, so ~0.1–0.25 s
   was lost per clip and a 40-clip video finished seconds early.

2. **Mixed geometry.** Wan on MPS silently clamps to ≤832×480 / 16 fps while the
   procedural and lip-sync paths render 25 fps at other sizes. Concatenating them
   produces corrupted frames or a hard failure.

`normalise_clip` fixes one clip to an exact duration *and* a common geometry, so
assembly is a straight concat with no drift and no format surprises.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger("milimo.video.normalise")

# Allow the delivered video to differ slightly without triggering a report.
DURATION_TOLERANCE_S = 0.05


def probe_duration(path: str) -> Optional[float]:
    """Duration in seconds via ffprobe, or None when it cannot be read."""
    if not path or not os.path.isfile(path):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def normalise_clip(
    src: str,
    dest: str,
    *,
    duration_s: float,
    width: int,
    height: int,
    fps: int,
) -> bool:
    """Rewrite `src` as `dest` with an exact duration and uniform geometry.

    Strategy: scale/pad to the project frame size, force the project fps and a
    square pixel aspect, then trim-or-pad the video track to `duration_s`. Padding
    the last frame (rather than stretching) keeps motion speed honest — a clip that
    rendered short is a clip that rendered short.
    """
    if not os.path.isfile(src) or os.path.getsize(src) == 0:
        return False
    if duration_s <= 0:
        return False

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},setsar=1,"
        # tpad duplicates the final frame if the clip is short; trim cuts if long.
        f"tpad=stop_mode=clone:stop_duration={duration_s:.3f},"
        f"trim=duration={duration_s:.3f},setpts=PTS-STARTPTS"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-an",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        logger.warning(
            "Clip normalisation failed for %s: %s", os.path.basename(src),
            detail.decode("utf-8", "ignore")[:200] if isinstance(detail, bytes) else detail,
        )
        return False

    got = probe_duration(dest)
    if got is None or abs(got - duration_s) > DURATION_TOLERANCE_S:
        logger.warning(
            "Normalised clip %s has duration %s (wanted %.3f)",
            os.path.basename(dest), got, duration_s,
        )
    return os.path.isfile(dest) and os.path.getsize(dest) > 0


def verify_assembly(total_video_s: Optional[float], song_s: Optional[float]) -> Optional[dict]:
    """Report whether the assembled picture matches the song length.

    Never raises: a mismatch is surfaced to the task status, not fatal.
    """
    if total_video_s is None or song_s is None or song_s <= 0:
        return None
    drift = total_video_s - song_s
    return {
        "video_duration_s": round(total_video_s, 3),
        "song_duration_s": round(song_s, 3),
        "drift_s": round(drift, 3),
        "ok": abs(drift) <= DURATION_TOLERANCE_S,
    }
