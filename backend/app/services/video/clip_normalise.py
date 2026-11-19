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


#: Intermediate quality. These clips are stitched and then encoded once more for
#: delivery, so the intermediate must not be the lossy step (audit D1).
INTERMEDIATE_CRF = 12


def normalise_clip(
    src: str,
    dest: str,
    *,
    duration_s: float,
    width: int,
    height: int,
    fps: int,
    extra_vf: Optional[str] = None,
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
    if extra_vf:
        # Beat-synced exposure pulse, applied here instead of in a second encode.
        vf = f"{vf},{extra_vf}"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-an",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(INTERMEDIATE_CRF),
        "-pix_fmt", "yuv420p", "-r", str(fps),
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


def interpolation_filter(src_fps: Optional[float], target_fps: Optional[float]) -> Optional[str]:
    """Frame-rate conversion filter for delivery (audit G5).

    Diffusion models emit 16 fps (Wan 2.1) or 24/25 fps; delivery is 24 or 30. The
    old path forced the output rate with `fps=` — frame duplication, which reads as
    judder on pans and on beat cuts. `minterpolate` synthesises intermediate frames
    instead. Returns None when no conversion is needed.
    """
    if not src_fps or not target_fps:
        return None
    if abs(float(src_fps) - float(target_fps)) < 0.01:
        return None
    # Motion-compensated interpolation (mci) synthesises real intermediate frames;
    # `blend` merely dissolves neighbours, which is what softened fast pans. mci is
    # slower, so `MILIMO_INTERP_MODE=blend` remains available as the fast path.
    import os as _os

    mode = (_os.environ.get("MILIMO_INTERP_MODE") or "mci").strip().lower()
    if mode == "blend":
        return f"minterpolate=fps={float(target_fps):.3f}:mi_mode=blend"
    return (
        f"minterpolate=fps={float(target_fps):.3f}:mi_mode=mci:"
        f"mc_mode=obmc:me_mode=bidir:vsbmc=1"
    )


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


def make_preview_proxy(src: str, dest: str, *, height: int = 480) -> bool:
    """Encode a small H.264 proxy for timeline scrubbing (plan 4.15).

    The editor streamed the full 1080p master with `preload="auto"`, so scrubbing
    a 3-minute video meant pulling the whole file. A 480p proxy makes seeking
    responsive and cheap.
    """
    if not os.path.isfile(src) or os.path.getsize(src) == 0:
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        dest,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        logger.warning(
            "Proxy encode failed for %s: %s", os.path.basename(src),
            detail.decode("utf-8", "ignore")[:200] if isinstance(detail, bytes) else detail,
        )
        return False
    return os.path.isfile(dest) and os.path.getsize(dest) > 0


def first_frame_poster(src: str, dest: str) -> bool:
    """Extract the first frame as a JPEG poster (the UI used album art instead)."""
    if not os.path.isfile(src):
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src, "-frames:v", "1", "-q:v", "3", dest,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return os.path.isfile(dest) and os.path.getsize(dest) > 0
