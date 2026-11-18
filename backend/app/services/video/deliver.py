"""Delivery operations on an existing master (plans 7.15, 7.17).

`/videos/{job}/export/{preset}` re-delivers a finished master for a destination,
and `/videos/{job}/variants` derives other aspect ratios from the same master.
Both are non-destructive: they write new files and new takes, never overwrite the
master they read.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional, Tuple

from app.services.video.export_presets import DeliveryPreset, video_args, audio_args, container_args, probe_streams

logger = logging.getLogger("milimo.video.deliver")

ASPECT_SIZES = {
    "16:9": {720: (1280, 720), 1080: (1920, 1080), 2160: (3840, 2160)},
    "9:16": {720: (720, 1280), 1080: (1080, 1920)},
    "1:1": {720: (720, 720), 1080: (1080, 1080)},
    "21:9": {720: (1680, 720), 1080: (2560, 1080)},
}


def size_for(aspect: str, resolution: str = "1080p") -> Optional[Tuple[int, int]]:
    """Target pixels for an aspect + resolution, or None when unsupported."""
    height = 2160 if resolution == "2160p" else 1080 if resolution == "1080p" else 720
    table = ASPECT_SIZES.get(aspect)
    if not table:
        return None
    return table.get(height) or table.get(720)


def aspect_filter(width: int, height: int, *, mode: str = "crop") -> str:
    """Fit a source into width×height: `crop` fills the frame, `pad` letterboxes."""
    if mode == "pad":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )


def transcode(
    src: str,
    dest: str,
    preset: DeliveryPreset,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: str = "crop",
    start: Optional[float] = None,
    duration: Optional[float] = None,
) -> bool:
    """Re-deliver `src` as `dest` for a preset. Returns False (with a log) on failure."""
    if not src or not os.path.isfile(src):
        return False

    src_streams = probe_streams(src) or {}
    target_w = width or preset.max_width or src_streams.get("width")
    target_h = height or preset.max_height or src_streams.get("height")
    if not target_w or not target_h:
        return False

    cmd: List[str] = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", f"{float(start):.3f}"]
    cmd += ["-i", src]

    filters = [aspect_filter(target_w, target_h, mode=mode)]
    if preset.loudness_lufs is not None and src_streams.get("audio_codec"):
        filters.append(
            f"loudnorm=I={preset.loudness_lufs}:TP={preset.loudness_tp or -1.0}:LRA=11"
        )
    # Video and audio filters travel in one graph so the delivery is a single encode.
    graph = filters[0]
    cmd += ["-vf", graph]
    if len(filters) > 1:
        cmd += ["-af", filters[1]]

    # `-af` and `-c:a copy` are mutually exclusive: when loudness is applied the
    # audio must be re-encoded, so the AAC passthrough is disabled.
    apply_loudness = preset.loudness_lufs is not None and bool(src_streams.get("audio_codec"))
    copy_aac = (src_streams.get("audio_codec") or "").lower() == "aac" and not apply_loudness

    cmd += video_args(preset)
    cmd += audio_args(preset, source_is_aac=copy_aac)
    cmd += container_args(preset)
    if preset.id == "canvas":
        # Spotify Canvas is 3-8s and silent.
        cmd += ["-t", f"{min(8.0, float(duration or 8.0)):.3f}", "-an"]
    elif duration:
        cmd += ["-t", f"{float(duration):.3f}"]
    cmd += [dest]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Export failed to start: %s", exc)
        return False
    if proc.returncode != 0 or not os.path.isfile(dest) or os.path.getsize(dest) == 0:
        logger.warning(
            "Export %s failed (%s): %s",
            preset.id, proc.returncode,
            (proc.stderr or b"")[-400:].decode("utf-8", errors="ignore"),
        )
        return False
    return True
