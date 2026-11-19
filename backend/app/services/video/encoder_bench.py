"""Measure the encoder instead of trusting the word "hardware" (plan 9.5).

`detect_hardware_encoder()` used to prefer `h264_videotoolbox` because the encoder
*exists* and could encode a black frame. Measured on the reference machine it is
**1.9× slower** than `libx264 -preset veryfast -crf 18` for 3 minutes of 1080p and
produces a larger file at a fixed bitrate. On an M1 Pro the literature agrees on wall
time and disagrees on cores: VideoToolbox uses ~0.9 cores against x264's ~6, which
makes it the right choice for *parallel* renders and the wrong one when the user is
waiting on a single render.

So: benchmark both on this machine, once, and cache the winner with the machine
fingerprint. An explicit override always wins.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("milimo.video.encoder")

CACHE_NAME = "encoder_choice.json"
BENCH_SECONDS = 1.0
BENCH_SIZE = "1920x1080"
#: Cached decisions older than this are re-measured (driver/OS updates move things).
CACHE_TTL_DAYS = 30

#: Encoder id -> ffmpeg arguments. VideoToolbox uses `-q:v 65`, which the literature
#: measures at ~VMAF 94 on camera content versus 93.2 at a higher bitrate.
CANDIDATES: Dict[str, list] = {
    "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"],
    "videotoolbox": ["-c:v", "h264_videotoolbox", "-q:v", "65"],
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"],
}


def _cache_path() -> str:
    from app.core.paths import get_data_dir

    return os.path.join(str(get_data_dir()), "video_cache", CACHE_NAME)


def machine_fingerprint() -> str:
    """What a benchmark result is valid for: machine + ffmpeg build."""
    version = ""
    binary = shutil.which("ffmpeg") or "ffmpeg"
    try:
        out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=10)
        version = (out.stdout or "").splitlines()[0][:80]
    except (subprocess.SubprocessError, OSError):
        version = "unknown"
    return f"{platform.platform()}|{platform.machine()}|{os.cpu_count()}|{version}"


def _encoder_available(encoder_id: str, binary: str = "ffmpeg") -> bool:
    """A cheap capability probe: can this encoder run at all?"""
    args = CANDIDATES.get(encoder_id)
    if not args:
        return False
    try:
        proc = subprocess.run(
            [binary, "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=black:s=64x64:d=0.1", *args, "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0


def _time_encode(encoder_id: str, workdir: str, binary: str = "ffmpeg") -> Optional[float]:
    """Seconds to encode BENCH_SECONDS of 1080p with this encoder."""
    args = CANDIDATES.get(encoder_id)
    if not args:
        return None
    out_path = os.path.join(workdir, f"bench_{encoder_id}.mp4")
    cmd = [
        binary, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size={BENCH_SIZE}:rate=24:duration={BENCH_SECONDS}",
        *args,
        # Match our real encodes: a capped thread budget, so the benchmark measures
        # what production will actually do.
        *(["-threads", "4"] if encoder_id == "x264" else []),
        "-pix_fmt", "yuv420p", "-an", out_path,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Encoder benchmark failed for %s: %s", encoder_id, exc)
        return None
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not os.path.isfile(out_path):
        logger.warning(
            "Encoder %s produced no output: %s", encoder_id,
            (proc.stderr or b"")[-200:].decode("utf-8", errors="ignore"),
        )
        return None
    return round(elapsed, 4)


def benchmark(*, force: bool = False, binary: str = "ffmpeg") -> Dict[str, Any]:
    """Measure the available encoders and record the winner. Cached per machine."""
    fingerprint = machine_fingerprint()

    cached = _read_cache()
    if cached and not force:
        age_days = (time.time() - float(cached.get("at", 0))) / 86400
        if cached.get("fingerprint") == fingerprint and age_days < CACHE_TTL_DAYS:
            return cached

    import tempfile

    results: Dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="encoder-bench-") as workdir:
        for encoder_id in ("x264", "videotoolbox", "nvenc"):
            if not _encoder_available(encoder_id, binary):
                continue
            elapsed = _time_encode(encoder_id, workdir, binary)
            if elapsed is not None:
                results[encoder_id] = elapsed

    # x264 is the reference: it always exists, so an empty result still yields a choice.
    winner = min(results, key=results.get) if results else "x264"
    record = {
        "winner": winner,
        "results": results,
        "fingerprint": fingerprint,
        "at": time.time(),
        "bench_seconds": BENCH_SECONDS,
        "bench_size": BENCH_SIZE,
    }
    _write_cache(record)
    logger.info("Encoder benchmark: %s (winner=%s)", results, winner)
    return record


def _read_cache() -> Optional[Dict[str, Any]]:
    path = _cache_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_cache(record: Dict[str, Any]) -> None:
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
    except OSError as exc:
        logger.debug("Could not persist the encoder choice: %s", exc)


def override() -> Optional[str]:
    """`MILIMO_VIDEO_ENCODER=x264|videotoolbox|nvenc` skips the measurement."""
    value = (os.environ.get("MILIMO_VIDEO_ENCODER") or "").strip().lower()
    if value in ("", "auto", "measure"):
        return None
    return value if value in CANDIDATES else None


def chosen_encoder(*, force: bool = False) -> Dict[str, Any]:
    """The encoder to use, why, and the supporting numbers."""
    forced = override()
    if forced:
        return {"encoder": forced, "reason": "MILIMO_VIDEO_ENCODER override", "results": {}}
    record = benchmark(force=force)
    return {
        "encoder": record["winner"],
        "reason": (
            "measured fastest on this machine"
            if record.get("results") else
            "only x264 is available"
        ),
        "results": record.get("results", {}),
    }
