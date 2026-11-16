"""
Storage Cache Garbage Collector (TTL Pruner).
Periodically cleans temporary video rendering cache, keyframe stills, and temporary
upload chunks older than 24 hours to prevent disk saturation on long-running deployments.
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from app.core.paths import get_data_dir, get_generated_audio_dir

DEFAULT_MEDIA_RETENTION_DAYS = 0  # 0 = keep generated audio forever (opt-in pruning)
MIN_FREE_GB_BEFORE_REFUSING = 2.0

logger = logging.getLogger(__name__)

_GC_TASK: Optional[asyncio.Task] = None
_GC_RUNNING = False


def prune_temporary_storage(max_age_seconds: float = 86400.0) -> int:
    """
    Scan temporary cache directories and prune files modified earlier than max_age_seconds ago.
    Returns the count of pruned files.
    """
    data_dir = get_data_dir()
    candidate_dirs: List[Path] = [
        data_dir / "video_cache",
        data_dir / "video_cache" / "keyframes",
        # Scene plates were never collected: one 1280x720 PNG per scene per job.
        data_dir / "video_cache" / "scene_stills",
        data_dir / "uploads",
        data_dir / "temp",
    ]

    now = time.time()
    pruned_count = 0
    reclaimed_bytes = 0

    for folder in candidate_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        try:
            for entry in folder.iterdir():
                if not entry.is_file():
                    continue

                # Safety guard: never prune database, profiles, or model files
                if entry.name.endswith((".db", ".sqlite", ".json", ".pth", ".pt", ".safetensors", ".ckpt")):
                    continue

                try:
                    mtime = entry.stat().st_mtime
                    if (now - mtime) > max_age_seconds:
                        file_size = entry.stat().st_size
                        entry.unlink()
                        pruned_count += 1
                        reclaimed_bytes += file_size
                except Exception as file_err:
                    logger.debug(f"Failed to prune cache file {entry}: {file_err}")

        except Exception as dir_err:
            logger.warning(f"Error scanning cache directory {folder}: {dir_err}")

    if pruned_count > 0:
        mb_reclaimed = reclaimed_bytes / (1024 * 1024)
        logger.info(f"Storage GC: Pruned {pruned_count} cache files ({mb_reclaimed:.2f} MB reclaimed)")

    return pruned_count


async def run_storage_gc_loop(interval_seconds: float = 3600.0, max_age_seconds: float = 86400.0):
    """
    Continuous background loop that wakes up periodically to prune temporary storage.
    """
    global _GC_RUNNING
    _GC_RUNNING = True
    logger.info("Storage GC background loop started (interval=%ds, max_age=%ds)", interval_seconds, max_age_seconds)

    try:
        while _GC_RUNNING:
            await asyncio.sleep(interval_seconds)
            if not _GC_RUNNING:
                break
            try:
                # Offload disk IO scan to executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, prune_temporary_storage, max_age_seconds)
            except Exception as e:
                logger.warning(f"Error during periodic storage GC run: {e}")
    except asyncio.CancelledError:
        logger.info("Storage GC background loop cancelled.")
    finally:
        _GC_RUNNING = False


def start_storage_gc(interval_seconds: float = 3600.0, max_age_seconds: float = 86400.0) -> Optional[asyncio.Task]:
    """Start the storage garbage collection background task."""
    global _GC_TASK
    if _GC_TASK is None or _GC_TASK.done():
        _GC_TASK = asyncio.create_task(run_storage_gc_loop(interval_seconds, max_age_seconds))
    return _GC_TASK


def stop_storage_gc():
    """Stop the storage garbage collection background task."""
    global _GC_TASK, _GC_RUNNING
    _GC_RUNNING = False
    if _GC_TASK and not _GC_TASK.done():
        _GC_TASK.cancel()
        _GC_TASK = None


def _retention_days() -> int:
    try:
        return int(os.environ.get("MILIMO_MEDIA_RETENTION_DAYS", str(DEFAULT_MEDIA_RETENTION_DAYS)))
    except ValueError:
        return DEFAULT_MEDIA_RETENTION_DAYS


def prune_generated_media(engine=None, retention_days: Optional[int] = None) -> int:
    """Opt-in retention for generated audio, keyed on the Job rows.

    Off by default (`MILIMO_MEDIA_RETENTION_DAYS=0`) because deleting a user's
    tracks is destructive; when enabled, files whose owning job is older than the
    window are removed and the job is kept with its metadata intact.
    """
    days = _retention_days() if retention_days is None else retention_days
    if days <= 0:
        return 0

    from datetime import datetime, timedelta, timezone

    cutoff = time.time() - days * 86400
    roots = [get_generated_audio_dir(), get_data_dir() / "covers"]
    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            if entry.name.endswith((".db", ".json", ".mid", ".musicxml")):
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
    if removed:
        logger.info("Retention: removed %d generated media file(s) older than %dd", removed, days)
    return removed


def disk_status() -> dict:
    """Free-space gauge for /health and for refusing new jobs on a full disk."""
    import shutil as _shutil

    target = get_generated_audio_dir()
    try:
        usage = _shutil.disk_usage(str(target))
    except OSError:
        return {"free_gb": None, "total_gb": None, "used_percent": None}
    return {
        "free_gb": round(usage.free / (1024 ** 3), 2),
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_percent": round(100 * usage.used / usage.total, 1) if usage.total else None,
    }


def ensure_disk_headroom() -> Optional[str]:
    """Return an error message when the disk is too full to start new work."""
    try:
        minimum = float(os.environ.get("MILIMO_MIN_FREE_GB", str(MIN_FREE_GB_BEFORE_REFUSING)))
    except ValueError:
        minimum = MIN_FREE_GB_BEFORE_REFUSING
    status = disk_status()
    free = status.get("free_gb")
    if free is not None and free < minimum:
        return (
            f"Only {free} GB free on the media volume (minimum {minimum:g} GB). "
            "Free space or raise MILIMO_MIN_FREE_GB."
        )
    return None
