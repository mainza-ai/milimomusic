"""
Immersive Gallery & 1-Click Media Bridge for Milimo Music Studio.

Implements 1-click gallery-to-input routing, cached first-frame video posters,
and before/after split comparison metadata for seamless media reuse.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("milimo.gallery.media_bridge")


@dataclass
class RoutedMediaPayload:
    """Dispatched gallery asset prepared for downstream Studio / Director inputs."""

    source_path: str
    target_slot: str  # "references" | "frames" | "animate" | "edit" | "upscale" | "lip_sync"
    media_type: str   # "image" | "video" | "audio"
    target_job_or_session_id: Optional[str] = None
    parameters: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MediaBridge:
    """Dispatches gallery media into active generation inputs and generates poster thumbnails."""

    THUMBNAIL_CACHE_DIR = Path(".milimo/thumbnails")

    @classmethod
    def get_thumbnail_dir(cls) -> Path:
        cls.THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls.THUMBNAIL_CACHE_DIR

    @classmethod
    def get_first_frame_poster(cls, video_path: str) -> Optional[str]:
        """
        Extracts and caches the first frame of a video as a JPEG poster.
        Enables instant gallery feed loading without container decoding lag.
        """
        v_path = Path(video_path)
        if not v_path.exists():
            return None

        # Derive stable cache key from path and mtime
        cache_key = hashlib.md5(f"{v_path.resolve()}_{v_path.stat().st_mtime}".encode("utf-8")).hexdigest()
        thumb_path = cls.get_thumbnail_dir() / f"{cache_key}.jpg"

        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            return str(thumb_path)

        # Extract first frame using ffmpeg
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                "00:00:00",
                "-i",
                str(v_path),
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-vf",
                "scale=640:-1",
                str(thumb_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and thumb_path.exists():
                return str(thumb_path)
            else:
                logger.warning(f"FFmpeg poster extraction failed for {video_path}: {res.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error generating first-frame poster: {e}")
            return None

    @classmethod
    def route_to_input(
        cls,
        media_path: str,
        target_slot: str,
        target_job_or_session_id: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> RoutedMediaPayload:
        """
        1-Click Gallery-to-Input media routing.
        Validates target slot and packages asset for Studio and Director forms.
        """
        p = Path(media_path)
        if not p.exists():
            raise FileNotFoundError(f"Media asset does not exist: {media_path}")

        ext = p.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".webp"]:
            media_type = "image"
        elif ext in [".mp4", ".mov", ".webm", ".mkv"]:
            media_type = "video"
        elif ext in [".wav", ".mp3", ".flac", ".ogg"]:
            media_type = "audio"
        else:
            media_type = "unknown"

        allowed_slots = ["references", "frames", "animate", "edit", "upscale", "lip_sync"]
        if target_slot not in allowed_slots:
            raise ValueError(f"Invalid target slot '{target_slot}'. Must be one of {allowed_slots}")

        payload = RoutedMediaPayload(
            source_path=str(p.resolve()),
            target_slot=target_slot,
            media_type=media_type,
            target_job_or_session_id=target_job_or_session_id,
            parameters=extra_params or {},
        )
        logger.info(f"Routed gallery media {p.name} ({media_type}) -> Slot '{target_slot}'")
        return payload

    @classmethod
    def create_before_after_split_manifest(
        cls,
        source_url: str,
        generated_url: str,
        split_ratio: float = 0.5,
        comparison_label: str = "Source vs AI Take",
    ) -> Dict[str, Any]:
        """
        Produce payload for interactive side-by-side / split divider comparison in UI.
        """
        return {
            "source_url": source_url,
            "generated_url": generated_url,
            "initial_split_ratio": max(0.1, min(0.9, split_ratio)),
            "label": comparison_label,
            "type": "split_comparison",
        }
