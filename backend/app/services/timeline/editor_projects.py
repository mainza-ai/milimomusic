"""
Non-Destructive Multi-Track DAW & Video Editor Project Engine.

Implements atomic multi-track timeline schema (video, audio stems, subtitles),
single-pass hardware-accelerated FFmpeg compiler, and round-trip AI take replacement.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.win_safe_files import atomic_write_json, safe_load_json

logger = logging.getLogger("milimo.timeline.editor_projects")


@dataclass
class ClipTransform:
    """Canvas positioning, scale, and opacity."""

    scale: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    opacity: float = 1.0
    crop: Optional[Dict[str, float]] = None  # {top, bottom, left, right} in fractions 0.0-1.0


@dataclass
class TimelineClip:
    """An individual media item positioned non-destructively on a track."""

    clip_id: str
    asset_path: str
    start_time: float
    duration: float
    source_in: float = 0.0
    source_out: Optional[float] = None
    volume: float = 1.0
    transform: ClipTransform = field(default_factory=ClipTransform)
    transition_in: Optional[str] = None  # "crossfade", "dissolve", "fade_black"
    transition_out: Optional[str] = None
    ai_take_parent_id: Optional[str] = None
    take_version: int = 1


@dataclass
class TimelineTrack:
    """A layer containing sequential or overlapping media clips."""

    track_id: str
    track_type: str  # "video" | "audio" | "subtitle"
    name: str
    muted: bool = False
    solo: bool = False
    volume: float = 1.0
    clips: List[TimelineClip] = field(default_factory=list)


@dataclass
class EditorProject:
    """Complete multi-track composition specification."""

    project_id: str
    title: str
    duration: float
    aspect_ratio: str = "16:9"  # "16:9" | "9:16" | "21:9"
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 24
    tracks: List[TimelineTrack] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EditorProject:
        tracks = []
        for t_data in data.get("tracks", []):
            clips = []
            for c_data in t_data.get("clips", []):
                t_info = c_data.get("transform", {})
                transform = ClipTransform(
                    scale=t_info.get("scale", 1.0),
                    x_offset=t_info.get("x_offset", 0.0),
                    y_offset=t_info.get("y_offset", 0.0),
                    opacity=t_info.get("opacity", 1.0),
                    crop=t_info.get("crop"),
                )
                clips.append(
                    TimelineClip(
                        clip_id=c_data["clip_id"],
                        asset_path=c_data["asset_path"],
                        start_time=c_data["start_time"],
                        duration=c_data["duration"],
                        source_in=c_data.get("source_in", 0.0),
                        source_out=c_data.get("source_out"),
                        volume=c_data.get("volume", 1.0),
                        transform=transform,
                        transition_in=c_data.get("transition_in"),
                        transition_out=c_data.get("transition_out"),
                        ai_take_parent_id=c_data.get("ai_take_parent_id"),
                        take_version=c_data.get("take_version", 1),
                    )
                )
            tracks.append(
                TimelineTrack(
                    track_id=t_data["track_id"],
                    track_type=t_data["track_type"],
                    name=t_data["name"],
                    muted=t_data.get("muted", False),
                    solo=t_data.get("solo", False),
                    volume=t_data.get("volume", 1.0),
                    clips=clips,
                )
            )
        return cls(
            project_id=data["project_id"],
            title=data.get("title", "Untitled Composition"),
            duration=data.get("duration", 0.0),
            aspect_ratio=data.get("aspect_ratio", "16:9"),
            resolution=tuple(data.get("resolution", (1920, 1080))),  # type: ignore[arg-type]
            fps=data.get("fps", 24),
            tracks=tracks,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


class EditorProjectManager:
    """Persistence, rendering compilation, and AI take round-trip manager."""

    @classmethod
    def detect_hardware_video_encoder(cls) -> str:
        """Auto-detect available hardware accelerated FFmpeg encoder."""
        sys_os = platform.system()
        if sys_os == "Darwin":
            # Apple Silicon VideoToolbox
            return "h264_videotoolbox"

        # Check for NVIDIA NVENC
        try:
            res = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
            )
            if "h264_nvenc" in res.stdout:
                return "h264_nvenc"
            if "h264_qsv" in res.stdout:
                return "h264_qsv"
            if "h264_vaapi" in res.stdout:
                return "h264_vaapi"
        except Exception:
            pass

        return "libx264"

    @classmethod
    def compile_editor_render_command(
        cls,
        project: EditorProject,
        output_file: str,
        video_encoder: Optional[str] = None,
    ) -> List[str]:
        """
        Compile non-destructive project timeline into a single-pass FFmpeg command.
        Uses complex filter graphs for video overlays, timing offsets, and audio mixing.
        """
        encoder = video_encoder or cls.detect_hardware_video_encoder()
        width, height = project.resolution

        inputs: List[str] = []
        input_index_map: Dict[str, int] = {}

        def get_input_id(path: str) -> int:
            if path not in input_index_map:
                idx = len(input_index_map)
                input_index_map[path] = idx
                inputs.extend(["-i", path])
            return input_index_map[path]

        v_filters: List[str] = []
        a_filters: List[str] = []

        # Background black canvas
        v_filters.append(
            f"color=c=black:s={width}x{height}:r={project.fps}:d={project.duration}[base_canvas]"
        )
        current_canvas = "base_canvas"

        video_overlay_count = 0
        audio_stream_labels: List[str] = []

        # Collect tracks
        video_tracks = [t for t in project.tracks if t.track_type == "video" and not t.muted]
        audio_tracks = [t for t in project.tracks if t.track_type == "audio" and not t.muted]

        # Video tracks compilation
        for t_idx, track in enumerate(video_tracks):
            for c_idx, clip in enumerate(track.clips):
                in_idx = get_input_id(clip.asset_path)
                clip_label = f"v_{t_idx}_{c_idx}"
                overlay_label = f"canvas_{video_overlay_count}"

                # Scale and setpts for in-point and start_time
                s_in = clip.source_in
                dur = clip.duration
                start_pts = clip.start_time

                v_filters.append(
                    f"[{in_idx}:v]trim=start={s_in}:duration={dur},setpts=PTS-STARTPTS,"
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                    f"format=yuva420p[{clip_label}]"
                )

                # Overlay onto canvas at start_pts
                v_filters.append(
                    f"[{current_canvas}][{clip_label}]overlay=enable='between(t,{start_pts},{start_pts + dur})':eof_action=pass[{overlay_label}]"
                )
                current_canvas = overlay_label
                video_overlay_count += 1

        v_filters.append(f"[{current_canvas}]format=yuv420p[outv]")

        # Audio tracks compilation
        for t_idx, track in enumerate(audio_tracks):
            for c_idx, clip in enumerate(track.clips):
                in_idx = get_input_id(clip.asset_path)
                a_label = f"a_{t_idx}_{c_idx}"
                delay_ms = int(clip.start_time * 1000)
                vol = clip.volume * track.volume

                a_filters.append(
                    f"[{in_idx}:a]atrim=start={clip.source_in}:duration={clip.duration},"
                    f"asetpts=PTS-STARTPTS,volume={vol},"
                    f"adelay={delay_ms}|{delay_ms}[{a_label}]"
                )
                audio_stream_labels.append(f"[{a_label}]")

        if audio_stream_labels:
            mix_inputs = "".join(audio_stream_labels)
            a_filters.append(
                f"{mix_inputs}amix=inputs={len(audio_stream_labels)}:duration=longest:dropout_transition=0[outa]"
            )
        else:
            # Synthetic silent audio track
            a_filters.append(f"anullsrc=r=44100:cl=stereo:d={project.duration}[outa]")

        full_filter = ";".join(v_filters + a_filters)

        cmd = ["ffmpeg", "-y"]
        cmd.extend(inputs)
        cmd.extend(["-filter_complex", full_filter])
        cmd.extend(["-map", "[outv]", "-map", "[outa]"])

        # Encoder-specific settings
        if "videotoolbox" in encoder:
            cmd.extend(["-c:v", encoder, "-b:v", "8000k"])
        elif "nvenc" in encoder:
            cmd.extend(["-c:v", encoder, "-preset", "p5", "-b:v", "8000k"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "20"])

        cmd.extend(["-c:a", "aac", "-b:a", "320k"])
        cmd.extend(["-t", str(project.duration)])
        cmd.append(output_file)

        return cmd

    @classmethod
    def apply_retake_to_clip(
        cls,
        project: EditorProject,
        clip_id: str,
        new_asset_path: str,
    ) -> bool:
        """
        Round-trip AI take replacement: Replaces asset reference and tracks take ancestry
        while strictly preserving start_time, duration, source_in, volume, and transitions.
        """
        for track in project.tracks:
            for clip in track.clips:
                if clip.clip_id == clip_id:
                    clip.ai_take_parent_id = clip.asset_path
                    clip.asset_path = new_asset_path
                    clip.take_version += 1
                    project.updated_at = datetime.now(timezone.utc).isoformat()
                    logger.info(
                        f"Applied retake to clip {clip_id} -> {new_asset_path} (Take v{clip.take_version})"
                    )
                    return True
        return False
