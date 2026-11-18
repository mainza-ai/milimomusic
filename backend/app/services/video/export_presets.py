"""Delivery presets for rendered video (plans 7.12-7.15).

One hardcoded profile used to serve every destination: 8-bit 4:2:0 H.264 at the
x264 default, no faststart, AAC 256k, no loudness target. That is not a
deliverable (audit D1-D3, D5). Presets make the destination explicit and give the
encoder something to aim at.

Numbers come from the platform specs in
`wiki/concepts/music-video-delivery-specs.md` (YouTube ladder, Vevo HD, Spotify
Canvas, EBU R 128 for broadcast).
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger("milimo.video.presets")


@dataclass(frozen=True)
class DeliveryPreset:
    """A concrete, ffprobe-verifiable delivery target."""

    id: str
    label: str
    # video
    video_codec: str = "libx264"
    crf: Optional[int] = 18
    video_bitrate: Optional[str] = None      # used instead of crf when set
    preset: str = "medium"                   # x264 speed/efficiency
    pix_fmt: str = "yuv420p"
    profile: Optional[str] = "high"
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    # audio
    audio_codec: str = "aac"
    audio_bitrate: str = "320k"
    audio_rate: int = 48000
    copy_audio_when_aac: bool = True
    # loudness (integrated LUFS / true peak dBTP). None = leave the master alone.
    loudness_lufs: Optional[float] = -14.0
    loudness_tp: Optional[float] = -1.0
    # container
    faststart: bool = True
    extra_args: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Presets users can choose. `default` is what an ordinary render gets.
PRESETS: Dict[str, DeliveryPreset] = {
    "default": DeliveryPreset(
        id="default",
        label="Standard MP4 (H.264 high quality, faststart, −14 LUFS)",
        crf=18,
        preset="medium",
    ),
    "youtube_1080p": DeliveryPreset(
        id="youtube_1080p",
        label="YouTube 1080p",
        crf=None,
        video_bitrate="8M",
        preset="medium",
        max_width=1920,
        max_height=1080,
    ),
    "youtube_2160p": DeliveryPreset(
        id="youtube_2160p",
        label="YouTube 2160p (4K)",
        crf=None,
        video_bitrate="40M",
        preset="medium",
        max_width=3840,
        max_height=2160,
    ),
    "vevo_hd": DeliveryPreset(
        id="vevo_hd",
        label="Vevo HD (20 Mbps, 44.1 kHz 320k audio, no slates)",
        crf=None,
        video_bitrate="20M",
        preset="medium",
        max_width=1920,
        max_height=1080,
        audio_rate=44100,
        audio_bitrate="320k",
        copy_audio_when_aac=False,
        loudness_lufs=None,          # Vevo takes the master as delivered
        loudness_tp=None,
    ),
    "canvas": DeliveryPreset(
        id="canvas",
        label="Spotify Canvas (silent, 9:16, ≤8 s)",
        crf=20,
        preset="medium",
        max_width=1080,
        max_height=1920,
        audio_codec="none",
        audio_bitrate="",
        loudness_lufs=None,
        loudness_tp=None,
    ),
    "broadcast_prores": DeliveryPreset(
        id="broadcast_prores",
        label="Broadcast ProRes 422 HQ (−23 LUFS)",
        video_codec="prores_ks",
        crf=None,
        video_bitrate=None,
        preset="medium",
        pix_fmt="yuv422p10le",
        profile=None,
        audio_codec="pcm_s16le",
        audio_bitrate="",
        audio_rate=48000,
        copy_audio_when_aac=False,
        loudness_lufs=-23.0,
        loudness_tp=-1.0,
        extra_args=("-profile:v", "3"),
    ),
}

DEFAULT_PRESET = "default"


def get_preset(name: Optional[str]) -> DeliveryPreset:
    """Resolve a preset by id, falling back to `default` (never to nothing)."""
    if not name:
        return PRESETS[DEFAULT_PRESET]
    return PRESETS.get(str(name).strip().lower(), PRESETS[DEFAULT_PRESET])


def list_presets() -> list[Dict[str, Any]]:
    return [p.to_dict() for p in PRESETS.values()]


def video_args(preset: DeliveryPreset) -> list[str]:
    """Encoder arguments for the delivery encode."""
    args: list[str] = ["-c:v", preset.video_codec]
    if preset.video_codec == "libx264":
        args += ["-preset", preset.preset]
        if preset.video_bitrate:
            args += ["-b:v", preset.video_bitrate, "-maxrate", preset.video_bitrate,
                     "-bufsize", _double_bitrate(preset.video_bitrate)]
        elif preset.crf is not None:
            args += ["-crf", str(preset.crf)]
        if preset.profile:
            args += ["-profile:v", preset.profile]
    elif preset.video_codec == "prores_ks":
        args += ["-profile:v", "3"]
    if preset.pix_fmt:
        args += ["-pix_fmt", preset.pix_fmt]
    args += list(preset.extra_args)
    return args


def audio_args(preset: DeliveryPreset, source_is_aac: bool = False) -> list[str]:
    """Audio arguments, or `[]` when the preset is silent."""
    if preset.audio_codec in ("", "none"):
        return ["-an"]
    if preset.copy_audio_when_aac and source_is_aac:
        return ["-c:a", "copy"]
    args = ["-c:a", preset.audio_codec]
    if preset.audio_bitrate:
        args += ["-b:a", preset.audio_bitrate]
    if preset.audio_rate:
        args += ["-ar", str(preset.audio_rate)]
    return args


def loudnorm_filter(preset: DeliveryPreset) -> Optional[str]:
    """Loudness normalisation filter, or None when the preset wants the master as-is."""
    if preset.loudness_lufs is None:
        return None
    tp = preset.loudness_tp if preset.loudness_tp is not None else -1.0
    return f"loudnorm=I={preset.loudness_lufs}:TP={tp}:LRA=11:print_format=json"


def container_args(preset: DeliveryPreset) -> list[str]:
    """Container flags — `+faststart` is what makes HTTP playback seekable."""
    args: list[str] = []
    if preset.video_codec == "libx264" and preset.faststart:
        args += ["-movflags", "+faststart"]
    return args


def _double_bitrate(bitrate: str) -> str:
    """`8M` -> `16M` for the VBV buffer."""
    text = str(bitrate).strip()
    if text.endswith(("k", "K")):
        return f"{int(float(text[:-1]) * 2)}k"
    if text.endswith(("M", "m")):
        return f"{int(float(text[:-1]) * 2)}M"
    return text


def probe_streams(path: str) -> Dict[str, Any]:
    """ffprobe a file into a flat dict (codec, size, bitrates, duration)."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    import json as _json

    try:
        payload = _json.loads(out.stdout or "{}")
    except ValueError:
        return {}
    video = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = payload.get("format", {})
    return {
        "duration": _float(fmt.get("duration")),
        "size": _int(fmt.get("size")),
        "bit_rate": _int(fmt.get("bit_rate")),
        "video_codec": video.get("codec_name"),
        "width": _int(video.get("width")),
        "height": _int(video.get("height")),
        "fps": _parse_rate(video.get("r_frame_rate")),
        "video_bit_rate": _int(video.get("bit_rate")),
        "audio_codec": audio.get("codec_name"),
        "audio_bit_rate": _int(audio.get("bit_rate")),
        "audio_sample_rate": _int(audio.get("sample_rate")),
        "audio_channels": _int(audio.get("channels")),
    }


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> Optional[float]:
    if not value or value == "0/0":
        return None
    try:
        num, _, den = str(value).partition("/")
        return round(float(num) / float(den or 1), 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
