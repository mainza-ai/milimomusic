"""Video capability matrix (plan 8.10).

The studio silently clamps on MPS (832×480, ≤33 frames) and refuses unimplemented
engines, but nothing told the user *before* a 40-minute render. This module answers
"what will this machine actually produce for this engine" from the same specs the
renderer uses.

Rules mirror `DiffusersWanGenerator`: on MPS the Wan family is clamped; the frame
lattice comes from `model_specs`. Nothing here guesses — an unknown device reports
`unknown` limits rather than optimistic ones.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger("milimo.video.capability")

#: Mirrors the clamp in `generators/diffusers_wan.py` (MPS attention budget).
MPS_MAX_W = 832
MPS_MAX_H = 480
MPS_MAX_FRAMES = 33


def detect_device() -> str:
    """`mps` | `cuda` | `cpu` — the device the local diffusion path would use."""
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def interpolation_backend() -> Dict[str, Any]:
    """Which delivery frame-rate path is usable here.

    `minterpolate` ships with ffmpeg and is always available; RIFE (MIT) is the
    quality upgrade and is detected, never assumed.
    """
    rife_cmd = os.environ.get("MILIMO_RIFE_CMD")
    rife_present = bool(rife_cmd and (os.path.isfile(rife_cmd) or shutil.which(rife_cmd)))
    return {
        "active": "rife" if rife_present else "minterpolate",
        "rife_available": rife_present,
        "note": (
            "RIFE detected via MILIMO_RIFE_CMD."
            if rife_present
            else "minterpolate (ffmpeg) — blend-based stopgap; set MILIMO_RIFE_CMD to use RIFE."
        ),
    }


def _effective_for_spec(spec: Any, device: str) -> Dict[str, Any]:
    """Clamp a spec to what the current device can actually render."""
    native_w = int(getattr(spec, "default_width", 0) or 0)
    native_h = int(getattr(spec, "default_height", 0) or 0)
    fps = int(getattr(spec, "fps", 0) or 0)
    max_duration = float(getattr(spec, "max_duration", 0.0) or 0.0)
    max_frames = int(max_duration * fps) if fps else 0

    width, height, frames = native_w, native_h, max_frames
    clamped = False
    notes: List[str] = []

    family = str(getattr(spec, "family", ""))
    if device == "mps" and family in ("dit", "t2v", "3d-vae"):
        if native_w > MPS_MAX_W or native_h > MPS_MAX_H:
            width, height = min(native_w, MPS_MAX_W), min(native_h, MPS_MAX_H)
            clamped = True
        if max_frames > MPS_MAX_FRAMES:
            frames = MPS_MAX_FRAMES
            clamped = True
        if clamped:
            notes.append(
                f"MPS clamps Wan-class attention to {MPS_MAX_W}×{MPS_MAX_H} and "
                f"{MPS_MAX_FRAMES} frames; larger requests are rendered then upscaled."
            )

    if not getattr(spec, "implemented", True):
        notes.append(
            getattr(spec, "unimplemented_reason", "")
            or "No local pipeline is wired for this engine."
        )

    return {
        "engine": getattr(spec, "model_id", None),
        "name": getattr(spec, "canonical_name", None),
        "family": family,
        "implemented": bool(getattr(spec, "implemented", True)),
        "native": {"width": native_w, "height": native_h, "fps": fps,
                   "max_seconds": max_duration, "max_frames": max_frames},
        "effective": {"width": width, "height": height, "fps": fps, "max_frames": frames},
        "clamped": clamped,
        "notes": notes,
    }


def matrix() -> Dict[str, Any]:
    from app.services.video.model_specs import VIDEO_MODEL_SPECS

    device = detect_device()
    engines = [_effective_for_spec(spec, device) for spec in VIDEO_MODEL_SPECS.values()]
    return {
        "device": device,
        "interpolation": interpolation_backend(),
        "engines": engines,
        "notes": (
            f"Rendering on {device.upper()}. Limits below are what this machine will "
            f"actually produce, not what the model can do on a datacentre GPU."
        ),
    }


def describe(engine_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The effective limits for one engine (for the picker)."""
    for entry in matrix()["engines"]:
        if entry["engine"] == engine_id:
            return entry
    return None
