"""
The capability matrix: what this build can actually do on THIS machine.

Decision D6: MiniMax Music 3 generation is Apple-Silicon-MLX only; CUDA is a
supported target for everything else; CPU runs the DAW/transcription path. Rather
than documenting that in a README and failing at runtime, the matrix is computed
here and served to the UI so it can disable engines with a stated reason.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger("milimo.capabilities")

APPLE_SILICON = "apple_silicon_mlx"
CUDA = "nvidia_cuda"
CPU = "cpu"

# Columns of the published matrix, in display order.
COLUMNS = [APPLE_SILICON, CUDA, CPU]

MATRIX: Dict[str, Dict[str, Any]] = {
    "minimax_music3": {
        "label": "MiniMax Music 3 (flagship generation)",
        "columns": {APPLE_SILICON: True, CUDA: False, CPU: False},
        "reason": {
            CUDA: "Requires Apple Silicon MLX (mlx-audio). Use the cloud engine or HeartMuLa on CUDA.",
            CPU: "Requires Apple Silicon MLX (mlx-audio).",
        },
    },
    "heartmula": {
        "label": "HeartMuLa (legacy generation)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: False},
        "reason": {CPU: "29 GB of weights and minutes-per-track on CPU; not practical."},
    },
    "mulacover": {
        "label": "MuLaCover (cover / remix)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: False},
        "reason": {CPU: "Requires an accelerator for usable latency."},
    },
    "separation": {
        "label": "Stem separation (HTDemucs)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: True},
        "reason": {},
    },
    "transcription": {
        "label": "Transcription (MuScriptor)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: True},
        "reason": {},
    },
    "mastering": {
        "label": "Mastering / taste/synthesis DSP",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: True},
        "reason": {},
    },
    "image": {
        "label": "Image generation (FLUX.2)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: False},
        "reason": {CPU: "Diffusion needs an accelerator to be usable."},
    },
    "video": {
        "label": "Video generation (Wan / LTX diffusers)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: False},
        "reason": {CPU: "Diffusion needs an accelerator to be usable."},
    },
    "cloud": {
        "label": "Cloud generation / lip sync (fal.ai)",
        "columns": {APPLE_SILICON: True, CUDA: True, CPU: True},
        "reason": {},
    },
}


def _has_mlx() -> bool:
    return importlib.util.find_spec("mlx_audio") is not None or importlib.util.find_spec("mlx") is not None


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def platform_class() -> str:
    """Which column of the matrix describes this machine."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return APPLE_SILICON
    if _has_cuda():
        return CUDA
    return CPU


def _engine_installed(feature: str) -> Optional[bool]:
    """Whether the software/weights for a feature are present (None = unknown)."""
    try:
        if feature == "minimax_music3":
            return _has_mlx()
        if feature == "cloud":
            return bool(os.environ.get("FAL_KEY") or os.environ.get("REPLICATE_API_TOKEN"))
        if feature in ("image", "video"):
            from app.services.model_manager import model_manager

            cat = "image" if feature == "image" else "video"
            installed = [
                m for m in model_manager.get_model_tree()
                if m.get("category") == cat and m.get("is_installed")
            ]
            return bool(installed)
    except Exception:
        return None
    return None


def build_matrix() -> Dict[str, Any]:
    """Return the matrix plus this machine's verdicts."""
    column = platform_class()
    rows: List[Dict[str, Any]] = []
    for feature, spec in MATRIX.items():
        supported = bool(spec["columns"].get(column))
        installed = _engine_installed(feature)
        reason = spec["reason"].get(column)
        if supported and installed is False:
            supported = False
            if feature == "cloud":
                reason = "No cloud API key configured (FAL_KEY)."
            else:
                reason = reason or "Engine or weights not installed."
        rows.append({
            "feature": feature,
            "label": spec["label"],
            "supported": supported,
            "installed": installed,
            "reason": None if supported else (reason or f"Not supported on {column}."),
            "matrix": spec["columns"],
        })

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return {
        "platform": column,
        "platform_label": {
            APPLE_SILICON: "Apple Silicon (MLX)",
            CUDA: "NVIDIA CUDA",
            CPU: "CPU",
        }[column],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "ffmpeg": ffmpeg_ok,
        "features": rows,
        "columns": COLUMNS,
    }


def feature_supported(feature: str) -> bool:
    """Cheap gate for call sites that must refuse rather than degrade."""
    spec = MATRIX.get(feature)
    if not spec:
        return True
    return bool(spec["columns"].get(platform_class()))
