"""
Model Tree and Hardware Capability Management Service.
Inspects local and HuggingFace model trees for MiniMax Music 3, HeartMuLa, and adapters.
"""

import os
import shutil
import platform
import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from app.providers.base import HardwareTier

# Load .env so MINIMAX_MODEL_PATH is honoured without hardcoding the snapshot path in code.
try:
    from dotenv import load_dotenv
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

logger = logging.getLogger(__name__)

DEFAULT_MINIMAX_SNAPSHOT = os.environ.get("MINIMAX_MODEL_PATH", "")


@dataclass
class ModelVariant:
    id: str
    name: str
    architecture: str
    quantization: str
    size_gb: float
    is_installed: bool
    local_path: Optional[str]
    license: str
    recommended_hardware: str
    is_default: bool = False


@dataclass
class HardwareProfile:
    os_name: str
    architecture: str
    processor: str
    has_cuda: bool
    has_mps: bool
    hardware_tier: str
    tier_description: str
    can_run_minimax_full: bool
    can_run_heartmula: bool


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def detect_hardware(self) -> HardwareProfile:
        """Detect system hardware profile and compute performance tiers."""
        has_cuda = False
        has_mps = False

        try:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            # On macOS ARM, Apple Silicon Metal acceleration is natively available
            if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
                has_mps = True

        tier = HardwareTier.MID_SINGLE_GPU
        desc = "Apple Silicon GPU (Metal/MPS) detected. Optimal for MiniMax Music 3 & HeartMuLa."

        if has_cuda:
            tier = HardwareTier.HIGH_DUAL_GPU
            desc = "NVIDIA CUDA GPU detected. Full multi-GPU & batch acceleration enabled."
        elif not has_mps:
            tier = HardwareTier.ENTRY_CPU
            desc = "CPU-only execution detected. Generation will run slower."

        return HardwareProfile(
            os_name=platform.system(),
            architecture=platform.machine(),
            processor=platform.processor() or "Apple Silicon / ARM64",
            has_cuda=has_cuda,
            has_mps=has_mps,
            hardware_tier=tier.value,
            tier_description=desc,
            can_run_minimax_full=True,
            can_run_heartmula=True
        )

    def get_model_tree(self) -> List[Dict[str, Any]]:
        """Return full catalog of supported generation models, variants, and local install state."""
        minimax_installed = os.path.isdir(DEFAULT_MINIMAX_SNAPSHOT)
        heartlib_ckpt_dir = os.path.expanduser("../heartlib/ckpt")
        heartmula_installed = os.path.isdir(heartlib_ckpt_dir) and os.path.isdir(os.path.join(heartlib_ckpt_dir, "HeartMuLa-oss-3B"))

        variants: List[ModelVariant] = [
            ModelVariant(
                id="minimax_music3_bf16",
                name="MiniMax Music 3 (bfloat16 Base)",
                architecture="Qwen3 + RVQ8 + Flow Matching DiT",
                quantization="BF16 (High Quality)",
                size_gb=28.5,
                is_installed=minimax_installed,
                local_path=DEFAULT_MINIMAX_SNAPSHOT if minimax_installed else None,
                license="MiniMax Open Weights",
                recommended_hardware="Apple Silicon (MPS) / 16GB+ VRAM",
                is_default=True
            ),
            ModelVariant(
                id="minimax_music3_int8",
                name="MiniMax Music 3 (8-bit Quantized)",
                architecture="Qwen3 + RVQ8 + Flow Matching DiT",
                quantization="INT8 (Memory Optimized)",
                size_gb=14.2,
                is_installed=False,
                local_path=None,
                license="MiniMax Open Weights",
                recommended_hardware="Single GPU 8GB-12GB VRAM",
                is_default=False
            ),
            ModelVariant(
                id="heartmula_3b",
                name="HeartMuLa-3B (Legacy/Local)",
                architecture="Autoregressive LM + HeartCodec",
                quantization="FP16 / BF16",
                size_gb=6.2,
                is_installed=heartmula_installed,
                local_path=heartlib_ckpt_dir if heartmula_installed else None,
                license="Apache-2.0",
                recommended_hardware="Apple Silicon / 8GB+ VRAM",
                is_default=False
            )
        ]

        return [asdict(v) for v in variants]

    def check_missing_dependencies(self, model_id: str) -> Dict[str, Any]:
        """Check if selected model requires download before starting generation."""
        tree = self.get_model_tree()
        match = next((m for m in tree if m["id"] == model_id or m["id"].startswith(model_id)), None)
        if not match:
            return {"missing": False, "model_id": model_id, "message": "Standard provider"}

        return {
            "missing": not match["is_installed"],
            "model_id": match["id"],
            "name": match["name"],
            "size_gb": match["size_gb"],
            "local_path": match["local_path"],
            "message": "Model downloaded and ready" if match["is_installed"] else f"{match['name']} ({match['size_gb']} GB) is not yet downloaded."
        }


model_manager = ModelManager()
