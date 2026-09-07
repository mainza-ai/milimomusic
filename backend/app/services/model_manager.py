"""
Model Tree, Multi-Modal Catalog, and Hardware Capability Management Service.
Inspects local and Hugging Face model trees for MiniMax Music 3, HeartMuLa,
FLUX.1 image models, and MiniMax H3 / Wan2.1 open video models.
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
    category: str = "audio"  # "audio", "image", "video"
    repo_id: Optional[str] = None
    is_default: bool = False
    is_active: bool = False


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


def resolve_hf_snapshot(repo_id: str) -> Optional[str]:
    """Find local snapshot directory for a huggingface repo ID.
    Supports local ./data/models, HF hub cache, or explicit snapshots.
    """
    try:
        # 1. Check data/models/{org}__{repo}
        data_path = os.path.join("data", "models", repo_id.replace("/", "__"))
        if os.path.isdir(data_path):
            try:
                if os.listdir(data_path):
                    return os.path.abspath(data_path)
            except (OSError, PermissionError):
                pass

        # 2. Check ~/.cache/huggingface/hub/models--{org}--{repo}
        hf_hub_name = f"models--{repo_id.replace('/', '--')}"
        hub_dir = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", hf_hub_name))
        if os.path.isdir(hub_dir):
            snapshots_dir = os.path.join(hub_dir, "snapshots")
            if os.path.isdir(snapshots_dir):
                try:
                    snaps = [s for s in os.listdir(snapshots_dir) if not s.startswith(".")]
                    if snaps:
                        snaps.sort(key=lambda s: os.path.getmtime(os.path.join(snapshots_dir, s)), reverse=True)
                        candidate = os.path.join(snapshots_dir, snaps[0])
                        if os.path.isdir(candidate) and len(os.listdir(candidate)) > 0:
                            return candidate
                except (OSError, PermissionError):
                    pass
    except Exception:
        pass
    return None


class ModelManager:
    _instance = None
    _active_model_id = "minimax_music3_bf16"

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
            if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
                has_mps = True

        tier = HardwareTier.MID_SINGLE_GPU
        desc = "Apple Silicon GPU (Metal/MPS) detected. Optimal for MiniMax Music 3."
        can_minimax = True

        if has_cuda:
            try:
                import torch
                count = torch.cuda.device_count()
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                dev_name = torch.cuda.get_device_name(0)
                if count > 1:
                    tier = HardwareTier.HIGH_DUAL_GPU
                    desc = f"Multi-GPU NVIDIA CUDA detected ({count}x {dev_name}, {vram_gb:.1f}GB VRAM). Full acceleration enabled."
                else:
                    tier = HardwareTier.MID_SINGLE_GPU
                    desc = f"NVIDIA CUDA GPU detected: {dev_name} ({vram_gb:.1f}GB VRAM)."
                can_minimax = vram_gb >= 12.0
            except Exception:
                tier = HardwareTier.MID_SINGLE_GPU
                desc = "NVIDIA CUDA GPU detected. Acceleration enabled."
        elif not has_mps:
            tier = HardwareTier.ENTRY_CPU
            desc = "CPU-only execution detected. Generation will run slower."
            can_minimax = False

        return HardwareProfile(
            os_name=platform.system(),
            architecture=platform.machine(),
            processor=platform.processor() or "Apple Silicon / ARM64",
            has_cuda=has_cuda,
            has_mps=has_mps,
            hardware_tier=tier.value,
            tier_description=desc,
            can_run_minimax_full=can_minimax,
            can_run_heartmula=True
        )

    def get_model_tree(self) -> List[Dict[str, Any]]:
        """Return full catalog of supported audio, image, and video models with local install state."""
        hw = self.detect_hardware()
        is_apple_silicon = (hw.os_name == "Darwin" and hw.architecture in ["arm64", "aarch64"])

        # Definitions catalog
        catalog = [
            # -------------------------------------------------------------
            # AUDIO MODELS: MiniMax Music 3 MLX (Apple Silicon) & Non-MLX
            # -------------------------------------------------------------
            {
                "id": "minimax_music3_mxfp4",
                "name": "MiniMax Music 3 (MLX mxfp4 Small)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "mxfp4 (Fastest / Low Memory)",
                "size_gb": 8.28,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "Apple Silicon (8GB+ RAM)",
                "category": "audio",
                "repo_id": "mlx-community/MiniMax-Music3-mxfp4",
                "is_default": is_apple_silicon
            },
            {
                "id": "minimax_music3_4bit",
                "name": "MiniMax Music 3 (MLX 4-bit)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "4-bit Quantized",
                "size_gb": 14.5,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "Apple Silicon (16GB+ RAM)",
                "category": "audio",
                "repo_id": "mlx-community/MiniMax-Music3-4bit",
                "is_default": False
            },
            {
                "id": "minimax_music3_6bit",
                "name": "MiniMax Music 3 (MLX 6-bit)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "6-bit Quantized",
                "size_gb": 19.8,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "Apple Silicon (24GB+ RAM)",
                "category": "audio",
                "repo_id": "mlx-community/MiniMax-Music3-6bit",
                "is_default": False
            },
            {
                "id": "minimax_music3_8bit",
                "name": "MiniMax Music 3 (MLX 8-bit)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "8-bit Quantized",
                "size_gb": 22.1,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "Apple Silicon (32GB+ RAM)",
                "category": "audio",
                "repo_id": "mlx-community/MiniMax-Music3-8bit",
                "is_default": False
            },
            {
                "id": "minimax_music3_bf16",
                "name": "MiniMax Music 3 (MLX bfloat16 Studio)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "BF16 (Reference Master Quality)",
                "size_gb": 26.55,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "Apple Silicon (36GB+ RAM)",
                "category": "audio",
                "repo_id": "mlx-community/MiniMax-Music3-bf16",
                "is_default": False
            },
            {
                "id": "minimax_music3_comfy_int8",
                "name": "MiniMax Music 3 (PyTorch / CUDA INT8)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "INT8 (CUDA TensorRT / PyTorch)",
                "size_gb": 11.3,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "NVIDIA GPU (12GB+ VRAM) / Linux / Windows",
                "category": "audio",
                "repo_id": "Comfy-Org/MiniMax-Music-3",
                "is_default": not is_apple_silicon and hw.has_cuda
            },
            {
                "id": "minimax_music3_gguf_q4",
                "name": "MiniMax Music 3 (GGUF Q4 Universal)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "GGUF Q4_K_M (Cross-Platform)",
                "size_gb": 7.7,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "CPU / Windows / Linux (16GB RAM)",
                "category": "audio",
                "repo_id": "molbal/Minimax-Music3-GGUF",
                "is_default": not is_apple_silicon and not hw.has_cuda
            },
            {
                "id": "minimax_music3_official_pytorch",
                "name": "MiniMax Music 3 (Official PyTorch Weights)",
                "architecture": "Qwen3 + RVQ8 + Flow Matching DiT",
                "quantization": "BF16 Full",
                "size_gb": 28.5,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "NVIDIA A100 / Dual 3090/4090",
                "category": "audio",
                "repo_id": "MiniMaxAI/MiniMax-Music3",
                "is_default": False
            },
            {
                "id": "heartmula_3b",
                "name": "HeartMuLa-3B (Offline Multitrack)",
                "architecture": "Autoregressive LM + HeartCodec",
                "quantization": "FP16 / BF16",
                "size_gb": 6.2,
                "license": "Apache-2.0",
                "recommended_hardware": "Apple Silicon / 8GB+ VRAM",
                "category": "audio",
                "repo_id": None,
                "is_default": False
            },

            # -------------------------------------------------------------
            # IMAGE MODELS: FLUX.1 & SDXL Turbo (On-Demand Visual Studio)
            # -------------------------------------------------------------
            {
                "id": "flux_1_schnell",
                "name": "FLUX.1 [schnell] (Official 12B)",
                "architecture": "Flow Transformer (12B Rectified Flow)",
                "quantization": "BF16 / FP8",
                "size_gb": 12.0,
                "license": "Apache-2.0",
                "recommended_hardware": "CUDA (16GB+ VRAM) / High RAM",
                "category": "image",
                "repo_id": "black-forest-labs/FLUX.1-schnell",
                "is_default": False
            },
            {
                "id": "flux_1_dev",
                "name": "FLUX.1 [dev] (12B Studio Reference)",
                "architecture": "Flow Transformer (12B Rectified Flow)",
                "quantization": "BF16",
                "size_gb": 12.0,
                "license": "FLUX.1-dev Non-Commercial License",
                "recommended_hardware": "CUDA (24GB+ VRAM) / High RAM",
                "category": "image",
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "is_default": False
            },
            {
                "id": "flux_1_schnell_mlx",
                "name": "FLUX.1 [schnell] (MLX 4-bit)",
                "architecture": "Flow Transformer (12B Rectified Flow)",
                "quantization": "4-bit Quantized (Apple Silicon)",
                "size_gb": 6.5,
                "license": "Apache-2.0",
                "recommended_hardware": "Apple Silicon (16GB+ RAM)",
                "category": "image",
                "repo_id": "mlx-community/FLUX.1-schnell-4bit",
                "is_default": is_apple_silicon
            },
            {
                "id": "sdxl_turbo",
                "name": "SDXL Turbo (1-Step Realtime)",
                "architecture": "Latent Diffusion (Adversarial Diffusion Distillation)",
                "quantization": "FP16",
                "size_gb": 3.5,
                "license": "Stability AI Non-Commercial Research",
                "recommended_hardware": "Any GPU / Apple Silicon (8GB+ RAM)",
                "category": "image",
                "repo_id": "stabilityai/sdxl-turbo",
                "is_default": not is_apple_silicon
            },

            # -------------------------------------------------------------
            # VIDEO MODELS: MiniMax H3, Wan2.1, HunyuanVideo & CogVideoX
            # -------------------------------------------------------------
            {
                "id": "minimax_h3",
                "name": "MiniMax Hailuo 3 (Official 33B Omni-Modal Video)",
                "architecture": "Hailuo DiT Video + Audio Synthesis",
                "quantization": "BF16 Full",
                "size_gb": 24.0,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "High-End GPU (24GB+ VRAM) / Cloud",
                "category": "video",
                "repo_id": "MiniMaxAI/MiniMax-H3",
                "is_default": False
            },
            {
                "id": "minimax_h3_gguf",
                "name": "MiniMax Hailuo 3 (GGUF Q4 Consumer GPU)",
                "architecture": "Hailuo DiT Video + Audio Synthesis",
                "quantization": "GGUF Q4 Quantized",
                "size_gb": 14.2,
                "license": "MiniMax Open Weights",
                "recommended_hardware": "Consumer GPU (16GB VRAM) / Apple Silicon",
                "category": "video",
                "repo_id": "unsloth/MiniMax-H3-GGUF",
                "is_default": False
            },
            {
                "id": "wan2_1_t2v_1_3b",
                "name": "Wan2.1 T2V (1.3B Open Lightweight)",
                "architecture": "Wan DiT Text-to-Video",
                "quantization": "BF16 / FP8",
                "size_gb": 3.3,
                "license": "Apache-2.0",
                "recommended_hardware": "Consumer GPU / Apple Silicon (16GB+ RAM)",
                "category": "video",
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "is_default": True
            },
            {
                "id": "wan2_1_t2v_14b",
                "name": "Wan2.1 T2V (14B Open Flagship)",
                "architecture": "Wan DiT 14B High-Fidelity Text-to-Video",
                "quantization": "BF16 / FP8",
                "size_gb": 28.0,
                "license": "Apache-2.0",
                "recommended_hardware": "NVIDIA GPU (24GB+ VRAM) / Dual GPU",
                "category": "video",
                "repo_id": "Wan-AI/Wan2.1-T2V-14B",
                "is_default": False
            },
            {
                "id": "hunyuan_video",
                "name": "Tencent HunyuanVideo (13B Open DiT)",
                "architecture": "Hunyuan DiT 13B Video Generation",
                "quantization": "BF16 / FP8",
                "size_gb": 24.0,
                "license": "Apache-2.0",
                "recommended_hardware": "NVIDIA GPU (24GB+ VRAM)",
                "category": "video",
                "repo_id": "tencent/HunyuanVideo",
                "is_default": False
            },
            {
                "id": "cogvideox_5b",
                "name": "CogVideoX-1.5-5B (High-Fidelity Video)",
                "architecture": "3D DiT Video Synthesis",
                "quantization": "BF16",
                "size_gb": 10.0,
                "license": "Apache-2.0",
                "recommended_hardware": "GPU (16GB+ VRAM) / Apple Silicon (32GB+ RAM)",
                "category": "video",
                "repo_id": "THUDM/CogVideoX1.5-5B",
                "is_default": False
            }
        ]

        # Scan install status
        results = []
        heartlib_ckpt_dir = os.path.expanduser("../heartlib/ckpt")
        heartmula_installed = os.path.isdir(heartlib_ckpt_dir) and os.path.isdir(os.path.join(heartlib_ckpt_dir, "HeartMuLa-oss-3B"))

        for item in catalog:
            local_path = None
            is_installed = False

            if item["id"] == "heartmula_3b":
                is_installed = heartmula_installed
                local_path = heartlib_ckpt_dir if heartmula_installed else None
            else:
                # First check if this matches DEFAULT_MINIMAX_SNAPSHOT
                if DEFAULT_MINIMAX_SNAPSHOT and os.path.isdir(DEFAULT_MINIMAX_SNAPSHOT):
                    repo_id = item.get("repo_id") or ""
                    if repo_id and (repo_id in DEFAULT_MINIMAX_SNAPSHOT or repo_id.replace("/", "--") in DEFAULT_MINIMAX_SNAPSHOT):
                        is_installed = True
                        local_path = DEFAULT_MINIMAX_SNAPSHOT

                # Next check huggingface cache / data directory
                if not is_installed and item.get("repo_id"):
                    resolved = resolve_hf_snapshot(item["repo_id"])
                    if resolved:
                        is_installed = True
                        local_path = resolved

            is_active = (item["id"] == self._active_model_id) or (local_path and DEFAULT_MINIMAX_SNAPSHOT and os.path.abspath(local_path) == os.path.abspath(DEFAULT_MINIMAX_SNAPSHOT))

            variant = ModelVariant(
                id=item["id"],
                name=item["name"],
                architecture=item["architecture"],
                quantization=item["quantization"],
                size_gb=item["size_gb"],
                is_installed=is_installed,
                local_path=local_path,
                license=item["license"],
                recommended_hardware=item["recommended_hardware"],
                category=item["category"],
                repo_id=item.get("repo_id"),
                is_default=item.get("is_default", False),
                is_active=is_active
            )
            results.append(asdict(variant))

        return results

    def get_active_model(self) -> Dict[str, Any]:
        """Get the currently active model."""
        tree = self.get_model_tree()
        active = next((m for m in tree if m.get("is_active")), None)
        if not active:
            installed = [m for m in tree if m["is_installed"] and m["category"] == "audio"]
            active = installed[0] if installed else tree[0]
        return active

    def set_active_model(self, model_id: str) -> Dict[str, Any]:
        """Set the active model and update runtime path."""
        tree = self.get_model_tree()
        match = next((m for m in tree if m["id"] == model_id or m.get("repo_id") == model_id), None)
        if not match:
            raise ValueError(f"Model ID '{model_id}' not found in catalog.")

        self._active_model_id = match["id"]
        if match["local_path"]:
            global DEFAULT_MINIMAX_SNAPSHOT
            DEFAULT_MINIMAX_SNAPSHOT = match["local_path"]
            os.environ["MINIMAX_MODEL_PATH"] = match["local_path"]
            logger.info(f"Active model switched to {match['name']} at {match['local_path']}")

        return match

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

    def check_auto_download_needed(self) -> Optional[str]:
        """
        Check if an automatic audio download is needed on fresh install.
        RULE: If ANY MiniMax Music 3 model is installed (or active), do NOT download anything!
        Only if ZERO audio models are installed, recommend the smallest audio model.
        Image and video models are STRICTLY user-initiated / on-demand and NEVER auto-downloaded.
        """
        tree = self.get_model_tree()
        audio_installed = [m for m in tree if m["category"] == "audio" and m["is_installed"]]
        if audio_installed:
            logger.info(f"Existing audio models detected ({len(audio_installed)} found, active: {audio_installed[0]['name']}). Skipping auto-download.")
            return None

        # Determine smallest audio model for current platform
        hw = self.detect_hardware()
        if hw.os_name == "Darwin" and hw.architecture in ["arm64", "aarch64"]:
            smallest = "mlx-community/MiniMax-Music3-mxfp4"
        elif hw.has_cuda:
            smallest = "Comfy-Org/MiniMax-Music-3"
        else:
            smallest = "molbal/Minimax-Music3-GGUF"

        logger.info(f"Fresh installation detected with zero installed audio models. Smallest model recommended: {smallest}")
        return smallest


model_manager = ModelManager()

