"""
Model Tree, Multi-Modal Catalog, and Hardware Capability Management Service.
Inspects local and Hugging Face model trees for MiniMax Music 3, HeartMuLa,
FLUX.1 image models, and MiniMax H3 / Wan2.1 open video models.
"""

import os
import json
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


CUSTOM_MODELS_PATH = os.path.join("data", "models", "custom_models.json")


def resolve_hf_snapshot(repo_id: str) -> Optional[str]:
    """Find local snapshot directory for a huggingface repo ID.
    Supports local ./data/models, MODEL_DIRECTORY, heartlib/ckpt, HF hub cache, or custom registry.
    """
    try:
        # 0. Check custom_models.json for explicit existing local_path
        if os.path.exists(CUSTOM_MODELS_PATH):
            try:
                with open(CUSTOM_MODELS_PATH, "r", encoding="utf-8") as f:
                    c_list = json.load(f)
                    for c in c_list:
                        if c.get("repo_id") == repo_id and c.get("local_path"):
                            if os.path.isdir(c["local_path"]) and os.listdir(c["local_path"]):
                                return os.path.abspath(c["local_path"])
            except Exception:
                pass

        # 1. Search candidate directories where models are stored
        escaped_repo = repo_id.replace("/", "__")
        search_roots = [
            os.environ.get("MODEL_DIRECTORY"),
            os.path.join("data", "models"),
            os.path.join("backend", "data", "models"),
            os.path.join("..", "data", "models"),
            os.path.join("heartlib", "ckpt"),
            os.path.join("..", "heartlib", "ckpt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "heartlib", "ckpt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")),
        ]
        for root in search_roots:
            if not root:
                continue
            data_path = os.path.join(root, escaped_repo)
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
            # IMAGE MODELS: FLUX.2, FLUX.1 & SDXL Turbo (Visual Studio)
            # -------------------------------------------------------------
            {
                "id": "flux_2_klein_4b_mlx",
                "name": "FLUX.2 [klein] 4B (MLX 4-bit Apple Silicon)",
                "architecture": "Flow Transformer (4B Distilled Rectified Flow)",
                "quantization": "4-bit Quantized (Apple Silicon)",
                "size_gb": 2.8,
                "license": "Apache-2.0",
                "recommended_hardware": "Apple Silicon (8GB+ unified memory)",
                "category": "image",
                "repo_id": "mlx-community/FLUX.2-Klein-4B-4bit",
                "is_default": is_apple_silicon
            },
            {
                "id": "flux_2_klein_4b",
                "name": "FLUX.2 [klein] 4B (Sub-Second Fast Generator)",
                "architecture": "Flow Transformer (4B Distilled Rectified Flow)",
                "quantization": "BF16 Full",
                "size_gb": 7.8,
                "license": "Apache-2.0",
                "recommended_hardware": "CUDA (8GB-12GB VRAM) / Consumer GPU",
                "category": "image",
                "repo_id": "black-forest-labs/FLUX.2-klein-4B",
                "is_default": not is_apple_silicon and not hw.has_cuda
            },
            {
                "id": "flux_2_klein_9b",
                "name": "FLUX.2 [klein] 9B (Distilled High-Fidelity)",
                "architecture": "Flow Transformer (9B Distilled, Qwen3 Embedder)",
                "quantization": "BF16 / FP8",
                "size_gb": 18.0,
                "license": "FLUX.2 Non-Commercial License",
                "recommended_hardware": "CUDA (16GB-24GB VRAM) / High RAM",
                "category": "image",
                "repo_id": "black-forest-labs/FLUX.2-klein-9B",
                "is_default": hw.has_cuda
            },
            {
                "id": "flux_2_dev",
                "name": "FLUX.2 [dev] (32B Flagship Next-Gen)",
                "architecture": "Flow Transformer (32B Rectified Flow + Multi-Ref Editing)",
                "quantization": "BF16",
                "size_gb": 64.0,
                "license": "FLUX.2-dev Non-Commercial License",
                "recommended_hardware": "High-End GPU (24GB+ VRAM with offload / 90GB+ VRAM)",
                "category": "image",
                "repo_id": "black-forest-labs/FLUX.2-dev",
                "is_default": False
            },
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
                "is_default": False
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
                "is_default": False
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

        # Extend with user-downloaded custom models
        catalog.extend(self._load_custom_models())

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

                # Check if item has explicit local_path that exists on disk
                if not is_installed and item.get("local_path") and os.path.isdir(item["local_path"]):
                    try:
                        if os.listdir(item["local_path"]):
                            is_installed = True
                            local_path = os.path.abspath(item["local_path"])
                    except (OSError, PermissionError):
                        pass

                # Next check huggingface cache / data directory / candidate directories
                if not is_installed and item.get("repo_id"):
                    resolved = resolve_hf_snapshot(item["repo_id"])
                    if resolved:
                        is_installed = True
                        local_path = resolved

            cat = item.get("category", "audio")
            if cat == "image":
                is_active = (item["id"] == getattr(self, "_active_image_model_id", None)) or (
                    not getattr(self, "_active_image_model_id", None) and item.get("is_default", False)
                )
            elif cat == "video":
                is_active = (item["id"] == getattr(self, "_active_video_model_id", None)) or (
                    not getattr(self, "_active_video_model_id", None) and item.get("is_default", False)
                )
            else:
                is_active = (item["id"] == self._active_model_id) or (
                    local_path and DEFAULT_MINIMAX_SNAPSHOT and os.path.abspath(local_path) == os.path.abspath(DEFAULT_MINIMAX_SNAPSHOT)
                )

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

    def get_active_model(self, category: str = "audio") -> Dict[str, Any]:
        """Get the currently active model for the given category (audio, image, video)."""
        tree = self.get_model_tree()
        cat_models = [m for m in tree if m.get("category") == category]
        active = next((m for m in cat_models if m.get("is_active")), None)
        if not active:
            installed = [m for m in cat_models if m["is_installed"]]
            active = installed[0] if installed else (cat_models[0] if cat_models else tree[0])
        return active

    def set_active_model(self, model_id: str) -> Dict[str, Any]:
        """Set the active model and update runtime path based on its category."""
        tree = self.get_model_tree()
        match = next((m for m in tree if m["id"] == model_id or m.get("repo_id") == model_id), None)
        if not match:
            raise ValueError(f"Model ID '{model_id}' not found in catalog.")

        cat = match.get("category", "audio")
        if cat == "image":
            self._active_image_model_id = match["id"]
            if match.get("local_path"):
                os.environ["IMAGE_MODEL_PATH"] = match["local_path"]
            logger.info(f"Active image/cover model switched to {match['name']}")
        elif cat == "video":
            self._active_video_model_id = match["id"]
            if match.get("local_path"):
                os.environ["VIDEO_MODEL_PATH"] = match["local_path"]
            logger.info(f"Active video model switched to {match['name']}")
        else:
            self._active_model_id = match["id"]
            if match.get("local_path"):
                global DEFAULT_MINIMAX_SNAPSHOT
                DEFAULT_MINIMAX_SNAPSHOT = match["local_path"]
                os.environ["MINIMAX_MODEL_PATH"] = match["local_path"]
            logger.info(f"Active audio model switched to {match['name']} at {match.get('local_path')}")

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

    def _load_custom_models(self) -> List[Dict[str, Any]]:
        """Load user-registered custom models from disk."""
        if not os.path.isfile(CUSTOM_MODELS_PATH):
            return []
        try:
            with open(CUSTOM_MODELS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read custom_models.json: {e}")
            return []

    def _save_custom_models(self, models: List[Dict[str, Any]]) -> None:
        """Persist user-registered custom models to disk."""
        os.makedirs(os.path.dirname(CUSTOM_MODELS_PATH), exist_ok=True)
        try:
            with open(CUSTOM_MODELS_PATH, "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save custom_models.json: {e}")

    def register_custom_model(self, repo_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a custom downloaded model from Hugging Face."""
        custom_models = self._load_custom_models()
        existing = next((m for m in custom_models if m.get("repo_id") == repo_id), None)
        if existing:
            if metadata and "local_path" in metadata and metadata["local_path"]:
                existing["local_path"] = metadata["local_path"]
                self._save_custom_models(custom_models)
            elif not existing.get("local_path"):
                resolved = resolve_hf_snapshot(repo_id)
                if resolved:
                    existing["local_path"] = resolved
                    self._save_custom_models(custom_models)
            return existing

        name = repo_id.split("/")[-1]
        category = "custom"
        size_gb = 0.0
        architecture = "Hugging Face Model"
        license_type = "Open Weights"

        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo_id=repo_id, files_metadata=True)
            pipe = getattr(info, "pipeline_tag", "") or ""
            if pipe in ["text-to-audio", "audio-to-audio", "automatic-speech-recognition", "voice-conversion"]:
                category = "audio"
            elif pipe in ["text-to-image", "image-to-image"]:
                category = "image"
            elif pipe in ["text-to-video", "image-to-video", "video-to-video"]:
                category = "video"
            else:
                low = repo_id.lower()
                if any(k in low for k in ["music", "audio", "sound", "voice"]):
                    category = "audio"
                elif any(k in low for k in ["flux", "sdxl", "image", "diffusion", "paint"]):
                    category = "image"
                elif any(k in low for k in ["video", "wan", "cogvideo", "hailuo", "hunyuan"]):
                    category = "video"

            total_bytes = sum(getattr(s, "size", 0) or 0 for s in (info.siblings or []))
            if total_bytes > 0:
                size_gb = round(total_bytes / (1024 ** 3), 2)
            architecture = pipe or "Custom HF Architecture"
        except Exception as e:
            logger.warning(f"Could not query HF metadata for {repo_id}: {e}")

        local_path = None
        if metadata:
            if "name" in metadata: name = metadata["name"]
            if "category" in metadata: category = metadata["category"]
            if "size_gb" in metadata: size_gb = metadata["size_gb"]
            if "architecture" in metadata: architecture = metadata["architecture"]
            if "license" in metadata: license_type = metadata["license"]
            if "local_path" in metadata: local_path = metadata["local_path"]

        if not local_path:
            local_path = resolve_hf_snapshot(repo_id)

        model_entry = {
            "id": "custom_" + repo_id.replace("/", "_").replace("-", "_").lower(),
            "name": name,
            "architecture": architecture,
            "quantization": "Custom / FP16 / BF16",
            "size_gb": size_gb,
            "license": license_type,
            "recommended_hardware": "System Compatible",
            "category": category,
            "repo_id": repo_id,
            "local_path": local_path,
            "is_default": False,
            "is_custom": True
        }
        custom_models.append(model_entry)
        self._save_custom_models(custom_models)
        logger.info(f"Registered custom model {repo_id} ({category}) with local_path={local_path}")
        return model_entry

    def update_custom_model(self, model_id_or_repo: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update metadata (category, name, local_path) for a registered custom model."""
        custom_models = self._load_custom_models()
        target = next((m for m in custom_models if m["id"] == model_id_or_repo or m.get("repo_id") == model_id_or_repo), None)
        if not target:
            return None
        for k, v in updates.items():
            if k in ["name", "category", "architecture", "license", "local_path"]:
                target[k] = v
        self._save_custom_models(custom_models)
        logger.info(f"Updated custom model {target['id']}: {updates}")
        return target

    def delete_custom_model(self, model_id_or_repo: str) -> bool:
        """Delete custom model from registry and disk."""
        custom_models = self._load_custom_models()
        target = next((m for m in custom_models if m["id"] == model_id_or_repo or m.get("repo_id") == model_id_or_repo), None)
        if not target:
            return False

        custom_models = [m for m in custom_models if m["id"] != target["id"]]
        self._save_custom_models(custom_models)

        # Remove local disk files if present
        import shutil
        paths_to_check = []
        if target.get("local_path"):
            paths_to_check.append(target["local_path"])
        repo_id = target.get("repo_id")
        if repo_id:
            paths_to_check.append(os.path.join("data", "models", repo_id.replace("/", "__")))
            paths_to_check.append(os.path.join("heartlib", "ckpt", repo_id.replace("/", "__")))
            paths_to_check.append(os.path.join("..", "heartlib", "ckpt", repo_id.replace("/", "__")))

        for p in paths_to_check:
            if p and os.path.isdir(p):
                try:
                    shutil.rmtree(p, ignore_errors=True)
                    logger.info(f"Removed custom model directory: {p}")
                except Exception as e:
                    logger.warning(f"Could not remove {p}: {e}")
        return True

    def search_huggingface(self, query: str, pipeline_tag: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search Hugging Face Hub for models and calculate repository file sizes."""
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import HfApi
        api = HfApi()
        limit = min(max(1, limit), 50)
        kwargs: Dict[str, Any] = {"search": query, "limit": limit, "sort": "downloads"}
        if pipeline_tag:
            kwargs["pipeline_tag"] = pipeline_tag

        models = list(api.list_models(**kwargs))

        # Check known catalog models for pre-computed weights sizes
        catalog_sizes: Dict[str, float] = {}
        try:
            for m in self.get_model_tree():
                if m.get("repo_id") and m.get("size_gb"):
                    catalog_sizes[m["repo_id"]] = float(m["size_gb"])
        except Exception:
            pass

        def _get_size_tuple(repo_id: str) -> tuple[int, float, str]:
            if repo_id in catalog_sizes and catalog_sizes[repo_id] > 0:
                gb = catalog_sizes[repo_id]
                b = int(gb * (1024**3))
                return b, gb, f"{gb:.2f} GB"
            try:
                info = api.model_info(repo_id, files_metadata=True)
                total_b = sum(getattr(s, "size", 0) or 0 for s in (info.siblings or []))
                if total_b >= 1024**3:
                    fmt = f"{total_b / (1024**3):.2f} GB"
                elif total_b >= 1024**2:
                    fmt = f"{total_b / (1024**2):.1f} MB"
                elif total_b > 0:
                    fmt = f"{total_b / 1024:.1f} KB"
                else:
                    fmt = "Unknown"
                return total_b, round(total_b / (1024**3), 2), fmt
            except Exception:
                return 0, 0.0, "Unknown"

        # Concurrently fetch model file sizes
        repo_ids = [m.id for m in models]
        size_map: Dict[str, tuple[int, float, str]] = {}
        if repo_ids:
            with ThreadPoolExecutor(max_workers=min(12, len(repo_ids))) as executor:
                size_results = list(executor.map(_get_size_tuple, repo_ids))
                for rid, sz_tup in zip(repo_ids, size_results):
                    size_map[rid] = sz_tup

        results = []
        for m in models:
            pipe = getattr(m, "pipeline_tag", "") or ""
            category = "custom"
            if pipe in ["text-to-audio", "audio-to-audio", "automatic-speech-recognition", "voice-conversion"]:
                category = "audio"
            elif pipe in ["text-to-image", "image-to-image"]:
                category = "image"
            elif pipe in ["text-to-video", "image-to-video", "video-to-video"]:
                category = "video"
            else:
                low = m.id.lower()
                if any(k in low for k in ["music", "audio", "sound", "voice"]):
                    category = "audio"
                elif any(k in low for k in ["flux", "sdxl", "image", "diffusion", "paint"]):
                    category = "image"
                elif any(k in low for k in ["video", "wan", "cogvideo", "hailuo", "hunyuan"]):
                    category = "video"

            is_installed = resolve_hf_snapshot(m.id) is not None
            sz_bytes, sz_gb, sz_fmt = size_map.get(m.id, (0, 0.0, "Unknown"))

            results.append({
                "repo_id": m.id,
                "name": m.id.split("/")[-1],
                "author": m.id.split("/")[0] if "/" in m.id else "community",
                "downloads": getattr(m, "downloads", 0) or 0,
                "likes": getattr(m, "likes", 0) or 0,
                "pipeline_tag": pipe,
                "category": category,
                "is_installed": is_installed,
                "size_bytes": sz_bytes,
                "size_gb": sz_gb,
                "size_formatted": sz_fmt,
                "last_modified": str(getattr(m, "last_modified", "") or "")
            })
        return results


model_manager = ModelManager()

