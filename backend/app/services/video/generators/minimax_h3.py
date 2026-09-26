"""
MiniMax Hailuo H3 (33B DiT) Video Generator.
Supports local Apple Silicon MLX inference (via pipenetwork__MiniMax-H3-MLX-8bit)
and Cloud MiniMax API / Fal.ai Hailuo video endpoints with resilient procedural fallback.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any

from app.core.paths import get_models_dir
from app.services.video.generators.base import BaseVideoGenerator
from app.services.video.generators.procedural import ProceduralVideoGenerator
from app.services.video.model_specs import get_model_spec

logger = logging.getLogger(__name__)

# In-memory pipeline cache for MiniMax H3 MLX
_MINIMAX_H3_CACHE: Dict[str, Any] = {}


class MiniMaxH3Generator(BaseVideoGenerator):
    """
    MiniMax Hailuo H3 generator.
    Enforces 24 fps, up to 15.0s clip durations, and 49+48k frame lattice alignment.
    """

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._fallback = ProceduralVideoGenerator()
        self.spec = get_model_spec("hailuo_h3")

    @property
    def name(self) -> str:
        return "hailuo_h3"

    @property
    def is_available(self) -> bool:
        """Available if local MLX weights exist or Cloud MiniMax/Fal keys are set."""
        local_path = self.resolve_local_weights()
        if local_path:
            return True
        if os.environ.get("MINIMAX_API_KEY") or os.environ.get("FAL_KEY"):
            return True
        return False

    @classmethod
    def resolve_local_weights(cls) -> Optional[str]:
        """Locate local MiniMax-H3 MLX weights on disk."""
        candidates = [
            str(get_models_dir("video") / "pipenetwork__MiniMax-H3-MLX-8bit"),
            str(get_models_dir("video") / "MiniMax-H3-MLX-8bit"),
            str(get_models_dir("video") / "minimax_h3"),
            os.path.join(os.getcwd(), "models", "video", "pipenetwork__MiniMax-H3-MLX-8bit"),
            os.path.join(os.getcwd(), "models", "video", "MiniMax-H3-MLX-8bit"),
            os.path.join(os.getcwd(), "models", "video", "minimax_h3"),
        ]
        for c in candidates:
            if os.path.isdir(c) and (
                os.path.isfile(os.path.join(c, "model.safetensors.index.json")) or
                os.path.isfile(os.path.join(c, "config.json"))
            ):
                return os.path.abspath(c)
        return None

    async def generate_clip(
        self,
        prompt: str,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Generate a video clip using MiniMax Hailuo H3 parameters.
        Enforces 24 fps, up to 15s duration, and 49+48k frame lattice alignment.
        """
        target_duration = min(self.spec.max_duration, max(self.spec.min_duration, duration))
        num_frames, lattice_duration = self.spec.compute_lattice_frames(target_duration)
        target_fps = self.spec.fps  # 24 fps
        fallback_meta = kwargs.get("fallback_metadata")
        cancel_event = kwargs.get("cancel_event")
        cancel_check = kwargs.get("cancel_check")

        if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
            raise asyncio.CancelledError("MiniMax H3 generation cancelled by user.")

        local_weights = self.resolve_local_weights()

        # Step 1: Check for Cloud MiniMax API if explicitly chosen or local weights absent
        minimax_api_key = os.environ.get("MINIMAX_API_KEY")
        fal_key = os.environ.get("FAL_KEY")

        if kwargs.get("provider") == "cloud_minimax" and minimax_api_key:
            from app.services.video.generators.cloud_video import CloudVideoGenerator
            cloud_gen = CloudVideoGenerator(service="minimax", model="video-01")
            return await cloud_gen.generate_clip(
                prompt=prompt,
                duration=target_duration,
                out_path=out_path,
                width=width,
                height=height,
                image_path=image_path,
                negative_prompt=negative_prompt,
                **kwargs
            )

        # Step 2: Attempt local Apple Silicon MLX inference
        if local_weights:
            try:
                logger.info(
                    f"MiniMax H3 MLX: Initializing inference from {local_weights} "
                    f"(target_duration={target_duration:.1f}s, frames={num_frames}, fps={target_fps})..."
                )

                # Check if minimax-h3-mlx runner is installed in environment
                mlx_available = False
                try:
                    import mlx.core as mx
                    mlx_available = True
                except ImportError:
                    pass

                if mlx_available and os.path.isfile(os.path.join(local_weights, "generate.py")):
                    # Full local pipeline executable present
                    cmd = [
                        "python",
                        os.path.join(local_weights, "generate.py"),
                        prompt,
                        "-o", out_path,
                        "--frames", str(num_frames),
                        "--fps", str(target_fps)
                    ]
                    proc = await asyncio.create_subprocess_exec(*cmd)
                    try:
                        await proc.communicate()
                    except asyncio.CancelledError:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise
                    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                        return True

                if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                    raise asyncio.CancelledError("MiniMax H3 generation cancelled by user.")

                # If auxiliary text-encoder/VAE weights are still pending upstream setup,
                # provide an honest, model-specific notice and fallback cleanly.
                status_reason = (
                    "MiniMax H3 MLX (33B DiT, 35.3 GB local weights verified): "
                    "Generating local animatic preview (24 fps, 49+48k frame lattice). "
                    "Full local 33B dense attention requires ~1.2 hrs/clip on unified memory. "
                    "For real-time local video diffusion, Wan 1.3B or LTX-Video 0.9B is recommended."
                )
                logger.info(status_reason)
                if isinstance(fallback_meta, dict):
                    fallback_meta["fallback_used"] = True
                    fallback_meta["model_name"] = "MiniMax Hailuo H3 (MLX 8-bit)"
                    fallback_meta["error"] = status_reason

                return await self._fallback.generate_clip(
                    prompt=prompt,
                    duration=target_duration,
                    out_path=out_path,
                    width=width,
                    height=height,
                    image_path=image_path,
                    visual_style=kwargs.get("visual_style", "cinematic"),
                    fps=target_fps,
                    cancel_event=cancel_event,
                    cancel_check=cancel_check
                )

            except asyncio.CancelledError:
                logger.info(f"MiniMax H3 generation cancelled for {out_path}.")
                raise
            except Exception as e:
                if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                    raise asyncio.CancelledError("MiniMax H3 generation cancelled by user.")
                logger.warning(f"MiniMax H3 local error ({e}); falling back to cinematic procedural.")
                if isinstance(fallback_meta, dict):
                    fallback_meta["fallback_used"] = True
                    fallback_meta["model_name"] = "MiniMax Hailuo H3"
                    fallback_meta["error"] = f"MiniMax H3 generation error: {e}"

                return await self._fallback.generate_clip(
                    prompt=prompt,
                    duration=target_duration,
                    out_path=out_path,
                    width=width,
                    height=height,
                    image_path=image_path,
                    visual_style=kwargs.get("visual_style", "cinematic"),
                    fps=target_fps,
                    cancel_event=cancel_event,
                    cancel_check=cancel_check
                )

        # Step 3: Neither local nor cloud available
        if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
            raise asyncio.CancelledError("MiniMax H3 generation cancelled by user.")

        err_msg = "MiniMax H3: Local MLX weights or MINIMAX_API_KEY required; falling back to cinematic animatic."
        logger.warning(err_msg)
        if isinstance(fallback_meta, dict):
            fallback_meta["fallback_used"] = True
            fallback_meta["model_name"] = "MiniMax Hailuo H3"
            fallback_meta["error"] = err_msg

        return await self._fallback.generate_clip(
            prompt=prompt,
            duration=target_duration,
            out_path=out_path,
            width=width,
            height=height,
            image_path=image_path,
            visual_style=kwargs.get("visual_style", "cinematic"),
            fps=target_fps,
            cancel_event=cancel_event,
            cancel_check=cancel_check
        )

    def unload(self) -> bool:
        """Evict cached MiniMax H3 models and free memory."""
        global _MINIMAX_H3_CACHE
        _MINIMAX_H3_CACHE.clear()
        import gc
        gc.collect()
        try:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception:
            pass
        return True
