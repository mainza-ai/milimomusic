"""
Lightricks LTX-Video (0.9B) Real-time Video Diffusion Provider.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any

from app.core.paths import get_data_dir, get_models_dir
from app.services.video.generators.base import BaseVideoGenerator
from app.services.video.generators.procedural import ProceduralVideoGenerator

logger = logging.getLogger(__name__)


class DiffusersLTXGenerator(BaseVideoGenerator):
    def __init__(self):
        self._fallback = ProceduralVideoGenerator()

    @property
    def name(self) -> str:
        return "ltx_video"

    @property
    def is_available(self) -> bool:
        try:
            import torch
            import diffusers
            return bool(getattr(diffusers, "LTXPipeline", None))
        except Exception:
            return False

    def _resolve_model_path(self) -> Optional[str]:
        """Find local LTX-Video weights or return None if weights are not installed locally."""
        repo_id = "Lightricks/LTX-Video"
        local_cand = [
            str(get_models_dir("video") / "Lightricks__LTX-Video"),
            str(get_models_dir("video") / "LTX-Video"),
            str(get_models_dir("video") / "ltx_video"),
            os.path.join(os.getcwd(), "models", "video", "Lightricks__LTX-Video"),
            os.path.join(os.getcwd(), "models", "video", "LTX-Video"),
            os.path.join(os.getcwd(), "models", "video", "ltx_video"),
        ]
        for c in local_cand:
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "model_index.json")):
                return os.path.abspath(c)

        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(repo_id, "model_index.json")
            if isinstance(cached, str):
                return repo_id
        except Exception:
            pass

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
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        **kwargs
    ) -> bool:
        """
        Generates fast cinematic video diffusion with LTX-Video.
        Never triggers silent background model downloads during task execution.
        """
        model_id = self._resolve_model_path()
        if not model_id:
            logger.info("LTX-Video local weights not installed; skipping background download and falling back to procedural animatic.")
            return await self._fallback.generate_clip(
                prompt=prompt,
                duration=duration,
                out_path=out_path,
                width=width,
                height=height,
                image_path=image_path
            )

        pipe = None
        try:
            import torch
            from diffusers import LTXPipeline
            from diffusers.utils import export_to_video

            device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

            logger.info(f"Loading LTX-Video on {device} (local files only)...")
            pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True)
            pipe.to(device)

            fps = 24
            num_frames = max(25, int(round(duration * fps)))
            # Align width and height to 32
            mod = 32
            w = (min(1280, width) // mod) * mod
            h = (min(720, height) // mod) * mod

            cancel_event = kwargs.get("cancel_event")
            cancel_check = kwargs.get("cancel_check")

            def step_end_callback(pipeline, step_index: int, timestep: int, callback_kwargs: dict):
                if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                    logger.info(f"DiffusersLTXGenerator: Immediate cancellation triggered at step {step_index}.")
                    raise asyncio.CancelledError("LTX-Video diffusion cancelled by user.")
                return callback_kwargs

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "worst quality, inconsistent motion, blurry, jittery",
                width=w,
                height=h,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=step_end_callback,
            ).frames[0]

            export_to_video(output, out_path, fps=fps)
            return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

        except asyncio.CancelledError:
            logger.info(f"DiffusersLTXGenerator: Task cancelled for {out_path}.")
            raise
        except Exception as e:
            if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                raise asyncio.CancelledError("LTX-Video diffusion cancelled by user.")
            logger.warning(f"LTX-Video diffusion error ({e}), falling back to procedural scene generation.")
            return await self._fallback.generate_clip(
                prompt=prompt,
                duration=duration,
                out_path=out_path,
                width=width,
                height=height,
                image_path=image_path,
                negative_prompt=negative_prompt,
                **kwargs
            )
        finally:
            if pipe is not None:
                try:
                    if hasattr(pipe, "remove_all_hooks"):
                        pipe.remove_all_hooks()
                except Exception:
                    pass
                del pipe
                import gc
                gc.collect()
                try:
                    from app.core.hardware_lock import GlobalHardwareCoordinator
                    GlobalHardwareCoordinator.flush_memory()
                except Exception:
                    pass

    def unload(self) -> bool:
        """Purge LTX pipeline memory and flush caches."""
        import gc
        gc.collect()
        try:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception:
            pass
        return True

