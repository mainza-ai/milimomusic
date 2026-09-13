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
            return hasattr(diffusers, "LTXPipeline")
        except ImportError:
            return False

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
        """
        try:
            import torch
            from diffusers import LTXPipeline
            from diffusers.utils import export_to_video

            device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

            model_id = "Lightricks/LTX-Video"
            logger.info(f"Loading LTX-Video on {device}...")
            pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)

            fps = 24
            num_frames = max(25, int(round(duration * fps)))
            # Align width and height to 32
            mod = 32
            w = (min(1280, width) // mod) * mod
            h = (min(720, height) // mod) * mod

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "worst quality, inconsistent motion, blurry, jittery",
                width=w,
                height=h,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).frames[0]

            export_to_video(output, out_path, fps=fps)
            return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

        except Exception as e:
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
