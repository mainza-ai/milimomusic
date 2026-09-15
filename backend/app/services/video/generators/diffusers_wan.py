"""
Wan 2.1 (14B / 1.3B) Video Diffusion Provider using Hugging Face Diffusers.
Supports both Text-to-Video (WanPipeline) and Image-to-Video (WanImageToVideoPipeline).
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any

from app.core.paths import get_data_dir, get_models_dir
from app.services.video.generators.base import BaseVideoGenerator
from app.services.video.generators.procedural import ProceduralVideoGenerator

logger = logging.getLogger(__name__)


class DiffusersWanGenerator(BaseVideoGenerator):
    def __init__(self, model_size: str = "14b"):
        self.model_size = model_size.lower()
        self._fallback = ProceduralVideoGenerator()
        self._pipe_t2v = None
        self._pipe_i2v = None

    @property
    def name(self) -> str:
        return f"wan_{self.model_size}"

    @property
    def is_available(self) -> bool:
        try:
            import torch
            import diffusers
            return bool(getattr(diffusers, "WanPipeline", None) or getattr(diffusers, "WanImageToVideoPipeline", None))
        except Exception:
            return False

    def _resolve_model_path(self, mode: str = "t2v") -> str:
        """Find local weights or return Hugging Face Hub model ID."""
        repo_prefix = f"Wan-AI/Wan2.1-{'I2V-14B-720P' if mode == 'i2v' else ('T2V-14B' if self.model_size == '14b' else 'T2V-1.3B')}-Diffusers"
        escaped = repo_prefix.replace("/", "__")
        local_cand = [
            str(get_models_dir("video") / escaped),
            str(get_models_dir("video") / "wan2.1"),
            os.path.join(os.getcwd(), "models", "video", escaped),
        ]
        for c in local_cand:
            if os.path.isdir(c) and len(os.listdir(c)) > 0:
                return os.path.abspath(c)
        return repo_prefix

    async def generate_clip(
        self,
        prompt: str,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        **kwargs
    ) -> bool:
        """
        Generate true video diffusion clip using Wan 2.1.
        Uses Image-to-Video if a keyframe image is supplied, otherwise Text-to-Video.
        """
        # Calculate target frame count at 16 fps (Wan default)
        target_fps = 16
        num_frames = max(17, min(81, int(round(duration * target_fps))))

        # If diffusers or weights cannot be loaded, fallback gracefully
        try:
            import torch
            from diffusers.utils import export_to_video, load_image

            device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

            # Check if Image-to-Video mode
            if image_path and os.path.isfile(image_path):
                from diffusers import WanImageToVideoPipeline, AutoencoderKLWan

                model_id = self._resolve_model_path(mode="i2v")
                logger.info(f"Loading Wan 2.1 I2V ({self.model_size}) from {model_id} on {device}...")

                ref_img = load_image(image_path)
                # Compute aspect ratio dimensions conforming to VAE patch size
                mod_value = 16
                target_w = (width // mod_value) * mod_value
                target_h = (height // mod_value) * mod_value
                ref_img = ref_img.resize((target_w, target_h))

                pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.to(device)

                logger.info(f"Diffusing I2V video: prompt='{prompt[:60]}...', frames={num_frames}, size={target_w}x{target_h}")
                output = pipe(
                    image=ref_img,
                    prompt=prompt,
                    negative_prompt=negative_prompt or "blurry, low quality, distorted, watermark",
                    height=target_h,
                    width=target_w,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).frames[0]

                export_to_video(output, out_path, fps=target_fps)
                return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

            else:
                from diffusers import WanPipeline

                model_id = self._resolve_model_path(mode="t2v")
                logger.info(f"Loading Wan 2.1 T2V ({self.model_size}) from {model_id} on {device}...")

                pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.to(device)

                mod_value = 16
                target_w = (width // mod_value) * mod_value
                target_h = (height // mod_value) * mod_value

                logger.info(f"Diffusing T2V video: prompt='{prompt[:60]}...', frames={num_frames}")
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "blurry, low quality, distorted, watermark",
                    height=target_h,
                    width=target_w,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                ).frames[0]

                export_to_video(output, out_path, fps=target_fps)
                return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

        except Exception as e:
            logger.warning(f"Wan 2.1 local diffusion error or offline weights ({e}). Falling back to cinematic procedural scene generation.")
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
