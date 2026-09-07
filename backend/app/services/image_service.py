"""
AI Image Generation Studio Service.
Provides multi-modal visual artwork generation for song and project covers,
supporting Black Forest Labs FLUX.2 (klein 4B/9B, dev 32B), FLUX.1, and SDXL Turbo,
with graceful high-definition procedural synthesis fallback when weights are offline.
"""

import os
import uuid
import hashlib
import logging
from typing import Optional, Dict, Any
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)

COVERS_DIR = os.path.join("data", "covers")
os.makedirs(COVERS_DIR, exist_ok=True)


class ImageService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageService, cls).__new__(cls)
        return cls._instance

    def get_default_image_model(self) -> str:
        """Get recommended image generation model for current hardware."""
        tree = model_manager.get_model_tree()
        image_models = [m for m in tree if m["category"] == "image"]
        default = next((m for m in image_models if m.get("is_default")), None)
        if default:
            return default["id"]
        return "flux_2_klein_4b"

    def _generate_raster_cover(
        self,
        prompt: str,
        style: str,
        width: int,
        height: int,
        dest_path: str
    ) -> None:
        """Synthesize a high-definition studio album cover raster image (PNG) using Pillow."""
        from PIL import Image, ImageDraw, ImageFilter
        import math

        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        hue1 = (h % 360)
        hue2 = ((h >> 4) % 360)
        hue3 = ((h >> 8) % 360)

        # Base background canvas
        img = Image.new("RGBA", (width, height), (12, 14, 20, 255))
        draw = ImageDraw.Draw(img)

        # Draw multi-stop vertical/diagonal gradient backdrop
        for y in range(height):
            ratio = y / height
            r = int(10 + 25 * math.sin(ratio * math.pi + (hue1 / 60.0)))
            g = int(14 + 35 * math.cos(ratio * math.pi + (hue2 / 60.0)))
            b = int(24 + 45 * math.sin(ratio * math.pi + (hue3 / 60.0)))
            draw.line([(0, y), (width, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255))

        # Ambient color glow orbs
        glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        cx1 = int(width * 0.4 + (h % 150))
        cy1 = int(height * 0.4 + ((h >> 3) % 150))
        rad1 = int(min(width, height) * 0.45)
        glow_draw.ellipse(
            [(cx1 - rad1, cy1 - rad1), (cx1 + rad1, cy1 + rad1)],
            fill=(20, 184, 166, 160)  # Studio Teal
        )

        cx2 = int(width * 0.7 - ((h >> 5) % 150))
        cy2 = int(height * 0.6 - ((h >> 7) % 150))
        rad2 = int(min(width, height) * 0.35)
        glow_draw.ellipse(
            [(cx2 - rad2, cy2 - rad2), (cx2 + rad2, cy2 + rad2)],
            fill=(6, 182, 212, 140)  # Cyan
        )

        # Heavy studio Gaussian blur to create smooth photographic lighting
        glow_blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=60))
        img = Image.alpha_composite(img, glow_blurred)

        # Studio geometric rim lines and album framing
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ol_draw = ImageDraw.Draw(overlay)
        inset = 28
        ol_draw.rounded_rectangle(
            [(inset, inset), (width - inset, height - inset)],
            radius=16,
            outline=(255, 255, 255, 45),
            width=2
        )

        # Concentric vinyl groove accents
        center_x, center_y = width // 2, height // 2
        for r_step in range(40, min(width, height) // 3, 25):
            ol_draw.ellipse(
                [(center_x - r_step, center_y - r_step), (center_x + r_step, center_y + r_step)],
                outline=(255, 255, 255, 20),
                width=1
            )

        img = Image.alpha_composite(img, overlay)
        img.convert("RGB").save(dest_path, "PNG", quality=95)

    def generate_cover(
        self,
        prompt: str,
        style: str = "cinematic album cover",
        aspect_ratio: str = "1:1",
        model_id: Optional[str] = None,
        visual_style: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate or synthesize visual artwork for track or project cover.
        Executes real diffusion models (FLUX.2, FLUX.1, SDXL Turbo) via Diffusers/MFlux
        when weights are downloaded, or renders high-resolution studio raster PNG artwork.
        """
        style = visual_style or style
        chosen_model_id = model_id or self.get_default_image_model()
        tree = model_manager.get_model_tree()
        model_info = next((m for m in tree if m["id"] == chosen_model_id), None)
        
        is_installed = model_info.get("is_installed", False) if model_info else False
        local_path = model_info.get("local_path") if model_info else None
        repo_id = model_info.get("repo_id") if model_info else None

        filename = f"ai_cover_{uuid.uuid4().hex[:10]}.png"
        dest_path = os.path.join(COVERS_DIR, filename)

        width, height = 1024, 1024
        if aspect_ratio == "16:9":
            width, height = 1024, 576
        elif aspect_ratio == "9:16":
            width, height = 576, 1024
        elif aspect_ratio == "4:3":
            width, height = 1024, 768

        engine_used = "studio_procedural_raster"
        diffusion_error: Optional[str] = None

        # Attempt genuine neural diffusion if model weights or repo are available
        if is_installed and (local_path and os.path.exists(local_path) or repo_id):
            try:
                import torch
                from diffusers import AutoPipelineForText2Image

                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

                model_source = local_path if (local_path and os.path.exists(local_path)) else repo_id
                logger.info(f"Running real image diffusion with {model_source} on {device}...")

                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
                pipe.to(device)

                steps = 4 if ("turbo" in chosen_model_id.lower() or "schnell" in chosen_model_id.lower()) else 20
                guidance = 0.0 if ("turbo" in chosen_model_id.lower() or "schnell" in chosen_model_id.lower()) else 3.5

                full_prompt = f"{prompt}, {style}, professional album cover artwork, 8k, photorealistic"
                image = pipe(
                    prompt=full_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height
                ).images[0]
                image.save(dest_path, "PNG")
                engine_used = "diffusers_neural_diffusion"
                logger.info(f"Real diffusion cover rendered successfully at {dest_path}")
            except Exception as e:
                diffusion_error = str(e)
                logger.warning(f"Diffusion generation encountered error ({e}); rendering high-res studio raster PNG.")

        # Fallback to high-definition studio raster PNG if diffusion weights are not installed or threw
        if engine_used != "diffusers_neural_diffusion":
            self._generate_raster_cover(
                prompt=prompt,
                style=style,
                width=width,
                height=height,
                dest_path=dest_path
            )

        return {
            "url": f"/covers/{filename}",
            "file_path": dest_path,
            "dest_path": dest_path,
            "prompt": prompt,
            "style": style,
            "model_id": chosen_model_id,
            "model_name": model_info["name"] if model_info else "FLUX.2 Image Studio",
            "is_installed": is_installed,
            "local_path": local_path,
            "engine": engine_used,
            "format": "png",
            "diffusion_error": diffusion_error
        }


image_service = ImageService()
