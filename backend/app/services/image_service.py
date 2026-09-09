"""
AI Image Generation Studio Service.
Provides multi-modal visual artwork generation for song and project covers
(plus cinematic per-scene video background stills via a separate entrypoint),
supporting Black Forest Labs FLUX.2 (klein 4B/9B, dev 32B), FLUX.1, and SDXL Turbo,
with graceful high-definition procedural synthesis fallback when weights are offline.
"""

import os
import logging
import uuid
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from app.services.model_manager import model_manager
from app.core.paths import get_data_dir

logger = logging.getLogger(__name__)

COVERS_DIR = str(get_data_dir() / "covers")
os.makedirs(COVERS_DIR, exist_ok=True)
os.makedirs(os.path.join("data", "covers"), exist_ok=True)

# Per-scene video backgrounds live here — never in COVERS_DIR — so B-roll
# stills are not served as public cover art and never get title overlays.
# Kept on disk for render debugging (no auto-deletion).
SCENE_STILLS_DIR = str(get_data_dir() / "video_cache" / "scene_stills")
os.makedirs(SCENE_STILLS_DIR, exist_ok=True)

# Prompt suffixes are per-surface: album-cover language must never leak into
# cinematic B-roll stills (and vice versa).
COVER_PROMPT_SUFFIX = (
    "professional album cover artwork, 8k, photorealistic, "
    "no text, no words, no letters, no watermark, no typography"
)
SCENE_PROMPT_SUFFIX = (
    "cinematic film still, wide establishing shot, natural composition, "
    "no text, no watermark, photorealistic"
)

# Title overlay bounds (production guard: never render unbounded text).
MAX_TITLE_LENGTH = 120

# Dedicated single-thread worker to isolate MLX Core stream allocation and thread context
_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_image_gen")


class ImageService:
    _instance = None
    _loaded_mlx_pipeline = None
    _loaded_mlx_model_id = None
    _loaded_diffusers_pipeline = None
    _loaded_diffusers_model_id = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageService, cls).__new__(cls)
        return cls._instance

    def get_default_image_model(self) -> str:
        """Get currently active or recommended image generation model."""
        active = model_manager.get_active_model("image")
        if active and active.get("id"):
            return active["id"]
        tree = model_manager.get_model_tree()
        image_models = [m for m in tree if m["category"] == "image"]
        default = next((m for m in image_models if m.get("is_default")), None)
        if default:
            return default["id"]
        return "custom_aitrader_flux2_klein_9b_mlx_4bit"

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

    def _run_mlx_diffusion(
        self,
        model_source: str,
        is_9b: bool,
        prompt: str,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        dest_path: str,
    ) -> None:
        """Executes MLX FLUX.2 inference strictly inside the dedicated _mlx_executor thread."""
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein

        if self._loaded_mlx_model_id != model_source or self._loaded_mlx_pipeline is None:
            logger.info(f"Loading Flux2Klein model from {model_source} in dedicated MLX thread (is_9b={is_9b})...")
            cfg = ModelConfig.flux2_klein_9b() if is_9b else ModelConfig.flux2_klein_4b()
            self._loaded_mlx_pipeline = Flux2Klein(model_path=model_source, model_config=cfg)
            self._loaded_mlx_model_id = model_source
        else:
            logger.info(f"Reusing in-memory Flux2Klein model ({model_source}) in dedicated MLX thread...")

        logger.info(f"Generating image via mflux Flux2Klein (steps={steps}, seed={seed}, size={width}x{height})...")
        image = self._loaded_mlx_pipeline.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=guidance,
        )
        image.save(dest_path)

    def _resolve_image_model(
        self, model_id: Optional[str] = None
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str], Optional[str], bool]:
        """Resolve an image model id to (chosen_id, info, local_path, repo_id, installed).

        Falls back to the active installed image model when the requested one
        has no local weights. Shared by covers and scene backgrounds.
        """
        chosen_model_id = model_id or self.get_default_image_model()
        tree = model_manager.get_model_tree()
        model_info = next(
            (m for m in tree if m["id"] == chosen_model_id or m.get("repo_id") == chosen_model_id),
            None,
        )

        is_installed = False
        local_path = None
        repo_id = None
        if model_info:
            local_path = model_info.get("local_path")
            repo_id = model_info.get("repo_id")
            is_installed = bool(model_info.get("is_installed") or (local_path and os.path.exists(local_path)))

        if not is_installed:
            active_info = model_manager.get_active_model("image")
            if active_info:
                act_path = active_info.get("local_path")
                act_installed = bool(active_info.get("is_installed") or (act_path and os.path.exists(act_path)))
                if act_installed:
                    model_info = active_info
                    chosen_model_id = active_info["id"]
                    local_path = act_path
                    repo_id = active_info.get("repo_id")
                    is_installed = True

        return chosen_model_id, model_info, local_path, repo_id, is_installed

    def _render_diffusion_image(
        self,
        full_prompt: str,
        width: int,
        height: int,
        chosen_model_id: str,
        local_path: Optional[str],
        repo_id: Optional[str],
        is_installed: bool,
        dest_path: str,
        log_label: str = "image",
    ) -> Dict[str, Any]:
        """Attempt MLX FLUX.2 then PyTorch-diffusers text-to-image rendering.

        Shared diffusion core for covers and video scene backgrounds. Performs
        NO raster fallback and writes NO response envelope — the caller decides
        what a diffusion miss means (cover: raster fallback; scene: procedural
        ffmpeg branch in the video service).
        """
        engine_used = "none"
        diffusion_error: Optional[str] = None
        model_source = local_path if (local_path and os.path.exists(local_path)) else repo_id

        if is_installed and model_source:
            is_mlx_flux2 = (
                "flux2" in chosen_model_id.lower()
                or "flux_2" in chosen_model_id.lower()
                or "klein" in chosen_model_id.lower()
                or "mlx" in chosen_model_id.lower()
                or (local_path and "mlx" in local_path.lower())
                or (local_path and "FLUX2" in local_path)
                or (repo_id and "mlx" in repo_id.lower())
                or (repo_id and "FLUX2" in repo_id)
            )

            if is_mlx_flux2:
                try:
                    is_9b = (
                        "9b" in chosen_model_id.lower()
                        or (repo_id and "9b" in repo_id.lower())
                        or (local_path and "9b" in local_path.lower())
                    )
                    seed = int(uuid.uuid4().hex[:8], 16) % 1000000
                    future = _mlx_executor.submit(
                        self._run_mlx_diffusion,
                        model_source=model_source,
                        is_9b=is_9b,
                        prompt=full_prompt,
                        steps=4,
                        guidance=1.0,
                        width=width,
                        height=height,
                        seed=seed,
                        dest_path=dest_path,
                    )
                    future.result(timeout=180)
                    engine_used = "mflux_flux2_mlx"
                    logger.info(f"MLX FLUX.2 Klein diffusion {log_label} rendered at {dest_path}")
                except Exception as e:
                    self._loaded_mlx_pipeline = None
                    self._loaded_mlx_model_id = None
                    diffusion_error = str(e)
                    logger.warning(f"MLX diffusion {log_label} error ({e}); trying diffusers fallback.")

            if engine_used == "none":
                try:
                    import torch
                    from diffusers import AutoPipelineForText2Image

                    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

                    logger.info(f"Running diffusers text2image with {model_source} on {device}...")
                    if self._loaded_diffusers_model_id == model_source and self._loaded_diffusers_pipeline is not None:
                        pipe = self._loaded_diffusers_pipeline
                    else:
                        pipe = AutoPipelineForText2Image.from_pretrained(
                            model_source,
                            torch_dtype=dtype,
                            use_safetensors=True,
                        )
                        pipe.to(device)
                        self._loaded_diffusers_pipeline = pipe
                        self._loaded_diffusers_model_id = model_source

                    lowered = chosen_model_id.lower()
                    steps = 4 if ("turbo" in lowered or "schnell" in lowered) else 20
                    guidance = 0.0 if ("turbo" in lowered or "schnell" in lowered) else 3.5

                    image = pipe(
                        prompt=full_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                    ).images[0]
                    image.save(dest_path, "PNG")
                    engine_used = "diffusers_neural_diffusion"
                    logger.info(f"Diffusers diffusion {log_label} rendered at {dest_path}")
                except Exception as e:
                    diffusion_error = f"{diffusion_error}; Diffusers fallback error: {e}" if diffusion_error else str(e)
                    logger.warning(f"Diffusion {log_label} error ({e}).")

        return {"engine": engine_used, "diffusion_error": diffusion_error}

    @staticmethod
    def _sanitize_title(title: str) -> str:
        """Strip control chars and bound length so overlay text is always safe."""
        cleaned = "".join(ch for ch in title.strip() if ch.isprintable() or ch in (" ", "\t"))
        cleaned = " ".join(cleaned.split())
        return cleaned[:MAX_TITLE_LENGTH]

    @staticmethod
    def strip_text_instructions(prompt: str) -> str:
        """Remove quoted substrings and typography-adjacent clauses from a visual prompt.

        Defense-in-depth behind the LLM system prompt: quoted song titles, lyric
        lines, or 'with the text …' instructions fed verbatim to diffusion invite
        garbled AI-rendered lettering that competes with the Pillow overlay.
        """
        import re

        cleaned = re.sub(r"'[^']*'", "", prompt)
        cleaned = re.sub(r'"[^"]*"', "", cleaned)
        cleaned = re.sub(
            r"[^.,;]*\b(text|typography|typographic|font|lettering|letters|words|"
            r"written|inscription|caption|headline|by\s+[A-Z][\w ]*)\b[^.,;]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return " ".join(cleaned.split())

    def _overlay_title(
        self,
        image_path: str,
        title: str,
        width: int,
        height: int,
        artist: Optional[str] = None,
    ) -> None:
        """Render the track title (plus optional artist byline) onto a finished cover.

        Cover-only: never called on the video scene-background path. Bottom-center
        placement with a legibility band and drop shadow. Raises on failure so the
        caller can log it — overlay failure must never fail cover generation.
        """
        from PIL import Image, ImageDraw, ImageFont

        clean_title = self._sanitize_title(title)
        if not clean_title:
            return
        clean_artist = self._sanitize_title(artist) if artist else ""

        def _fit_font(text: str, size: int, max_w: int):
            size = max(12, size)
            while size > 12:
                try:
                    f = ImageFont.truetype(font_path, size)
                except (OSError, IOError):
                    return ImageFont.load_default(), size
                bbox = f.getbbox(text)
                if bbox[2] - bbox[0] <= max_w:
                    return f, size
                size -= 2
            try:
                return ImageFont.truetype(font_path, 12), 12
            except (OSError, IOError):
                return ImageFont.load_default(), 12

        img = Image.open(image_path).convert("RGBA")

        font_path = os.path.join(os.path.dirname(__file__), "fonts", "Inter-Bold.ttf")
        max_text_width = int(width * 0.80)

        font, _ = _fit_font(clean_title, int(height * 0.08), max_text_width)
        if clean_artist:
            artist_font, _ = _fit_font(clean_artist, int(height * 0.045), max_text_width)
        else:
            artist_font = None

        title_bbox = font.getbbox(clean_title)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        if clean_artist and artist_font is not None:
            artist_bbox = artist_font.getbbox(clean_artist)
            artist_w = artist_bbox[2] - artist_bbox[0]
            artist_h = artist_bbox[3] - artist_bbox[1]
            gap = int(height * 0.008)
        else:
            artist_w, artist_h, gap = 0, 0, 0

        block_h = title_h + gap + artist_h
        bottom_margin = int(height * 0.08)
        y_top = height - bottom_margin - block_h
        x_title = (width - title_w) // 2
        x_artist = (width - artist_w) // 2 if clean_artist else 0

        band_padding = int(height * 0.02)
        band = Image.new("RGBA", (width, block_h + band_padding * 2), (0, 0, 0, 120))
        img.paste(band, (0, y_top - band_padding), band)

        draw = ImageDraw.Draw(img)
        draw.text((x_title + 2, y_top + 2), clean_title, font=font, fill=(0, 0, 0, 180))
        draw.text((x_title, y_top), clean_title, font=font, fill=(255, 255, 255, 240))
        if clean_artist and artist_font is not None:
            y_artist = y_top + title_h + gap
            draw.text((x_artist + 1, y_artist + 1), clean_artist, font=artist_font, fill=(0, 0, 0, 160))
            draw.text((x_artist, y_artist), clean_artist, font=artist_font, fill=(255, 255, 255, 200))

        img.convert("RGB").save(image_path, "PNG", quality=95)

    def generate_cover(
        self,
        prompt: str,
        style: str = "cinematic album cover",
        aspect_ratio: str = "1:1",
        model_id: Optional[str] = None,
        visual_style: Optional[str] = None,
        title: Optional[str] = None,
        artist: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate or synthesize visual artwork for track or project cover.
        Executes real diffusion models (FLUX.2, FLUX.1, SDXL Turbo) via MLX (mflux) / Diffusers
        when weights are downloaded, or renders high-resolution studio raster PNG artwork.
        When `title` is provided it is composited onto the finished image with
        Pillow (never by the diffusion model), with `artist` as an optional
        smaller byline beneath it. Overlay failure is non-fatal. When `title`
        is absent no text is rendered at all.
        """
        style = visual_style or style
        chosen_model_id, model_info, local_path, repo_id, is_installed = self._resolve_image_model(model_id)

        filename = f"ai_cover_{uuid.uuid4().hex[:10]}.png"
        dest_path = os.path.join(COVERS_DIR, filename)

        width, height = 1024, 1024
        if aspect_ratio == "16:9":
            width, height = 1024, 576
        elif aspect_ratio == "9:16":
            width, height = 576, 1024
        elif aspect_ratio == "4:3":
            width, height = 1024, 768

        safe_prompt = self.strip_text_instructions(prompt)
        full_prompt = f"{safe_prompt}, {style}, {COVER_PROMPT_SUFFIX}"
        diffusion_res = self._render_diffusion_image(
            full_prompt=full_prompt,
            width=width,
            height=height,
            chosen_model_id=chosen_model_id,
            local_path=local_path,
            repo_id=repo_id,
            is_installed=is_installed,
            dest_path=dest_path,
            log_label="cover",
        )
        engine_used = diffusion_res["engine"]
        diffusion_error = diffusion_res["diffusion_error"]
        if engine_used == "none":
            engine_used = "studio_procedural_raster"

        # Fallback to high-definition studio raster PNG if diffusion weights are not installed or threw
        if engine_used == "studio_procedural_raster":
            self._generate_raster_cover(
                prompt=prompt,
                style=style,
                width=width,
                height=height,
                dest_path=dest_path
            )

        # Overlay title (+ optional artist byline) on the cover (Pillow composite).
        # Non-fatal by design: a cover without text beats a failed request.
        overlay_title = self._sanitize_title(title) if title else ""
        overlay_artist = self._sanitize_title(artist) if artist else ""
        if overlay_title:
            try:
                self._overlay_title(dest_path, overlay_title, width, height, artist=overlay_artist or None)
                logger.info(f"Title '{overlay_title}' overlaid on cover at {dest_path}")
            except Exception as e:
                logger.warning(f"Title overlay failed (continuing without text): {e}")

        # Mirror to backend/data/covers for backwards compatibility
        backend_dest = os.path.abspath(os.path.join("data", "covers", filename))
        if os.path.abspath(dest_path) != backend_dest and os.path.exists(dest_path):
            try:
                shutil.copy2(dest_path, backend_dest)
            except Exception:
                pass

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
            "diffusion_error": diffusion_error,
            "title": overlay_title or None,
            "artist": overlay_artist or None,
        }

    def generate_scene_background(
        self,
        prompt: str,
        style: str = "cinematic film still",
        width: int = 1280,
        height: int = 720,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Render a per-scene video background still (B-roll / Ken Burns source).

        Distinct from `generate_cover`: cinematic prompt suffix, exact video
        dimensions, output under `video_cache/scene_stills/` (never `covers/`),
        NO public URL, NO raster album-art fallback, NO title overlay. When
        diffusion is unavailable the result reports `ok=False` and the video
        service falls back to its procedural ffmpeg branch.
        """
        chosen_model_id, model_info, local_path, repo_id, is_installed = self._resolve_image_model(model_id)

        width = max(320, min(1920, int(width)))
        height = max(320, min(1920, int(height)))

        filename = f"scene_bg_{uuid.uuid4().hex[:10]}.png"
        dest_path = os.path.join(SCENE_STILLS_DIR, filename)

        full_prompt = f"{prompt}, {style}, {SCENE_PROMPT_SUFFIX}"
        diffusion_res = self._render_diffusion_image(
            full_prompt=full_prompt,
            width=width,
            height=height,
            chosen_model_id=chosen_model_id,
            local_path=local_path,
            repo_id=repo_id,
            is_installed=is_installed,
            dest_path=dest_path,
            log_label="scene background",
        )
        engine_used = diffusion_res["engine"]
        diffusion_error = diffusion_res["diffusion_error"]

        ok = engine_used != "none" and os.path.isfile(dest_path)
        if ok:
            logger.info(f"Scene background still rendered at {dest_path}")
        else:
            logger.info("Scene background diffusion unavailable; caller should use procedural fallback.")

        return {
            "ok": ok,
            "dest_path": dest_path if ok else None,
            "prompt": prompt,
            "style": style,
            "model_id": chosen_model_id,
            "model_name": model_info["name"] if model_info else "FLUX.2 Image Studio",
            "engine": engine_used,
            "diffusion_error": diffusion_error,
        }


image_service = ImageService()
