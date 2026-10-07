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

# Upstream diffusers bug safeguard: diffusers.pipelines.wan.pipeline_wan_i2v unconditionally
# calls ftfy.fix_text in basic_clean() without checking is_ftfy_available().
try:
    import ftfy
except ImportError:
    import types
    dummy_ftfy = types.ModuleType("ftfy")
    dummy_ftfy.fix_text = lambda x: str(x)
    sys.modules["ftfy"] = dummy_ftfy
    import builtins
    setattr(builtins, "ftfy", dummy_ftfy)

_WAN_PIPELINE_CACHE: Dict[str, Any] = {}


class DiffusersWanGenerator(BaseVideoGenerator):
    def __init__(self, model_size: str = "14b"):
        self.model_size = model_size.lower()
        self._fallback = ProceduralVideoGenerator()

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

    def _resolve_model_path(self, mode: str = "t2v") -> Optional[str]:
        """Find local weights or return None if weights are not installed locally."""
        if mode == "i2v" and self.model_size == "14b":
            repo_prefix = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        elif self.model_size == "1.3b":
            repo_prefix = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        else:
            repo_prefix = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        escaped = repo_prefix.replace("/", "__")
        no_diffusers = escaped.replace("-Diffusers", "")
        clean_tag = repo_prefix.replace("-Diffusers", "")
        local_cand = [
            str(get_models_dir("video") / escaped),
            str(get_models_dir("video") / no_diffusers),
            str(get_models_dir("video") / clean_tag),
            str(get_models_dir("video") / "wan2.1"),
            os.path.join(os.getcwd(), "models", "video", escaped),
            os.path.join(os.getcwd(), "models", "video", no_diffusers),
            os.path.join(os.getcwd(), "models", "video", clean_tag),
            os.path.join(os.getcwd(), "models", "video", "wan2.1"),
        ]
        for c in local_cand:
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "model_index.json")):
                return os.path.abspath(c)

        # Check if already present in huggingface cache
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(repo_prefix, "model_index.json")
            if isinstance(cached, str):
                return repo_prefix
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
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        **kwargs
    ) -> bool:
        """
        Generate true video diffusion clip using Wan 2.1.
        Uses Image-to-Video if a keyframe image is supplied, otherwise Text-to-Video.
        Never triggers silent background model downloads during task execution.
        """
        # Calculate target frame count at 16 fps (Wan default)
        target_fps = 16
        num_frames = max(17, min(81, int(round(duration * target_fps))))

        cancel_event = kwargs.get("cancel_event")
        cancel_check = kwargs.get("cancel_check")

        if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
            raise asyncio.CancelledError("Diffusion cancelled by user.")

        def _configure_pipeline_memory(p):
            if hasattr(p, "enable_attention_slicing"):
                try:
                    p.enable_attention_slicing(slice_size="auto")
                    logger.info("DiffusersWanGenerator: Enabled attention slicing (auto).")
                except Exception as ex:
                    logger.debug(f"Could not enable attention slicing: {ex}")
            if hasattr(p, "vae") and p.vae is not None:
                if hasattr(p.vae, "enable_tiling"):
                    try:
                        p.vae.enable_tiling()
                        logger.info("DiffusersWanGenerator: Enabled VAE tiling.")
                    except Exception as ex:
                        logger.debug(f"Could not enable VAE tiling: {ex}")
                if hasattr(p.vae, "enable_slicing"):
                    try:
                        p.vae.enable_slicing()
                    except Exception:
                        pass
                if device == "mps" and hasattr(p.vae, "to"):
                    try:
                        p.vae.to(dtype=torch.float32)
                        logger.info("DiffusersWanGenerator: Ensured VAE on MPS uses float32 precision.")
                    except Exception as ex:
                        logger.debug(f"Could not convert VAE to float32 on MPS: {ex}")

        def step_end_callback(pipeline, step_index: int, timestep: int, callback_kwargs: dict):
            if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                logger.info(f"DiffusersWanGenerator: Immediate cancellation triggered at step {step_index}.")
                raise asyncio.CancelledError("Diffusion cancelled by user.")
            return callback_kwargs

        # If diffusers or weights cannot be loaded, fallback gracefully
        try:
            import torch
            from diffusers.utils import export_to_video, load_image

            device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

            # Wan architecture constraints:
            # 1. (num_frames - 1) % 4 == 0 (e.g. 17, 21, 25, 29, 33, 49, 65, 81)
            # 2. width and height divisible by 16
            # 3. Dense un-fused self-attention on MPS (Metal) scales as O(S^2)
            # where S = (W/16) * (H/16) * ((num_frames - 1)/4 + 1).
            # At 1280x720 with 65 frames: S = 80 * 45 * 17 = 61,200 tokens -> (40 heads * 61200^2 * 4 bytes) = 558.11 GB!
            # Metal terminates with "Invalid buffer size: 558.11 GB".
            # Clamping on MPS ensures S <= 14,040 tokens, keeping attention memory under safe Metal limits (<8 GB).
            if device == "mps":
                if width >= height:
                    target_w = min(width, 832)
                    target_h = min(height, 480)
                else:
                    target_w = min(width, 480)
                    target_h = min(height, 832)
                num_frames = min(num_frames, 33)
            else:
                target_w = width
                target_h = height

            mod_value = 16
            target_w = max(16, (target_w // mod_value) * mod_value)
            target_h = max(16, (target_h // mod_value) * mod_value)
            num_frames = max(17, ((num_frames - 1) // 4) * 4 + 1)

            # Check if true Image-to-Video is available (Wan 2.1 official I2V is 14B)
            can_i2v = (self.model_size == "14b") and bool(image_path and os.path.isfile(image_path))

            if can_i2v:
                from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
                wan_i2v_mod = sys.modules.get("diffusers.pipelines.wan.pipeline_wan_i2v")
                if wan_i2v_mod and not getattr(wan_i2v_mod, "ftfy", None):
                    import ftfy
                    setattr(wan_i2v_mod, "ftfy", ftfy)

                model_id = self._resolve_model_path(mode="i2v")
                if not model_id:
                    logger.info(f"Wan 2.1 I2V ({self.model_size}) local weights not installed; skipping background download and falling back to procedural animatic.")
                    return False

                cache_key = f"i2v:{model_id}"

                if cache_key in _WAN_PIPELINE_CACHE:
                    logger.info(f"Reusing cached Wan 2.1 I2V pipeline ({model_id}).")
                    pipe = _WAN_PIPELINE_CACHE[cache_key]
                else:
                    logger.info(f"Loading Wan 2.1 I2V ({self.model_size}) from {model_id} on {device} (local files only)...")
                    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True)
                    if device == "cuda":
                        try:
                            pipe.enable_model_cpu_offload()
                            logger.info("Enabled model CPU offload for Wan 2.1 I2V on CUDA.")
                        except Exception as e:
                            logger.warning(f"Could not enable model CPU offload: {e}. Moving to {device}.")
                            pipe.to(device)
                    else:
                        pipe.to(device)
                    _configure_pipeline_memory(pipe)
                    _WAN_PIPELINE_CACHE[cache_key] = pipe

                ref_img = load_image(image_path)
                ref_img = ref_img.resize((target_w, target_h))

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
                    callback_on_step_end=step_end_callback,
                ).frames[0]

                export_to_video(output, out_path, fps=target_fps)

            else:
                from diffusers import WanPipeline

                model_id = self._resolve_model_path(mode="t2v")
                if not model_id:
                    logger.info(f"Wan 2.1 T2V ({self.model_size}) local weights not installed; skipping background download and falling back to procedural animatic.")
                    return False

                cache_key = f"t2v:{model_id}"

                if cache_key in _WAN_PIPELINE_CACHE:
                    logger.info(f"Reusing cached Wan 2.1 T2V pipeline ({model_id}).")
                    pipe = _WAN_PIPELINE_CACHE[cache_key]
                else:
                    logger.info(f"Loading Wan 2.1 T2V ({self.model_size}) from {model_id} on {device} (local files only)...")
                    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True)
                    if device == "cuda":
                        try:
                            pipe.enable_model_cpu_offload()
                            logger.info("Enabled model CPU offload for Wan 2.1 T2V on CUDA.")
                        except Exception as e:
                            logger.warning(f"Could not enable model CPU offload: {e}. Moving to {device}.")
                            pipe.to(device)
                    else:
                        pipe.to(device)
                    _configure_pipeline_memory(pipe)
                    _WAN_PIPELINE_CACHE[cache_key] = pipe

                logger.info(f"Diffusing T2V video: prompt='{prompt[:60]}...', frames={num_frames}, size={target_w}x{target_h}")
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "blurry, low quality, distorted, watermark",
                    height=target_h,
                    width=target_w,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    callback_on_step_end=step_end_callback,
                ).frames[0]

                export_to_video(output, out_path, fps=target_fps)

            # Evict working activation tensors after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

        except asyncio.CancelledError:
            logger.info(f"DiffusersWanGenerator: Task cancelled for {out_path}.")
            raise
        except Exception as e:
            if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                logger.info(f"DiffusersWanGenerator: Task aborted due to cancellation.")
                raise asyncio.CancelledError("Diffusion cancelled by user.")
            logger.warning(f"Wan 2.1 local diffusion error or offline weights ({e}). Falling back to cinematic procedural scene generation.")
            fallback_meta = kwargs.get("fallback_metadata")
            if isinstance(fallback_meta, dict):
                fallback_meta["fallback_used"] = True
                fallback_meta["error"] = str(e)
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

    def unload(self) -> bool:
        """Completely release all cached Wan 2.1 pipelines and accelerator memory."""
        global _WAN_PIPELINE_CACHE
        if not _WAN_PIPELINE_CACHE:
            return True
        for key, pipe in list(_WAN_PIPELINE_CACHE.items()):
            try:
                if hasattr(pipe, "remove_all_hooks"):
                    pipe.remove_all_hooks()
            except Exception:
                pass
        _WAN_PIPELINE_CACHE.clear()
        import gc
        gc.collect()
        try:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            GlobalHardwareCoordinator.flush_memory()
        except Exception:
            pass
        logger.info("DiffusersWanGenerator: Evicted all cached Wan 2.1 pipelines and flushed VRAM.")
        return True


def _evict_wan_pipelines():
    global _WAN_PIPELINE_CACHE
    if not _WAN_PIPELINE_CACHE:
        return
    for key, pipe in list(_WAN_PIPELINE_CACHE.items()):
        try:
            if hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()
        except Exception:
            pass
    _WAN_PIPELINE_CACHE.clear()


try:
    from app.core.hardware_lock import GlobalHardwareCoordinator
    GlobalHardwareCoordinator.register_eviction_hook("video_gen", _evict_wan_pipelines)
except Exception:
    pass

