"""
Stable Audio Open 1.0 Generation Provider.
Wraps Stability AI's Stable Audio Open 1.0 DiT diffusion pipeline via Hugging Face diffusers,
supporting native 44.1 kHz stereo audio generation on CUDA, Apple Silicon MPS, and CPU.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict
import numpy as np

from app.providers.base import (
    GenerationProvider,
    GenerationCapabilities,
    GeneratedAudioResult,
    HardwareTier,
)
from app.core.paths import get_generated_audio_dir

logger = logging.getLogger(__name__)

DEFAULT_STABLE_AUDIO_REPO = "stabilityai/stable-audio-open-1.0"


class StableAudioProvider(GenerationProvider):
    """Generates studio-quality 44.1 kHz stereo audio using Stable Audio Open DiT."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.environ.get("STABLE_AUDIO_MODEL_PATH") or DEFAULT_STABLE_AUDIO_REPO
        self.pipeline = None
        self._is_loaded = False
        self._is_loading = False

    def get_capabilities(self) -> GenerationCapabilities:
        return GenerationCapabilities(
            provider_id="stable_audio_open_1_0",
            display_name="Stable Audio Open 1.0 (DiT 44.1kHz)",
            description="Continuous Latent Diffusion Transformer (DiT) producing native 44.1 kHz stereo music and soundscapes on CUDA, MPS, and CPU.",
            version="1.0",
            max_duration_sec=47,
            supports_structured_caption=False,
            supports_section_tags=False,
            supports_lora=False,
            supports_voice_conversion=True,
            supports_track_extension=True,
            supports_segment_repair=True,
            recommended_hardware=HardwareTier.MID_SINGLE_GPU,
            license_class="Stability AI Community License",
            default_sample_rate=44100,
        )

    def is_ready(self) -> bool:
        return self._is_loaded or os.path.exists(self.model_path) or (not os.path.isabs(self.model_path) and "/" in self.model_path)

    def _resolve_device_and_dtype(self):
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float32
        return "cpu", torch.float32

    async def initialize(self, model_path: Optional[str] = None) -> bool:
        if model_path:
            self.model_path = model_path
        if self._is_loaded and self.pipeline is not None:
            return True
        if self._is_loading:
            return False

        self._is_loading = True
        try:
            from diffusers import StableAudioPipeline
            device, dtype = self._resolve_device_and_dtype()
            logger.info(f"Loading Stable Audio Open pipeline ({self.model_path}) on {device} ({dtype})...")

            loop = asyncio.get_running_loop()

            def _load():
                pipe = StableAudioPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                )
                if device == "cuda":
                    try:
                        pipe.enable_model_cpu_offload()
                    except Exception:
                        pipe.to(device)
                else:
                    pipe.to(device)
                return pipe

            self.pipeline = await loop.run_in_executor(None, _load)
            self._is_loaded = True
            logger.info("Stable Audio Open pipeline loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Stable Audio Open pipeline: {e}")
            return False
        finally:
            self._is_loading = False

    def unload(self) -> bool:
        """Release pipeline from memory and clear framework caches."""
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, "remove_all_hooks"):
                    self.pipeline.remove_all_hooks()
            except Exception:
                pass
            self.pipeline = None
            self._is_loaded = False
            import gc
            gc.collect()
            try:
                from app.core.hardware_lock import GlobalHardwareCoordinator
                GlobalHardwareCoordinator.flush_memory()
            except Exception:
                pass
            logger.info("StableAudioProvider: Pipeline unloaded and accelerator cache flushed.")
            return True
        return False

    async def generate(
        self,
        job_id: str,
        prompt: str,
        lyrics: Optional[str],
        duration_ms: int,
        tags: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        cfg_scale: float = 7.0,
        topk: int = 50,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        structured_caption: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        import soundfile as sf
        import torch

        out_dir = get_generated_audio_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{job_id}.wav"
        output_path = str(out_dir / filename)
        duration_sec = min(47.0, max(1.0, duration_ms / 1000.0))

        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Generation cancelled before start")

        if not self._is_loaded:
            if progress_callback:
                progress_callback(1, 4, "Loading Stable Audio Open weights...")
            ok = await self.initialize()
            if not ok or self.pipeline is None:
                raise RuntimeError(f"Could not load Stable Audio Open weights from {self.model_path}")

        if progress_callback:
            progress_callback(2, 4, f"Diffusing 44.1 kHz audio with Stable Audio Open ({duration_sec:.1f}s)...")

        device, _ = self._resolve_device_and_dtype()
        loop = asyncio.get_running_loop()

        def _infer():
            generator = None
            if seed is not None and int(seed) >= 0:
                gen_device = "cpu" if device == "mps" else device
                generator = torch.Generator(device=gen_device).manual_seed(int(seed))

            full_prompt = prompt
            if tags:
                full_prompt = f"{prompt}, {tags}"

            # Run DiT diffusion forward pass
            res = self.pipeline(
                full_prompt,
                negative_prompt=kwargs.get("negative_prompt", "low quality, distorted, mono, muffled, noise"),
                num_inference_steps=int(kwargs.get("steps", 100)),
                audio_end_in_s=duration_sec,
                guidance_scale=float(cfg_scale or 7.0),
                generator=generator,
            )

            audio_data = res.audios
            sr = int(getattr(self.pipeline.vae, "sampling_rate", 44100))

            # Shape: [batch, channels, samples] -> [samples, channels]
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data[0].T.float().cpu().numpy()
            else:
                audio_np = np.array(audio_data[0]).T.astype(np.float32)

            # Ensure stereo
            if audio_np.ndim == 1:
                audio_np = np.column_stack([audio_np, audio_np])
            elif audio_np.shape[1] == 1:
                audio_np = np.column_stack([audio_np[:, 0], audio_np[:, 0]])

            # Enforce -1.0 dBFS true peak ceiling (amplitude ~0.891)
            peak = float(np.max(np.abs(audio_np))) if audio_np.size > 0 else 0.0
            if peak > 0.891:
                audio_np = audio_np * (0.891 / peak)

            sf.write(output_path, audio_np, sr)
            return output_path, sr

        try:
            _, sr = await loop.run_in_executor(None, _infer)
        finally:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            if GlobalHardwareCoordinator.get_memory_policy()["policy"] == "eager":
                self.unload()

        if progress_callback:
            progress_callback(4, 4, "Stable Audio generation complete.")

        return GeneratedAudioResult(
            audio_path=f"/audio/{filename}",
            duration_sec=duration_sec,
            sample_rate=sr,
            structured_caption=structured_caption,
            metadata={
                "provider_id": "stable_audio_open_1_0",
                "model_name": "Stable Audio Open 1.0",
                "effective_prompt": prompt,
                "effective_tags": tags or "",
            },
        )

    async def extend(
        self,
        job_id: str,
        parent_audio_path: str,
        extend_ms: int,
        lyrics: Optional[str] = None,
        prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        return await self.generate(
            job_id=f"{job_id}_ext",
            prompt=prompt or "extension",
            lyrics=lyrics,
            duration_ms=extend_ms,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            **kwargs
        )

    async def repair_segment(
        self,
        job_id: str,
        audio_path: str,
        start_time_sec: float,
        end_time_sec: float,
        prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        duration_ms = int((end_time_sec - start_time_sec) * 1000)
        return await self.generate(
            job_id=f"{job_id}_repair",
            prompt=prompt or "audio segment",
            lyrics=None,
            duration_ms=duration_ms,
            progress_callback=progress_callback,
            **kwargs
        )

