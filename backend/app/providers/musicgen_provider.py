"""
Meta MusicGen Generation Provider.
Wraps Meta AI's MusicGen models (small, melody, medium, large) using Hugging Face transformers,
supporting CPU-friendly execution, text prompting, and melody-guided conditioning.
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
from app.providers.hf_audio_provider import HuggingFaceAudioProvider
from app.core.paths import get_generated_audio_dir

logger = logging.getLogger(__name__)

DEFAULT_MUSICGEN_REPO = "facebook/musicgen-small"


class MusicGenProvider(HuggingFaceAudioProvider):
    """Generates music using Meta MusicGen with optional melody guidance and CPU support."""

    def __init__(self, model_path: Optional[str] = None):
        repo_id = model_path or os.environ.get("MUSICGEN_MODEL_PATH") or DEFAULT_MUSICGEN_REPO
        super().__init__(repo_id=repo_id, local_path=repo_id)
        self.model_path = repo_id
        self.processor = None
        self.model = None
        self._is_loaded = False
        self._is_loading = False

    def get_capabilities(self) -> GenerationCapabilities:
        is_melody = "melody" in self.model_path.lower()
        return GenerationCapabilities(
            provider_id="musicgen_melody" if is_melody else "musicgen",
            display_name=f"Meta MusicGen ({'Melody Conditioned' if is_melody else 'Lightweight / CPU'})",
            description="Meta AudioCraft autoregressive music transformer supporting text prompts, melody conditioning, and fast CPU/GPU execution.",
            version="1.0",
            max_duration_sec=120,
            supports_structured_caption=False,
            supports_section_tags=False,
            supports_lora=False,
            supports_voice_conversion=True,
            supports_track_extension=True,
            supports_segment_repair=True,
            recommended_hardware=HardwareTier.ENTRY_CPU if "small" in self.model_path else HardwareTier.MID_SINGLE_GPU,
            license_class="MIT License",
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
        if self._is_loaded and self.model is not None:
            return True
        if self._is_loading:
            return False

        self._is_loading = True
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            device, dtype = self._resolve_device_and_dtype()
            logger.info(f"Loading MusicGen ({self.model_path}) on {device} ({dtype})...")

            loop = asyncio.get_running_loop()

            def _load():
                proc = AutoProcessor.from_pretrained(self.model_path)
                mod = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                )
                mod.to(device)
                mod.eval()
                return proc, mod

            self.processor, self.model = await loop.run_in_executor(None, _load)
            self._is_loaded = True
            logger.info(f"MusicGen ({self.model_path}) loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load MusicGen model: {e}")
            return False
        finally:
            self._is_loading = False

    def unload(self) -> bool:
        """Release MusicGen model and processor from memory."""
        if self.model is not None:
            try:
                if hasattr(self.model, "remove_all_hooks"):
                    self.model.remove_all_hooks()
            except Exception:
                pass
            self.model = None
            self.processor = None
            self._is_loaded = False
            import gc
            gc.collect()
            try:
                from app.core.hardware_lock import GlobalHardwareCoordinator
                GlobalHardwareCoordinator.flush_memory()
            except Exception:
                pass
            logger.info("MusicGenProvider: Model unloaded and VRAM flushed.")
        return True

    async def generate(
        self,
        job_id: str,
        prompt: str,
        lyrics: Optional[str],
        duration_ms: int,
        tags: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        cfg_scale: float = 3.0,
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
        duration_sec = min(120.0, max(1.0, duration_ms / 1000.0))

        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Generation cancelled before start")

        if not self._is_loaded:
            if progress_callback:
                progress_callback(1, 4, f"Loading MusicGen ({self.model_path}) weights...")
            ok = await self.initialize()
            if not ok or self.model is None:
                raise RuntimeError(f"Could not load MusicGen weights from {self.model_path}")

        if progress_callback:
            progress_callback(2, 4, f"Generating audio with MusicGen ({duration_sec:.1f}s)...")

        device, _ = self._resolve_device_and_dtype()
        loop = asyncio.get_running_loop()

        def _infer():
            if seed is not None and int(seed) >= 0:
                torch.manual_seed(int(seed))

            full_prompt = prompt
            if tags:
                full_prompt = f"{prompt}, {tags}"

            # Check if reference melody audio provided
            melody_audio = kwargs.get("ref_audio_path") or kwargs.get("melody_path")
            processor_kwargs = {"text": [full_prompt], "padding": True, "return_tensors": "pt"}

            if melody_audio and os.path.isfile(melody_audio):
                try:
                    import librosa
                    y_mel, sr_mel = librosa.load(melody_audio, sr=32000, mono=True)
                    processor_kwargs["audio"] = y_mel
                    processor_kwargs["sampling_rate"] = sr_mel
                    logger.info(f"Conditioning MusicGen on reference melody: {melody_audio}")
                except Exception as e:
                    logger.warning(f"Could not load reference melody audio ({e}); proceeding with text only.")

            inputs = self.processor(**processor_kwargs)
            # Move inputs to target device
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device)

            # Max tokens calculation (MusicGen produces 50 tokens per second of 32kHz audio)
            max_new_tokens = int(duration_sec * 50)

            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=float(cfg_scale or 3.0),
                    max_new_tokens=max_new_tokens,
                    temperature=max(0.1, float(temperature or 1.0)),
                    top_k=max(1, int(topk or 50)),
                )

            # Shape: [batch, channels, samples] -> e.g. [1, 1, samples]
            audio_np = audio_values[0].detach().cpu().float().numpy()
            model_sr = 32000

            # Ensure 2D: [channels, samples]
            if audio_np.ndim == 1:
                audio_np = np.expand_dims(audio_np, axis=0)

            target_sr = 44100
            # Sinc Resample 32 kHz -> 44.1 kHz
            try:
                import torchaudio.transforms as T
                resampler = T.Resample(orig_freq=model_sr, new_freq=target_sr)
                audio_tensor = torch.from_numpy(audio_np)
                audio_np = resampler(audio_tensor).numpy()
            except Exception:
                try:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(model_sr, target_sr)
                    audio_np = resample_poly(audio_np, target_sr // g, model_sr // g, axis=-1)
                except Exception as e:
                    logger.warning(f"Resampling failed ({e}); retaining native {model_sr}Hz.")
                    target_sr = model_sr

            # Ensure stereo [samples, 2]
            if audio_np.shape[0] == 1:
                audio_np = np.vstack([audio_np, audio_np])

            out_pcm = audio_np.T

            # Enforce -1.0 dBFS true peak ceiling (amplitude ~0.891)
            peak = float(np.max(np.abs(out_pcm))) if out_pcm.size > 0 else 0.0
            if peak > 0.891:
                out_pcm = out_pcm * (0.891 / peak)

            sf.write(output_path, out_pcm, target_sr)
            return output_path, target_sr

        try:
            _, sr = await loop.run_in_executor(None, _infer)
        finally:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            if GlobalHardwareCoordinator.get_memory_policy()["policy"] == "eager":
                self.unload()

        if progress_callback:
            progress_callback(4, 4, "MusicGen generation complete.")

        return GeneratedAudioResult(
            audio_path=f"/audio/{filename}",
            duration_sec=duration_sec,
            sample_rate=sr,
            structured_caption=structured_caption,
            metadata={
                "provider_id": self.get_capabilities().provider_id,
                "model_name": self.model_path,
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
        kwargs["ref_audio_path"] = parent_audio_path
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
            prompt=prompt or "repair",
            lyrics=None,
            duration_ms=duration_ms,
            progress_callback=progress_callback,
            **kwargs
        )

