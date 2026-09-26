"""
Generic Hugging Face Audio Generation Provider.
Enables execution of any Hugging Face audio/music model (e.g. MusicGen, AudioCraft,
or custom text-to-audio models downloaded via Hugging Face Hub) with real Transformers pipelines.
"""

import os
import asyncio
import logging
from typing import Optional, Callable, Any, Dict
import numpy as np

from app.providers.base import (
    GenerationProvider,
    GenerationCapabilities,
    GeneratedAudioResult,
    HardwareTier
)

logger = logging.getLogger(__name__)


class HuggingFaceAudioProvider(GenerationProvider):
    def __init__(self, repo_id: str, local_path: Optional[str] = None):
        self.repo_id = repo_id
        self.model_id = repo_id
        self.name = f"hf:{repo_id}"
        self.local_path = local_path or repo_id
        self.pipeline = None
        self._is_loaded = False
        self._is_loading = False

    def get_capabilities(self) -> GenerationCapabilities:
        return GenerationCapabilities(
            provider_id=f"hf_{self.repo_id.replace('/', '_')}",
            display_name=f"Hugging Face: {self.repo_id.split('/')[-1]}",
            description=f"Hugging Face text-to-audio generation engine using {self.repo_id}.",
            version="1.0",
            max_duration_sec=120,
            supports_structured_caption=False,
            supports_section_tags=False,
            supports_lora=False,
            supports_voice_conversion=False,
            supports_track_extension=False,
            supports_segment_repair=False,
            recommended_hardware=HardwareTier.MID_SINGLE_GPU,
            license_class="Hugging Face Open Weights",
            default_sample_rate=32000
        )

    def is_ready(self) -> bool:
        return self._is_loaded or os.path.exists(self.local_path)

    def _looks_like_video_weights(self) -> bool:
        """Detect video-diffusion weight dirs so they fail fast with a clear message."""
        try:
            import json as _json
            cfg_path = os.path.join(self.local_path, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                cls = str(cfg.get("_class_name", ""))
                if any(k in cls for k in ("MiniMaxH3", "HunyuanVideo", "CogVideo", "Wan", "LTXVideo", "Mochi")):
                    return True
                if "audio_latents_dim" in cfg and "patch_size" in cfg:
                    return True  # joint video+audio DiT layout (H3 family)
            try:
                names = os.listdir(self.local_path)
            except OSError:
                return False
            blob = " ".join(names).lower()
            if any(k in blob for k in ("minimax-h3", "hailuo", "cogvideo", "hunyuanvideo", "wan2")):
                return True
            return False
        except Exception:
            return False

    async def initialize(self) -> bool:
        if self._is_loaded or self._is_loading:
            return True
        self._is_loading = True
        try:
            if self._looks_like_video_weights():
                logger.error(
                    f"Refusing to load '{self.repo_id}' as audio: directory looks like "
                    f"video-diffusion weights (use the video pipeline, not text-to-audio)."
                )
                return False
            from transformers import pipeline
            import torch

            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading Hugging Face audio pipeline for {self.local_path} on {device}...")
            
            loop = asyncio.get_running_loop()
            self.pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline("text-to-audio", model=self.local_path, device=device)
            )
            self._is_loaded = True
            logger.info(f"Hugging Face audio pipeline loaded for {self.repo_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Hugging Face pipeline for {self.repo_id}: {e}")
            return False
        finally:
            self._is_loading = False

    async def generate(
        self,
        job_id: str,
        prompt: str,
        lyrics: Optional[str],
        duration_ms: int,
        tags: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        cfg_scale: float = 1.5,
        topk: int = 50,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        structured_caption: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        os.makedirs("generated_audio", exist_ok=True)
        filename = f"{job_id}.wav"
        output_path = os.path.join("generated_audio", filename)
        duration_sec = duration_ms / 1000.0

        if not self._is_loaded:
            if progress_callback:
                progress_callback(1, 4, f"Loading weights for {self.repo_id}...")
            ok = await self.initialize()
            if not ok:
                raise RuntimeError(f"Could not load Hugging Face model {self.repo_id}")

        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Generation cancelled before inference")

        if progress_callback:
            progress_callback(2, 4, f"Generating audio with {self.repo_id}...")

        loop = asyncio.get_running_loop()

        def _infer():
            import torch
            import soundfile as sf
            
            if seed is not None and seed >= 0:
                torch.manual_seed(seed)

            # Build conditioning prompt
            full_prompt = prompt
            if tags:
                full_prompt = f"{prompt}, {tags}"

            # Run text-to-audio pipeline with parameters
            # max_new_tokens controls duration (roughly 50 tokens per sec for MusicGen)
            max_tokens = int(duration_sec * 50)
            res = self.pipeline(
                full_prompt,
                forward_params={
                    "max_new_tokens": max_tokens,
                    "temperature": max(0.1, temperature),
                    "guidance_scale": max(1.0, cfg_scale),
                    "top_k": max(1, topk)
                }
            )

            audio_data = res["audio"]
            sr = int(res.get("sampling_rate", 32000))

            # Format raw waveform numpy array
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.detach().cpu().numpy()
            else:
                audio_np = np.array(audio_data, dtype=np.float32)

            # Ensure 2D (channels, samples) or (samples, channels)
            if audio_np.ndim == 1:
                audio_np = np.expand_dims(audio_np, axis=0)
            elif audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
                audio_np = audio_np.T

            target_sr = 44100
            if sr != target_sr:
                try:
                    import torchaudio.transforms as T
                    resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                    audio_tensor = torch.from_numpy(audio_np)
                    audio_np = resampler(audio_tensor).numpy()
                    sr = target_sr
                except Exception:
                    try:
                        from scipy.signal import resample_poly
                        from math import gcd
                        g = gcd(sr, target_sr)
                        audio_np = resample_poly(audio_np, target_sr // g, sr // g, axis=-1)
                        sr = target_sr
                    except Exception as e:
                        logger.warning(f"Could not resample audio from {sr} to {target_sr}: {e}")

            # Ensure stereo (2 channels)
            if audio_np.shape[0] == 1:
                audio_np = np.vstack([audio_np, audio_np])

            # Peak ceiling normalization to -1.0 dBFS (amplitude ~0.891)
            peak = float(np.max(np.abs(audio_np))) if audio_np.size > 0 else 0.0
            if peak > 0.891:
                audio_np = audio_np * (0.891 / peak)

            # Transpose to (samples, channels) for soundfile write
            out_pcm = audio_np.T
            sf.write(output_path, out_pcm, sr)
            return output_path

        try:
            await loop.run_in_executor(None, _infer)
        finally:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            if GlobalHardwareCoordinator.get_memory_policy()["policy"] == "eager":
                self.unload()

        if progress_callback:
            progress_callback(4, 4, "Generation complete.")

        return GeneratedAudioResult(
            audio_path=f"/audio/{filename}",
            duration_sec=duration_sec,
            sample_rate=44100,
            structured_caption=structured_caption,
            metadata={
                "provider_id": self.get_capabilities().provider_id,
                "model_version": self.repo_id,
                "prompt_used": prompt,
            }
        )

    def unload(self) -> bool:
        """Evict pipeline and release PyTorch accelerator memory."""
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "remove_all_hooks"):
                    self.pipeline.model.remove_all_hooks()
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
            logger.info(f"HuggingFaceAudioProvider: Unloaded {self.repo_id} from memory.")
        return True


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
        return GeneratedAudioResult(
            audio_path=audio_path,
            duration_sec=end_time_sec - start_time_sec,
            metadata={"repaired_range": [start_time_sec, end_time_sec]}
        )

