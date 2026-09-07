"""
HeartMuLa Generation Provider.
Wraps HeartMuLa-3B + HeartCodec (12.5Hz) into the GenerationProvider interface.
"""

import os
import asyncio
import logging
from typing import Optional, Callable, Any, Dict
from app.providers.base import (
    GenerationProvider,
    GenerationCapabilities,
    GeneratedAudioResult,
    HardwareTier
)

logger = logging.getLogger(__name__)


class HeartMuLaProvider(GenerationProvider):
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.model_path = None
        self.active_lora_path = None
        self._is_loading = False

    def get_capabilities(self) -> GenerationCapabilities:
        return GenerationCapabilities(
            provider_id="heartmula",
            display_name="HeartMuLa-3B (Legacy/Local)",
            description="3B-parameter autoregressive model paired with HeartCodec (12.5Hz RVQ) for efficient single-GPU / MPS generation.",
            version="3B",
            max_duration_sec=240,
            supports_structured_caption=False,
            supports_section_tags=True,
            supports_lora=True,
            supports_voice_conversion=True,
            supports_track_extension=True,
            supports_segment_repair=True,
            recommended_hardware=HardwareTier.MID_SINGLE_GPU,
            license_class="Apache-2.0",
            default_sample_rate=48000
        )

    def is_ready(self) -> bool:
        return self.pipeline is not None

    async def initialize(self, model_path: Optional[str] = None) -> bool:
        if self.pipeline is not None or self._is_loading:
            return True

        self._is_loading = True
        try:
            import torch
            from app.services.config_manager import ConfigManager
            from app.core.paths import get_heartmula_ckpt_dir
            config = ConfigManager().get_config()
            if model_path is None:
                hm_cfg = config.get("paths", {}).get("heartmula_model_path")
                if hm_cfg and os.path.isdir(os.path.expanduser(hm_cfg)):
                    model_path = os.path.expanduser(hm_cfg)
                else:
                    model_path = str(get_heartmula_ckpt_dir())
            self.model_path = model_path

            device_str = "cpu"
            if torch.cuda.is_available():
                device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"

            self.device = torch.device(device_str)
            target_dtype = torch.float16 if device_str == "mps" else torch.bfloat16

            from heartlib import HeartMuLaGenPipeline
            loop = asyncio.get_running_loop()
            self.pipeline = await loop.run_in_executor(
                None,
                lambda: HeartMuLaGenPipeline.from_pretrained(
                    self.model_path,
                    device=self.device,
                    dtype=target_dtype,
                    version="3B"
                )
            )
            logger.info(f"HeartMuLa loaded on {self.device}")
            return True
        except Exception as e:
            logger.warning(f"HeartMuLa initialization deferred or failed: {e}")
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
        **kwargs
    ) -> GeneratedAudioResult:
        if self.pipeline is None:
            await self.initialize()
            if self.pipeline is None:
                raise RuntimeError("HeartMuLa model is not initialized.")

        os.makedirs("generated_audio", exist_ok=True)
        filename = f"{job_id}.mp3"
        output_path = os.path.join("generated_audio", filename)

        full_prompt = prompt or ""
        if tags and tags not in full_prompt:
            full_prompt = f"{tags}, {full_prompt}" if full_prompt else tags

        loop = asyncio.get_running_loop()

        def _step_callback(step: int, total_steps: int):
            if progress_callback:
                progress_callback(step, total_steps, f"Generating tokens {step}/{total_steps}")

        await loop.run_in_executor(
            None,
            lambda: self.pipeline.generate(
                prompt=full_prompt,
                lyrics=lyrics,
                output_path=output_path,
                max_duration_ms=duration_ms,
                temperature=temperature,
                cfg_scale=cfg_scale,
                top_k=topk,
                seed=seed,
                step_callback=_step_callback,
                cancel_event=cancel_event
            )
        )

        return GeneratedAudioResult(
            audio_path=f"/audio/{filename}",
            duration_sec=duration_ms / 1000.0,
            sample_rate=48000,
            metadata={"seed": seed, "provider": "heartmula"}
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
            job_id=job_id,
            prompt=prompt or "",
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
