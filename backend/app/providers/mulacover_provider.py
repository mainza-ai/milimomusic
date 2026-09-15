"""
MuLaCover Generation Provider.
Wraps MuLaCover into the unified Milimo Music GenerationProvider interface.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Union

from app.providers.base import (
    GenerationProvider,
    GenerationCapabilities,
    GeneratedAudioResult,
    HardwareTier,
)
from app.services.mulacover.mulacover_engine import mulacover_engine
from app.services.mulacover.symbolic_hub import symbolic_hub
from app.services.mulacover.formatters import format_style_tags, sanitize_lyrics_for_mulacover

logger = logging.getLogger(__name__)


class MuLaCoverProvider(GenerationProvider):
    """Production GenerationProvider implementation for MuLaCover cover & remix generation."""

    def __init__(self):
        self.engine = mulacover_engine
        self.symbolic_hub = symbolic_hub
        self._is_initialized = False

    def get_capabilities(self) -> GenerationCapabilities:
        return GenerationCapabilities(
            provider_id="mulacover",
            display_name="MuLaCover (Cover & Remix Engine)",
            description="Controllable symbolic cover-song and music-remix model conditioning on melody, harmony, drums, and lyrics.",
            version="3B",
            max_duration_sec=300,
            supports_structured_caption=True,
            supports_section_tags=True,
            supports_lora=False,
            supports_voice_conversion=True,
            supports_track_extension=True,
            supports_segment_repair=False,
            recommended_hardware=HardwareTier.MID_SINGLE_GPU,
            license_class="CC BY-NC 4.0",
            default_sample_rate=48000,
        )

    def is_ready(self) -> bool:
        return self.engine.model_root.is_dir() and (self.engine.model_root / "MuLaCover").is_dir()

    async def initialize(self, model_path: Optional[str] = None) -> bool:
        if model_path:
            self.engine.model_root = Path(model_path)
        self._is_initialized = self.is_ready()
        if self._is_initialized:
            logger.info(f"MuLaCoverProvider initialized at {self.engine.model_root}")
        else:
            logger.warning(f"MuLaCover checkpoints not found at {self.engine.model_root}")
        return self._is_initialized

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
        topk: int = 250,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        structured_caption: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        """Standard text generation fallback if invoked as a general provider."""
        # Check if caller provided symbolic inputs in kwargs
        ref_audio = kwargs.get("ref_audio_path")
        melody_midi = kwargs.get("melody_midi_path")
        chord_midi = kwargs.get("chord_midi_path")
        drum_midi = kwargs.get("drum_midi_path")
        bpm = kwargs.get("bpm")

        return await self.generate_cover(
            job_id=job_id,
            ref_audio_path=ref_audio,
            melody_midi_path=melody_midi,
            chord_midi_path=chord_midi,
            drum_midi_path=drum_midi,
            bpm=bpm,
            prompt=prompt,
            tags=tags,
            lyrics=lyrics,
            duration_ms=duration_ms,
            temperature=temperature,
            cfg_scale=cfg_scale,
            topk=topk,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            transcription_engine=kwargs.get("transcription_engine", "milimo_neural"),
        )

    async def generate_cover(
        self,
        job_id: str,
        ref_audio_path: Optional[str] = None,
        melody_midi_path: Optional[str] = None,
        chord_midi_path: Optional[str] = None,
        drum_midi_path: Optional[str] = None,
        bpm: Optional[float] = None,
        prompt: Optional[str] = None,
        tags: Optional[str] = None,
        lyrics: Optional[str] = None,
        duration_ms: int = 120_000,
        temperature: float = 1.0,
        cfg_scale: float = 1.5,
        topk: int = 250,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        transcription_engine: str = "milimo_neural",
    ) -> GeneratedAudioResult:
        """Full cover song / remix synthesis workflow."""
        if not self.is_ready():
            await self.initialize()

        # Step 0: Obtain or Transcribe SymbolicCondition
        condition = None
        symbolic_dir = Path("generated_audio/symbolic") / job_id
        symbolic_dir.mkdir(parents=True, exist_ok=True)

        if melody_midi_path and chord_midi_path:
            if progress_callback:
                progress_callback(5, 100, "Loading symbolic MIDI lead sheet...")
            condition = self.symbolic_hub.create_condition_from_midi(
                melody_path=melody_midi_path,
                chord_path=chord_midi_path,
                drum_path=drum_midi_path,
            )
        elif ref_audio_path:
            if progress_callback:
                progress_callback(5, 100, f"Transcribing reference audio ({transcription_engine})...")

            if transcription_engine == "upstream":
                condition = self.symbolic_hub.transcribe_upstream(ref_audio_path, bpm=bpm)
            else:
                condition = await self.symbolic_hub.transcribe_milimo_neural(
                    audio_path=ref_audio_path,
                    job_id=job_id,
                    bpm=bpm,
                )
        else:
            raise ValueError("MuLaCover requires either ref_audio_path or both melody_midi_path and chord_midi_path")

        # Export exact condition as replayable MIDI for DAW inspection
        exported_midis = self.symbolic_hub.export_lead_sheet(condition, symbolic_dir)

        # Output audio file path
        filename = f"{job_id}.wav"
        output_file = Path("generated_audio") / filename

        # Synthesize audio with cancellable engine
        res = await self.engine.generate_cover(
            condition=condition,
            lyrics=lyrics or "",
            tags=tags or prompt or "",
            output_path=output_file,
            duration_ms=duration_ms,
            temperature=temperature,
            cfg_scale=cfg_scale,
            topk=topk,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        metadata = res.get("metadata", {})
        metadata.update({
            "symbolic_midi": exported_midis,
            "ref_audio_path": ref_audio_path,
            "melody_midi_path": melody_midi_path,
            "chord_midi_path": chord_midi_path,
            "is_cover": True,
        })

        return GeneratedAudioResult(
            audio_path=f"/audio/{filename}",
            duration_sec=res.get("duration_sec", duration_ms / 1000.0),
            sample_rate=res.get("sample_rate", 48000),
            metadata=metadata,
            used_fallback_synth=False,
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
        """Extend an existing track using MuLaCover."""
        return await self.generate(
            job_id=job_id,
            prompt=prompt or "",
            lyrics=lyrics,
            duration_ms=extend_ms,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            ref_audio_path=parent_audio_path,
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


mulacover_provider = MuLaCoverProvider()
