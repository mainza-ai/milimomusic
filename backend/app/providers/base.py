"""
Generation Provider Base Interface & Capability Definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pydantic import BaseModel, ConfigDict


class HardwareTier(str, Enum):
    ENTRY_CPU = "entry_cpu"
    MID_SINGLE_GPU = "mid_single_gpu"       # e.g., Apple M-Series 16GB+, RTX 3060/4060
    HIGH_DUAL_GPU = "high_dual_gpu"         # e.g., 24GB+ VRAM, Dual GPU split / SGLang


class GenerationCapabilities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider_id: str
    display_name: str
    description: str
    version: str = "v1"
    max_duration_sec: int = 240
    supports_structured_caption: bool = False
    supports_section_tags: bool = True
    supports_lora: bool = False
    supports_voice_conversion: bool = True
    supports_track_extension: bool = True
    supports_segment_repair: bool = True
    recommended_hardware: HardwareTier = HardwareTier.MID_SINGLE_GPU
    license_class: str = "open-weights"
    default_sample_rate: int = 44100
    # Legacy providers stay registered (for reproducibility of old projects) but
    # are hidden from the default engine picker and receive no new features.
    legacy: bool = False


class GeneratedAudioResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    audio_path: str
    duration_sec: float
    sample_rate: int = 44100
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_caption: Optional[Dict[str, str]] = None
    # Provenance: whether real model inference failed/skipped and the track was
    # produced by the procedural fallback synth instead. Surfaced to the UI and
    # persisted on the Job so users can always tell real MiniMax output apart.
    used_fallback_synth: bool = False
    fallback_reason: Optional[str] = None
    # What actually produced the audio, independent of the provider's identity:
    #   "model" | "dsp" | "placeholder"
    source: str = "model"
    # True when the produced audio is shorter than what the caller asked for.
    truncated: bool = False
    requested_duration_sec: Optional[float] = None
    truncation_reason: Optional[str] = None

    @property
    def degraded(self) -> bool:
        return self.used_fallback_synth or self.source != "model" or self.truncated

    def provenance(self) -> Dict[str, Any]:
        """Stage-level provenance record persisted on the Job."""
        return {
            "stage": "generation",
            "source": self.source,
            "degraded": self.degraded,
            "reason": self.fallback_reason,
            "truncated": self.truncated,
            "requested_duration_sec": self.requested_duration_sec,
            "sample_rate": self.sample_rate,
            "metadata": {k: v for k, v in (self.metadata or {}).items()
                         if k in ("provider", "seed", "extension_mode", "channels")},
        }


class GenerationProvider(ABC):
    """
    Abstract interface for music generation engines (MiniMax Music 3, HeartMuLa, etc.).
    """

    @abstractmethod
    def get_capabilities(self) -> GenerationCapabilities:
        """Return the capability manifest for this provider."""
        pass

    @abstractmethod
    async def initialize(self, model_path: Optional[str] = None) -> bool:
        """Load or connect to model weights."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if provider is loaded and ready for inference."""
        pass

    def unload(self) -> bool:
        """Release loaded model weights and clear framework caches.

        Subclasses override this to perform accelerator and runtime specific
        deallocations (e.g. MLX Metal cache clearing, PyTorch CUDA IPC collection,
        and accelerate hook stripping).
        """
        return True

    @abstractmethod
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
        """Generate full audio track."""
        pass

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
        """Extend an existing audio track.

        Concrete default: refuse honestly. Providers used to implement this by
        generating a brand-new unrelated song and returning it as an "extension";
        a provider may only override it if it can genuinely continue audio.
        """
        raise NotImplementedError(
            f"{self.get_capabilities().provider_id} cannot continue existing audio. "
            "Generate a new track, or use a provider that supports track extension."
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
        """In-paint / repair a specific audio time window.

        Concrete default: refuse honestly. Returning the input file untouched and
        calling it "repaired" was how four providers silently no-oped; segment
        repair is owned by ``inpainting_service``.
        """
        raise NotImplementedError(
            f"{self.get_capabilities().provider_id} does not implement segment repair. "
            "Use the inpainting service."
        )

