"""
Generation Provider Base Interface & Capability Definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pydantic import BaseModel


class HardwareTier(str, Enum):
    ENTRY_CPU = "entry_cpu"
    MID_SINGLE_GPU = "mid_single_gpu"       # e.g., Apple M-Series 16GB+, RTX 3060/4060
    HIGH_DUAL_GPU = "high_dual_gpu"         # e.g., 24GB+ VRAM, Dual GPU split / SGLang


class GenerationCapabilities(BaseModel):
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


class GeneratedAudioResult(BaseModel):
    audio_path: str
    duration_sec: float
    sample_rate: int = 44100
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_caption: Optional[Dict[str, str]] = None


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
        **kwargs
    ) -> GeneratedAudioResult:
        """Generate full audio track."""
        pass

    @abstractmethod
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
        """Extend an existing audio track."""
        pass

    @abstractmethod
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
        """In-paint / repair a specific audio time window."""
        pass
