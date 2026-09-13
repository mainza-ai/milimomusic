"""
Base interface for generative video diffusion providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseVideoGenerator(ABC):
    """
    Abstract interface for Text-to-Video (T2V) and Image-to-Video (I2V) generation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the video diffusion model."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this model is available in the current environment."""
        pass

    @abstractmethod
    async def generate_clip(
        self,
        prompt: str,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Generates a video clip from text prompt (T2V) or image keyframe (I2V).
        Returns True on success, False on failure.
        """
        pass
