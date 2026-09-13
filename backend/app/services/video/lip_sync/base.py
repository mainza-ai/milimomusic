"""
Base interface for neural singing lip-sync providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLipSyncProvider(ABC):
    """
    Abstract interface for driving facial singing performance from vocal audio.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the lip-sync engine."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider can execute in the current environment."""
        pass

    @abstractmethod
    async def render_lip_sync(
        self,
        face_image_path: str,
        vocal_audio_path: str,
        start_time: float,
        duration: float,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        **kwargs
    ) -> bool:
        """
        Render an audio-driven singing vocal performance clip.
        Returns True on success, False on failure.
        """
        pass
