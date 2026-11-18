"""
Base interface for neural singing lip-sync providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLipSyncProvider(ABC):
    """
    Abstract interface for driving facial singing performance from vocal audio.
    """

    #: Filled in by every `render_lip_sync` call so callers never have to guess
    #: what actually ran. Keys: ``engine`` (str), ``used_fallback`` (bool),
    #: ``reason`` (str|None). A provider that silently delegates to another engine
    #: must set ``used_fallback=True`` — that is the whole point (plan 7.1).
    last_outcome: Dict[str, Any] = {}

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
        Must set ``self.last_outcome`` before returning.
        """
        pass

    def _set_outcome(self, engine: str, used_fallback: bool, reason: Optional[str] = None) -> None:
        self.last_outcome = {
            "engine": engine,
            "used_fallback": bool(used_fallback),
            "reason": reason,
        }
