"""
LivePortrait Neural Singing Lip-Sync & Avatar Animation Provider.
Implements deep audio-driven facial keypoint deformation, natural eye blinks,
head sway, and mouth viseme rendering.
"""

import os
import sys
import math
import uuid
import shutil
import asyncio
import logging
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

from app.core.paths import get_data_dir, get_models_dir
from app.services.video.lip_sync.base import BaseLipSyncProvider
from app.services.video.lip_sync.fallback import SmoothVisemeFallbackProvider

logger = logging.getLogger(__name__)
TEMP_DIR = str(get_data_dir() / "video_cache")


class LivePortraitProvider(BaseLipSyncProvider):
    def __init__(self):
        self._fallback = SmoothVisemeFallbackProvider()
        self._is_initialized = False
        self._device = None
        self._pipeline = None

    @property
    def name(self) -> str:
        return "live_portrait"

    @property
    def is_available(self) -> bool:
        """True only when a real audio-driven LivePortrait run is possible.

        Upstream LivePortrait is *video*-driven: it has no `--driving_audio`
        entry point. Claiming availability on `torch+device` alone is what made
        the studio advertise a neural singing avatar while every clip was the
        procedural jaw-warp (audit finding E4 / plan 7.1).
        """
        if self._get_weights_path() is None or self._audio_entry_point() is None:
            return False
        try:
            import torch
            return torch.cuda.is_available() or (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except ImportError:
            return False

    def _audio_entry_point(self) -> Optional[str]:
        """Locate an audio-driven inference script for this weights bundle.

        We only run what exists: a custom audio-driven wrapper (`inference_audio.py`
        or `MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT`) is required. The upstream
        `inference.py` takes a driving *video*, so it is not used for singing.
        """
        override = os.environ.get("MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT")
        if override:
            return override if os.path.isfile(override) else None
        weights = self._get_weights_path()
        if not weights:
            return None
        for candidate in ("inference_audio.py", "audio2video.py"):
            path = os.path.join(weights, candidate)
            if os.path.isfile(path):
                return path
        return None

    def _get_weights_path(self) -> Optional[str]:
        candidates = [
            str(get_models_dir("video") / "liveportrait"),
            str(get_models_dir() / "liveportrait"),
            os.path.join(os.getcwd(), "models", "video", "liveportrait"),
            os.path.join(os.getcwd(), "models", "liveportrait"),
            os.path.expanduser("~/.cache/huggingface/hub/models--KwaiVGI--LivePortrait"),
        ]
        for c in candidates:
            if os.path.isdir(c) and len(os.listdir(c)) > 0:
                return os.path.abspath(c)
        return None

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
        Executes LivePortrait neural avatar animation driven by vocal audio stem.
        If local neural weights are missing, seamlessly executes smooth fallback.
        """
        weights_path = self._get_weights_path()
        audio_script = self._audio_entry_point()
        if not weights_path or not audio_script or not self.is_available:
            missing = (
                "weights" if not weights_path
                else "an audio-driven entry point (inference_audio.py)"
                if not audio_script
                else "an MPS/CUDA device"
            )
            logger.info(
                f"LivePortrait unavailable ({missing}); rendering the procedural "
                f"jaw-warp animatic instead."
            )
            ok = await self._fallback.render_lip_sync(
                face_image_path=face_image_path,
                vocal_audio_path=vocal_audio_path,
                start_time=start_time,
                duration=duration,
                out_path=out_path,
                width=width,
                height=height,
                **kwargs
            )
            self._set_outcome(
                "jaw_warp",
                True,
                f"LivePortrait neural lip sync unavailable: missing {missing}. "
                f"Rendered the procedural jaw-warp animatic.",
            )
            return ok

        try:
            import torch
            device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Executing LivePortrait inference on device: {device} with weights from {weights_path}")

            liveportrait_script = audio_script
            if os.path.isfile(liveportrait_script):
                slice_audio = os.path.join(TEMP_DIR, f"live_slice_{uuid.uuid4().hex[:8]}.wav")
                cmd_cut = [
                    "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
                    "-i", vocal_audio_path, "-ar", "16000", "-ac", "1", slice_audio
                ]
                proc_cut = await asyncio.create_subprocess_exec(*cmd_cut, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc_cut.communicate()

                cmd_run = [
                    sys.executable, liveportrait_script,
                    "--source_image", face_image_path,
                    "--driving_audio", slice_audio,
                    "--output_video", out_path,
                    "--device", device
                ]
                proc = await asyncio.create_subprocess_exec(*cmd_run, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await proc.communicate()

                if os.path.isfile(slice_audio):
                    os.remove(slice_audio)

                if proc.returncode == 0 and os.path.isfile(out_path):
                    self._set_outcome("live_portrait", False)
                    return True
                logger.warning(
                    f"LivePortrait script exited with {proc.returncode}: "
                    f"{stderr.decode('utf-8', errors='ignore')[:200]}"
                )
                script_error = f"LivePortrait exited with {proc.returncode}"

            # Fallback to smooth provider if the script produced nothing
            ok = await self._fallback.render_lip_sync(
                face_image_path=face_image_path,
                vocal_audio_path=vocal_audio_path,
                start_time=start_time,
                duration=duration,
                out_path=out_path,
                width=width,
                height=height,
                **kwargs
            )
            self._set_outcome(
                "jaw_warp",
                True,
                locals().get("script_error", "LivePortrait produced no output")
                + "; rendered the procedural jaw-warp animatic.",
            )
            return ok

        except asyncio.CancelledError:
            raise
        except Exception as e:
            cancel_event = kwargs.get("cancel_event")
            cancel_check = kwargs.get("cancel_check")
            if (cancel_event and cancel_event.is_set()) or (cancel_check and cancel_check()):
                raise asyncio.CancelledError("LivePortrait execution cancelled by user.")
            logger.error(f"LivePortrait execution error: {e}", exc_info=True)
            ok = await self._fallback.render_lip_sync(
                face_image_path=face_image_path,
                vocal_audio_path=vocal_audio_path,
                start_time=start_time,
                duration=duration,
                out_path=out_path,
                width=width,
                height=height,
                **kwargs
            )
            self._set_outcome(
                "jaw_warp",
                True,
                f"LivePortrait failed ({e}); rendered the procedural jaw-warp animatic.",
            )
            return ok
