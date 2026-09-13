"""
Types, dataclasses, and schemas for the AI Music Video Studio.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum


class SceneType(str, Enum):
    VOCAL_PERFORMANCE = "VOCAL_PERFORMANCE"
    CINEMATIC_BROLL = "CINEMATIC_BROLL"


class VideoProviderType(str, Enum):
    LOCAL = "local"
    CLOUD_FAL = "cloud_fal"
    CLOUD_REPLICATE = "cloud_replicate"
    CLOUD_MINIMAX = "cloud_minimax"


class LipSyncEngineType(str, Enum):
    LIVE_PORTRAIT = "live_portrait"
    ECHOMIMIC = "echomimic"
    WAV2LIP = "wav2lip"
    FALLBACK = "fallback"


class VideoModelType(str, Enum):
    WAN_14B = "wan_14b"
    WAN_1_3B = "wan_1.3b"
    LTX_VIDEO = "ltx_video"
    COGVIDEOX = "cogvideox"
    AUDIOREACTIVE = "audioreactive"


@dataclass
class SceneClip:
    clip_index: int
    start_time: float
    end_time: float
    duration: float
    time_str: str
    is_vocal: bool
    scene_type: str
    lyrics: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    camera: str = ""
    lighting: str = ""
    keyframe_image_path: Optional[str] = None
    rendered_clip_path: Optional[str] = None
    status: str = "pending"  # pending, rendering, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoPlan:
    job_id: str
    total_clips: int
    vocal_clips_count: int
    broll_clips_count: int
    max_clip_duration: float
    model_max_duration: float
    model_name: str
    bpm: float
    visual_style: str
    clips: List[SceneClip] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clips"] = [c.to_dict() if isinstance(c, SceneClip) else c for c in self.clips]
        return d


@dataclass
class VideoTaskStatusInfo:
    id: str
    job_id: str
    status: str = "processing"
    step: str = "Initializing Video Pipeline"
    progress: int = 0
    total_clips: int = 0
    current_clip: int = 0
    current_clip_type: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    clips: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoProviderInfo:
    id: str
    name: str
    provider_type: VideoProviderType
    description: str
    is_available: bool
    supported_models: List[str]
    has_api_key: bool = False
    requires_api_key: bool = False
    default_for_tier: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provider_type"] = self.provider_type.value
        return d
