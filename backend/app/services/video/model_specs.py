"""
Unified Video Model Specifications & Lattice Constraints.
Single source of truth for duration constraints, frame rates, lattice snapping,
and engine families across scene planning, audio sync, and diffusion rendering.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class VideoModelSpec:
    model_id: str
    canonical_name: str
    family: str
    fps: int
    min_duration: float
    max_duration: float
    default_duration: float
    min_frames: int
    frame_step: int
    default_width: int
    default_height: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "canonical_name": self.canonical_name,
            "family": self.family,
            "fps": self.fps,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "default_duration": self.default_duration,
            "min_frames": self.min_frames,
            "frame_step": self.frame_step,
            "default_width": self.default_width,
            "default_height": self.default_height,
            "description": self.description,
        }

    def compute_lattice_frames(self, target_duration: float) -> Tuple[int, float]:
        """
        Computes the closest valid lattice frame count and trimmed duration
        for this model's temporal architecture.
        """
        raw_frames = max(self.min_frames, int(round(target_duration * self.fps)))
        if self.frame_step > 1:
            step_offset = (raw_frames - self.min_frames) % self.frame_step
            if step_offset != 0:
                raw_frames += (self.frame_step - step_offset)
        trimmed_duration = round(raw_frames / self.fps, 3)
        return raw_frames, trimmed_duration


VIDEO_MODEL_SPECS: Dict[str, VideoModelSpec] = {
    "hailuo_h3": VideoModelSpec(
        model_id="hailuo_h3",
        canonical_name="MiniMax Hailuo H3 (33B DiT)",
        family="dit",
        fps=24,
        min_duration=5.0,
        max_duration=15.0,
        default_duration=15.0,
        min_frames=49,
        frame_step=48,  # 1 + 8k at 24fps (49, 97, 145, 193...)
        default_width=1280,
        default_height=720,
        description="MiniMax Hailuo H3 flagship DiT with high visual fidelity and synchronized rhythm",
    ),
    "minimax_h3": VideoModelSpec(
        model_id="minimax_h3",
        canonical_name="MiniMax Hailuo H3 (33B DiT)",
        family="dit",
        fps=24,
        min_duration=5.0,
        max_duration=15.0,
        default_duration=15.0,
        min_frames=49,
        frame_step=48,
        default_width=1280,
        default_height=720,
        description="MiniMax Hailuo H3 flagship DiT with high visual fidelity and synchronized rhythm",
    ),
    "wan_14b": VideoModelSpec(
        model_id="wan_14b",
        canonical_name="Wan 2.1 14B Flagship",
        family="dit",
        fps=16,
        min_duration=2.0,
        max_duration=5.0,
        default_duration=5.0,
        min_frames=16,
        frame_step=4,  # multiples of 4 at 16fps
        default_width=1280,
        default_height=720,
        description="Alibaba Wan 2.1 14B DiT with 3D temporal diffusion & keyframe I2V",
    ),
    "wan_1.3b": VideoModelSpec(
        model_id="wan_1.3b",
        canonical_name="Wan 2.1 1.3B Fast",
        family="t2v",
        fps=16,
        min_duration=2.0,
        max_duration=5.0,
        default_duration=5.0,
        min_frames=16,
        frame_step=4,
        default_width=832,
        default_height=480,
        description="Lightweight text-to-video diffusion for rapid local preview",
    ),
    "ltx_video": VideoModelSpec(
        model_id="ltx_video",
        canonical_name="LTX-Video 0.9B Realtime",
        family="dit",
        fps=25,
        min_duration=3.0,
        max_duration=10.0,
        default_duration=5.0,
        min_frames=25,
        frame_step=8,
        default_width=1280,
        default_height=720,
        description="Lightricks 0.9B real-time DiT (25 fps) for quick scene rendering",
    ),
    "cogvideox": VideoModelSpec(
        model_id="cogvideox",
        canonical_name="THUDM CogVideoX 1.5 (5B)",
        family="3d-vae",
        fps=24,
        min_duration=3.0,
        max_duration=10.0,
        default_duration=10.0,
        min_frames=24,
        frame_step=8,
        default_width=1280,
        default_height=720,
        description="5B 3D causal VAE model with emotive cinematic depth zooms",
    ),
    "hunyuan": VideoModelSpec(
        model_id="hunyuan",
        canonical_name="Tencent HunyuanVideo (13B)",
        family="dit",
        fps=24,
        min_duration=4.0,
        max_duration=15.0,
        default_duration=15.0,
        min_frames=24,
        frame_step=4,
        default_width=1280,
        default_height=720,
        description="Tencent HunyuanVideo 13B DiT for extended visual takes",
    ),
    "audioreactive": VideoModelSpec(
        model_id="audioreactive",
        canonical_name="Audio-Reactive Full",
        family="procedural",
        fps=30,
        min_duration=5.0,
        max_duration=120.0,
        default_duration=120.0,
        min_frames=30,
        frame_step=1,
        default_width=1280,
        default_height=720,
        description="Continuous full-timeline audio reactive spectrum & waveform visualizer",
    ),
}


def normalize_model_key(raw_name: Optional[str]) -> str:
    """Resolve a raw model string to a canonical VideoModelSpec key."""
    if not raw_name:
        return "wan_14b"
    clean = raw_name.lower().strip()
    if clean in VIDEO_MODEL_SPECS:
        return clean
    if "hailuo" in clean or "h3" in clean or "minimax" in clean:
        return "hailuo_h3"
    if "1.3" in clean or "1_3" in clean:
        return "wan_1.3b"
    if "wan" in clean:
        return "wan_14b"
    if "ltx" in clean:
        return "ltx_video"
    if "cog" in clean:
        return "cogvideox"
    if "hunyuan" in clean:
        return "hunyuan"
    if "reactive" in clean:
        return "audioreactive"
    return "wan_14b"


def get_model_spec(model_name: Optional[str]) -> VideoModelSpec:
    """Retrieve the authoritative VideoModelSpec for any model identifier."""
    canonical_key = normalize_model_key(model_name)
    return VIDEO_MODEL_SPECS.get(canonical_key, VIDEO_MODEL_SPECS["wan_14b"])
