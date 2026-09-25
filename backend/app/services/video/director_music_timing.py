"""
Musical Anchor Snapping & Model-Native Frame Lattice Boundary Planner.

Implements Maestro v2.4.0 scored accent clipping with cut speed pacing (-2 to +2)
and frame-accurate model lattice duration snapping with zero-drift trimming.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.video.audio_analysis import AudioAnalysisResult

logger = logging.getLogger("milimo.video.music_timing")


@dataclass
class PlannedMusicClip:
    """Frame-accurate musical video clip boundary."""

    clip_index: int
    start_time: float
    end_time: float
    target_duration: float
    model_render_duration: float
    music_output_trim: float
    cut_type: str
    accent_score: float
    section_label: str
    is_vocal_active: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DirectorMusicTiming:
    """Plans clip boundaries aligned with musical accents and video model lattices."""

    # Model lattice definitions: (fps, min_frames, frame_step)
    MODEL_LATTICE_CONFIGS: Dict[str, Dict[str, Any]] = {
        "minimax_h3": {"fps": 24, "min_frames": 49, "frame_step": 48},  # 1 + 8k at 24fps
        "wan2.1": {"fps": 16, "min_frames": 16, "frame_step": 4},       # Multiples of 4
        "wan_14b": {"fps": 16, "min_frames": 16, "frame_step": 4},
        "ltx_video": {"fps": 25, "min_frames": 25, "frame_step": 8},
        "cogvideox": {"fps": 24, "min_frames": 24, "frame_step": 8},
        "default": {"fps": 24, "min_frames": 24, "frame_step": 1},
    }

    @classmethod
    def resolve_model_lattice(
        cls,
        target_duration: float,
        model_name: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculates the closest valid model lattice duration and the required trim duration.
        Returns: (model_render_duration, music_output_trim)
        """
        m = (model_name or "default").lower()
        key = "default"
        for k in cls.MODEL_LATTICE_CONFIGS:
            if k in m:
                key = k
                break

        cfg = cls.MODEL_LATTICE_CONFIGS[key]
        fps = cfg["fps"]
        min_frames = cfg["min_frames"]
        frame_step = cfg["frame_step"]

        target_frames = int(math.ceil(target_duration * fps))

        if target_frames <= min_frames:
            render_frames = min_frames
        else:
            excess = target_frames - min_frames
            step_count = int(math.ceil(excess / frame_step))
            render_frames = min_frames + step_count * frame_step

        model_render_duration = round(render_frames / fps, 3)
        music_output_trim = round(max(0.0, model_render_duration - target_duration), 3)

        return model_render_duration, music_output_trim

    @classmethod
    def plan_capped_music_clips(
        cls,
        analysis: AudioAnalysisResult,
        model_name: str = "wan2.1",
        pacing_bias: int = 0,
        model_max_duration: float = 5.0,
        min_clip_duration: Optional[float] = None,
    ) -> List[PlannedMusicClip]:
        """
        Partitions song audio into frame-accurate video clips using the scored accent system.

        Pacing bias:
          -2: Slow, sweeping (section & lyric snaps, larger clip spans)
          -1: Moderate cinematic
           0: Balanced (bars/downbeats, ~3-5s)
          +1: Rhythmic (beats/downbeats, ~2-3s)
          +2: Rapid montage cuts (~1.5-2.5s)
        """
        duration = analysis.duration_sec
        if duration <= 0:
            return []

        # Derive target clip min/max duration based on pacing bias
        pacing_bias = max(-2, min(2, pacing_bias))

        if min_clip_duration is None:
            if pacing_bias == -2:
                target_min = 4.0
                target_max = min(model_max_duration, 10.0)
            elif pacing_bias == -1:
                target_min = 3.0
                target_max = min(model_max_duration, 7.5)
            elif pacing_bias == 0:
                target_min = 2.5
                target_max = min(model_max_duration, 5.0)
            elif pacing_bias == 1:
                target_min = 1.8
                target_max = min(model_max_duration, 3.5)
            else:  # +2
                target_min = 1.2
                target_max = min(model_max_duration, 2.5)
        else:
            target_min = min_clip_duration
            target_max = model_max_duration

        # Compile candidate cut points with hierarchical accent scores
        candidates: List[Tuple[float, str, float]] = []

        # Downbeats: 1.8 weight
        for db in analysis.downbeats:
            candidates.append((db, "downbeat", 1.8))

        # Beats: 0.5 weight
        for b in analysis.beats:
            candidates.append((b, "beat", 0.5))

        # Section boundaries: 3.0 weight
        for sec in analysis.sections:
            candidates.append((sec["start"], "section_change", 3.0))

        # Percussion drops: 2.2 weight
        for cue in analysis.percussion_cues:
            candidates.append((cue["timestamp"], "percussion_drop", 2.2))

        # Sort and deduplicate candidates within 0.15s
        candidates.sort(key=lambda x: x[0])
        filtered_candidates: List[Tuple[float, str, float]] = []
        for cand in candidates:
            t, cut_type, score = cand
            if t <= 0.1 or t >= duration - 0.2:
                continue
            if not filtered_candidates:
                filtered_candidates.append(cand)
            else:
                last_t = filtered_candidates[-1][0]
                if abs(t - last_t) < 0.15:
                    # Keep candidate with higher score
                    if score > filtered_candidates[-1][2]:
                        filtered_candidates[-1] = cand
                else:
                    filtered_candidates.append(cand)

        # Plan sequential cuts
        current_time = 0.0
        clip_index = 0
        planned_clips: List[PlannedMusicClip] = []

        while current_time < duration - 0.2:
            earliest_cut = current_time + target_min
            latest_cut = min(duration, current_time + target_max)

            # If remaining duration is smaller than target_min * 1.3, absorb into final clip
            if duration - current_time <= target_max:
                cut_time = duration
                cut_type = "song_end"
                best_score = 5.0
            else:
                # Find highest scoring candidate in window [earliest_cut, latest_cut]
                valid = [c for c in filtered_candidates if earliest_cut <= c[0] <= latest_cut]
                if valid:
                    # Pick highest score
                    valid.sort(key=lambda x: x[2], reverse=True)
                    cut_time, cut_type, best_score = valid[0]
                else:
                    # Fallback to downbeat or latest_cut
                    cut_time = latest_cut
                    cut_type = "max_duration_snap"
                    best_score = 1.0

            target_dur = round(cut_time - current_time, 3)
            model_render_dur, trim_dur = cls.resolve_model_lattice(target_dur, model_name)

            # Determine active section
            sec_label = "Verse"
            for s in analysis.sections:
                if s["start"] <= current_time < s["end"]:
                    sec_label = s["label"]
                    break

            # Determine vocal activity
            vocal_active = any(
                s <= (current_time + target_dur * 0.5) <= e for s, e in analysis.vocal_intervals
            )

            planned_clips.append(
                PlannedMusicClip(
                    clip_index=clip_index,
                    start_time=round(current_time, 3),
                    end_time=round(cut_time, 3),
                    target_duration=target_dur,
                    model_render_duration=model_render_dur,
                    music_output_trim=trim_dur,
                    cut_type=cut_type,
                    accent_score=round(best_score, 2),
                    section_label=sec_label,
                    is_vocal_active=vocal_active,
                )
            )

            current_time = cut_time
            clip_index += 1

        return planned_clips
