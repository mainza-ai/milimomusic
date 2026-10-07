"""Stem-Driven Audio-Reactive Video Modulation Engine.

Extracts frame-synchronized modulation curves from isolated stems:
- Vocal Stem -> Drives clean LivePortrait lip sync and facial openness without drum bleed.
- Drums & Bass Stem -> Drives Wan 2.1 camera motion, zoom pulses, and beat-locked shake.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import numpy as np

logger = logging.getLogger("milimo.video.stem_reactive")


def extract_stem_reactive_modulation(
    vocal_stem_path: Optional[str] = None,
    drums_stem_path: Optional[str] = None,
    bass_stem_path: Optional[str] = None,
    fps: int = 24,
    target_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute frame-level audio modulation curves for Wan 2.1 and LivePortrait.

    Args:
        vocal_stem_path: Path to isolated vocal WAV.
        drums_stem_path: Path to isolated drums WAV.
        bass_stem_path: Path to isolated bass WAV.
        fps: Video frames per second (default 24).
        target_duration: Total duration in seconds.

    Returns:
        Dict containing vocal_envelope, camera_zoom, beat_hits, and frame_count.
    """
    import librosa

    # Determine duration
    sr = 22050
    duration = target_duration or 5.0

    # 1. Vocal Envelope for Lip-Sync
    vocal_curve: List[float] = []
    if vocal_stem_path and Path(vocal_stem_path).is_file():
        try:
            y_vox, _ = librosa.load(vocal_stem_path, sr=sr, mono=True)
            if target_duration:
                max_samples = int(target_duration * sr)
                y_vox = y_vox[:max_samples]
            duration = max(duration, len(y_vox) / sr)

            frame_samples = int(sr / fps)
            num_frames = int(np.ceil(len(y_vox) / frame_samples))
            for f in range(num_frames):
                chunk = y_vox[f * frame_samples : (f + 1) * frame_samples]
                rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
                vocal_curve.append(round(rms, 4))

            # Normalize vocal curve to 0.0 - 1.0
            max_v = max(vocal_curve) if vocal_curve else 1.0
            if max_v > 0:
                vocal_curve = [round(v / max_v, 4) for v in vocal_curve]
        except Exception as e:
            logger.warning(f"Failed to process vocal stem: {e}")

    # 2. Drums & Bass for Camera Motion
    total_frames = int(np.ceil(duration * fps))
    camera_zoom = [1.0] * total_frames
    beat_hits = [False] * total_frames

    rhythm_path = drums_stem_path or bass_stem_path
    if rhythm_path and Path(rhythm_path).is_file():
        try:
            y_rhythm, _ = librosa.load(rhythm_path, sr=sr, mono=True)
            onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, wait=2)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            for t in onset_times:
                v_frame = int(t * fps)
                if 0 <= v_frame < total_frames:
                    beat_hits[v_frame] = True
                    # Create a decaying zoom pulse over 3-4 frames
                    for decay_i, zoom_val in enumerate([1.08, 1.05, 1.02, 1.01]):
                        target_f = v_frame + decay_i
                        if target_f < total_frames:
                            camera_zoom[target_f] = max(camera_zoom[target_f], zoom_val)
        except Exception as e:
            logger.warning(f"Failed to process rhythm stem: {e}")

    # Pad vocal curve if shorter than total frames
    if len(vocal_curve) < total_frames:
        vocal_curve.extend([0.0] * (total_frames - len(vocal_curve)))
    else:
        vocal_curve = vocal_curve[:total_frames]

    return {
        "duration": round(duration, 2),
        "fps": fps,
        "total_frames": total_frames,
        "vocal_envelope": vocal_curve,
        "camera_zoom": camera_zoom,
        "beat_hits": beat_hits,
    }


def build_stem_reactive_filter(
    start_time: float,
    duration: float,
    stem_reactivity: Optional[Dict[str, Any]],
    fps: int = 24,
) -> Optional[str]:
    """Build an FFmpeg video filter expression modulating brightness and contrast
    synchronously with detected kick and snare onsets.
    """
    if not stem_reactivity or not stem_reactivity.get("beat_hits"):
        return None

    beat_hits = stem_reactivity["beat_hits"]
    start_frame = max(0, int(round(start_time * fps)))
    end_frame = min(len(beat_hits), int(round((start_time + duration) * fps)))

    if start_frame >= end_frame:
        return None

    clip_hits = beat_hits[start_frame:end_frame]
    hit_times = [round(i / fps, 3) for i, hit in enumerate(clip_hits) if hit]

    if not hit_times:
        return None

    # Thin out hits that are too close (< 0.2s) to prevent strobe flicker
    thinned_hits: List[float] = []
    for t in hit_times:
        if not thinned_hits or (t - thinned_hits[-1] >= 0.20):
            thinned_hits.append(t)
        if len(thinned_hits) >= 16:  # Capped for compact FFmpeg command line
            break

    if not thinned_hits:
        return None

    # Construct decay pulse expression: between(t, ti, ti+0.12)*(1-(t-ti)/0.12)
    pulse_terms = [f"(between(t,{t},{t+0.12:.3f})*(1-(t-{t})/0.12))" for t in thinned_hits]
    pulse_sum = "+".join(pulse_terms)

    # Mild rhythmic exposure boost (+4% brightness, +5% contrast on kick/snare)
    filter_expr = f"eq=brightness='0.035*({pulse_sum})':contrast='1.0+0.05*({pulse_sum})'"
    return filter_expr


async def apply_stem_reactive_fx(
    clip_path: str,
    out_path: str,
    start_time: float,
    duration: float,
    stem_reactivity: Optional[Dict[str, Any]],
    fps: int = 24,
) -> bool:
    """Apply beat-synced reactive contrast/exposure pulse to a rendered clip."""
    import asyncio
    import os

    filter_expr = build_stem_reactive_filter(start_time, duration, stem_reactivity, fps)
    if not filter_expr:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-vf", filter_expr,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
        "-c:a", "copy",
        out_path
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, err = await proc.communicate()
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            logger.info(f"Applied stem audio-reactive pulse filter to {os.path.basename(clip_path)}")
            return True
        logger.warning(f"Reactive pulse filter failed: {err.decode('utf-8', errors='ignore')[:150]}")
        return False
    except Exception as ex:
        logger.warning(f"Error applying audio-reactive filter: {ex}")
        return False

