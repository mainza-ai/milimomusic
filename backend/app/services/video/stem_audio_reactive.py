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
