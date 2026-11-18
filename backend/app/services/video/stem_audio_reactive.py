"""Stem-driven audio-reactive video modulation (audit M4 / plan 7.10).

Only one curve here is actually consumed: `beat_hits`, which drives the
beat-synced exposure pulse applied to each rendered clip. This module used to
also compute a full-song `vocal_envelope` and `camera_zoom` that **nothing read**
(two librosa passes per render, and a docstring claiming the vocal envelope drove
lip sync when it did not). They are gone: a curve nothing consumes is a lie with
a CPU bill.

If a future feature needs the vocal envelope, compute it where it is used — the
lip-sync providers already analyse their own audio slice.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import numpy as np

logger = logging.getLogger("milimo.video.stem_reactive")

#: Keys `extract_stem_reactive_modulation` returns. Anything not in here is dead
#: weight; a test asserts this set stays in sync with what the orchestrator reads.
RETURNED_CURVES = ("beat_hits",)


def extract_stem_reactive_modulation(
    drums_stem_path: Optional[str] = None,
    bass_stem_path: Optional[str] = None,
    fps: int = 24,
    target_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute frame-level beat hits from the drums (or bass) stem.

    Args:
        drums_stem_path: Path to isolated drums WAV.
        bass_stem_path: Path to isolated bass WAV (used when drums are absent).
        fps: Video frames per second (default 24).
        target_duration: Total duration in seconds (default: the stem's length).

    Returns:
        ``{"duration", "fps", "total_frames", "beat_hits"}`` — see RETURNED_CURVES.
    """
    import librosa

    sr = 22050
    duration = target_duration or 5.0

    total_frames = max(1, int(np.ceil(duration * fps)))
    beat_hits: List[bool] = [False] * total_frames

    rhythm_path = drums_stem_path or bass_stem_path
    if rhythm_path and Path(rhythm_path).is_file():
        try:
            y_rhythm, _ = librosa.load(rhythm_path, sr=sr, mono=True)
            if target_duration:
                y_rhythm = y_rhythm[: int(target_duration * sr)]
            # The stem may be longer/shorter than the plan; size the grid to it.
            duration = max(duration, len(y_rhythm) / sr)
            total_frames = max(1, int(np.ceil(duration * fps)))
            beat_hits = [False] * total_frames

            onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, wait=2)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            for t in onset_times:
                frame = int(t * fps)
                if 0 <= frame < total_frames:
                    beat_hits[frame] = True
        except Exception as exc:
            logger.warning("Failed to process rhythm stem: %s", exc)

    return {
        "duration": round(duration, 2),
        "fps": fps,
        "total_frames": total_frames,
        "beat_hits": beat_hits,
    }


def build_stem_reactive_filter(
    start_time: float,
    duration: float,
    stem_reactivity: Optional[Dict[str, Any]],
    fps: int = 24,
) -> Optional[str]:
    """Build an FFmpeg video filter that pulses exposure/contrast on kick/snare.

    Returns None when there is nothing to pulse (no hits in this window), so the
    caller can skip the re-encode entirely.
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

    # Thin out hits that are too close (< 0.2s) to prevent strobe flicker.
    thinned_hits: List[float] = []
    for t in hit_times:
        if not thinned_hits or (t - thinned_hits[-1] >= 0.20):
            thinned_hits.append(t)
        if len(thinned_hits) >= 16:  # Capped for compact FFmpeg command line
            break

    if not thinned_hits:
        return None

    # Decay pulse: between(t, ti, ti+0.12) * (1 - (t-ti)/0.12)
    pulse_terms = [f"(between(t,{t},{t + 0.12:.3f})*(1-(t-{t})/0.12))" for t in thinned_hits]
    pulse_sum = "+".join(pulse_terms)

    # Mild rhythmic exposure boost (~+4% brightness, +5% contrast).
    return f"eq=brightness='0.035*({pulse_sum})':contrast='1.0+0.05*({pulse_sum})'"
