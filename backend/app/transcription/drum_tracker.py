"""Drum stem onset and rhythmic event extraction for MuLaCover conditioning.

Extracts rhythmic transients from isolated BS-Roformer drum stems, classifying
hits into Kick (36), Snare (38), and Hi-Hat/Cymbals (42) in General MIDI drum format.
"""

from pathlib import Path
from typing import Dict, List, Union
import logging
import numpy as np

logger = logging.getLogger("milimo.transcription.drum_tracker")


def transcribe_drums_from_stem(
    drum_wav_path: Union[str, Path],
    bpm: float = 120.0,
    min_velocity: int = 40,
    max_velocity: int = 120,
) -> List[Dict[str, Union[float, int, bool]]]:
    """Extract rhythmic events from an isolated drum stem for MuLaCover conditioning.

    Args:
        drum_wav_path: Path to the separated drum stem WAV/FLAC file.
        bpm: Estimated or detected song BPM.
        min_velocity: Lower bound for normalized MIDI velocity.
        max_velocity: Upper bound for normalized MIDI velocity.

    Returns:
        List of event dicts with onset, offset, pitch, program=128, and is_drum=True.
    """
    path = Path(drum_wav_path)
    if not path.is_file():
        logger.warning(f"Drum stem does not exist: {path}")
        return []

    try:
        import librosa

        # Load audio at standard 22050Hz for efficient onset and STFT calculation
        sr = 22050
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        if y is None or len(y) == 0:
            return []

        # Check for absolute silence / digital zero
        rms = float(np.sqrt(np.mean(y**2)))
        if rms < 1e-4:
            logger.info(f"Drum stem {path.name} is silent (RMS={rms:.6f}).")
            return []

        # Calculate spectral flux onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if onset_env.max() <= 0:
            return []

        # Detect transient peaks with a small wait window (avoids double-triggering)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True,
            wait=3,
            pre_avg=3,
            post_avg=3,
            pre_max=3,
            post_max=3,
        )
        if len(onset_frames) == 0:
            return []

        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # STFT for sub-band frequency distribution
        hop_length = 512
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        low_mask = freqs < 220
        mid_mask = (freqs >= 220) & (freqs < 2800)
        high_mask = freqs >= 2800

        low_band = np.mean(S[low_mask, :], axis=0) if np.any(low_mask) else np.zeros(S.shape[1])
        mid_band = np.mean(S[mid_mask, :], axis=0) if np.any(mid_mask) else np.zeros(S.shape[1])
        high_band = np.mean(S[high_mask, :], axis=0) if np.any(high_mask) else np.zeros(S.shape[1])

        max_env = float(onset_env.max()) or 1.0
        drum_events: List[Dict[str, Union[float, int, bool]]] = []

        # Fixed duration for percussive hits on 16th-note grid (approx 100ms or 1/16th)
        beat_duration = 60.0 / max(bpm, 20.0)
        sixteenth_dur = max(0.06, beat_duration / 4.0)

        for frame, t in zip(onset_frames, onset_times):
            f_idx = min(frame, S.shape[1] - 1)
            e_low = float(low_band[f_idx])
            e_mid = float(mid_band[f_idx])
            e_high = float(high_band[f_idx])

            # Classify based on dominant spectral energy
            if e_low > (e_mid * 1.35) and e_low > (e_high * 1.5):
                pitch = 36  # General MIDI Bass / Kick Drum
            elif e_mid >= e_low and e_mid > (e_high * 0.9):
                pitch = 38  # General MIDI Acoustic Snare
            else:
                pitch = 42  # General MIDI Closed Hi-Hat

            # Normalized velocity based on envelope strength
            norm_str = float(onset_env[min(frame, len(onset_env) - 1)]) / max_env
            vel = int(np.clip(norm_str * (max_velocity - min_velocity) + min_velocity, min_velocity, max_velocity))

            drum_events.append({
                "onset": round(float(t), 4),
                "offset": round(float(t + sixteenth_dur), 4),
                "pitch": int(pitch),
                "velocity": int(vel),
                "program": 128,  # General MIDI Drum Channel Designator
                "is_drum": True,
            })

        logger.info(f"Transcribed {len(drum_events)} drum events from {path.name} (BPM={bpm:.1f}).")
        return drum_events

    except Exception as e:
        logger.error(f"Failed to transcribe drum stem {drum_wav_path}: {e}", exc_info=True)
        return []
