"""
Production Audio Signal Analyzer for Milimo Music Director Mode v2.

Extracts beat grids, musical downbeats, structural sections, vocal presence,
and transient percussion cues to drive frame-accurate music video directing.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from app.core.hardware_autotune import cpu_scoped_audio

logger = logging.getLogger("milimo.video.audio_analysis")


@dataclass
class AudioAnalysisResult:
    """Comprehensive musical and structural analysis of a song."""

    duration_sec: float
    tempo_bpm: float
    beats: List[float] = field(default_factory=list)
    downbeats: List[float] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    vocal_intervals: List[Tuple[float, float]] = field(default_factory=list)
    percussion_cues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioSignalAnalyzer:
    """Multi-signal audio feature extraction scoped strictly to CPU."""

    @classmethod
    @cpu_scoped_audio
    def analyze_audio(
        cls,
        audio_path: str,
        vocal_path: Optional[str] = None,
        timed_lyrics: Optional[List[Dict[str, Any]]] = None,
        target_sr: int = 22050,
    ) -> AudioAnalysisResult:
        """
        Extract rhythm, structural sections, vocal presence, and percussion cues.
        """
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if librosa is None:
            logger.warning("Librosa not installed, falling back to synthetic 120 BPM grid.")
            return cls._fallback_analysis(audio_path)

        # 1. Load audio on CPU
        try:
            y, sr = librosa.load(str(p), sr=target_sr, mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return cls._fallback_analysis(audio_path)

        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration <= 0.1:
            return cls._fallback_analysis(audio_path)

        # 2. Beat and Downbeat Tracking
        tempo, beats, downbeats = cls._extract_beats_and_downbeats(y, sr)

        # 3. Structural Sections and Energy
        sections = cls._extract_sections(y, sr, duration, timed_lyrics)

        # 4. Vocal Activity Intervals
        vocal_intervals = cls._extract_vocal_intervals(vocal_path, timed_lyrics, duration)

        # 5. Percussion and Transient Entrances
        percussion_cues = cls._extract_percussion_cues(y, sr)

        return AudioAnalysisResult(
            duration_sec=round(duration, 3),
            tempo_bpm=round(float(tempo), 1),
            beats=[round(float(b), 3) for b in beats],
            downbeats=[round(float(db), 3) for db in downbeats],
            sections=sections,
            vocal_intervals=[(round(float(s), 3), round(float(e), 3)) for s, e in vocal_intervals],
            percussion_cues=percussion_cues,
        )

    @classmethod
    def _extract_beats_and_downbeats(
        cls, y: np.ndarray, sr: int
    ) -> Tuple[float, List[float], List[float]]:
        """Compute beat frames and estimate 4/4 downbeats via beat tracker."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_arr, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, onset_envelope=onset_env, trim=False
        )

        tempo = float(np.mean(tempo_arr)) if np.size(tempo_arr) > 0 else 120.0
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        if not beat_times:
            # Synthetic 120 BPM fallback
            step = 60.0 / max(1.0, tempo)
            dur = len(y) / sr
            beat_times = [float(i * step) for i in range(int(dur / step))]

        # Downbeat estimation: group into 4-beat bars
        downbeats = [beat_times[i] for i in range(0, len(beat_times), 4)]

        return tempo, beat_times, downbeats

    @classmethod
    def _extract_sections(
        cls,
        y: np.ndarray,
        sr: int,
        duration: float,
        timed_lyrics: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment song into [Intro], [Verse], [Chorus], [Bridge], [Outro]
        using RMS energy dynamics and lyric timing boundaries.
        """
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        max_rms = float(np.max(rms)) if len(rms) > 0 and np.max(rms) > 0 else 1.0
        norm_rms = rms / max_rms

        # Use 15-second windows to compute regional energy
        window_sec = 16.0
        num_windows = max(1, int(math.ceil(duration / window_sec)))
        raw_sections: List[Dict[str, Any]] = []

        for idx in range(num_windows):
            start = idx * window_sec
            end = min(duration, (idx + 1) * window_sec)
            mask = (times >= start) & (times < end)
            energy = float(np.mean(norm_rms[mask])) if np.any(mask) else 0.5

            raw_sections.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "energy": round(energy, 2),
            })

        # Classify labels based on progression and energy
        sections: List[Dict[str, Any]] = []
        for idx, sec in enumerate(raw_sections):
            e = sec["energy"]
            if idx == 0:
                label = "Intro"
            elif idx == len(raw_sections) - 1:
                label = "Outro"
            elif e > 0.65:
                label = "Chorus"
            elif e < 0.35:
                label = "Bridge"
            else:
                label = "Verse"

            sections.append({
                "label": label,
                "start": sec["start"],
                "end": sec["end"],
                "energy": sec["energy"],
            })

        return sections

    @classmethod
    def _extract_vocal_intervals(
        cls,
        vocal_path: Optional[str],
        timed_lyrics: Optional[List[Dict[str, Any]]],
        duration: float,
    ) -> List[Tuple[float, float]]:
        """
        Identify vocal activity intervals. Prefers isolated vocals.wav stem,
        falling back to word-level lyric timestamps.
        """
        intervals: List[Tuple[float, float]] = []

        # 1. Check isolated vocal stem
        if vocal_path and os.path.exists(vocal_path) and librosa is not None:
            try:
                vy, vsr = librosa.load(vocal_path, sr=16000, mono=True)
                v_rms = librosa.feature.rms(y=vy, hop_length=512)[0]
                v_times = librosa.times_like(v_rms, sr=vsr, hop_length=512)
                threshold = 0.05 * np.max(v_rms) if np.max(v_rms) > 0 else 0.01

                active = v_rms > threshold
                in_vocal = False
                start_t = 0.0

                for t, is_act in zip(v_times, active):
                    if is_act and not in_vocal:
                        in_vocal = True
                        start_t = float(t)
                    elif not is_act and in_vocal:
                        in_vocal = False
                        if float(t) - start_t >= 0.5:
                            intervals.append((start_t, float(t)))

                if in_vocal and duration - start_t >= 0.5:
                    intervals.append((start_t, duration))

                if intervals:
                    return intervals
            except Exception as e:
                logger.warning(f"Failed processing vocal stem {vocal_path}: {e}")

        # 2. Fallback to timed_lyrics
        if timed_lyrics:
            for item in timed_lyrics:
                s = item.get("start") or item.get("start_time")
                e = item.get("end") or item.get("end_time")
                if s is not None and e is not None and float(e) > float(s):
                    intervals.append((float(s), float(e)))

            # Merge overlapping or close intervals (< 0.5s apart)
            if intervals:
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for s, e in intervals[1:]:
                    last_s, last_e = merged[-1]
                    if s <= last_e + 0.5:
                        merged[-1] = (last_s, max(last_e, e))
                    else:
                        merged.append((s, e))
                return merged

        return intervals

    @classmethod
    def _extract_percussion_cues(cls, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Detect kick drum drops and transient bursts on CPU."""
        if len(y) == 0:
            return []

        cues: List[Dict[str, Any]] = []
        hop = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=10
        )
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)

        for t, p_idx in zip(peak_times, peaks):
            intensity = float(onset_env[p_idx])
            cues.append({
                "timestamp": round(float(t), 3),
                "intensity": round(intensity, 2),
                "type": "transient_drop" if intensity > 2.0 else "beat_accent",
            })

        return cues

    @classmethod
    def _fallback_analysis(cls, audio_path: str, duration_sec: Optional[float] = None) -> AudioAnalysisResult:
        """Synthetic fallback when audio packages or files are unavailable."""
        dur = float(duration_sec) if duration_sec is not None else 30.0
        if audio_path and sf is not None and os.path.exists(audio_path):
            try:
                info = sf.info(audio_path)
                dur = float(info.duration)
            except Exception:
                pass

        step = 0.5  # 120 BPM
        beats = [round(i * step, 3) for i in range(max(1, int(dur / step)))]
        downbeats = [round(i * step * 4, 3) for i in range(max(1, int(dur / (step * 4))))]

        v1_end = min(dur, max(8.0, dur * 0.4))
        ch_end = min(dur, max(v1_end + 4.0, dur * 0.8))

        return AudioAnalysisResult(
            duration_sec=round(dur, 3),
            tempo_bpm=120.0,
            beats=beats,
            downbeats=downbeats,
            sections=[
                {"label": "Intro", "start": 0.0, "end": min(dur, 8.0), "energy": 0.4},
                {"label": "Verse", "start": min(dur, 8.0), "end": v1_end, "energy": 0.6},
                {"label": "Chorus", "start": v1_end, "end": ch_end, "energy": 0.85},
                {"label": "Outro", "start": ch_end, "end": dur, "energy": 0.3},
            ],
            vocal_intervals=[(2.0, min(dur, dur - 2.0))],
            percussion_cues=[],
        )
