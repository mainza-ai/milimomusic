"""
Symbolic Conditioning Hub for MuLaCover.
Provides dual audio-to-symbolic transcription pipelines:
1. Upstream YourMT3 + ChordNet ensemble.
2. Milimo Neural Stem-Separated transcription (BS-Roformer vocal isolation + harmonic extraction).
3. Lossless MIDI ingestion and export for DAW / PianoRoll bi-directional sync.
"""

import os
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


def detect_tempo(audio_path: Union[str, Path], user_bpm: Optional[float] = None) -> float:
    """Estimate audio tempo with octave error mitigation.
    Returns BPM bounded within reasonable musical limits [40, 240].
    """
    if user_bpm is not None and user_bpm > 0 and math.isfinite(user_bpm):
        return float(user_bpm)

    import librosa
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    raw_bpm = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)

    # Heuristic octave normalization: common pop/rock/electronic ranges
    if raw_bpm < 55.0:
        raw_bpm *= 2.0
    elif raw_bpm > 200.0:
        raw_bpm /= 2.0

    return max(40.0, min(240.0, round(raw_bpm, 2)))


class SymbolicHub:
    """Manages symbolic lead-sheet creation, transcription, and MIDI interchange."""

    def __init__(self, checkpoints_dir: Optional[Union[str, Path]] = None, device: Optional[torch.device] = None):
        if checkpoints_dir is None:
            # Default to canonical Milimo models location
            from app.core.paths import get_models_dir
            checkpoints_dir = get_models_dir("audio") / "HeartMuLa__MuLaCover"

        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    def create_condition_from_midi(
        self,
        melody_path: Union[str, Path],
        chord_path: Union[str, Path],
        drum_path: Optional[Union[str, Path]] = None,
    ):
        """Create SymbolicCondition from pre-existing MIDI files."""
        from mulacover.symbolic import SymbolicCondition
        return SymbolicCondition.from_midi(melody_path, chord_path, drum_path)

    def transcribe_upstream(
        self,
        audio_path: Union[str, Path],
        bpm: Optional[float] = None,
    ):
        """Execute upstream YourMT3 + 5-fold ChordNet ensemble transcription."""
        from mulacover._symbolic_transcription import SymbolicTranscriber

        transcriptor_dir = self.checkpoints_dir / "SymbolicTranscriptor"
        if not transcriptor_dir.is_dir():
            raise FileNotFoundError(f"SymbolicTranscriptor checkpoint directory missing: {transcriptor_dir}")

        resolved_bpm = detect_tempo(audio_path, user_bpm=bpm)
        # YourMT3 and ChordNet require float32 on CPU / MPS
        dtype = torch.float32 if self.device.type in ("cpu", "mps") else torch.bfloat16

        transcriber = SymbolicTranscriber(
            checkpoint_dir=transcriptor_dir,
            device=self.device,
            dtype=dtype,
            lazy_load=True,
        )
        return transcriber.transcribe(str(audio_path), bpm=resolved_bpm)

    async def transcribe_milimo_neural(
        self,
        audio_path: Union[str, Path],
        job_id: str,
        bpm: Optional[float] = None,
    ):
        """High-precision Milimo transcription:
        1. Separate master into clean stems via BS-Roformer.
        2. Extract clean vocal melody notes via MuScriptor on the isolated vocal stem.
        3. Extract harmonic progression from isolated accompaniment stems.
        4. Quantize notes and chords into SymbolicCondition on a 16th-note grid.
        """
        import asyncio
        from app.transcription.real_separator import separate_sources, unload_model
        from app.transcription.muscriptor_provider import muscriptor_provider
        from mulacover.symbolic import SymbolicCondition

        resolved_bpm = detect_tempo(audio_path, user_bpm=bpm)
        stems_dir = Path("generated_audio/stems") / job_id
        stems_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        sep_res = await loop.run_in_executor(
            None, separate_sources, str(audio_path), str(stems_dir.parent), job_id, 1
        )
        stems = dict(sep_res.stems if hasattr(sep_res, "stems") else sep_res)
        unload_model()

        vocal_path = stems.get("vocals") or str(audio_path)
        # MuScriptor transcription of isolated vocal stem for lead vocal melody
        transcription_res = await muscriptor_provider.transcribe(vocal_path, job_id=f"{job_id}_lead")

        raw_notes = transcription_res.notes or []
        lead_notes = []
        for n in raw_notes:
            if isinstance(n, dict):
                onset = n.get("start_time", n.get("start", n.get("onset", 0.0)))
                offset = n.get("end_time", n.get("end", n.get("offset", 0.0)))
                pitch = n.get("pitch", 60)
            else:
                onset = getattr(n, "start_time", getattr(n, "start", getattr(n, "onset", 0.0)))
                offset = getattr(n, "end_time", getattr(n, "end", getattr(n, "offset", 0.0)))
                pitch = getattr(n, "pitch", 60)
            lead_notes.append({
                "onset": float(onset),
                "offset": float(offset),
                "pitch": int(pitch),
                "program": 100,  # 100 designates lead melody in YourMT3/MuLaCover schema
                "is_drum": False
            })

        # Run harmony transcription on the mixed audio / accompaniment
        transcriptor_dir = self.checkpoints_dir / "SymbolicTranscriptor"
        if (transcriptor_dir / "chord").is_dir():
            from mulacover._symbolic_transcription.harmony import ChordTranscriber
            chord_dtype = torch.float32 if self.device.type in ("cpu", "mps") else torch.bfloat16
            chord_transcriber = ChordTranscriber(transcriptor_dir / "chord", self.device, chord_dtype)
            chord_events = chord_transcriber.transcribe(str(audio_path), resolved_bpm)
        else:
            chord_events = []

        return SymbolicCondition.from_transcription(lead_notes, chord_events, resolved_bpm)

    def export_lead_sheet(self, condition, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Export SymbolicCondition as replayable/editable MIDI files for DAW inspection."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = condition.save_midi(out)
        return {key: str(val) for key, val in paths.items()}


symbolic_hub = SymbolicHub()
