"""
Per-Instrument Stem Renderer (Dynamic Stems).

Derives a dynamic, per-instrument stem set directly from the MuScriptor
note-level transcription instead of a fixed generic kit.  Every distinct
``instrument`` that appears in the song's notes becomes its own audio stem,
rendered with a dependency-free (numpy + soundfile) tone generator whose
timbre follows the instrument family (drums, bass, plucked, keys, sustained).

The rendered timeline matches the source transcription exactly: each note's
pitch, start time, duration and velocity map 1:1 into the stem, so the mixdown
of all stems reconstructs the harmonic/rhythmic content that MuScriptor heard.

NOTE: these are *synthesized* instrument parts (a faithful, listening-aid
rendering of the transcribed notes), not a neural source-separation of the
original recording.  That is the honest, environment-realistic way to make
stems track the song's actual instruments.
"""

import os
import re
import logging
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
STEM_DIR = "generated_audio/stems"


# --- instrument family detection -------------------------------------------

_FAMILY_KEYWORDS = {
    "drums":      ("drum", "percussion", "hi hat", "hi-hat", "cymbal", "kick", "snare", "toms", "shaker", "woodblock", "timpani", "congas", "bongo"),
    "bass":       ("bass",),
    "guitar":     ("guitar", "mandolin", "banjo", "ukulele", "sitar"),
    "keys":       ("piano", "keys", "keyboard", "organ", "synth", "ep", "electric piano", "rhodes", "clavinet", "harpsichord", "accordion"),
    "sustained":  ("strings", "violin", "viola", "cello", "horns", "brass", "trumpet", "sax", "flute", "choir", "voice", "vocal", "oboe", "clarinet", "pad", "bell"),
}


def _family_for(instrument: str) -> str:
    """Map an instrument name to a synthesis family (defaults to 'plucked')."""
    name = (instrument or "").lower()
    for family, kws in _FAMILY_KEYWORDS.items():
        if any(k in name for k in kws):
            return family
    return "plucked"


def _slug(name: str) -> str:
    """Turn an instrument name into a safe url/file slug."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or "instrument"


# General MIDI program (instrument) numbers, keyed by the MuScriptor instrument
# group names (mirrors the GM_PROGRAM table in MuScriptor's own web app). Used
# for correct track/instrument labeling and downstream MIDI/notation fidelity.
GM_PROGRAM: dict[str, int] = {
    "acoustic_piano": 0, "electric_piano": 4, "chromatic_percussion": 9,
    "organ": 19, "acoustic_guitar": 24, "clean_electric_guitar": 27,
    "distorted_electric_guitar": 30, "acoustic_bass": 32, "electric_bass": 33,
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43,
    "orchestral_harp": 46, "timpani": 47, "string_ensemble": 48,
    "synth_strings": 50, "voice": 52, "orchestra_hit": 55, "trumpet": 56,
    "trombone": 57, "tuba": 58, "french_horn": 60, "brass_section": 61,
    "soprano_and_alto_sax": 65, "tenor_sax": 66, "baritone_sax": 67,
    "oboe": 68, "english_horn": 69, "bassoon": 70, "clarinet": 71,
    "flutes": 73, "synth_lead": 80, "synth_pad": 89,
}


def gm_program_for(instrument: str) -> int:
    """Resolve a MuScriptor instrument name to a General MIDI program number.

    Snapshots the translatable name (e.g. ``Voice``) back to the canonical
    group key, then to its GM program. Drums/unknown fall back to 0 (piano).
    """
    key = _slug(instrument)  # "clean electric guitar" -> "clean_electric_guitar"
    return GM_PROGRAM.get(key, 0)


def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# --- note -> audio rendering -----------------------------------------------

def _render_note(family: str, freq: float, dur_sec: float, velocity: float, sr: int) -> np.ndarray:
    """Render a single note as a float array shaped by its instrument family."""
    n = max(1, int(dur_sec * sr))
    t = np.linspace(0, dur_sec, n, endpoint=False)
    amp = 0.16 * (velocity / 127.0)

    # Drums: transient noise burst with a tonal body.
    if family == "drums":
        body = np.sin(2 * np.pi * freq * t) * np.exp(-t * 9.0)
        noise = np.random.default_rng(int(freq * 1000)).standard_normal(n) * np.exp(-t * 42.0)
        out = (body * 0.6 + noise * 0.5)
        env = np.minimum(1.0, t * 400.0) * np.exp(-t * 8.0)
        return (out * env * amp)

    # Bass: low sawtooth with a touch of fundamental sine.
    if family == "bass":
        saw = 2.0 * (t * freq - np.floor(0.5 + t * freq))
        out = 0.55 * saw + 0.45 * np.sin(2 * np.pi * freq * t)
        env = np.minimum(1.0, t * 60.0) * np.exp(-t * 2.2)
        return out * env * amp

    # Plucked (guitar etc.): fast-decay harmonic-rich pluck.
    if family == "plucked":
        f = freq
        out = (np.sin(2 * np.pi * f * t)
               + 0.5 * np.sin(2 * np.pi * 2 * f * t)
               + 0.25 * np.sin(2 * np.pi * 3 * f * t))
        env = np.minimum(1.0, t * 250.0) * np.exp(-t * 6.0)
        return out * env * amp

    # Keys (piano): bell-ish attack, medium decay.
    if family == "keys":
        f = freq
        out = (np.sin(2 * np.pi * f * t)
               + 0.4 * np.sin(2 * np.pi * 2 * f * t)
               + 0.18 * np.sin(2 * np.pi * 3 * f * t))
        env = np.minimum(1.0, t * 180.0) * np.exp(-t * 3.0)
        return out * env * amp

    # Sustained (strings, voices, organs...): steady tone with soft attack/release.
    attack = np.clip(t * 25.0, 0.0, 1.0)
    release_end = np.clip((dur_sec - t) * 8.0, 0.0, 1.0)
    env = attack * release_end
    f = freq
    out = (0.7 * np.sin(2 * np.pi * f * t)
           + 0.3 * np.sin(2 * np.pi * 2 * f * t)
           + 0.1 * np.sin(2 * np.pi * 3 * f * t))
    return out * env * amp


def render_instrument_stems(
    notes: list[dict],
    job_id: str,
    duration_sec: float | None = None,
    sr: int = SAMPLE_RATE,
) -> dict[str, str]:
    """
    Build one audio stem per distinct instrument present in the transcription.

    Args:
        notes: MuScriptor note dicts (pitch, start_time, end_time/duration,
               velocity, instrument).
        job_id: The generation job id, used to name stem files.
        duration_sec: Optional total track length (default: last note end).

    Returns:
        Mapping of original instrument name -> "/audio/stems/<job>_<slug>.wav".
    """
    if not notes:
        logger.info("instrument_stems: no notes, returning empty instrument map.")
        return {}

    os.makedirs(STEM_DIR, exist_ok=True)

    # Group notes by instrument (preserve first-seen order for a stable DAW).
    by_instrument: dict[str, list[dict]] = {}
    for note in notes:
        inst = note.get("instrument") or "Instrument"
        by_instrument.setdefault(inst, []).append(note)

    if duration_sec is None:
        duration_sec = max((n.get("end_time") or n.get("start_time") or 0.0) for n in notes) + 2.0
    total = max(1, int(duration_sec * sr))

    result: dict[str, str] = {}
    for inst, inst_notes in by_instrument.items():
        family = _family_for(inst)
        slug = _slug(inst)
        buffer = np.zeros(total, dtype=np.float64)

        for note in inst_notes:
            start = max(0.0, float(note.get("start_time") or 0.0))
            dur = float(note.get("duration") or (note.get("end_time", start) - start) or 0.4)
            dur = max(0.08, min(dur, 20.0))
            pid = int(note.get("pitch") or 60)
            vel = float(note.get("velocity") or 90)
            freq = _midi_to_freq(pid)

            seg = _render_note(family, freq, dur, vel, sr)
            s = int(start * sr)
            e = min(total, s + len(seg))
            if s < total:
                buffer[s:e] += seg[: e - s]

        # Normalize the stem so it doesn't clip, preserving relative loudness.
        peak = np.max(np.abs(buffer)) + 1e-6
        buffer = buffer / peak * 0.92

        # Namespace per-instrument files with a "part_" prefix so they never
        # collide with the HTDemucs 4-master stems (e.g. a "Drums" instrument
        # part would otherwise clobber the real separated <job>_drums.wav).
        stem_path = f"{STEM_DIR}/{job_id}_part_{slug}.wav"
        sf.write(stem_path, buffer.astype(np.float32), sr, subtype="PCM_16")
        result[inst] = f"/audio/stems/{job_id}_part_{slug}.wav"
        logger.info("instrument_stems: rendered stem '%s' (%d notes, family=%s)", inst, len(inst_notes), family)

    return result


def render_instrument_parts(
    notes: list[dict],
    job_id: str,
    duration_sec: float | None = None,
    sr: int = SAMPLE_RATE,
) -> tuple[dict[str, str], dict[str, int]]:
    """Render per-instrument stems AND their GM program numbers.

    Convenience wrapper around :func:`render_instrument_stems` that returns
    both the instrument audio mapping and, for each instrument, its General
    MIDI program number (used for accurate DAW/notation instrument labeling).

    Returns:
        (parts, programs) where
        - parts: {instrument_name: "/audio/stems/..."}
        - programs: {instrument_name: gm_program_int}
    """
    parts = render_instrument_stems(notes, job_id, duration_sec, sr)
    programs: dict[str, int] = {}
    for inst in parts:
        programs[inst] = gm_program_for(inst)
    return parts, programs
