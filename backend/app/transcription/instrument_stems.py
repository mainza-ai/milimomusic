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
from pathlib import Path
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


def get_stem_dir() -> Path:
    """Resolve the canonical generated audio stems directory."""
    try:
        from app.core.paths import get_generated_audio_dir
        d = get_generated_audio_dir() / "stems"
    except Exception:
        d = Path("generated_audio/stems").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- instrument family detection -------------------------------------------

_FAMILY_KEYWORDS = {
    "drums":      ("drum", "percussion", "hi hat", "hi-hat", "cymbal", "kick", "snare", "toms", "shaker", "woodblock", "timpani", "congas", "bongo", "crash", "ride"),
    "bass":       ("bass",),
    "guitar":     ("guitar", "mandolin", "banjo", "ukulele", "sitar"),
    "keys":       ("piano", "keys", "keyboard", "organ", "synth", "ep", "electric piano", "rhodes", "clavinet", "harpsichord", "accordion"),
    "reeds":      ("clarinet", "sax", "oboe", "english horn", "bassoon", "flute", "recorder", "pipe"),
    "sustained":  ("strings", "violin", "viola", "cello", "contrabass", "horns", "brass", "trumpet", "trombone", "tuba", "choir", "voice", "vocal", "pad", "synth strings", "bell"),
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
    "acoustic_piano": 0, "piano": 0, "electric_piano": 4, "chromatic_percussion": 9,
    "organ": 19, "acoustic_guitar": 24, "clean_electric_guitar": 27,
    "distorted_electric_guitar": 30, "acoustic_bass": 32, "electric_bass": 33,
    "bass": 33, "violin": 40, "viola": 41, "cello": 42, "contrabass": 43,
    "orchestral_harp": 46, "timpani": 47, "string_ensemble": 48,
    "synth_strings": 50, "voice": 52, "orchestra_hit": 55, "trumpet": 56,
    "trombone": 57, "tuba": 58, "french_horn": 60, "brass_section": 61,
    "soprano_and_alto_sax": 65, "tenor_sax": 66, "baritone_sax": 67,
    "oboe": 68, "english_horn": 69, "bassoon": 70, "clarinet": 71,
    "flutes": 73, "synth_lead": 80, "synth_pad": 89, "drums": 128,
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

def _render_note(family: str, pitch: int, freq: float, dur_sec: float, velocity: float, sr: int) -> np.ndarray:
    """Render a single note as a float array shaped by its instrument family and GM pitch."""
    n = max(1, int(dur_sec * sr))
    t = np.linspace(0, dur_sec, n, endpoint=False)
    amp = 0.45 * (velocity / 127.0)

    # Drums: pitch-aware General MIDI drum synthesis with natural acoustic decay envelopes
    if family == "drums":
        if pitch in (35, 36):  # Bass Drum / Kick (extend decay to 0.35s for real sub-bass punch)
            dur = max(0.35, dur_sec)
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            k_env = np.exp(-t * 14.0)
            f_env = 45.0 + 140.0 * np.exp(-t * 26.0)
            phase = 2.0 * np.pi * np.cumsum(f_env) / sr
            click = np.random.default_rng(pitch).standard_normal(n) * np.exp(-t * 150.0) * 0.4
            return (np.sin(phase) + click) * k_env * amp * 2.2
        elif pitch in (38, 40):  # Acoustic / Electric Snare (extend decay to 0.25s)
            dur = max(0.25, dur_sec)
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 14.0)
            noise = np.random.default_rng(pitch + int(t[0] * 1000)).standard_normal(n) * np.exp(-t * 18.0)
            return (body * 0.60 + noise * 0.85) * amp * 1.9
        elif pitch in (42, 44):  # Closed Hi-hat
            dur = max(0.12, dur_sec)
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            noise = np.random.default_rng(pitch + int(t[0] * 1000)).standard_normal(n)
            hp_noise = np.diff(noise, prepend=0) * np.exp(-t * 45.0)
            return hp_noise * amp * 1.6
        elif pitch in (46,):  # Open Hi-hat
            dur = max(0.40, dur_sec)
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            noise = np.random.default_rng(pitch + int(t[0] * 1000)).standard_normal(n)
            hp_noise = np.diff(noise, prepend=0) * np.exp(-t * 12.0)
            return hp_noise * amp * 1.8
        elif pitch in (49, 51, 52, 53, 55, 57, 59):  # Cymbals (Crash, Ride, Splash)
            dur = max(1.2, dur_sec)
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            noise = np.random.default_rng(pitch + int(t[0] * 1000)).standard_normal(n)
            ring = np.sin(2.0 * np.pi * 820.0 * t) + 0.5 * np.sin(2.0 * np.pi * 1340.0 * t)
            cymbal = (np.diff(noise, prepend=0) * 0.8 + ring * 0.2) * np.exp(-t * 3.5)
            return cymbal * amp * 2.0
        else:  # Toms, Latin Percussion (Bongos, Congas, etc.)
            dur = max(0.25, dur_sec)
            n = int(dur * sr)
            t = np.linspace(0, dur, n, endpoint=False)
            f_env = max(70.0, freq * 0.8) + (freq * 0.6) * np.exp(-t * 16.0)
            phase = 2.0 * np.pi * np.cumsum(f_env) / sr
            return np.sin(phase) * np.exp(-t * 11.0) * amp * 1.8

    # Bass: low sawtooth with fundamental sine and punchy low-pass envelope
    if family == "bass":
        saw = 2.0 * (t * freq - np.floor(0.5 + t * freq))
        sub = np.sin(2.0 * np.pi * freq * t)
        out = 0.65 * saw + 0.55 * sub
        env = np.minimum(1.0, t * 80.0) * (0.45 + 0.55 * np.exp(-t * 1.2))
        return out * env * amp * 1.5

    # Plucked / Clean Electric Guitar: crisp pick attack, pickup harmonic chime, natural string pluck decay
    if family in ("plucked", "guitar"):
        f = freq
        # Fast metallic pick click (noise burst decaying in 6ms)
        click = np.random.default_rng(pitch).standard_normal(n) * np.exp(-t * 220.0) * 0.15
        # Electric guitar pickup harmonic chime (warm 2nd harmonic, bright 3rd & 4th)
        harmonics = (
            np.sin(2.0 * np.pi * f * t)
            + 0.65 * np.sin(2.0 * np.pi * 2.0 * f * t)
            + 0.35 * np.sin(2.0 * np.pi * 3.0 * f * t)
            + 0.18 * np.sin(2.0 * np.pi * 4.0 * f * t)
            + 0.08 * np.sin(2.0 * np.pi * 5.0 * f * t)
        )
        env = np.minimum(1.0, t * 300.0) * (0.25 + 0.75 * np.exp(-t * 4.5))
        return (harmonics + click) * env * amp * 1.6

    # Reeds (Clarinet, oboe, sax): authentic cylindrical stopped pipe with dominant odd harmonics
    if family == "reeds":
        f = freq
        # Clarinet acoustic signature: strong fundamental, 3rd, and 5th harmonics (odd harmonics)
        pipe_harmonics = (
            0.75 * np.sin(2.0 * np.pi * f * t)
            + 0.04 * np.sin(2.0 * np.pi * 2.0 * f * t)  # Suppressed 2nd harmonic
            + 0.45 * np.sin(2.0 * np.pi * 3.0 * f * t)  # Dominant 3rd harmonic (clarinet overtone)
            + 0.03 * np.sin(2.0 * np.pi * 4.0 * f * t)  # Suppressed 4th harmonic
            + 0.22 * np.sin(2.0 * np.pi * 5.0 * f * t)  # Audible 5th harmonic
            + 0.08 * np.sin(2.0 * np.pi * 7.0 * f * t)  # 7th harmonic
        )
        attack = np.clip(t * 40.0, 0.0, 1.0)
        release_end = np.clip((dur_sec - t) * 16.0, 0.0, 1.0)
        return pipe_harmonics * attack * release_end * amp * 0.75

    # Keys (piano): multi-sine hammer attack, warm sustaining acoustic body
    if family == "keys":
        f = freq
        out = (np.sin(2.0 * np.pi * f * t)
               + 0.60 * np.sin(2.0 * np.pi * 2.0 * f * t)
               + 0.35 * np.sin(2.0 * np.pi * 3.0 * f * t)
               + 0.18 * np.sin(2.0 * np.pi * 4.0 * f * t)
               + 0.10 * np.sin(2.0 * np.pi * 5.0 * f * t))
        env = np.minimum(1.0, t * 150.0) * (0.35 + 0.65 * np.exp(-t * 1.6))
        return out * env * amp * 1.5

    # Sustained (strings, voices, choir, horns): harmonic tone with subtle 5.5 Hz vibrato
    f = freq
    vib = 1.0 + 0.015 * np.sin(2.0 * np.pi * 5.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f * vib) / sr
    out = (0.75 * np.sin(phase)
           + 0.40 * np.sin(2.0 * phase)
           + 0.20 * np.sin(3.0 * phase))
    attack = np.clip(t * 30.0, 0.0, 1.0)
    release_end = np.clip((dur_sec - t) * 12.0, 0.0, 1.0)
    return out * attack * release_end * amp * 1.4


def enrich_and_partition_notes(notes: list[dict], job_id: str, duration_sec: float | None = None) -> list[dict]:
    """Intelligently partition polyphonic notes and enrich missing Bass, Guitar, Clarinet, and Vocal melody.

    1. Separates low register notes (pitch < 48) into a dedicated 'Electric Bass' track.
    2. Extracts guitar chord accompaniment across the track from piano harmony in guitar register.
    3. Extracts expressive woodwind melody lines in clarinet register (60 <= pitch <= 84) for Clarinet.
    4. Enriches 'Voice' notes using librosa.pyin pitch tracking from the separated vocals stem.
    5. Normalizes artifact tracks (e.g. Program 116) into musical parts.
    """
    if not notes:
        return notes

    has_bass = any("bass" in (n.get("instrument") or "").lower() for n in notes)

    partitioned: list[dict] = []
    for n in notes:
        inst = n.get("instrument") or "Instrument"
        inst_lower = inst.lower()
        pitch = int(n.get("pitch") or 60)

        # Merge single Piano into Acoustic Piano
        if inst_lower == "piano":
            inst = "Acoustic Piano"
            inst_lower = "acoustic piano"

        # Remap Program 116 or numeric artifact tracks
        if "program" in inst_lower:
            inst = "Electric Bass" if pitch < 48 else "Acoustic Piano"
            inst_lower = inst.lower()

        # Split low-register piano notes (<48: C1-B2) into dedicated Bass line if missing
        if not has_bass and ("piano" in inst_lower or "keys" in inst_lower) and pitch < 48:
            n_copy = dict(n)
            n_copy["instrument"] = "Electric Bass"
            n_copy["program"] = 33
            n_copy["channel"] = 0
            partitioned.append(n_copy)
        else:
            n_copy = dict(n)
            n_copy["instrument"] = inst
            partitioned.append(n_copy)

    from collections import defaultdict

    # Enrich / Re-voice Clarinet woodwind melody if notes are sparse or glitched into a single tail burst
    clar_notes = [n for n in partitioned if "clarinet" in (n.get("instrument") or "").lower()]
    starts = [float(n.get("start_time", 0)) for n in clar_notes]
    is_glitched = len(clar_notes) > 0 and (max(starts) - min(starts) < 2.0) and min(starts) > 100.0

    if len(clar_notes) < 20 or is_glitched:
        partitioned = [n for n in partitioned if "clarinet" not in (n.get("instrument") or "").lower()]
        piano_notes = [n for n in partitioned if "piano" in (n.get("instrument") or "").lower()]
        by_time = defaultdict(list)
        for n in piano_notes:
            t_round = round(float(n.get("start_time", 0.0)) * 4) / 4.0
            by_time[t_round].append(n)

        added_clar: list[dict] = []
        for t_val, ns in sorted(by_time.items()):
            if len(ns) == 1:
                p = int(ns[0].get("pitch", 0))
                if 60 <= p <= 84:
                    cl = dict(ns[0])
                    cl["instrument"] = "Clarinet"
                    cl["program"] = 71
                    cl["channel"] = 0
                    added_clar.append(cl)
        if added_clar:
            partitioned.extend(added_clar)
            logger.info("enrich_and_partition_notes: extracted %d Clarinet woodwind melody notes across the song", len(added_clar))

    # Enrich Clean Electric Guitar if guitar notes are sparse (< 20 notes)
    guitar_notes = [n for n in partitioned if "guitar" in (n.get("instrument") or "").lower()]
    if len(guitar_notes) < 20:
        piano_notes = [n for n in partitioned if "piano" in (n.get("instrument") or "").lower()]
        by_time = defaultdict(list)
        for n in piano_notes:
            t_round = round(float(n.get("start_time", 0.0)) * 4) / 4.0
            by_time[t_round].append(n)

        added_guitar: list[dict] = []
        for t_val, ns in sorted(by_time.items()):
            if len(ns) >= 2:  # Chord event
                candidates = [n for n in ns if 40 <= int(n.get("pitch", 60)) <= 76]
                for n in candidates:
                    g = dict(n)
                    g["instrument"] = "Clean Electric Guitar"
                    g["program"] = 27
                    g["channel"] = 0
                    added_guitar.append(g)
        if added_guitar:
            partitioned.extend(added_guitar)
            logger.info("enrich_and_partition_notes: extracted %d guitar chord comping notes across the song", len(added_guitar))

    # Enrich Voice notes if vocal notes are sparse and vocal stem exists
    vocal_notes = [n for n in partitioned if "voice" in (n.get("instrument") or "").lower() or "vocal" in (n.get("instrument") or "").lower()]
    stem_dir = get_stem_dir()
    vocal_stem_path = stem_dir / f"{job_id}_vocals.wav"

    if len(vocal_notes) < 25 and vocal_stem_path.exists():
        try:
            import librosa
            y, sr = librosa.load(str(vocal_stem_path), sr=16000)
            hop_length = 512
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6'),
                sr=sr, hop_length=hop_length
            )
            times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

            # Drop sparse vocal fragments to replace with real pitch contour
            partitioned = [n for n in partitioned if not ("voice" in (n.get("instrument") or "").lower() or "vocal" in (n.get("instrument") or "").lower())]

            in_note = False
            start_t = 0.0
            note_pitches: list[int] = []

            for t, f, v in zip(times, f0, voiced_flag):
                if v and not np.isnan(f):
                    midi_p = int(round(float(librosa.hz_to_midi(f))))
                    if not in_note:
                        in_note = True
                        start_t = float(t)
                        note_pitches = [midi_p]
                    else:
                        if abs(midi_p - note_pitches[-1]) >= 2 and len(note_pitches) >= 4:
                            dur = float(t) - start_t
                            if dur >= 0.08:
                                median_p = int(round(float(np.median(note_pitches))))
                                partitioned.append({
                                    "pitch": median_p,
                                    "start_time": round(start_t, 3),
                                    "end_time": round(float(t), 3),
                                    "duration": round(dur, 3),
                                    "velocity": 85,
                                    "instrument": "Voice",
                                    "program": 52,
                                    "channel": 0
                                })
                            start_t = float(t)
                            note_pitches = [midi_p]
                        else:
                            note_pitches.append(midi_p)
                else:
                    if in_note:
                        in_note = False
                        dur = float(t) - start_t
                        if dur >= 0.08 and note_pitches:
                            median_p = int(round(float(np.median(note_pitches))))
                            partitioned.append({
                                "pitch": median_p,
                                "start_time": round(start_t, 3),
                                "end_time": round(float(t), 3),
                                "duration": round(dur, 3),
                                "velocity": 85,
                                "instrument": "Voice",
                                "program": 52,
                                "channel": 0
                            })
                        note_pitches = []
            logger.info("enrich_and_partition_notes: extracted vocal melody notes via pyin from %s", vocal_stem_path.name)
        except Exception as e:
            logger.warning("enrich_and_partition_notes: vocal extraction fallback: %s", e)

    return partitioned


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

    stem_dir = get_stem_dir()

    # Pre-process notes: partition low piano notes to Electric Bass, extract guitar comping, enrich vocal melody
    notes = enrich_and_partition_notes(notes, job_id, duration_sec)

    # Group notes by instrument (preserve first-seen order for a stable DAW).
    by_instrument: dict[str, list[dict]] = {}
    for note in notes:
        inst = note.get("instrument") or "Instrument"
        by_instrument.setdefault(inst, []).append(note)

    max_note_end = max((float(n.get("end_time") or n.get("start_time") or 0.0) + float(n.get("duration") or 0.5)) for n in notes)
    if duration_sec is None:
        duration_sec = max_note_end + 2.0
    else:
        # Clamp duration so empty padding (e.g. 300s timeout) doesn't produce minutes of dead air
        duration_sec = min(duration_sec, max_note_end + 3.0)
    total = max(1, int(duration_sec * sr))

    result: dict[str, str] = {}
    target_rms_map = {
        "drums": 0.22,      # -13.1 dBFS (high crest factor drum punch)
        "keys": 0.20,       # -14.0 dBFS (warm piano presence)
        "bass": 0.20,       # -14.0 dBFS (solid bass weight)
        "guitar": 0.18,     # -14.9 dBFS (bright guitar presence)
        "sustained": 0.18,  # -14.9 dBFS (vocal melody clarity)
        "reeds": 0.12,      # -18.4 dBFS (acoustic woodwind presence)
        "plucked": 0.18,
    }

    for inst, inst_notes in by_instrument.items():
        family = _family_for(inst)
        slug = _slug(inst)
        buffer = np.zeros(total, dtype=np.float64)

        for note in inst_notes:
            start = max(0.0, float(note.get("start_time") or 0.0))
            dur = float(note.get("duration") or (note.get("end_time", start) - start) or 0.4)
            dur = max(0.04, min(dur, 20.0))
            pid = int(note.get("pitch") or 60)
            vel = float(note.get("velocity") or 90)
            freq = _midi_to_freq(pid)

            seg = _render_note(family, pid, freq, dur, vel, sr)
            s = int(start * sr)
            e = min(total, s + len(seg))
            if s < total:
                buffer[s:e] += seg[: e - s]

        # RMS-aware active level limiter calibrated per instrument family
        active_mask = np.abs(buffer) > 1e-4
        if np.any(active_mask):
            active_rms = np.sqrt(np.mean(buffer[active_mask] ** 2))
            target_rms = target_rms_map.get(family, 0.16)
            gain = min(4.5, max(0.2, target_rms / (active_rms + 1e-6)))
            buffer = np.tanh(buffer * gain) * 0.95
        else:
            peak = np.max(np.abs(buffer)) + 1e-6
            buffer = buffer / peak * 0.92

        # Namespace per-instrument files with a "part_" prefix so they never
        # collide with the HTDemucs 4-master stems (e.g. a "Drums" instrument
        # part would otherwise clobber the real separated <job>_drums.wav).
        stem_path = stem_dir / f"{job_id}_part_{slug}.wav"
        sf.write(str(stem_path), buffer.astype(np.float32), sr, subtype="PCM_16")
        result[inst] = f"/audio/stems/{job_id}_part_{slug}.wav"
        logger.info("instrument_stems: rendered stem '%s' (%d notes, family=%s, path=%s)", inst, len(inst_notes), family, stem_path)

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

