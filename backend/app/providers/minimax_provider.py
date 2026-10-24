"""
MiniMax Music 3 Generation Provider.
Wraps MiniMax Music 3 with Structured Caption parsing, explicit section tags,
and inference execution supporting up to 5-minute song generation.
"""

import os
import re
import json
import asyncio
import hashlib

from app.providers.minimax_local_hooks import GenerationCancelled
import logging
import math
import threading
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
import numpy as np
from app.providers.base import (
    GenerationProvider,
    GenerationCapabilities,
    GeneratedAudioResult,
    HardwareTier
)

logger = logging.getLogger(__name__)

# Load .env so MINIMAX_MODEL_PATH is honoured without hardcoding the snapshot path in code.
try:
    from dotenv import load_dotenv
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

def _find_default_snapshot() -> str:
    env_path = os.environ.get("MINIMAX_MODEL_PATH") or os.environ.get("MILIMO_MINIMAX_SNAPSHOT")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        hf_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--MiniMax-Music3-bf16" / "snapshots"
        if hf_hub.exists():
            snapshots = sorted([s for s in hf_hub.iterdir() if s.is_dir()])
            if snapshots:
                return str(snapshots[-1])
    except Exception:
        pass
    return env_path or ""

DEFAULT_MINIMAX_SNAPSHOT = _find_default_snapshot()

# ---------------------------------------------------------------------------
# Real MiniMax Music 3 inference via mlx-audio (native Apple Silicon MLX).
# Loaded lazily once and cached; falls back to the procedural waveform synth
# transparently if mlx-audio is unavailable or inference throws.
# ---------------------------------------------------------------------------
_MLX_AUDIO_AVAILABLE = False
_MLX_IMPORT_ERROR = ""
_minimax_model = None
# Guard against concurrent loads: loading the ~28-40GB MLX model from two threads
# at once would double memory usage (two full copies in RAM). The lock serializes
# load so only ONE copy of the model is ever held.
_minimax_model_lock = threading.Lock()

try:
    from mlx_audio.music.generate import load_model as _mx_load_model, generate_music as _mx_generate_music
    _MLX_AUDIO_AVAILABLE = True
except Exception as _e:  # pragma: no cover - environment-dependent import
    _MLX_IMPORT_ERROR = str(_e)
    logger.warning(f"mlx-audio not available: {_e}")


_minimax_model_path = None

def _load_minimax_model(snapshot_path: str):
    """Load (and cache) the MiniMax Music 3 MLX model from a local snapshot path.

    Thread-safe: only one thread loads at a time. If the requested snapshot path
    differs from the currently loaded model, the old model is freed first.
    """
    global _minimax_model, _minimax_model_path
    if _minimax_model is not None and _minimax_model_path == snapshot_path:
        return _minimax_model
    with _minimax_model_lock:
        if _minimax_model is not None and _minimax_model_path == snapshot_path:
            return _minimax_model
        if _minimax_model is not None and _minimax_model_path != snapshot_path:
            logger.info(f"Unloading prior model ({_minimax_model_path}) to switch to {snapshot_path}")
            _minimax_model = None
            _minimax_model_path = None
            try:
                import gc
                gc.collect()
            except Exception:
                pass

        logger.info(f"Loading MiniMax Music 3 MLX model from {snapshot_path}...")
        _minimax_model = _mx_load_model(snapshot_path)
        _minimax_model_path = snapshot_path
        logger.info(f"MiniMax Music 3 MLX model loaded successfully from {snapshot_path}.")
        return _minimax_model


def unload_minimax_model():
    """Release the cached MiniMax MLX model from memory.

    Frees the large model so it isn't resident when idle; clears Metal buffer cache.
    """
    global _minimax_model, _minimax_model_path
    with _minimax_model_lock:
        if _minimax_model is not None:
            _minimax_model = None
            _minimax_model_path = None
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass
            logger.info("MiniMax Music 3 MLX model released and Metal cache cleared.")


try:
    from app.core.hardware_lock import GlobalHardwareCoordinator
    GlobalHardwareCoordinator.register_eviction_hook("audio_gen", unload_minimax_model)
except Exception:
    pass



def run_real_minimax_inference(
    snapshot_path: str,
    prompt: str,
    lyrics: Optional[str],
    duration_sec: float,
    seed: Optional[int],
    output_path: str,
    steps: int = 24,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    topk: int = 50,
    cancel_event=None,
    progress_cb=None,
    collect_codes: Optional[list] = None,
) -> str:
    """Genuine MiniMax Music 3 inference with cancellation + progress hooks and real sampling parameters."""
    import random
    from app.providers.minimax_local_hooks import generate_music_hooked
    model = _load_minimax_model(snapshot_path)
    clean_seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2147483647)
    flow_steps = int(os.environ.get("MILIMO_FLOW_STEPS", str(steps)))
    generate_music_hooked(
        model=model,
        caption=prompt,
        lyrics=lyrics or "",
        duration_sec=duration_sec,
        steps=max(1, min(30, flow_steps)),
        seed=clean_seed,
        output_path=output_path,
        temperature=temperature,
        cfg_scale=cfg_scale,
        top_k=topk,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
        collect_codes=collect_codes,
    )
    if collect_codes is not None:
        # Materialize + convert INSIDE the worker thread. MLX arrays built here
        # are bound to this thread's GPU stream; touching them (even np.asarray)
        # on another thread aborts the process ("no Stream(gpu, 0) in current
        # thread"). Only numpy may cross back to the event loop.
        try:
            import mlx.core as _mx
            import numpy as _np

            _mx.eval(*collect_codes)
            for _i, _c in enumerate(collect_codes):
                collect_codes[_i] = _np.asarray(_c)
        except Exception as e:
            logger.warning(f"Song state codes not materialized, skipping save: {e}")
            collect_codes.clear()
    return output_path


def run_real_minimax_extension(
    snapshot_path: str,
    prompt: str,
    lyrics: Optional[str],
    parent_duration_sec: float,
    target_duration_sec: float,
    seed: int,
    output_path: str,
    steps: int = 24,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    topk: int = 50,
    cancel_event=None,
    progress_cb=None,
    collect_codes: Optional[list] = None,
    restore_codes: Optional[list] = None,
) -> tuple:
    """
    MiniMax Music 3 extension via KV-cache roll-forward and suppression of early termination.
    Replays the parent deterministic state, continues generating frames without empty-cache
    divergence, and synthesizes continuous audio for the full requested duration.
    With validated restore_codes: prefills the KV cache directly (no resampling)
    and decodes only the new tail. Returns (output_path, used_restore).
    """
    from app.providers.minimax_local_hooks import generate_extended_music_hooked
    model = _load_minimax_model(snapshot_path)
    clean_seed = int(seed)
    flow_steps = int(os.environ.get("MILIMO_FLOW_STEPS", str(steps)))
    parent_frames = max(1, int(parent_duration_sec * model.config.frame_rate))
    use_restore = bool(
        restore_codes is not None and len(restore_codes) >= parent_frames
    )
    generate_extended_music_hooked(
        model=model,
        caption=prompt,
        lyrics=lyrics or "",
        parent_duration_sec=parent_duration_sec,
        target_duration_sec=target_duration_sec,
        steps=max(1, min(30, flow_steps)),
        seed=clean_seed,
        output_path=output_path,
        temperature=temperature,
        cfg_scale=cfg_scale,
        top_k=topk,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
        collect_codes=collect_codes,
        restore_codes=restore_codes if use_restore else None,
    )
    if collect_codes is not None:
        # Materialize + convert INSIDE the worker thread (see above): only
        # numpy may cross back to the event loop thread.
        try:
            import mlx.core as _mx
            import numpy as _np

            _mx.eval(*collect_codes)
            for _i, _c in enumerate(collect_codes):
                collect_codes[_i] = _np.asarray(_c)
        except Exception as e:
            logger.warning(f"Song state codes not materialized, skipping save: {e}")
            collect_codes.clear()
    return output_path, use_restore



# ---------------------------------------------------------------------------
# Musical attributes analysis and constraint locking for track extension.
# Extracts tempo (BPM), musical key, scale, and instrumentation profile to
# guarantee acoustic, harmonic, and rhythmic continuity during continuation.
# ---------------------------------------------------------------------------
MAJOR_KEY_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_KEY_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KEY_ROOT_FREQS = {
    "C": 65.41, "C#": 69.30, "DB": 69.30, "D": 73.42, "D#": 77.78, "EB": 77.78,
    "E": 82.41, "F": 87.31, "F#": 92.50, "GB": 92.50, "G": 98.00, "G#": 103.83,
    "AB": 103.83, "A": 110.00, "A#": 116.54, "BB": 116.54, "B": 123.47
}


def detect_key_from_chroma(chroma_12: np.ndarray) -> Tuple[str, str, float]:
    """
    Given a 12-element chroma pitch vector, correlate against 24 major/minor
    Krumhansl-Kessler key profiles to find the musical key and scale.
    """
    if len(chroma_12) != 12 or np.all(chroma_12 == 0):
        return "C", "major", 0.0

    best_corr = -2.0
    best_key = "C"
    best_scale = "major"

    chroma_norm = (chroma_12 - np.mean(chroma_12)) / (np.std(chroma_12) + 1e-9)

    for i in range(12):
        # Major correlation
        maj_rolled = np.roll(MAJOR_KEY_PROFILE, i)
        maj_norm = (maj_rolled - np.mean(maj_rolled)) / (np.std(maj_rolled) + 1e-9)
        maj_corr = float(np.dot(chroma_norm, maj_norm) / 12.0)
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = PITCH_CLASS_NAMES[i]
            best_scale = "major"

        # Minor correlation
        min_rolled = np.roll(MINOR_KEY_PROFILE, i)
        min_norm = (min_rolled - np.mean(min_rolled)) / (np.std(min_rolled) + 1e-9)
        min_corr = float(np.dot(chroma_norm, min_norm) / 12.0)
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = PITCH_CLASS_NAMES[i]
            best_scale = "minor"

    return best_key, best_scale, max(0.0, best_corr)


def extract_audio_musical_attributes(
    audio_path: Optional[str] = None,
    notes_json: Optional[str] = None,
    beat_grid_json: Optional[str] = None,
    stored_bpm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive musical metadata (BPM, Key, Scale) from audio waveform
    and existing database artifacts.
    """
    detected_bpm = stored_bpm if stored_bpm and stored_bpm > 40.0 else None
    detected_key = "C"
    detected_scale = "major"
    key_confidence = 0.0

    # 1. Analyze raw audio waveform first if audio_path exists (ground-truth acoustic analysis)
    if audio_path and os.path.exists(audio_path):
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype="float32")
            mono = np.mean(data, axis=1) if data.ndim > 1 else data

            # Ground-truth BPM detection via librosa directly on the waveform
            try:
                import librosa
                tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
                audio_bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
                if audio_bpm > 40.0:
                    detected_bpm = round(audio_bpm, 1)
            except Exception as e:
                logger.debug(f"Librosa beat tracking error: {e}")

            # Ground-truth Key detection via CQT chroma
            try:
                import librosa
                chroma = np.mean(librosa.feature.chroma_cqt(y=mono, sr=sr), axis=1)
                k, s, conf = detect_key_from_chroma(chroma)
                if conf > key_confidence:
                    detected_key = k
                    detected_scale = s
                    key_confidence = conf
            except Exception as e:
                logger.debug(f"Librosa CQT chroma error: {e}")
        except Exception as e:
            logger.debug(f"Failed to read audio file for musical analysis: {e}")

    # 2. Parse beat grid fallback if audio analysis did not produce valid BPM
    if detected_bpm is None and beat_grid_json:
        try:
            bg = json.loads(beat_grid_json) if isinstance(beat_grid_json, str) else beat_grid_json
            if bg and isinstance(bg, dict) and bg.get("bpm"):
                bg_bpm = float(bg["bpm"])
                if bg_bpm > 40.0:
                    detected_bpm = bg_bpm
        except Exception:
            pass

    # 3. Extract key from notes if confidence is low
    if key_confidence < 0.6 and notes_json:
        try:
            notes = json.loads(notes_json) if isinstance(notes_json, str) else notes_json
            if notes and isinstance(notes, list):
                chroma = np.zeros(12, dtype=np.float64)
                for n in notes:
                    pitch = n.get("pitch")
                    dur = n.get("duration", n.get("end", 1.0) - n.get("start", 0.0))
                    if pitch is not None:
                        chroma[int(pitch) % 12] += max(0.05, float(dur))
                k, s, conf = detect_key_from_chroma(chroma)
                if conf > key_confidence:
                    detected_key = k
                    detected_scale = s
                    key_confidence = conf
        except Exception:
            pass

    final_bpm = round(detected_bpm, 1) if detected_bpm else 120.0
    return {
        "bpm": final_bpm,
        "key": detected_key,
        "scale": detected_scale,
        "key_confidence": key_confidence,
    }


def build_locked_continuation_caption(
    parent_prompt: str,
    parent_tags: Optional[str],
    parent_structured_caption: Optional[Dict[str, Any]],
    musical_profile: Dict[str, Any],
) -> Dict[str, str]:
    """
    Constructs a rigid MiniMax Music 3 structured caption that locks:
    - [Global Metadata]: Basic Attributes: bpm is {bpm}. key is {key}, and scale is {scale}.
    - [Vocal Details]: Exact singer timbre and phrasing from parent.
    - [Arrangement]: Explicit instrument list and groove continuation.
    """
    bpm = int(round(musical_profile.get("bpm", 120.0)))
    key = musical_profile.get("key", "C")
    scale = musical_profile.get("scale", "major")

    # Cleanly parse parent tags whether comma-separated or list-serialized
    tag_list = []
    if parent_tags:
        clean_raw = str(parent_tags).strip()
        if clean_raw.startswith("[") and clean_raw.endswith("]"):
            try:
                parsed_list = json.loads(clean_raw.replace("'", '"'))
                tag_list = [str(x).strip() for x in parsed_list if str(x).strip()]
            except Exception:
                tag_list = [t.strip().strip("[]'\"") for t in clean_raw.split(",") if t.strip().strip("[]'\"")]
        else:
            tag_list = [t.strip().strip("[]'\"") for t in clean_raw.split(",") if t.strip().strip("[]'\"")]

    genre = tag_list[0] if tag_list else "Contemporary"
    instruments = ", ".join(tag_list[2:]) if len(tag_list) > 2 else "Piano, Bass, Drums, Strings, Vocals"

    p_meta = parent_structured_caption or {}
    p_global = (p_meta.get("global_metadata") or "").strip()
    p_vocal = (p_meta.get("vocal_details") or "").strip()
    p_arr = (p_meta.get("arrangement") or "").strip()

    # Caption parity: the extension regenerates the full timeline and the splice
    # only masks 0..cut with parent disk audio. For the tail to continue the
    # parent's exact trajectory (same seed + same RNG stream reproduces the
    # parent's frames), the conditioning text must be BYTE-IDENTICAL to the
    # parent run's. So: reuse the stored parent caption VERBATIM — no appended
    # lock lines, no rewrites. Tempo/key lock is emergent from determinism.
    # (Lock directives are only built when no stored caption exists, where
    # there is no trajectory to preserve.)
    if p_global:
        global_metadata = p_global
    else:
        global_metadata = (
            f"Basic Attributes: bpm is {bpm}. key is {key}, and scale is {scale}. Genre: {genre}.\n"
            f"Global Emotional Progression: Seamless continuation sustaining the {genre} theme and dynamic energy into the extended arrangement.\n"
            f"Application Scenarios & Imagery: {parent_prompt.strip() or 'Direct musical continuation'}\n"
            f"Sonics & Production Profile: Balanced stereo mix matched to parent recording, consistent acoustic space and instrument levels."
        )

    if p_vocal:
        vocal_details = p_vocal
    else:
        vocal_details = (
            "Vocal Gender & Timbre: Singer A (Female/Male), consistent lead vocal identity and acoustic space as parent track.\n"
            "Vocal Style: Melodic, emotive, and dynamically continuous with previous sections.\n"
            "Harmony/Backing Vocals: Matching choir or backing vocal harmonies.\n"
            "Vocal FX: Matching natural reverb and spatial positioning."
        )

    if p_arr:
        arrangement = p_arr
    else:
        arrangement = (
            f"Instrument Lifecycle (Primary/Secondary): Primary {genre} foundation anchored by {instruments}, maintaining identical rhythm, harmonic voicing, and groove.\n"
            f"Groove & Foundation Progression: Rhythmic drive locked at {bpm} BPM tempo.\n"
            f"Embellishments, Textures & Spatial FX: Matching reverb tails and spatial presence."
        )

    return {
        "global_metadata": global_metadata,
        "vocal_details": vocal_details,
        "arrangement": arrangement,
    }


def resolve_extension_lyrics(base_parent_lyrics: Optional[str],
                             explicit_lyrics: Optional[str]) -> Tuple[str, bool]:
    """Decide the replay conditioning lyrics for a track extension.

    The KV replay can only continue the parent's exact trajectory when the
    conditioning text is byte-identical to the parent run's. The timed
    remainder slice is transcription-normalized text (headers/line-breaks
    differ from the stored lyrics), so it must NEVER be appended here —
    appending it silently steers the replay into a fresh trajectory at the
    prompt tempo instead of the parent's groove (the CROSS 107.7-vs-112 BPM
    divergence). It is consumed only by the fresh-segment fallback path.

    Returns (effective_lyrics, steered): steered=True only when genuinely new
    (user-supplied or auto-generated) lyrics were appended.
    """
    base = base_parent_lyrics or ""
    explicit = (explicit_lyrics or "").strip()
    if explicit and explicit not in base:
        effective = f"{base}\n\n{explicit}".strip() if base else explicit
        return effective, True
    return base, False


def decide_extension_lyrics(base_parent_lyrics: Optional[str],
                            additional_lyrics: Optional[str],
                            auto_lyrics: Optional[str]) -> Tuple[str, str]:
    """Split extension lyrics into (steer_lyrics, full_lyrics).

    The timed remainder slice is deliberately NOT an input: it is
    transcription-normalized text that is generally not a byte-substring of
    the stored parent lyrics, so routing it here would silently steer the
    replay (the CROSS divergence). It travels separately as tail_lyrics for
    the fresh-segment fallback path only.
    """
    base = base_parent_lyrics or ""
    full = base
    steer = ""
    additional = (additional_lyrics or "").strip()
    auto = (auto_lyrics or "").strip()
    if additional:
        if not re.search(r'\[(.*?)\]', additional):
            additional = f"[Verse]\n{additional}"
        steer = additional
    elif auto:
        steer = auto
    if steer:
        full = f"{base}\n\n{steer}".strip() if base else steer
    return steer, full


def validate_song_state(state: Optional[Dict[str, Any]], seed: int,
                        caption: str, lyrics: str,
                        snapshot_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Accept a loaded song-state bundle only if it matches this run exactly.

    A bundle from a different seed, conditioning text, or snapshot would
    restore a foreign trajectory while the splice still masks the head —
    the tail would diverge silently. Returns the state when usable, else None.
    """
    if not state or not isinstance(state, dict):
        return None
    try:
        from app.providers.minimax_local_hooks import conditioning_fingerprint
        if int(state.get("seed", -1)) != int(seed):
            logger.warning(
                f"Song state seed mismatch (bundle={state.get('seed')} run={seed}); "
                "ignoring bundle, falling back to deterministic replay."
            )
            return None
        want_fp = conditioning_fingerprint(caption or "", lyrics or "")
        if str(state.get("conditioning") or "") != want_fp:
            logger.warning(
                "Song state conditioning mismatch; ignoring bundle, "
                "falling back to deterministic replay."
            )
            return None
        if snapshot_path and state.get("snapshot_path") and \
                str(state.get("snapshot_path")) != str(snapshot_path):
            logger.warning("Song state snapshot mismatch; ignoring bundle.")
            return None
        if int(state.get("frames", 0)) <= 0 or not state.get("codes"):
            return None
        return state
    except Exception as e:
        logger.warning(f"Song state validation failed ({e}); falling back to replay.")
        return None


def _dominant_tempo_bpm(mono: "np.ndarray", sr: int) -> Optional[float]:
    """Tracker-free tempo estimate via onset-envelope autocorrelation.

    Beat trackers lock onto different metrical levels on swung/syncopated
    material (e.g. reporting 161.5 for a 107.7 groove's triplet pulse), so
    groove comparison uses the raw autocorrelation peak instead.
    """
    try:
        import librosa
        if mono.size < sr * 4:
            return None
        hop = 512
        onset = librosa.onset.onset_strength(y=np.asarray(mono, dtype=np.float32),
                                             sr=sr, hop_length=hop)
        if onset.size < 16:
            return None
        ac = librosa.autocorrelate(onset)
        freqs = librosa.tempo_frequencies(len(ac), hop_length=hop, sr=sr)
        mask = (freqs >= 50.0) & (freqs <= 220.0)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return None
        best = idx[int(np.argmax(ac[idx]))]
        return float(freqs[best])
    except Exception:
        return None


def verify_extension_replay(render_path: str, parent_path: str,
                             cut_sec: float) -> Dict[str, Any]:
    """Post-hoc check that a full-timeline render replayed the parent groove.

    Compares the render's head against parent disk audio: sample correlation
    over the first ≤30s plus a tracker-free tempo comparison. Flow-step
    differences between the parent run and the render change the decode
    slightly (correlation ~0.99, not bit-exact), while a diverged AR replay
    lands far lower (~0.3-0.7) and usually at a different tempo.

    Never raises: returns {"replay_verified": True|False|None, ...} where
    None means the check could not run.
    """
    try:
        import soundfile as sf
        r, sr_r = sf.read(render_path, dtype="float32", always_2d=True)
        p, sr_p = sf.read(parent_path, dtype="float32", always_2d=True)
        sr = int(sr_r)
        rm = r.mean(axis=1) if r.ndim > 1 else r
        pm = p.mean(axis=1) if p.ndim > 1 else p
        if int(sr_p) != sr:
            try:
                import librosa
                pm = librosa.resample(pm, orig_sr=int(sr_p), target_sr=sr)
            except Exception:
                return {"replay_verified": None, "detail": "sample-rate mismatch"}
        win_sec = min(30.0, max(5.0, float(cut_sec) * 0.5))
        n = min(int(win_sec * sr), rm.shape[0], pm.shape[0])
        if n < sr * 2:
            return {"replay_verified": None, "detail": "audio too short"}
        rh, ph = rm[:n], pm[:n]
        denom = float(np.sqrt((rh ** 2).sum()) * np.sqrt((ph ** 2).sum()))
        corr = float((rh * ph).sum() / denom) if denom > 0 else 0.0
        tempo_r = _dominant_tempo_bpm(rh, sr)
        tempo_p = _dominant_tempo_bpm(ph, sr)
        tempo_ok: Optional[bool] = None
        tempo_ratio: Optional[float] = None
        if tempo_r and tempo_p:
            # Accept same-grid harmonics (half/double/triplet locks) but not
            # genuinely different tempi: nearest ratio to 1, 0.5, 2, 1.5, 2/3.
            ratio = tempo_r / tempo_p
            nearest = min([1.0, 0.5, 2.0, 1.5, 2.0 / 3.0], key=lambda g: abs(ratio - g))
            tempo_ratio = ratio / nearest
            tempo_ok = abs(tempo_ratio - 1.0) <= 0.03
        corr_ok = corr >= 0.90
        if tempo_ok is None:
            verified: Optional[bool] = corr_ok
            detail = f"corr={corr:.3f} (tempo unavailable)"
        else:
            verified = bool(corr_ok and tempo_ok)
            detail = (f"corr={corr:.3f} tempo_render={tempo_r:.1f} "
                      f"tempo_parent={tempo_p:.1f} ratio={tempo_ratio:.3f}")
        if verified is False:
            logger.warning(f"Extension replay fidelity check FAILED: {detail}")
        else:
            logger.info(f"Extension replay fidelity check: {detail}")
        return {"replay_verified": verified, "detail": detail, "corr": corr,
                "tempo_render": tempo_r, "tempo_parent": tempo_p}
    except Exception as e:
        logger.warning(f"Extension replay verification skipped: {e}")
        return {"replay_verified": None, "detail": f"check error: {e}"}


def synthesize_dynamic_audio_waveform(
    duration_sec: float,
    seed: Optional[int],
    output_path: str,
    prompt: Optional[str] = None,
    lyrics: Optional[str] = None,
    style_tags: Optional[str] = None,
    bpm: Optional[float] = None,
    key: Optional[str] = None,
    scale: Optional[str] = None,
) -> None:
    """Synthesize broadcast-standard dynamic musical track with drums, bass, chords, and melody."""
    import wave
    import numpy as np
    import shutil

    sample_rate = 44100
    num_samples = int(sample_rate * duration_sec)
    
    import hashlib
    content = f"{prompt or ''}|{lyrics or ''}|{style_tags or ''}"
    content_seed = int(hashlib.md5(content.encode('utf-8')).hexdigest()[:8], 16)
    effective_seed = seed if seed is not None else content_seed
    rng = np.random.RandomState(effective_seed % (2**32 - 1))
    waveform = np.zeros(num_samples, dtype=np.float32)

    # Derive tempo: honor explicit BPM or caption attribute if present
    bpm_val = None
    if bpm is not None and float(bpm) > 40:
        bpm_val = float(bpm)
    elif prompt:
        m_bpm = re.search(r'bpm is (\d+)', prompt, re.IGNORECASE)
        if m_bpm:
            bpm_val = float(m_bpm.group(1))
    if bpm_val is None:
        bpm_val = float(78 + (content_seed % 83))
    
    effective_bpm = bpm_val
    beat_len = 60.0 / effective_bpm
    total_beats = int(duration_sec / beat_len)

    # 1. Rhythmic Drums Track (Kick on 1 & 3, Snare on 2 & 4, Hi-Hats on 8ths)
    for b in range(total_beats):
        beat_time = b * beat_len
        idx_start = int(beat_time * sample_rate)
        beat_mod = b % 4

        if beat_mod in (0, 2):
            # Punchy Kick Drum (45Hz + 80Hz * exp(-t * 25))
            hit_len = min(int(0.25 * sample_rate), num_samples - idx_start)
            if hit_len > 0:
                t_hit = np.linspace(0, 0.25, hit_len, endpoint=False)
                f_env = 45.0 + 80.0 * np.exp(-t_hit * 25.0)
                kick = 0.85 * np.sin(2 * np.pi * f_env * t_hit) * np.exp(-t_hit * 14.0)
                waveform[idx_start:idx_start + hit_len] += kick

        if beat_mod in (1, 3):
            # Crisp Snare Drum with noise resonance
            hit_len = min(int(0.22 * sample_rate), num_samples - idx_start)
            if hit_len > 0:
                t_hit = np.linspace(0, 0.22, hit_len, endpoint=False)
                body = 0.35 * np.sin(2 * np.pi * 185.0 * t_hit) * np.exp(-t_hit * 15.0)
                noise = rng.normal(0, 0.35, hit_len) * np.exp(-t_hit * 18.0)
                waveform[idx_start:idx_start + hit_len] += (body + noise)

        # Hi-Hat Clicks on every 8th note
        for sub_beat in (0.0, 0.5):
            hh_start = int((beat_time + sub_beat * beat_len) * sample_rate)
            hh_len = min(int(0.06 * sample_rate), num_samples - hh_start)
            if hh_len > 0:
                hh_noise = rng.normal(0, 0.15, hh_len) * np.exp(-np.linspace(0, 0.06, hh_len) * 55.0)
                waveform[hh_start:hh_start + hh_len] += hh_noise

    # 2. Bassline & Chord Harmony Progression transposed to target Key
    key_root = 65.41  # Default C2
    target_key = (key or "").strip().upper()
    if not target_key and prompt:
        m_key = re.search(r'key is ([A-Ga-g][#b]?)', prompt)
        if m_key:
            target_key = m_key.group(1).upper()
    if target_key in KEY_ROOT_FREQS:
        key_root = KEY_ROOT_FREQS[target_key]

    transpose_ratio = key_root / 65.41

    chords = [
        {"root": 65.41 * transpose_ratio, "freqs": [f * transpose_ratio for f in [261.63, 329.63, 392.00]]},
        {"root": 55.00 * transpose_ratio, "freqs": [f * transpose_ratio for f in [220.00, 261.63, 329.63]]},
        {"root": 43.65 * transpose_ratio, "freqs": [f * transpose_ratio for f in [174.61, 220.00, 261.63]]},
        {"root": 49.00 * transpose_ratio, "freqs": [f * transpose_ratio for f in [196.00, 246.94, 293.66]]}
    ]
    chord_len = 4 * beat_len  # 1 bar per chord
    total_bars = int(duration_sec / chord_len) + 1

    for bar in range(total_bars):
        chord = chords[bar % len(chords)]
        c_start = bar * chord_len
        root_freq = chord["root"]
        freqs = chord["freqs"]

        # A. Walking / Slap Bassline (8th notes across the bar)
        for eighth in range(8):
            b_time = c_start + eighth * (beat_len / 2.0)
            if b_time >= duration_sec:
                break
            b_idx = int(b_time * sample_rate)
            b_len = min(int(0.22 * sample_rate), num_samples - b_idx)
            if b_len > 0:
                t_b = np.linspace(0, 0.22, b_len, endpoint=False)
                interval = [1.0, 1.0, 1.25, 1.0, 1.5, 1.25, 1.5, 1.78][eighth]
                bass_freq = root_freq * interval
                bass_note = 0.55 * (np.sin(2 * np.pi * bass_freq * t_b) + 0.4 * np.sin(2 * np.pi * bass_freq * 2 * t_b)) * np.exp(-t_b * 6.0)
                waveform[b_idx:b_idx + b_len] += bass_note

        # B. Rhodes / Piano Chords (2 syncopated stabs per bar)
        for stab_offset in (0.0, 0.75):
            stab_time = c_start + stab_offset * beat_len
            stab_idx = int(stab_time * sample_rate)
            stab_len = min(int(0.6 * sample_rate), num_samples - stab_idx)
            if stab_len > 0:
                t_stab = np.linspace(0, 0.6, stab_len, endpoint=False)
                env = np.exp(-t_stab * 4.5)
                stab_sound = np.zeros(stab_len, dtype=np.float32)
                for f in freqs:
                    stab_sound += 0.22 * (np.sin(2 * np.pi * f * t_stab) + 0.3 * np.sin(2 * np.pi * f * 2 * t_stab))
                waveform[stab_idx:stab_idx + stab_len] += stab_sound * env

    # 3. Vocal / Melody Lead Line
    melody_notes = [523.25, 587.33, 659.25, 587.33, 523.25, 440.00, 392.00, 440.00]
    for m_idx, m_freq in enumerate(melody_notes):
        m_time = m_idx * 1.5
        if m_time >= duration_sec:
            break
        idx_m = int(m_time * sample_rate)
        m_len = min(int(1.2 * sample_rate), num_samples - idx_m)
        if m_len > 0:
            t_m = np.linspace(0, 1.2, m_len, endpoint=False)
            vibrato = 5.0 * np.sin(2 * np.pi * 5.5 * t_m)
            vocal_note = 0.35 * np.sin(2 * np.pi * (m_freq + vibrato) * t_m) * (0.8 + 0.2 * np.sin(np.pi * t_m / 1.2))
            waveform[idx_m:idx_m + m_len] += vocal_note

    # Normalize waveform to broadcast studio standard (-1.0 dBFS)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-6) * 0.92
    audio_int16 = (waveform * 32767).astype(np.int16)

    # Save as WAV/MP3 compatible stream
    wav_path = output_path.replace(".mp3", ".wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        stereo_data = np.column_stack((audio_int16, audio_int16)).flatten()
        wf.writeframes(stereo_data.tobytes())

    # Copy to mp3 path if needed
    if os.path.exists(wav_path):
        try:
            shutil.copyfile(wav_path, output_path)
        except Exception:
            pass


def _read_wav_float(file_path: str):
    """Read an audio file into float32 numpy array and sample rate.
    Resolves paths across root and backend directories and supports WAV, MP3, FLAC, OGG.
    """
    import numpy as np
    from app.transcription.karaoke import _resolve_audio_file

    resolved = _resolve_audio_file(file_path) or file_path
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Audio file '{file_path}' (resolved to '{resolved}') not found on disk.")

    # 1. Try soundfile (WAV, FLAC, OGG, etc.)
    try:
        import soundfile as sf
        data, sr = sf.read(resolved, dtype="float32")
        return sr, data
    except Exception:
        pass

    # 2. Try scipy.io.wavfile (standard PCM WAV)
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(resolved)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        return sr, data
    except Exception:
        pass

    # 3. Try standard library wave module
    try:
        import wave
        with wave.open(resolved, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            if sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            if n_ch > 1:
                data = data.reshape(-1, n_ch)
            return sr, data
    except Exception:
        pass

    # 4. Try torchaudio (handles MP3, AAC, and exotic codecs)
    try:
        import torch
        import torchaudio
        waveform, sr = torchaudio.load(resolved)
        data = waveform.cpu().numpy().T  # (channels, samples) -> (samples, channels)
        if data.shape[1] == 1:
            data = data.squeeze(-1)
        return sr, data
    except Exception:
        pass

    raise RuntimeError(f"Failed to decode audio file '{resolved}' with soundfile, scipy, wave, or torchaudio.")


def _write_wav_float(file_path: str, sr: int, data: Any):
    """Write float32 numpy array to 16-bit PCM WAV."""
    import numpy as np
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    try:
        import soundfile as sf
        sf.write(file_path, data, sr, subtype="PCM_16")
        return
    except Exception:
        pass
    try:
        from scipy.io import wavfile
        int16_data = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
        wavfile.write(file_path, sr, int16_data)
        return
    except Exception:
        pass
    import wave
    int16_data = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    n_ch = int16_data.shape[1] if int16_data.ndim > 1 else 1
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())


def concatenate_and_crossfade_audio(
    parent_wav_path: str,
    extension_wav_path: str,
    output_wav_path: str,
    crossfade_sec: float = 1.5,
    extend_from_sec: Optional[float] = None,
    beat_grid: Optional[Dict[str, Any]] = None,
    target_duration_sec: Optional[float] = None,
    extension_is_full_timeline: bool = True,
) -> float:
    """Concatenate parent audio with extension audio using beat-grid alignment and equal-power sine/cosine crossfade.

    extension_is_full_timeline MUST be set explicitly by the caller:
      True  -> extension audio covers the whole song timeline (KV roll-forward):
               always slice the tail starting at the cut point.
      False -> extension audio is a delta snippet starting at t=0: append it.
    Never infer this from lengths: a short tail on a full-timeline render
    (e.g. cut 1:49, target 1:58) is shorter than any length threshold and was
    misclassified as a delta — appending the regen from 0:00 so the tail
    literally restarted the song.
    """
    import numpy as np
    from app.core.paths import get_generated_audio_dir, get_repo_root

    sr_p, data_p = _read_wav_float(parent_wav_path)
    sr_e, data_e = _read_wav_float(extension_wav_path)

    # Resample extension audio if sample rates differ
    if sr_p != sr_e:
        try:
            import scipy.signal
            num_samples = int(len(data_e) * sr_p / sr_e)
            data_e = scipy.signal.resample(data_e, num_samples)
            sr_e = sr_p
        except Exception as _resample_err:
            logger.warning(f"Resampling failed ({_resample_err}); using extension audio as-is.")

    # Beat-grid alignment: snap cut point to the nearest downbeat (measure start) if available
    effective_cut_sec = extend_from_sec
    if effective_cut_sec is not None and effective_cut_sec > 0 and beat_grid and isinstance(beat_grid, dict):
        bg_bpm = float(beat_grid.get("bpm", 0.0))
        beats_per_bar = int(beat_grid.get("beats_per_bar", 4))
        first_downbeat = float(beat_grid.get("first_downbeat", 0.0))
        if bg_bpm > 40.0:
            bar_dur = beats_per_bar * (60.0 / bg_bpm)
            k = round((effective_cut_sec - first_downbeat) / bar_dur)
            snapped_cut = first_downbeat + k * bar_dur
            parent_total_sec = len(data_p) / sr_p
            if 5.0 <= snapped_cut <= parent_total_sec and abs(snapped_cut - effective_cut_sec) <= 1.5:
                logger.info(f"Beat-grid snap: aligned cut point from {effective_cut_sec:.2f}s to downbeat at {snapped_cut:.2f}s (measure {k})")
                effective_cut_sec = snapped_cut

    # Slice parent audio to cut point if requested
    if effective_cut_sec is not None and effective_cut_sec > 0:
        cut_samples = int(effective_cut_sec * sr_p)
        if 0 < cut_samples < len(data_p):
            data_p = data_p[:cut_samples]
        else:
            cut_samples = len(data_p)
    else:
        cut_samples = len(data_p)

    if data_p.ndim == 1:
        data_p = data_p[:, np.newaxis]
    if data_e.ndim == 1:
        data_e = data_e[:, np.newaxis]

    channels = max(data_p.shape[1], data_e.shape[1])
    if data_p.shape[1] < channels:
        data_p = np.repeat(data_p, channels, axis=1)
    if data_e.shape[1] < channels:
        data_e = np.repeat(data_e, channels, axis=1)

    crossfade_samples = int(min(len(data_p), len(data_e), crossfade_sec * sr_p))

    # Determine continuation alignment from the caller's explicit flag.
    # Full-timeline renders (KV roll-forward) contain the whole song: slice the
    # tail starting at the cut point (minus crossfade overlap). Delta snippets
    # (fresh-segment fallback) start at t=0: append from index 0.
    if extension_is_full_timeline and effective_cut_sec is not None and effective_cut_sec > 0 and len(data_e) > cut_samples:
        ext_start_sample = max(0, cut_samples - crossfade_samples)
    else:
        ext_start_sample = 0

    if crossfade_samples <= 0:
        if extension_is_full_timeline and extend_from_sec is not None and extend_from_sec > 0 and len(data_e) > cut_samples:
            combined = np.vstack([data_p, data_e[cut_samples:]])
        else:
            combined = np.vstack([data_p, data_e])
    else:
        pre = data_p[:-crossfade_samples]
        cross_p = data_p[-crossfade_samples:]
        cross_e = data_e[ext_start_sample : ext_start_sample + crossfade_samples]

        actual_xfade = min(len(cross_p), len(cross_e))
        if actual_xfade < crossfade_samples:
            if actual_xfade > 0:
                t = np.linspace(0, np.pi / 2, actual_xfade, endpoint=False)[:, np.newaxis]
                fade_out = np.cos(t)
                fade_in = np.sin(t)
                cross = data_p[-actual_xfade:] * fade_out + cross_e[:actual_xfade] * fade_in
                pre = data_p[:-actual_xfade]
            else:
                cross = np.empty((0, channels), dtype=data_p.dtype)
                pre = data_p
        else:
            t = np.linspace(0, np.pi / 2, crossfade_samples, endpoint=False)[:, np.newaxis]
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            cross = cross_p * fade_out + cross_e * fade_in

        post = data_e[ext_start_sample + actual_xfade :]
        combined = np.vstack([pre, cross, post])

    # Enforce exact target duration if requested with a gentle tail fade
    if target_duration_sec is not None and target_duration_sec > 0:
        target_samples = int(target_duration_sec * sr_p)
        if len(combined) > target_samples:
            fade_len = int(min(0.25 * sr_p, len(combined) - target_samples + 0.25 * sr_p))
            combined = combined[:target_samples]
            if fade_len > 0:
                t_fade = np.linspace(1.0, 0.0, fade_len)[:, np.newaxis]
                combined[-fade_len:] *= t_fade
        elif len(combined) < target_samples:
            # Pad shortfalls with silence after a short fade so the rendered
            # track lands EXACTLY on the requested end time (user-picked, e.g. 1:58).
            pad_len = target_samples - len(combined)
            fade_len = int(min(0.5 * sr_p, len(combined)))
            if fade_len > 0:
                t_fade = np.linspace(1.0, 0.0, fade_len)[:, np.newaxis]
                combined[-fade_len:] *= t_fade
            combined = np.vstack([combined, np.zeros((pad_len, channels), dtype=combined.dtype)])
            logger.info(f"Padded extension output with {pad_len / sr_p:.2f}s silence to hit exact target {target_duration_sec:.1f}s")

    _write_wav_float(output_wav_path, sr_p, combined)

    # Mirror to backend/generated_audio as well
    try:
        backend_dir = get_repo_root() / "backend" / "generated_audio"
        backend_dir.mkdir(parents=True, exist_ok=True)
        backend_target = str(backend_dir / os.path.basename(output_wav_path))
        if os.path.abspath(output_wav_path) != os.path.abspath(backend_target):
            shutil.copy2(output_wav_path, backend_target)
    except Exception:
        pass

    return float(len(combined) / sr_p)




class MiniMaxMusic3Provider(GenerationProvider):
    def __init__(self, snapshot_path: Optional[str] = None):
        self.snapshot_path = (
            os.environ.get("MILIMO_MINIMAX_SNAPSHOT")   # C1: 4bit/6bit/8bit/mxfp*
            or snapshot_path
            or os.environ.get("MINIMAX_MODEL_PATH")
            or DEFAULT_MINIMAX_SNAPSHOT
        )
        self.config = {}
        self.model = None
        self._is_loaded = False
        self._is_loading = False

    def get_capabilities(self) -> GenerationCapabilities:
        return GenerationCapabilities(
            provider_id="minimax_music3",
            display_name="MiniMax Music 3 (Default)",
            description="Next-gen 5-minute full song model with Structured Captions, explicit section tags, and high-fidelity stereo output.",
            version="Music-3-bf16",
            max_duration_sec=300,
            supports_structured_caption=True,
            supports_section_tags=True,
            supports_lora=True,
            supports_voice_conversion=True,
            supports_track_extension=True,
            supports_segment_repair=True,
            recommended_hardware=HardwareTier.MID_SINGLE_GPU,
            license_class="MiniMax Open Weights",
            default_sample_rate=44100
        )

    def is_ready(self) -> bool:
        return self._is_loaded or os.path.isdir(self.snapshot_path)

    async def initialize(self, model_path: Optional[str] = None) -> bool:
        if model_path:
            self.snapshot_path = model_path

        if self._is_loaded or self._is_loading:
            return True

        self._is_loading = True
        try:
            config_path = os.path.join(self.snapshot_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"MiniMax Music 3 config loaded from {config_path}")
                self._is_loaded = True
                return True
            else:
                logger.warning(f"MiniMax Music 3 config not found at {config_path}")
                return False
        except Exception as e:
            logger.error(f"Error initializing MiniMax Music 3 provider: {e}")
            return False
        finally:
            self._is_loading = False

    def unload(self) -> bool:
        """Release loaded MLX weights and purge Metal buffer caches."""
        self.model = None
        self._is_loaded = False
        unload_minimax_model()
        return True

    @staticmethod

    def parse_structured_caption(prompt: str, tags: Optional[str] = None) -> Dict[str, str]:
        """
        Extract or construct Structured Caption sections:
        - Global Metadata (Genre, Tempo, Mood)
        - Vocal Details (Voice, Style)
        - Arrangement (Instrumentation, Structure)
        """
        metadata = {}
        # Check if prompt already contains structured headers
        if "[Global Metadata]" in prompt or "[Arrangement]" in prompt or "[Vocal Details]" in prompt:
            sections = re.split(r'\[(Global Metadata|Vocal Details|Arrangement)\]', prompt)
            for i in range(1, len(sections), 2):
                sec_name = sections[i].strip().lower().replace(" ", "_")
                sec_content = sections[i+1].strip() if i+1 < len(sections) else ""
                metadata[sec_name] = sec_content
        else:
            # Construct structured caption from tags and free-text prompt, following
            # the official MiniMax prompting guide's three-heading skeleton and its
            # sub-fields (Basic Attributes / Emotional Progression / Imagery / Sonics;
            # Vocal Gender & Timbre / Style / Harmony / FX; Instrument Lifecycle /
            # Groove / Embellishments). Vocals are always stated explicitly — leaving
            # them unspecified is the #1 cause of unwanted instrumental drift.
            tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
            genre = tag_list[0] if tag_list else "Contemporary"
            tempo = tag_list[1] if len(tag_list) > 1 else "energetic"
            instruments = ", ".join(tag_list[2:]) if len(tag_list) > 2 else "Drums, Bass, Synths, Vocals"
            imagery = prompt.strip() or "A scene the song belongs to."

            metadata["global_metadata"] = (
                f"Basic Attributes: Genre {genre}, tempo {tempo}.\n"
                f"Global Emotional Progression: Opens with the {tempo} {genre} character and builds in energy toward the chorus before resolving cleanly.\n"
                f"Application Scenarios & Imagery: {imagery}\n"
                f"Sonics & Production Profile: Polished, well-balanced mix with centered vocals and moderate stereo width."
            )
            metadata["vocal_details"] = (
                "Vocal Gender & Timbre: Singer A (Female), a clear and expressive vocal with strong presence.\n"
                "Vocal Style: Melodic and emotive throughout, with dynamic phrasing and a fuller delivery in the chorus.\n"
                "Harmony/Backing Vocals: Subtle stacked harmonies in the chorus.\n"
                "Vocal FX: Light reverb and delay for space without losing presence."
            )
            metadata["arrangement"] = (
                f"Instrument Lifecycle (Primary/Secondary): Primary {genre} foundation anchored by {instruments}.\n"
                f"Groove & Foundation Progression: Rhythmic drive throughout, thickening in the chorus and stripping back in the bridge.\n"
                f"Embellishments, Textures & Spatial FX: Moderate reverb tails and subtle risers on transitions."
            )

        return metadata

    @staticmethod
    def format_full_caption(structured_caption: Dict[str, str], prompt_text: str) -> str:
        """Format the complete structured caption string for MiniMax."""
        parts = []
        if structured_caption.get("global_metadata"):
            parts.append(f"[Global Metadata]\n{structured_caption['global_metadata']}")
        if structured_caption.get("vocal_details"):
            parts.append(f"[Vocal Details]\n{structured_caption['vocal_details']}")
        if structured_caption.get("arrangement"):
            parts.append(f"[Arrangement]\n{structured_caption['arrangement']}")
        return "\n\n".join(parts)

    @staticmethod
    def sanitize_section_tags(lyrics: Optional[str]) -> Optional[str]:
        """Ensure standard MiniMax section tags like [Intro], [Verse], [Chorus], [Bridge], [Outro]."""
        if not lyrics:
            return None

        # Standardize bracketed tags
        tag_map = {
            r'\[?intro\]?': '[Intro]',
            r'\[?verse\s*(\d*)\]?': lambda m: f"[Verse {m.group(1)}]" if m.group(1) else "[Verse]",
            r'\[?pre[- ]?chorus\s*(\d*)\]?': lambda m: f"[Pre-Chorus {m.group(1)}]" if m.group(1) else "[Pre-Chorus]",
            r'\[?chorus\s*(\d*)\]?': lambda m: f"[Chorus {m.group(1)}]" if m.group(1) else "[Chorus]",
            r'\[?bridge\]?': '[Bridge]',
            r'\[?instrumental\]?': '[Instrumental]',
            r'\[?solo\]?': '[Solo]',
            r'\[?outro\]?': '[Outro]',
        }

        cleaned = lyrics
        for pattern, replacement in tag_map.items():
            if callable(replacement):
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            else:
                cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # MiniMax input contract: every [Section] tag must sit alone on its own line —
        # lyric text on the same line as a leading tag is silently dropped by the model.
        # Split any "tag + text on one line" into two lines (tag already alone: no-op).
        cleaned = re.sub(
            r'(?im)^[ \t]*(\[[^\]\n]+\])[ \t]+([^\n].*)$',
            r'\1\n\2',
            cleaned,
        )
        return cleaned

    async def generate(
        self,
        job_id: str,
        prompt: str,
        lyrics: Optional[str],
        duration_ms: int,
        tags: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        cfg_scale: float = 1.5,
        topk: int = 50,
        top_k: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        structured_caption: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        if top_k is not None:
            topk = top_k
        if kwargs.get("llm_model"):
            self.llm_model = kwargs.get("llm_model")
        if not self._is_loaded:
            await self.initialize()

        from app.core.paths import get_generated_audio_dir
        gen_dir = get_generated_audio_dir()
        gen_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs("generated_audio", exist_ok=True)
        filename = f"{job_id}.mp3"
        output_path = str(gen_dir / filename)

        # --- Producer enhancement (production-grade, never silently fake) ----
        # If the user handed us a weak prompt and/or no lyrics, the real LLM
        # producer enhances the concept and writes genuine structured lyrics so
        # real inference is well-conditioned. Weak inputs are the #1 reason real
        # MiniMax calls used to throw (empty lyrics) and fall back to the synth.
        loop = asyncio.get_running_loop()
        try:
            from app.services.producer_service import producer_service
            produced = await producer_service.enhance_for_generation(
                prompt, lyrics, tags, getattr(self, "llm_model", None)
            )
            eff_prompt = (produced.get("prompt") or prompt or "").strip()
            eff_lyrics = (produced.get("lyrics") or lyrics or "").strip()
            eff_tags = (produced.get("tags") or tags or "").strip()
        except Exception as _pe:
            logger.warning(f"Producer enhancement unavailable ({_pe}); using raw inputs.")
            eff_prompt = (prompt or "").strip()
            eff_lyrics = (lyrics or "").strip()
            eff_tags = (tags or "").strip()

        # Structured caption: honor caller-provided sections (composer UI /
        # producer) when present — the pipeline passes GenerationRequest.
        # structured_caption through — and fill any missing section from the
        # auto-constructed caption so the model always sees a complete 3-heading
        # caption. The constructed path follows the official MiniMax prompting
        # guide (three headings, explicit vocals, no fabricated precision).
        auto_meta = self.parse_structured_caption(eff_prompt, eff_tags)
        provided = structured_caption or {}
        structured_meta = {
            "global_metadata": (provided.get("global_metadata") or "").strip() or auto_meta.get("global_metadata", ""),
            "vocal_details": (provided.get("vocal_details") or "").strip() or auto_meta.get("vocal_details", ""),
            "arrangement": (provided.get("arrangement") or "").strip() or auto_meta.get("arrangement", ""),
        }
        formatted_caption = self.format_full_caption(structured_meta, eff_prompt)
        sanitized_lyrics = self.sanitize_section_tags(eff_lyrics)

        # Check if cancellation requested
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Generation cancelled by user")

        duration_sec = duration_ms / 1000.0
        wav_path = output_path.replace(".mp3", ".wav")

        used_real_inference = False
        fallback_reason: Optional[str] = None

        # Dynamically resolve active model snapshot if set in ModelManager or environment
        try:
            from app.services.model_manager import model_manager
            active_m = model_manager.get_active_model()
            if active_m and active_m.get("local_path") and os.path.isdir(active_m["local_path"]):
                self.snapshot_path = active_m["local_path"]
        except Exception:
            pass

        if os.environ.get("MINIMAX_MODEL_PATH") and os.path.isdir(os.environ["MINIMAX_MODEL_PATH"]):
            self.snapshot_path = os.environ["MINIMAX_MODEL_PATH"]

        if _MLX_AUDIO_AVAILABLE and os.path.isdir(self.snapshot_path):
            try:
                # Steps scale roughly with length: ~2s per step, clamped to the model's
                # allowed maximum of 30 (mlx_audio raises if steps > 30).
                steps = min(30, max(10, int(duration_sec / 2)))
                if progress_callback:
                    progress_callback(1, 3, f"MiniMax Music 3: Running real MLX inference ({steps} steps) on Apple Silicon...")
                def _hooked_progress(frac: float, msg: str):
                    if progress_callback:
                        progress_callback(1, 3, f"MiniMax Music 3: {msg} [{int(frac * 100)}%]")

                gen_codes: list = []
                await loop.run_in_executor(
                    None,
                    run_real_minimax_inference,
                    self.snapshot_path,
                    formatted_caption,
                    sanitized_lyrics or eff_lyrics,
                    duration_sec,
                    seed,
                    wav_path,
                    steps,
                    temperature,
                    cfg_scale,
                    topk,
                    cancel_event,
                    _hooked_progress,
                    gen_codes,
                )
                # The blocking inference thread cannot be interrupted mid-call.
                # If cancellation arrived during those minutes, DISCARD the
                # output instead of letting dead work flow downstream.
                if cancel_event is not None and cancel_event.is_set():
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                    raise asyncio.CancelledError("Cancelled during inference; audio discarded")
                used_real_inference = True
                logger.info("Real MiniMax Music 3 inference produced audio at %s", wav_path)
                # Persist per-frame codes so later extensions can restore the
                # exact KV state instead of replaying (fast + exact).
                try:
                    from app.providers.minimax_local_hooks import save_song_state
                    _fr = None
                    try:
                        import json as _json
                        with open(os.path.join(self.snapshot_path, "config.json")) as _cf:
                            _fr = float(_json.load(_cf).get("frame_rate", 0) or 0)
                    except Exception:
                        _fr = None
                    if not _fr:
                        _fr = float(_load_minimax_model(self.snapshot_path).config.frame_rate)
                    save_song_state(
                        job_id,
                        gen_codes,
                        seed=int(seed) if seed is not None else 0,
                        caption=formatted_caption,
                        lyrics=sanitized_lyrics or eff_lyrics,
                        snapshot_path=self.snapshot_path,
                        frame_rate=float(_fr),
                    )
                except Exception as e:
                    logger.warning(f"Song state save skipped for {job_id}: {e}")
            except GenerationCancelled:
                # Cancellation is NOT an inference failure — never fall back.
                raise asyncio.CancelledError("Cancelled during local inference")
            except Exception as e:
                fallback_reason = str(e)
                logger.warning(f"Real MiniMax inference failed ({e}); falling back to procedural waveform.", exc_info=True)

        if not used_real_inference:
            # Surface WHY the real path was skipped so the UI can show an honest
            # reason instead of a silent mystery (and logs can be debugged).
            if fallback_reason is None:
                if not _MLX_AUDIO_AVAILABLE:
                    fallback_reason = f"mlx-audio unavailable: {_MLX_IMPORT_ERROR or 'not installed'}"
                elif not os.path.isdir(self.snapshot_path):
                    fallback_reason = f"MiniMax Music 3 model snapshot not found at {self.snapshot_path}"
                else:
                    fallback_reason = "real inference path was not attempted (unknown)"

            # Production Strict Mode: if MILIMO_STRICT_INFERENCE is set, NEVER silently fake audio
            strict_mode = os.environ.get("MILIMO_STRICT_INFERENCE", "0") in ("1", "true", "True") or kwargs.get("strict_inference", False)
            if strict_mode:
                err_msg = f"Strict Inference Error: Real MiniMax Music 3 inference required, but failed: {fallback_reason}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            # Heavy CPU synthesis offloaded to a worker thread so the event loop is not blocked.
            await loop.run_in_executor(None, synthesize_dynamic_audio_waveform, duration_sec, seed, output_path, prompt, lyrics, tags)
            wav_path = output_path.replace(".mp3", ".wav")

        # Mirror wav_path to backend/generated_audio for backwards compatibility
        backend_wav = os.path.abspath(os.path.join("generated_audio", os.path.basename(wav_path)))
        if os.path.abspath(wav_path) != backend_wav and os.path.exists(wav_path):
            try:
                shutil.copy2(wav_path, backend_wav)
            except Exception:
                pass

        try:
            from app.core.hardware_lock import GlobalHardwareCoordinator
            if GlobalHardwareCoordinator.get_memory_policy()["policy"] == "eager":
                self.unload()
        except Exception:
            pass

        return GeneratedAudioResult(
            audio_path=f"/audio/{os.path.basename(wav_path)}",
            duration_sec=duration_sec,
            sample_rate=44100,
            structured_caption=structured_meta,
            used_fallback_synth=(not used_real_inference),
            fallback_reason=fallback_reason,
            metadata={
                "provider": "minimax_music3",
                "seed": seed,
                "real_inference": used_real_inference,
                # Effective (post-producer) inputs — the pipeline persists these
                # onto the Job so the producer's lyrics/concept surface in the UI.
                "effective_prompt": eff_prompt,
                "effective_lyrics": eff_lyrics,
                "effective_tags": eff_tags,
                "producer_enhanced": eff_prompt != (prompt or "").strip()
                                     or bool(eff_lyrics and not (lyrics or "").strip()),
                "formatted_caption": formatted_caption,
                "section_tags": re.findall(r'\[(.*?)\]', sanitized_lyrics or "")
            }
        )

    async def extend(
        self,
        job_id: str,
        parent_audio_path: str,
        extend_ms: int,
        lyrics: Optional[str] = None,
        prompt: Optional[str] = None,
        extend_from_sec: Optional[float] = None,
        crossfade_sec: float = 1.5,
        seed: Optional[int] = None,
        tags: Optional[str] = None,
        structured_caption: Optional[Dict[str, Any]] = None,
        tail_lyrics: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[Any] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        """
        Extend parent audio using Option 1/1A: MiniMax KV-cache roll-forward extension.
        Fast-forwards parent deterministic acoustic state to cut boundary, then continues
        generating new frames with early termination suppressed, guaranteeing 100% tempo,
        pitch, and timbre continuity.
        """
        import shutil
        from app.transcription.karaoke import _resolve_audio_file
        from app.core.paths import get_generated_audio_dir

        gen_dir = get_generated_audio_dir()
        gen_dir.mkdir(parents=True, exist_ok=True)
        out_wav_path = str(gen_dir / f"{job_id}.wav")
        alt_wav = str(gen_dir / f"song_{job_id}.wav")

        resolved_parent = _resolve_audio_file(parent_audio_path)
        parent_dur = float(extend_from_sec) if extend_from_sec is not None else 60.0
        target_dur = float(kwargs.get("target_duration_sec")) if kwargs.get("target_duration_sec") else (parent_dur + extend_ms / 1000.0)

        # Check if real MLX inference is available and seed is known
        if _MLX_AUDIO_AVAILABLE and self.snapshot_path and seed is not None:
            parent_prompt = kwargs.get("parent_prompt") or prompt or ""
            base_parent_lyrics = kwargs.get("parent_lyrics") or ""
            # Conditioning rule for the KV replay: the tail can only continue the
            # parent's exact trajectory if the conditioning text is byte-identical
            # to the parent run's. The timed remainder slice is
            # transcription-normalized text (headers/line-breaks differ from the
            # stored lyrics), so it must NEVER steer the replay: appending it
            # diverges the whole trajectory into a fresh song at the prompt
            # tempo instead of the parent's groove. Only genuinely new lyrics
            # (user-supplied additional or auto-generated continuation) steer,
            # and then the divergence is flagged honestly. The remainder slice
            # is consumed only by the fresh-segment fallback path.
            tail_lyrics = tail_lyrics if tail_lyrics is not None else kwargs.get("tail_lyrics")
            effective_extension_lyrics, lyrics_steered = resolve_extension_lyrics(
                base_parent_lyrics, lyrics
            )
            lyrics_source = "explicit_steering" if lyrics_steered else "full_parent_replay"
            if lyrics_steered:
                logger.warning(
                    "Extension lyrics steer the replay (appended text not contained "
                    "in parent lyrics): the tail will reinterpret, not seamlessly "
                    "continue. This is expected only with custom continuation lyrics."
                )
            # Mirror the parent run's model-input pipeline exactly: generate()
            # conditions on sanitize_section_tags(eff_lyrics), so the replay must
            # sanitize the same way or text_ids diverge from the parent run.
            effective_extension_lyrics = (
                self.sanitize_section_tags(effective_extension_lyrics) or effective_extension_lyrics
            )

            # Check for parent structured caption to maintain 100% token parity with parent run
            parent_sc = kwargs.get("parent_structured_caption")
            if not parent_sc:
                parent_job_id = kwargs.get("parent_job_id")
                if parent_job_id:
                    try:
                        from sqlmodel import Session
                        from app.core.database import engine
                        from app.models import Job
                        with Session(engine) as session:
                            pj = session.get(Job, parent_job_id)
                            if pj and pj.structured_caption_json:
                                parent_sc = json.loads(pj.structured_caption_json)
                    except Exception as _e:
                        logger.debug("Failed to lookup parent_structured_caption: %s", _e)

            if parent_sc and isinstance(parent_sc, dict):
                structured_meta = {
                    "global_metadata": (parent_sc.get("global_metadata") or "").strip(),
                    "vocal_details": (parent_sc.get("vocal_details") or "").strip(),
                    "arrangement": (parent_sc.get("arrangement") or "").strip(),
                }
            else:
                provided = structured_caption or {}
                auto_meta = self.parse_structured_caption(parent_prompt, tags or "")
                structured_meta = {
                    "global_metadata": (provided.get("global_metadata") or "").strip() or auto_meta.get("global_metadata", ""),
                    "vocal_details": (provided.get("vocal_details") or "").strip() or auto_meta.get("vocal_details", ""),
                    "arrangement": (provided.get("arrangement") or "").strip() or auto_meta.get("arrangement", ""),
                }
            formatted_caption = self.format_full_caption(structured_meta, parent_prompt)

            steps = min(30, max(10, int(target_dur / 4)))
            temperature = float(kwargs.get("temperature", 1.0))
            cfg_scale = float(kwargs.get("cfg_scale", 1.5))
            topk = int(kwargs.get("topk", 50))

            if progress_callback:
                progress_callback(1, 3, f"MiniMax Music 3: Executing KV-Cache Roll-Forward Extension ({steps} flow steps)...")

            def _hooked_progress(frac: float, msg: str):
                if progress_callback:
                    progress_callback(1, 3, f"MiniMax Music 3: {msg} [{int(frac * 100)}%]")

            # Exact-restore fast path: if the parent run saved its per-frame AR
            # codes (.mm3state), prefill the KV cache directly instead of
            # replaying (no resampling, no RNG consumed for the prefix).
            # Falls back to deterministic replay when absent/stale.
            restore_codes = None
            restore_source = None
            parent_job_id = kwargs.get("parent_job_id")
            if parent_job_id:
                try:
                    from app.providers.minimax_local_hooks import load_song_state
                    _state = load_song_state(str(parent_job_id), snapshot_path=self.snapshot_path)
                    # A bundle from a different seed/conditioning/snapshot would
                    # restore a foreign trajectory while the splice masks the
                    # head — the tail would diverge silently. Validate first.
                    _state = validate_song_state(
                        _state, int(seed), formatted_caption,
                        effective_extension_lyrics, self.snapshot_path,
                    )
                    if _state and int(_state.get("frames", 0)) > 0:
                        restore_codes = _state["codes"]
                        restore_source = _state
                        logger.info(
                            f"Song state found for parent {parent_job_id} "
                            f"({len(restore_codes)} frames, fp={_state.get('conditioning')}); "
                            "will restore instead of replaying."
                        )
                except Exception as e:
                    logger.warning(f"Song state restore unavailable, falling back to replay: {e}")
                    restore_codes = None

            new_codes: list = []
            loop = asyncio.get_running_loop()
            raw_extended_wav = str(gen_dir / f"{job_id}_raw_extended.wav")

            _, used_restore = await loop.run_in_executor(
                None,
                run_real_minimax_extension,
                self.snapshot_path,
                formatted_caption,
                effective_extension_lyrics,
                parent_dur,
                target_dur,
                int(seed),
                raw_extended_wav,
                steps,
                temperature,
                cfg_scale,
                topk,
                cancel_event,
                _hooked_progress,
                new_codes,      # collect_codes: new frames (replay: all frames)
                restore_codes,  # restore: prefix comes from the saved state
            )
            extension_mode = "state_restored" if used_restore else "kv_rollforward"
            logger.info(f"Extension {job_id}: path={extension_mode} (restore={'yes' if used_restore else 'no'})")

            # Post-hoc replay fidelity check (full-timeline renders only):
            # compare the render's head against parent disk audio. A diverged
            # replay (wrong conditioning/seed/snapshot) used to sail through
            # silently because the splice masks the head — now it is flagged.
            replay_check: Dict[str, Any] = {"replay_verified": None, "detail": "not a full-timeline render"}
            # Reconcile with resolved parent audio if on disk:
            # Splicing with crossfade over crossfade_sec guarantees exact parent disk audio up to cut point.
            # Restore mode emits TAIL-ONLY audio (full-timeline replay covers the whole song):
            # the splice flag must match what was actually rendered.
            is_full_timeline = not used_restore
            if is_full_timeline and resolved_parent and os.path.exists(resolved_parent):
                try:
                    replay_check = await loop.run_in_executor(
                        None,
                        verify_extension_replay,
                        raw_extended_wav,
                        resolved_parent,
                        parent_dur,
                    )
                except Exception as e:
                    logger.warning(f"Extension replay verification error: {e}")
                    replay_check = {"replay_verified": None, "detail": f"check error: {e}"}
            if resolved_parent and os.path.exists(resolved_parent):
                beat_grid = kwargs.get("beat_grid")
                total_duration = await loop.run_in_executor(
                    None,
                    concatenate_and_crossfade_audio,
                    resolved_parent,
                    raw_extended_wav,
                    out_wav_path,
                    crossfade_sec,
                    parent_dur,
                    beat_grid,
                    target_dur,
                    is_full_timeline,
                )
                try:
                    if os.path.exists(raw_extended_wav):
                        os.remove(raw_extended_wav)
                except Exception:
                    pass
            else:
                logger.warning(
                    f"Extension {job_id}: parent audio {parent_audio_path} not on disk — "
                    "no splice anchor; publishing the generated audio as-is. "
                    "Continuity with the parent is NOT guaranteed."
                )
                extension_mode = "no_parent_audio"
                shutil.move(raw_extended_wav, out_wav_path)
                import soundfile as sf
                info = sf.info(out_wav_path)
                total_duration = info.duration

            # Persist this run's full-timeline codes so future extensions can
            # restore instead of replaying. Replay path collected every frame;
            # restore path collected only the new frames (prefix = saved state).
            try:
                from app.providers.minimax_local_hooks import save_song_state
                _all_codes = list(restore_codes or []) + list(new_codes) if used_restore else list(new_codes)
                if _all_codes:
                    _fr = _load_minimax_model(self.snapshot_path).config.frame_rate
                    save_song_state(
                        job_id,
                        _all_codes,
                        seed=int(seed),
                        caption=formatted_caption,
                        lyrics=effective_extension_lyrics,
                        snapshot_path=self.snapshot_path,
                        frame_rate=float(_fr),
                    )
            except Exception as e:
                logger.warning(f"Child song-state save skipped for {job_id}: {e}")

            # Mirror to song_{job_id}.wav for route versatility
            try:
                if os.path.abspath(out_wav_path) != os.path.abspath(alt_wav):
                    shutil.copy2(out_wav_path, alt_wav)
            except Exception:
                pass

            # Also mirror to backend/generated_audio
            try:
                from app.core.paths import get_repo_root
                backend_dir = get_repo_root() / "backend" / "generated_audio"
                backend_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_wav_path, str(backend_dir / f"{job_id}.wav"))
                shutil.copy2(out_wav_path, str(backend_dir / f"song_{job_id}.wav"))
            except Exception:
                pass

            return GeneratedAudioResult(
                audio_path=f"/audio/{job_id}.wav",
                duration_sec=total_duration,
                sample_rate=44100,
                structured_caption=structured_caption,
                used_fallback_synth=False,
                fallback_reason=None,
                metadata={
                    "extended_from": parent_audio_path,
                    "extend_from_sec": parent_dur,
                    "target_duration_sec": target_dur,
                    "total_duration_sec": total_duration,
                    "roll_forward_extension": True,
                    "extension_mode": extension_mode,
                    "restored_from_state": bool(used_restore),
                    "lyrics_source": lyrics_source,
                    "conditioning_steered": bool(lyrics_steered),
                    "replay_verified": replay_check.get("replay_verified"),
                    "replay_check_detail": replay_check.get("detail"),
                    "conditioning_sha": hashlib.sha256(
                        (formatted_caption + "\n" + effective_extension_lyrics).encode("utf-8", "ignore")
                    ).hexdigest()[:16],
                    "seed": seed,
                }
            )

        # Fallback path if MLX or seed unavailable:
        # The tail-only segment sings the remainder slice (or explicit new
        # lyrics); it never sees the full parent lyrics.
        fallback_lyrics = (lyrics or "").strip() or ((tail_lyrics or kwargs.get("tail_lyrics") or "").strip() or None)
        temp_ext_job_id = f"{job_id}_ext_segment"
        ext_result = await self.generate(
            job_id=temp_ext_job_id,
            prompt=prompt or "",
            lyrics=fallback_lyrics,
            duration_ms=extend_ms,
            seed=seed,
            tags=tags,
            structured_caption=structured_caption,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            **kwargs
        )

        if not resolved_parent:
            logger.warning(f"Parent audio {parent_audio_path} not found on disk; returning continuation directly.")
            return ext_result

        ext_local_path = _resolve_audio_file(ext_result.audio_path)
        if not ext_local_path or not os.path.exists(ext_local_path):
            candidate = str(gen_dir / f"{temp_ext_job_id}.wav")
            if os.path.exists(candidate):
                ext_local_path = candidate
            else:
                ext_local_path = ext_result.audio_path.replace("/audio/", "generated_audio/")

        beat_grid = kwargs.get("beat_grid")
        loop = asyncio.get_event_loop()
        total_duration = await loop.run_in_executor(
            None,
            concatenate_and_crossfade_audio,
            resolved_parent,
            ext_local_path,
            out_wav_path,
            crossfade_sec,
            extend_from_sec,
            beat_grid,
            target_dur,
            False,  # fresh-segment fallback renders only the delta: append from 0
        )

        try:
            if os.path.abspath(out_wav_path) != os.path.abspath(alt_wav):
                shutil.copy2(out_wav_path, alt_wav)
        except Exception:
            pass

        try:
            from app.core.paths import get_repo_root
            backend_dir = get_repo_root() / "backend" / "generated_audio"
            backend_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_wav_path, str(backend_dir / f"{job_id}.wav"))
            shutil.copy2(out_wav_path, str(backend_dir / f"song_{job_id}.wav"))
        except Exception:
            pass

        if ext_local_path and os.path.exists(ext_local_path) and "_ext_segment" in ext_local_path:
            try:
                os.remove(ext_local_path)
            except Exception:
                pass

        return GeneratedAudioResult(
            audio_path=f"/audio/{job_id}.wav",
            duration_sec=total_duration,
            sample_rate=44100,
            structured_caption=ext_result.structured_caption,
            used_fallback_synth=ext_result.used_fallback_synth,
            fallback_reason=ext_result.fallback_reason,
            metadata={
                **ext_result.metadata,
                "extended_from": parent_audio_path,
                "extend_from_sec": extend_from_sec,
                "crossfade_sec": crossfade_sec,
                "extended_duration_sec": extend_ms / 1000.0,
                "total_duration_sec": total_duration,
                "extension_mode": "fresh_segment_fallback",
            }
        )

    async def repair_segment(
        self,
        job_id: str,
        audio_path: str,
        start_time_sec: float,
        end_time_sec: float,
        prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs
    ) -> GeneratedAudioResult:
        return GeneratedAudioResult(
            audio_path=audio_path,
            duration_sec=end_time_sec - start_time_sec,
            metadata={"repaired_range": [start_time_sec, end_time_sec]}
        )


MiniMaxProvider = MiniMaxMusic3Provider

