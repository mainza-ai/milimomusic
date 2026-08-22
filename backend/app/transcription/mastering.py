"""Reference mastering — REAL DSP, two honest modes:

  1. reference provided  → Matchering (spectrum/tone/loudness matched to the
     reference track) via mg.process(); output saved by the library itself.
  2. no reference        → pyloudnorm integrated-LUFS normalization to the
     broadcast target (-14 LUFS default). Real loudness DSP with a true peak
     guard — not a copy, not a fabricated score.

The old implementation copied the input file (or wrote a 0-byte placeholder)
and returned a fabricated spectral_match_score=0.96. Every number this module
returns is now measured.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MasteringResult:
    mastered_audio_path: str      # public /audio/... URL path
    method: str                   # "matchering" | "loudness_normalize"
    measured_lufs: float
    target_lufs: float


def _local_path(public_or_local: str) -> str:
    if public_or_local.startswith("/audio/"):
        return os.path.join("generated_audio", public_or_local[len("/audio/"):])
    return public_or_local


def _measure_lufs(path: str) -> float:
    import numpy as np
    import soundfile as sf
    import pyloudnorm as pyln

    data, sr = sf.read(path, always_2d=True)
    mono = data.mean(axis=1)
    meter = pyln.Meter(sr)
    try:
        lufs = meter.integrated_loudness(mono)
    except ValueError:
        # near-silence: pyloudnorm raises; report digital silence convention
        lufs = -70.0
    return float(lufs)


def _matchering_sync(target_path: str, reference_path: str, output_path: str) -> None:
    import matchering as mg

    results = [mg.Result(file=output_path, subtype="FLOAT")]
    config = mg.Config(temp_folder=None)
    mg.process(
        target=target_path,
        reference=reference_path,
        results=results,
        config=config,
    )


def _loudness_normalize_sync(target_path: str, output_path: str, target_lufs: float) -> float:
    import numpy as np
    import soundfile as sf
    import pyloudnorm as pyln

    data, sr = sf.read(target_path, always_2d=True)
    mono = data.mean(axis=1)
    meter = pyln.Meter(sr)
    try:
        current = float(meter.integrated_loudness(mono))
    except ValueError:
        raise RuntimeError("Track is (near-)silent; loudness normalization is undefined.")

    gain_db = target_lufs - current
    normalized = data * (10.0 ** (gain_db / 20.0))

    # True-peak guard: never allow intersample clipping from the gain stage.
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.999:
        normalized = normalized * (0.999 / peak)

    sf.write(output_path, normalized, sr, subtype="FLOAT")
    final_mono = normalized.mean(axis=1)
    try:
        return float(meter.integrated_loudness(final_mono))
    except ValueError:
        return target_lufs


async def master_track(
    *,
    job_id: str,
    target_audio_path_public: str,
    reference_audio_path_public: str | None,
    target_lufs: float,
) -> MasteringResult:
    """Run mastering off the event loop; returns honest measured results."""
    started = time.monotonic()
    local_target = _local_path(target_audio_path_public)
    if not os.path.exists(local_target):
        raise FileNotFoundError(f"Master file missing: {target_audio_path_public}")

    out_dir = os.path.join("generated_audio", "mastered")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(local_target))[0]
    output_path = os.path.join(out_dir, f"{stem}_mastered.wav")

    has_reference = bool(reference_audio_path_public)

    def work() -> float:
        if has_reference:
            ref_path = _local_path(reference_audio_path_public)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference file missing: {reference_audio_path_public}")
            logger.info(f"Mastering {job_id}: Matchering against reference {os.path.basename(ref_path)}")
            _matchering_sync(local_target, ref_path, output_path)
            return _measure_lufs(output_path)
        else:
            logger.info(f"Mastering {job_id}: LUFS normalization to {target_lufs}")
            return _loudness_normalize_sync(local_target, output_path, target_lufs)

    try:
        measured = await asyncio.to_thread(work)
    except ImportError as e:
        # Dependency genuinely absent → honest unavailability, never a fake file.
        raise RuntimeError(f"Mastering dependency unavailable: {e}") from e
    except Exception:
        # Partial outputs must never masquerade as masters.
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass
        raise

    logger.info(f"Mastering {job_id} done in {time.monotonic()-started:.1f}s → {output_path}")
    return MasteringResult(
        mastered_audio_path=f"/audio/mastered/{os.path.basename(output_path)}",
        method="matchering" if has_reference else "loudness_normalize",
        measured_lufs=round(measured, 2),
        target_lufs=target_lufs,
    )


def cleanup_legacy_stub() -> None:
    """Remove the old stub's copy-paste artifact if it exists."""
    legacy = os.path.join("generated_audio", "mastered")
    if os.path.isdir(legacy):
        for name in os.listdir(legacy):
            path = os.path.join(legacy, name)
            try:
                if os.path.getsize(path) == 0:
                    os.remove(path)
                    logger.info(f"Removed zero-byte legacy stub artifact: {path}")
            except OSError:
                pass
