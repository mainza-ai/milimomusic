"""Candidate verification and reshoot (plan 8.1).

AutoMV's ablation (Nov 2026) shows the *verifier + reshoot* loop gives the largest
measured character-consistency gain of any technique they tried — larger than
adding an identity adapter — and it costs no model weights. Their pipeline renders
candidates, scores them, and re-rolls the losers.

This module provides the scoring. It deliberately uses dependency-free image
statistics (difference hash, palette distance, Laplacian sharpness) rather than a
face-embedding model: it must work on any machine with no downloads, and it must be
deterministic in tests. A learned identity score can be plugged in later behind the
same `score_candidate` signature.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("milimo.video.verifier")

#: A plate scoring below this against the reference is worth one re-roll.
DEFAULT_RESHoot_THRESHOLD = 0.55
#: Identity is the point of the exercise, so it has its own floor: a plate can look
#: sharp and colour-matched while depicting someone else.
IDENTITY_FLOOR = 0.70
MAX_CANDIDATES = 3


@dataclass
class CandidateScore:
    path: str
    identity: float     # perceptual-hash agreement with the reference, 0..1
    palette: float      # brightness/contrast agreement with the reference, 0..1
    sharpness: float    # Laplacian variance, normalised 0..1
    total: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _load_gray(path: str):
    """8-bit grayscale numpy array, or None when unreadable."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
    except Exception as exc:
        logger.debug("Could not read %s for verification: %s", path, exc)
        return None


def difference_hash(path: str, size: int = 8) -> Optional[int]:
    """64-bit difference hash (dHash) — stable under scale and mild grading."""
    gray = _load_gray(path)
    if gray is None:
        return None
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as img:
            small = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.float32)
    except Exception:
        return None
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (1 if small[row, col] > small[row, col + 1] else 0)
    return bits


def _hamming(a: int, b: int, bits: int = 64) -> int:
    return bin((a ^ b) & ((1 << bits) - 1)).count("1")


def _sharpness(path: str) -> float:
    """Normalised Laplacian variance; blurry plates should not win."""
    gray = _load_gray(path)
    if gray is None:
        return 0.0
    try:
        import numpy as np

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        h, w = gray.shape
        centre = gray[1:h - 1, 1:w - 1]
        laplacian = (
            kernel[1][0] * gray[1:h - 1, 0:w - 2]
            + kernel[0][1] * gray[0:h - 2, 1:w - 1]
            + kernel[1][1] * centre
            + kernel[2][1] * gray[2:h, 1:w - 1]
            + kernel[1][2] * gray[1:h - 1, 2:w]
        )
        variance = float(np.var(laplacian))
    except Exception:
        return 0.0
    # 800 is roughly "clearly textured"; clamp and normalise.
    return max(0.0, min(1.0, variance / 800.0))


def _palette(path: str) -> Tuple[float, float]:
    gray = _load_gray(path)
    if gray is None:
        return (0.0, 0.0)
    try:
        import numpy as np

        return (float(np.mean(gray)), float(np.std(gray)))
    except Exception:
        return (0.0, 0.0)


def score_candidate(
    candidate_path: str,
    reference_path: Optional[str] = None,
) -> CandidateScore:
    """Score one candidate plate.

    With a reference the score measures agreement (identity 55%, palette 25%,
    sharpness 20%). Without one, only composition quality is meaningful, so the
    score is the sharpness with a neutral identity/palette.
    """
    sharpness = _sharpness(candidate_path)
    if not reference_path or not os.path.isfile(reference_path):
        return CandidateScore(candidate_path, 1.0, 1.0, sharpness, sharpness)

    cand_hash = difference_hash(candidate_path)
    ref_hash = difference_hash(reference_path)
    if cand_hash is None or ref_hash is None:
        return CandidateScore(candidate_path, 0.0, 0.0, sharpness, sharpness * 0.2)

    identity = 1.0 - (_hamming(cand_hash, ref_hash) / 64.0)

    c_mean, c_std = _palette(candidate_path)
    r_mean, r_std = _palette(reference_path)
    palette = 1.0 - min(1.0, (abs(c_mean - r_mean) / 128.0 + abs(c_std - r_std) / 128.0) / 2.0)

    total = 0.55 * identity + 0.25 * palette + 0.20 * sharpness
    return CandidateScore(candidate_path, round(identity, 4), round(palette, 4),
                          round(sharpness, 4), round(total, 4))


def select_best(
    candidates: Sequence[str],
    reference_path: Optional[str] = None,
) -> Tuple[Optional[str], List[CandidateScore]]:
    """Return (best_path, scores) — the highest total, deterministic on ties."""
    scored = [score_candidate(path, reference_path) for path in candidates if path]
    if not scored:
        return None, []
    best = max(scored, key=lambda s: (s.total, s.path))
    return best.path, scored


def needs_reshoot(score: Optional[CandidateScore], threshold: float = DEFAULT_RESHoot_THRESHOLD) -> bool:
    """Whether a staged plate is worth one more diffusion attempt.

    Only when we have a reference to compare against: without one, "low score"
    just means "not very sharp", which is not a consistency problem.
    """
    if score is None:
        return False
    if score.identity >= 0.999 and score.palette >= 0.999:
        return False  # no reference was used
    return score.identity < IDENTITY_FLOOR or score.total < threshold
