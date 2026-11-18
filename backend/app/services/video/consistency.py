"""Subject consistency across scenes (audit finding E6).

Eight independently seeded scenes read as eight different people. Hosted tools solve
this with reference conditioning; the strongest open pipelines bound the drift with
two cheap mechanisms instead of a bigger model:

1. **Character bible** — one persisted physical description injected into every
   scene prompt, so the text conditioning describes the same person each time.
2. **Last-frame chaining** — the previous clip's final frame seeds the next scene's
   image-to-video pass, so lighting, grade and framing carry across a cut
   (the mechanism AutoMV's ablations credit for continuity).

Both are deterministic, need no extra weights, and are individually switchable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger("milimo.video.consistency")

CONSISTENCY_MODES = ("off", "character", "chain")
DEFAULT_MODE = "character"
#: How far chaining may reach. "section" keeps continuity inside a verse/chorus but
#: lets a cut into a new section reset the frame; "all" chains across everything.
CHAIN_SCOPES = ("section", "all")
DEFAULT_CHAIN_SCOPE = "section"


def normalise_mode(value: Optional[str]) -> str:
    mode = (value or DEFAULT_MODE).strip().lower()
    return mode if mode in CONSISTENCY_MODES else DEFAULT_MODE


def character_bible(config: Dict[str, Any], plan: Any = None) -> str:
    """Resolve the one description every scene must agree on.

    Explicit config wins; otherwise the director's persisted character profile;
    otherwise nothing (never an invented description).
    """
    explicit = str(config.get("character_desc") or config.get("characterPromptNote") or "").strip()
    if explicit:
        return explicit
    profile = getattr(plan, "character_profile", None) or ""
    profile = str(profile).strip()
    # The director's default filler is not a description worth propagating.
    if profile and profile.lower() not in ("lead performer", "the lead performer"):
        return profile
    return ""


def apply_character_bible(prompt: str, bible: str) -> str:
    """Append the bible to a prompt exactly once.

    Idempotent: re-rendering an already-stamped prompt must not stack the
    description N times.
    """
    text = (prompt or "").strip()
    if not bible:
        return text
    marker = f"consistent character: {bible}"
    if marker.lower() in text.lower():
        return text
    if not text:
        return f"{marker}."
    separator = "" if text.endswith((".", "!", "?")) else "."
    return f"{text}{separator} {marker}."


def extract_last_frame(clip_path: str, dest_path: str, *, timeout: int = 120) -> Optional[str]:
    """Write the final frame of `clip_path` as a PNG and return its path.

    `-sseof -0.1` seeks to 0.1 s before EOF, which is robust for clips whose
    duration is not an exact frame multiple. Returns None (with a log) when the
    frame cannot be produced — chaining is an optimisation, never a hard failure.
    """
    if not clip_path or not os.path.isfile(clip_path):
        return None
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-sseof", "-0.1", "-i", clip_path,
        "-frames:v", "1", "-q:v", "2", dest_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Could not extract the last frame of %s: %s", clip_path, exc)
        return None
    if proc.returncode != 0 or not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
        logger.debug(
            "ffmpeg produced no last frame for %s: %s",
            clip_path, (proc.stderr or b"")[-200:].decode("utf-8", errors="ignore"),
        )
        return None
    return dest_path


def scene_seed(
    *,
    approved_plate: Optional[str],
    chained_frame: Optional[str],
    face_image: Optional[str],
    mode: str,
) -> Optional[str]:
    """Choose the image-to-video seed for one scene, in priority order.

    1. A plate the user approved — an explicit human decision always wins.
    2. The previous clip's last frame, when chaining is on.
    3. The performer portrait, if one exists.
    4. None → pure text-to-video.
    """
    if approved_plate and os.path.isfile(approved_plate) and os.path.getsize(approved_plate) > 0:
        return approved_plate
    if normalise_mode(mode) == "chain" and chained_frame:
        return chained_frame
    if face_image and os.path.isfile(face_image):
        return face_image
    return None


def normalise_chain_scope(value: Optional[str]) -> str:
    scope = (value or DEFAULT_CHAIN_SCOPE).strip().lower()
    return scope if scope in CHAIN_SCOPES else DEFAULT_CHAIN_SCOPE


def chain_allowed(previous_section: Optional[str], current_section: Optional[str], scope: Optional[str]) -> bool:
    """Whether the previous clip's last frame may seed this scene.

    Chaining across a section change is what made `chain` mode feel monotonous: a
    chorus inherited the verse's framing. Unknown sections never block the chain —
    an unlabeled plan should not lose the feature.
    """
    if normalise_chain_scope(scope) == "all":
        return True
    before = str(previous_section or "").strip().lower()
    after = str(current_section or "").strip().lower()
    if not before or not after:
        return True
    return before == after
