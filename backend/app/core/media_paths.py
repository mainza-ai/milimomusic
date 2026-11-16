"""
Guard for request-supplied media references.

A request may *name* media. It may not name an arbitrary file on the host.

Every path that arrives from a client is routed through here before it can be
read, served, or persisted onto a Job row. The rules are deliberately
deterministic — no "search a dozen directories for the basename" heuristics —
so an absolute path is either inside an allow-listed media root or it is
rejected.

History: `POST /tracks/{id}/commit-vocal` used to accept any `master_path`,
store it as `job.audio_path`, and `GET /download_track/{id}` would serve
whatever that path pointed at — an unauthenticated arbitrary-file-read chain
that could exfiltrate `.env` / `llm_config.json`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from app.core.paths import get_data_dir, get_generated_audio_dir, get_repo_root

AUDIO_EXTS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".mid", ".midi"}
)
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"})
ALL_MEDIA_EXTS = AUDIO_EXTS | IMAGE_EXTS | VIDEO_EXTS
PUBLIC_PREFIXES = ("audio", "stems", "covers", "generated_audio", "videos", "mastered")


class MediaPathError(ValueError):
    """Raised when a client-supplied media reference cannot be honoured.

    ``kind`` lets the caller pick the right status code:
      * ``missing``   — a legitimate reference that does not exist  → 404
      * ``forbidden`` — an absolute path outside the media roots     → 400
      * ``invalid``   — traversal, NUL, unsupported extension        → 400
    """

    def __init__(self, message: str, kind: str = "invalid"):
        super().__init__(message)
        self.kind = kind


def _is_public_media_url(path: str) -> bool:
    trimmed = path.lstrip("/")
    return any(trimmed.startswith(prefix + "/") for prefix in PUBLIC_PREFIXES)


def media_roots() -> List[Path]:
    """Directories from which the API is ever allowed to read media.

    Specific subdirectories of ``data/`` are listed rather than ``data/`` itself
    so that databases and config files sitting beside them stay unreachable.
    """
    repo = get_repo_root()
    data = get_data_dir()
    raw = [
        get_generated_audio_dir(),
        repo / "generated_audio",
        repo / "backend" / "generated_audio",
        repo / "generated_midi",
        repo / "backend" / "generated_midi",
        data / "uploads",
        data / "covers",
        data / "audio",
        data / "video_cache",
        data / "voice_profiles",
        repo / "backend" / "data" / "covers",
        repo / "backend" / "data" / "uploads",
        repo / "backend" / "data" / "video_cache",
        repo / "backend" / "data" / "voice_profiles",
        repo / "assets",
    ]
    roots: List[Path] = []
    seen = set()
    for r in raw:
        try:
            p = Path(r).resolve()
        except OSError:
            continue
        if p in seen:
            continue
        seen.add(p)
        roots.append(p)
    return roots


def _normalise(value: str) -> str:
    """Strip scheme/host/query and normalise separators. Never touches the FS."""
    text = str(value).strip()
    if "://" in text:
        text = urlparse(text).path or ""
    else:
        text = text.split("?", 1)[0].split("#", 1)[0]
    return text.replace("\\", "/")


def _has_traversal(value: str) -> bool:
    if "\x00" in value:
        return True
    return any(part == ".." for part in value.split("/"))


def is_allowed_media_path(path: "str | os.PathLike[str]", *, allow_exts: Optional[frozenset] = None) -> bool:
    """True when ``path`` resolves inside an allow-listed media root."""
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return False

    if allow_exts is not None and resolved.suffix.lower() not in allow_exts:
        return False

    for root in media_roots():
        if resolved == root or root in resolved.parents:
            return True
    return False


def resolve_media_reference(
    value: Optional[str],
    *,
    allow_exts: Optional[frozenset] = None,
    must_exist: bool = True,
    field: str = "path",
) -> Optional[str]:
    """Resolve a client-supplied media reference to an absolute path inside the
    allow-listed roots, or raise ``ValueError``.

    Accepted forms (all confined to media roots):
      * ``/audio/track.wav``            web URL emitted by this API
      * ``http://host/audio/track.wav`` same, with an absolute URL
      * ``generated_audio/track.wav``  repo-relative path stored on old Job rows
      * ``/abs/path/in/generated_audio/track.wav`` server-side absolute path

    Rejected: anything containing ``..`` or NUL, any absolute path outside the
    media roots, and any extension outside ``allow_exts``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise MediaPathError(f"{field} must be a string")

    clean = _normalise(value)
    if not clean.strip():
        return None
    if _has_traversal(clean):
        raise MediaPathError(f"{field} contains an illegal path segment", kind="invalid")

    ext_filter = allow_exts

    # NOTE: `os.path.isabs("/audio/x.wav")` is True on POSIX. A leading slash in a
    # URL is *not* a filesystem path, so public media URLs take the relative branch.
    if os.path.isabs(clean) and not _is_public_media_url(clean):
        candidate = Path(clean)
        if not is_allowed_media_path(candidate, allow_exts=ext_filter):
            raise MediaPathError(f"{field} is outside the media directories", kind="forbidden")
        if must_exist and not candidate.is_file():
            raise MediaPathError(f"{field} does not exist", kind="missing")
        return str(candidate.resolve())

    # Relative form (or public URL): try each media root, then the repo root, and
    # require the result to still land inside a media root. Public URL prefixes
    # (/audio/, /stems/, /covers/, /generated_audio/, /videos/) are stripped
    # because that is how this API names its own media.
    rel = clean.lstrip("/")
    variants = [rel]
    for prefix in PUBLIC_PREFIXES:
        if rel.startswith(prefix + "/"):
            variants.append(rel[len(prefix) + 1:])

    bases = media_roots() + [get_repo_root()]
    for variant in variants:
        for base in bases:
            candidate = base / variant
            if not candidate.is_file():
                continue
            if not is_allowed_media_path(candidate, allow_exts=ext_filter):
                continue
            return str(candidate.resolve())

    if must_exist:
        raise MediaPathError(f"{field} does not exist", kind="missing")
    return None


def to_public_media_reference(resolved: str) -> str:
    """Map a validated media file back to the public URL form this API serves.

    Stored job paths stay in URL form (`/audio/<name>`) so the frontend can build
    `<audio src>` without knowing where the file physically lives.
    """
    path = Path(resolved)
    lowered = path.as_posix().lower()
    name = path.name
    if "/covers/" in lowered:
        return f"/covers/{name}"
    if "/video_cache/" in lowered or "/videos/" in lowered:
        return f"/videos/{name}"
    if "/uploads/" in lowered:
        return f"/audio/{name}"
    return f"/audio/{name}"


def safe_media_name(value: Optional[str], *, allow_exts: Optional[frozenset] = None) -> str:
    """Sanitise a single path segment used in a URL (e.g. a thumbnail filename).

    Returns a bare filename — never a path — so callers cannot be walked out of
    the directory they intend to read.
    """
    if not value:
        raise MediaPathError("filename is required", kind="invalid")
    clean = _normalise(value)
    if _has_traversal(clean):
        raise MediaPathError("filename contains an illegal path segment", kind="invalid")
    name = os.path.basename(clean)
    if not name or name in {".", ".."}:
        raise MediaPathError("filename is invalid", kind="invalid")
    if allow_exts is not None and Path(name).suffix.lower() not in allow_exts:
        raise MediaPathError("filename has an unsupported extension", kind="invalid")
    return name
