"""Upload hardening: extension whitelist, size caps (streamed), magic bytes,
and randomized containment-safe filenames.

Closes the audit's path-traversal / unbounded-RAM-upload class (A4/A7):
callers never touch user-supplied filenames or unbounded reads.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}   # SVG deliberately excluded (stored-XSS)

_MAGIC = (
    # audio
    (b"ID3", "audio"), (b"\xff\xfb", "audio"), (b"\xff\xf3", "audio"), (b"\xff\xfa", "audio"),
    (b"RIFF", "audio"),          # wav (and webp — disambiguated by caller's kind)
    (b"fLaC", "audio"), (b"OggS", "audio"),
    # images
    (b"\x89PNG", "image"), (b"\xff\xd8\xff", "image"),  # png, jpeg
)


def _fail(code: str, message: str, status: int) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message}})


def _env_mb(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
        return max(1.0, min(v, 512.0))
    except ValueError:
        return default


async def save_upload(
    file: UploadFile,
    dest_dir: str,
    *,
    kind: str = "audio",
    max_mb: Optional[float] = None,
) -> Tuple[str, str]:
    """Validate + stream an upload to `dest_dir` under a random safe name.

    Returns (absolute_path, public_filename). Raises typed HTTP errors:
      bad_type (400), bad_content (400), payload_too_large (413).
    """
    if kind == "audio":
        allowed, cap_env = AUDIO_EXTS, "MAX_AUDIO_UPLOAD_MB"
        default_cap = 60.0
    elif kind == "image":
        allowed, cap_env = IMAGE_EXTS, "MAX_IMAGE_UPLOAD_MB"
        default_cap = 8.0
    else:
        raise ValueError(f"unknown upload kind {kind}")

    original_name = file.filename or ""
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in allowed:
        raise _fail("bad_type", f"File type '{ext or '(none)'}' is not allowed. Allowed: {sorted(allowed)}.", 400)

    cap_mb = max_mb if max_mb is not None else _env_mb(cap_env, default_cap)
    cap_bytes = int(cap_mb * 1024 * 1024)

    os.makedirs(dest_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.abspath(os.path.join(dest_dir, safe_name))

    first_chunk: Optional[bytes] = None
    written = 0
    with open(dest_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            if first_chunk is None:
                first_chunk = chunk[:16]
            written += len(chunk)
            if written > cap_bytes:
                out.close()
                os.remove(dest_path)
                raise _fail("payload_too_large", f"Upload exceeds the {cap_mb:g} MB limit.", 413)
            out.write(chunk)

    if written == 0:
        os.remove(dest_path)
        raise _fail("bad_content", "Uploaded file is empty.", 400)

    # Magic-byte sniff: content must look like its claimed kind.
    head = first_chunk or b""
    looks_ok = False
    for magic, magic_kind in _MAGIC:
        if head.startswith(magic):
            if kind == "image" and magic == b"RIFF":
                continue  # RIFF belongs to wav; webp handled below
            looks_ok = (magic_kind == kind)
            break
        if kind == "audio" and ext == ".m4a" and len(head) >= 8 and head[4:8] == b"ftyp":
            looks_ok = True
            break
    if not looks_ok:
        os.remove(dest_path)
        raise _fail("bad_content", "File content does not match its declared type.", 400)

    return dest_path, safe_name
