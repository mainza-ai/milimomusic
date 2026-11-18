"""Character library operations (plan 8.3).

A character is the reusable half of consistency: a description plus a reference
plate, referenced by id from any render. The plate lives under `data/characters/`
and is content-addressed, so re-importing the same portrait does not duplicate it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Session, select

logger = logging.getLogger("milimo.characters")

SUPPORTED_PLATE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def characters_dir() -> str:
    from app.core.paths import get_data_dir

    path = os.path.join(str(get_data_dir()), "characters")
    os.makedirs(path, exist_ok=True)
    return path


def _sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _uuid(value: Any) -> Optional[UUID]:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def list_characters(session: Session) -> List[Dict[str, Any]]:
    from app.models import Character

    rows = session.exec(select(Character).order_by(Character.updated_at.desc())).all()
    return [row.to_dict() for row in rows]


def get_character(session: Session, character_id: Any) -> Optional["Character"]:  # noqa: F821
    from app.models import Character

    uid = _uuid(character_id)
    return session.get(Character, uid) if uid else None


def create_character(
    session: Session,
    *,
    name: str,
    description: Optional[str] = None,
    plate_source: Optional[str] = None,
) -> "Character":  # noqa: F821
    """Create a character, copying its reference plate into the library.

    `plate_source` is a filesystem path (upload, gallery asset or cover). Anything
    that is not a readable image is refused rather than stored broken.
    """
    from app.models import Character

    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("name is required")

    plate_path = None
    plate_sha = None
    if plate_source:
        src = plate_source
        if not os.path.isfile(src):
            try:
                from app.core.paths import resolve_image_file

                src = resolve_image_file(plate_source) or plate_source
            except Exception:
                src = plate_source
        if not os.path.isfile(src):
            raise ValueError(f"reference image not found: {plate_source}")
        ext = os.path.splitext(src)[1].lower()
        if ext not in SUPPORTED_PLATE_EXTS:
            raise ValueError(f"unsupported image type: {ext or 'unknown'}")

        plate_sha = _sha256_file(src)
        existing = session.exec(
            select(Character).where(Character.plate_sha256 == plate_sha)
        ).first()
        if existing and existing.plate_path and os.path.isfile(existing.plate_path):
            # Same portrait already in the library — reuse the file, keep both rows.
            plate_path = existing.plate_path
        else:
            digest_name = (plate_sha or hashlib.sha1(clean_name.encode()).hexdigest())[:32]
            plate_path = os.path.join(characters_dir(), f"{digest_name}{ext}")
            if not os.path.isfile(plate_path):
                shutil.copyfile(src, plate_path)

    character = Character(
        name=clean_name,
        description=(description or "").strip() or None,
        plate_path=plate_path,
        plate_sha256=plate_sha,
        updated_at=datetime.now(timezone.utc),
    )
    session.add(character)
    session.commit()
    session.refresh(character)
    return character


def update_character(
    session: Session,
    character_id: Any,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    plate_source: Optional[str] = None,
) -> Optional["Character"]:  # noqa: F821
    character = get_character(session, character_id)
    if not character:
        return None
    if name is not None and name.strip():
        character.name = name.strip()
    if description is not None:
        character.description = description.strip() or None
    if plate_source:
        # Replacing the plate re-imports through the same path (dedup included).
        character.plate_path = None
        character.plate_sha256 = None
        session.add(character)
        session.commit()
        refreshed = create_character(
            session, name=character.name, description=character.description,
            plate_source=plate_source,
        )
        character.plate_path = refreshed.plate_path
        character.plate_sha256 = refreshed.plate_sha256
        session.delete(refreshed)
        session.commit()
    character.updated_at = datetime.now(timezone.utc)
    session.add(character)
    session.commit()
    session.refresh(character)
    return character


def delete_character(session: Session, character_id: Any) -> bool:
    character = get_character(session, character_id)
    if not character:
        return False
    # Shared plates (same sha used by another character) must survive.
    shared = False
    if character.plate_sha256:
        from app.models import Character

        others = session.exec(
            select(Character).where(
                Character.plate_sha256 == character.plate_sha256,
                Character.id != character.id,
            )
        ).all()
        shared = bool(others)
    if character.plate_path and not shared and os.path.isfile(character.plate_path):
        try:
            os.remove(character.plate_path)
        except OSError as exc:
            logger.warning("Could not remove character plate %s: %s", character.plate_path, exc)
    session.delete(character)
    session.commit()
    return True


def resolve_for_render(session: Session, character_id: Any) -> Dict[str, Any]:
    """What a render needs from a character: description + plate path."""
    character = get_character(session, character_id)
    if not character:
        return {}
    return {
        "description": character.description or "",
        "plate_path": character.plate_path if character.plate_path and os.path.isfile(character.plate_path) else None,
    }
