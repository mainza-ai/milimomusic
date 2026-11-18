"""Artifact manifest: one reproducible record per session (plan 7.22).

The platform already knew the seed, sampling parameters, model id, per-stage
provenance, quality verdict and the exact prompt it sent — and emitted none of it
in a portable form. `build_manifest` is the single builder those facts flow
through; packs, reports and any future DOI/Zenodo export consume it instead of
re-inventing a shape.

Design rules:
- **Never invent.** A missing fact is `None`, not a default.
- **Hash what is shipped.** Every artifact carries size + sha256.
- **Say what produced it.** Stage provenance and the quality verdict travel with
  the artifacts they describe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("milimo.manifest")

SCHEMA_VERSION = "1.0"


def _git_commit() -> Optional[str]:
    """Best-effort code revision; None when this is not a git checkout."""
    try:
        from app.core.paths import get_repo_root

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(get_repo_root()), capture_output=True, text=True, timeout=10,
        )
        commit = (out.stdout or "").strip()
        return commit if out.returncode == 0 and commit else None
    except (subprocess.SubprocessError, OSError):
        return None


def _sha256(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _json_field(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _artifact(path: Optional[str], kind: str, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    candidate = path
    if not os.path.isfile(candidate):
        try:
            from app.core.paths import resolve_audio_file

            resolved = resolve_audio_file(candidate)
            candidate = resolved or candidate
        except Exception:
            pass
    if not os.path.isfile(candidate):
        return {
            "kind": kind,
            "label": label or os.path.basename(str(path)),
            "path": str(path),
            "present": False,
        }
    return {
        "kind": kind,
        "label": label or os.path.basename(candidate),
        "path": candidate,
        "present": True,
        "bytes": os.path.getsize(candidate),
        "sha256": _sha256(candidate),
    }


def build_manifest(
    job: Any,
    *,
    takes: Optional[Iterable[Dict[str, Any]]] = None,
    extra_artifacts: Optional[Iterable[tuple]] = None,
) -> Dict[str, Any]:
    """Build the manifest for one job.

    `takes` is the list from `services.video.takes.list_takes` (optional);
    `extra_artifacts` are `(path, kind)` pairs the caller wants included.
    """
    provenance = _json_field(getattr(job, "provenance_json", None))
    video_config = _json_field(getattr(job, "video_config_json", None))
    stem_sources = _json_field(getattr(job, "stems_json", None)) or {}

    artifacts: List[Dict[str, Any]] = []
    for path, kind, label in (
        (getattr(job, "audio_path", None), "master_audio", "generated master"),
        (getattr(job, "mastered_path", None), "mastered_audio", "mastered master"),
        (getattr(job, "midi_path", None), "midi", "transcription MIDI"),
        (getattr(job, "musicxml_path", None), "musicxml", "transcription MusicXML"),
        (getattr(job, "cover_image_path", None), "cover", "cover artwork"),
        (getattr(job, "video_path", None), "video", "selected video take"),
    ):
        entry = _artifact(path, kind, label)
        if entry:
            artifacts.append(entry)

    if isinstance(stem_sources, dict):
        for name, path in stem_sources.items():
            if isinstance(path, str) and name not in ("instrument_programs", "sources_available", "default_source"):
                entry = _artifact(path, f"stem:{name}")
                if entry:
                    artifacts.append(entry)

    for path, kind in (extra_artifacts or []):
        entry = _artifact(path, kind)
        if entry:
            artifacts.append(entry)

    selected_take = None
    take_list = list(takes or [])
    for take in take_list:
        if take.get("is_selected"):
            selected_take = take
            break

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "application": {
            "name": "Milimo Music",
            "version": _app_version(),
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "job": {
            "id": str(getattr(job, "id", "")),
            "title": getattr(job, "title", None),
            "status": str(getattr(getattr(job, "status", None), "value", getattr(job, "status", None))),
            "created_at": _iso(getattr(job, "created_at", None)),
            "duration_ms": getattr(job, "duration_ms", None),
            "quality": getattr(job, "quality", None),
            "used_fallback_synth": getattr(job, "used_fallback_synth", None),
            "fallback_reason": getattr(job, "fallback_reason", None),
        },
        "generation": {
            "provider": getattr(job, "model_provider", None),
            "model_id": getattr(job, "model_id", None),
            "seed": getattr(job, "seed", None),
            "temperature": getattr(job, "temperature", None),
            "cfg_scale": getattr(job, "cfg_scale", None),
            "topk": getattr(job, "topk", None),
            "bpm": getattr(job, "bpm", None),
            "model_prompt": getattr(job, "model_prompt", None) or getattr(job, "prompt", None),
            "model_lyrics": getattr(job, "model_lyrics", None) or getattr(job, "lyrics", None),
            "structured_caption": _json_field(getattr(job, "structured_caption_json", None)),
        },
        "provenance": provenance,
        "beat_grid": _json_field(getattr(job, "beat_grid_json", None)),
        "lyrics": {
            "has_timed_lyrics": bool(getattr(job, "timed_lyrics_json", None)),
            "timed_line_count": len(_json_field(getattr(job, "timed_lyrics_json", None)) or []),
        },
        "video": {
            "config": video_config,
            "selected_take": selected_take,
            "takes": take_list,
        },
        "artifacts": artifacts,
        "reproduction": {
            "note": (
                "Generated by Milimo Music. Seeds and sampling parameters are recorded "
                "verbatim; model weights are identified by provider/model id — record the "
                "checkpoint revisions your provider reported if you publish results."
            ),
        },
    }


def _app_version() -> Optional[str]:
    try:
        from app import release_state

        return getattr(release_state, "VERSION", None) or getattr(release_state, "__version__", None)
    except Exception:
        return None


def _iso(value: Any) -> Optional[str]:
    try:
        return value.isoformat() if value else None
    except AttributeError:
        return str(value) if value else None
