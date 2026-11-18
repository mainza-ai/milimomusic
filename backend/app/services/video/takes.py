"""Render takes: every video render is kept, selectable and deletable (plan 7.4).

Before this, `Job.video_path` was a single string and every render overwrote the
previous master silently — so a retake or an A/B comparison was impossible and
the UI's "compare" widget had nothing to compare (audit W4/U1).

A *take* is one render attempt: its config, its artifacts (master, poster, proxy,
caption sidecars), its QC verdict, and whether the job currently shows it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID

from sqlmodel import Session, select

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models import VideoRender

logger = logging.getLogger("milimo.video.takes")

# Files belonging to a take live beside the master and share its stem.
_SIDECAR_SUFFIXES = (
    "_poster.jpg",
    "_480p.mp4",
    ".srt",
    ".vtt",
    ".ass",
    "_report.json",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid(value: Union[str, UUID]) -> UUID:
    """Coerce a 32-hex or hyphenated id to a UUID for UUID-typed columns."""
    return value if isinstance(value, UUID) else UUID(str(value))


def create_take(
    session: Session,
    *,
    job_id: str,
    task_id: Optional[str],
    kind: str,
    config: Optional[Dict[str, Any]] = None,
) -> "VideoRender":  # noqa: F821 - imported lazily to avoid a models import cycle
    """Insert a take row in `rendering` state and return it."""
    from app.models import VideoRender

    take = VideoRender(
        job_id=_uuid(job_id),
        task_id=task_id,
        kind=kind,
        status="rendering",
        config_json=json.dumps(config or {}, default=str),
    )
    session.add(take)
    session.commit()
    session.refresh(take)
    return take


def finish_take(
    session: Session,
    take: "VideoRender",  # noqa: F821
    *,
    status: str,
    video_path: Optional[str] = None,
    file_path: Optional[str] = None,
    error: Optional[str] = None,
    qc: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    poster_path: Optional[str] = None,
    proxy_path: Optional[str] = None,
) -> "VideoRender":  # noqa: F821
    take.status = status
    take.video_path = video_path
    take.file_path = file_path
    take.error = error
    take.qc_json = json.dumps(qc, default=str) if qc else None
    take.duration_ms = duration_ms
    take.width = width
    take.height = height
    take.fps = fps
    take.poster_path = poster_path
    take.proxy_path = proxy_path
    take.updated_at = _now()
    session.add(take)
    session.commit()
    session.refresh(take)
    return take


def reconcile_stale_renders(session: Session, reason: Optional[str] = None) -> int:
    """Fail renders that were in flight when the process died (plan 7.28).

    A killed server left `videorender` rows in `rendering` forever: the UI showed a
    take that never completed, and nothing ever moved it. On boot we mark them
    failed with a stated reason — an honest terminal state instead of a ghost.
    """
    from app.models import VideoRender

    message = reason or "The server restarted before this render finished."
    rows = session.exec(select(VideoRender).where(VideoRender.status == "rendering")).all()
    for row in rows:
        row.status = "failed"
        row.error = message
        row.updated_at = _now()
        session.add(row)
    if rows:
        session.commit()
        logger.info("Marked %d interrupted video render(s) as failed", len(rows))
    return len(rows)


def sweep_stalled_renders(session: Session, max_age_seconds: float) -> int:
    """Fail renders that have been `rendering` longer than any real render takes.

    Boot reconciliation (7.28) covers process death. This covers the other case: a
    render that is still "running" but wedged — the UI would show it forever.
    """
    from datetime import timedelta

    from app.models import VideoRender

    cutoff = _now() - timedelta(seconds=max(60.0, float(max_age_seconds)))
    rows = session.exec(select(VideoRender).where(VideoRender.status == "rendering")).all()
    swept = 0
    for row in rows:
        started = row.updated_at or row.created_at
        if started is None:
            continue
        started = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
        if started <= cutoff:
            row.status = "failed"
            row.error = (
                f"Render made no progress for {int(max_age_seconds)}s and was stopped "
                f"by the stall sweeper."
            )
            row.updated_at = _now()
            session.add(row)
            swept += 1
    if swept:
        session.commit()
        logger.info("Swept %d stalled video render(s)", swept)
    return swept


def list_takes(session: Session, job_id: str) -> List[Dict[str, Any]]:
    from app.models import VideoRender

    rows = session.exec(
        select(VideoRender)
        .where(VideoRender.job_id == _uuid(job_id))
        .order_by(VideoRender.created_at.desc())
    ).all()
    return [t.to_dict() for t in rows]


def get_take(session: Session, render_id: str) -> Optional["VideoRender"]:  # noqa: F821
    from app.models import VideoRender

    try:
        return session.get(VideoRender, _uuid(render_id))
    except (ValueError, AttributeError, TypeError):
        return None


def select_take(session: Session, job_id: str, render_id: str) -> Optional["VideoRender"]:
    """Point `Job.video_path` at this take, and only this one, selected."""
    from app.models import Job, VideoRender

    take = get_take(session, render_id)
    if not take or take.job_id != _uuid(job_id) or take.status != "completed":
        return None
    if not take.video_path:
        return None

    rows = session.exec(select(VideoRender).where(VideoRender.job_id == _uuid(job_id))).all()
    for row in rows:
        row.is_selected = row.id == take.id
        row.updated_at = _now()
        session.add(row)

    job = session.get(Job, _uuid(take.job_id))
    if job:
        job.video_path = take.video_path
        job.video_config_json = take.config_json
        session.add(job)

    session.commit()
    session.refresh(take)
    return take


def mark_selected(session: Session, job_id: str, render_id: str) -> None:
    from app.models import Job, VideoRender

    rows = session.exec(select(VideoRender).where(VideoRender.job_id == _uuid(job_id))).all()
    target = None
    for row in rows:
        row.is_selected = row.id == _uuid(render_id)
        session.add(row)
        if row.is_selected:
            target = row
    if target:
        job = session.get(Job, _uuid(job_id))
        if job:
            job.video_path = target.video_path
            job.video_config_json = target.config_json
            session.add(job)
    session.commit()


def _take_files(video_path: Optional[str]) -> List[str]:
    """Every artifact that belongs to a take, derived from its master path."""
    if not video_path:
        return []
    directory, name = os.path.split(video_path)
    stem, ext = os.path.splitext(name)
    candidates = [video_path]
    candidates += [os.path.join(directory, stem + suffix) for suffix in _SIDECAR_SUFFIXES]
    return [p for p in candidates if p and os.path.isfile(p)]


def delete_take(session: Session, job_id: str, render_id: str) -> Dict[str, Any]:
    """Delete a take, its artifacts, and its per-clip workspace."""
    from app.models import VideoRender

    take = get_take(session, render_id)
    if not take or take.job_id != _uuid(job_id):
        return {"deleted": False, "reason": "not_found"}

    removed = 0
    for path in _take_files(take.file_path or take.video_path):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue

    workspace = clip_workspace(render_id)
    if os.path.isdir(workspace):
        try:
            removed += sum(len(files) for _, _, files in os.walk(workspace))
            shutil.rmtree(workspace, ignore_errors=True)
        except OSError:
            pass

    was_selected = take.is_selected
    session.delete(take)
    session.commit()

    remaining = list_takes(session, job_id)
    if was_selected and remaining:
        select_take(session, job_id, remaining[0]["id"])
    elif was_selected:
        from app.models import Job
        job = session.get(Job, _uuid(job_id))
        if job:
            job.video_path = None
            job.video_config_json = None
            session.add(job)
            session.commit()

    return {"deleted": True, "removed_files": removed, "was_selected": was_selected}


def clip_workspace(render_id: str) -> str:
    """Per-take directory that keeps the rendered clips and the plan (plan 7.11)."""
    from app.core.paths import get_data_dir

    return str(get_data_dir() / "video_cache" / "takes" / str(render_id))


def save_clip_plan(render_id: str, plan: Dict[str, Any], clip_paths: List[str]) -> Optional[str]:
    """Persist the plan and copy the rendered clips into the take workspace."""
    workspace = clip_workspace(render_id)
    clips_dir = os.path.join(workspace, "clips")
    try:
        os.makedirs(clips_dir, exist_ok=True)
        kept: List[str] = []
        for idx, src in enumerate(clip_paths):
            if not src or not os.path.isfile(src):
                continue
            dest = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")
            # A retake re-saves a take whose clips already live here; copying a file
            # onto itself raises SameFileError and used to abort the whole plan.
            if os.path.abspath(src) != os.path.abspath(dest):
                try:
                    shutil.copyfile(src, dest)
                except OSError as copy_err:
                    logger.warning("Could not keep clip %s: %s", src, copy_err)
                    continue
            kept.append(dest)
        payload = dict(plan)
        payload["clips"] = kept
        with open(os.path.join(workspace, "plan.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return workspace
    except OSError as exc:
        logger.warning("Could not persist clips for take %s: %s", render_id, exc)
        return None


def load_clip_plan(render_id: str) -> Optional[Dict[str, Any]]:
    workspace = clip_workspace(render_id)
    path = os.path.join(workspace, "plan.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read plan for take %s: %s", render_id, exc)
        return None
