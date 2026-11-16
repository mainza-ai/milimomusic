"""
One recovery policy for every kind of interrupted work.

Three divergent reconcilers used to run at boot, each modelling "interrupted"
differently: jobs were marked FAILED, durable tasks were parked at `paused`
(which nothing ever resumed), and agent runs were marked `interrupted`. A
operator could not answer "what happens to my queued render after a crash?"
without reading three files.

This module is the single answer:

    jobs        queued      → re-dispatchable (kept QUEUED; the worker picks it up)
                processing  → INTERRUPTED (the in-flight work cannot be resumed)
    tasks       queued      → kept QUEUED
                running     → PAUSED + a resume hint (clips already rendered survive)
    agent runs  active      → INTERRUPTED (they cannot resume mid-track)

It also owns the stuck-job sweeper: `updated_at` cannot serve as a heartbeat, so
every progress tick writes `last_heartbeat_at`, and a job that stops breathing is
marked STALLED instead of holding the GPU lock forever.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlmodel import Session, select

from app.models import Job, JobStatus

logger = logging.getLogger("milimo.recovery")

ACTIVE_RUN_STATUSES = ("queued", "running", "awaiting_approval")

# A job with no heartbeat for this long is considered wedged.
DEFAULT_STALL_SECONDS = 900


def _now() -> datetime:
    return datetime.now(timezone.utc)


def recover_jobs(engine) -> Dict[str, int]:
    """Crash recovery for generation jobs.

    QUEUED jobs stay queued: nothing has started, so they are safe to dispatch
    again. The old implementation failed them with "Interrupted by server
    restart", which silently destroyed work the user had already paid for.
    """
    stats = {"queued_kept": 0, "interrupted": 0}
    with Session(engine) as session:
        queued = session.exec(select(Job).where(Job.status == JobStatus.QUEUED)).all()
        stats["queued_kept"] = len(queued)

        running = session.exec(select(Job).where(Job.status == JobStatus.PROCESSING)).all()
        for job in running:
            job.status = JobStatus.INTERRUPTED
            job.error_msg = (
                "Interrupted by server restart. The job was already running and cannot be "
                "resumed; retry it to regenerate."
            )
            session.add(job)
        stats["interrupted"] = len(running)
        session.commit()

    if stats["interrupted"] or stats["queued_kept"]:
        logger.warning(
            "Job recovery: %d queued job(s) preserved, %d in-flight job(s) marked interrupted.",
            stats["queued_kept"], stats["interrupted"],
        )
    return stats


def recover_tasks() -> Dict[str, int]:
    """Crash recovery for the durable task queue."""
    stats = {"paused": 0}
    try:
        from app.core.task_queue import task_queue

        recovered = task_queue.recover_interrupted_tasks()
        stats["paused"] = len(recovered)
        if recovered:
            logger.warning(
                "Durable queue: %d interrupted task(s) parked at 'paused' "
                "(completed clips preserved; POST /videos/tasks/{id}/resume re-queues them).",
                len(recovered),
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Durable queue recovery skipped: %s", exc)
    return stats


def recover_agent_runs(engine) -> Dict[str, int]:
    """Crash recovery for agent/album runs (every blocking status)."""
    from app.models import AgentRun

    stats = {"interrupted": 0}
    with Session(engine) as session:
        runs = session.exec(select(AgentRun).where(AgentRun.status.in_(ACTIVE_RUN_STATUSES))).all()
        seen = sorted({r.status for r in runs})
        for run in runs:
            run.status = "interrupted"
            run.error_type = "interrupted"
            run.error_message = "Interrupted by server restart."
            run.finished_at = _now()
            session.add(run)
        session.commit()
        stats["interrupted"] = len(runs)
        if runs:
            logger.warning(
                "Reconciled %d orphaned agent run(s) (statuses: %s).", len(runs), ", ".join(seen)
            )
    return stats


def recover_on_boot(engine) -> Dict[str, Dict[str, int]]:
    """The single boot-time recovery entry point."""
    return {
        "jobs": recover_jobs(engine),
        "tasks": recover_tasks(),
        "agent_runs": recover_agent_runs(engine),
    }


def sweep_stalled_jobs(engine, max_stall_seconds: int = DEFAULT_STALL_SECONDS) -> List[str]:
    """Mark jobs that stopped heart-beating as FAILED instead of leaving them 'processing'.

    Without this a wedged MLX call or a hung provider keeps a job `processing`
    forever and holds the GPU lock, so every subsequent job queues behind a ghost.
    """
    cutoff = _now() - timedelta(seconds=max_stall_seconds)
    stalled: List[str] = []
    with Session(engine) as session:
        candidates = session.exec(select(Job).where(Job.status == JobStatus.PROCESSING)).all()
        for job in candidates:
            beat = job.last_heartbeat_at or job.updated_at or job.created_at
            if beat is None:
                continue
            if beat.tzinfo is None:
                beat = beat.replace(tzinfo=timezone.utc)
            if beat > cutoff:
                continue
            job.status = JobStatus.FAILED
            job.error_msg = (
                f"Stalled: no progress for over {max_stall_seconds // 60} minutes. "
                "The worker was most likely wedged inside a model call."
            )
            session.add(job)
            stalled.append(str(job.id))
        session.commit()

    for job_id in stalled:
        logger.warning("Marked stalled job %s as failed (no heartbeat).", job_id)
    return stalled


def touch_heartbeat(engine, job_id) -> None:
    """Record liveness. Cheap enough to call on every progress tick."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.last_heartbeat_at = _now()
                session.add(job)
                session.commit()
    except Exception as exc:  # pragma: no cover - never fail a render over telemetry
        logger.debug("Heartbeat write failed for %s: %s", job_id, exc)


def resume_paused_task(task_id: str) -> bool:
    """Re-queue a task parked at `paused` by crash recovery."""
    try:
        from app.core.task_queue import task_queue

        task = task_queue.get_task(task_id)
        if not task or task.status != "paused":
            return False
        task_queue.update_task(
            task_id,
            status="queued",
            message="Re-queued after interruption.",
        )
        return True
    except Exception as exc:
        logger.warning("Could not resume task %s: %s", task_id, exc)
        return False


def repair_stalled_agent_runs(engine) -> int:  # pragma: no cover - thin alias
    """Backwards-compatible alias kept for callers importing the old name."""
    return recover_agent_runs(engine)["interrupted"]
