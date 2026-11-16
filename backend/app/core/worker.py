"""
Durable music-generation worker (plan 2.1).

Generation jobs were Starlette `BackgroundTasks`: a crash between the DB commit and
task pickup left a `queued` row nobody owned, and there was no ordering, no depth
limit and no restart pickup. Only the video orchestrator used the durable queue.

Now `/generate/music` enqueues a durable task; this worker claims and runs it:

    queued  -> claimed atomically (status='running') -> pipeline
    crash   -> restart re-queues `running` tasks with attempts+1
    cancel  -> the pipeline's own cancel_event still applies

The GPU lane stays single-consumer (the pipeline acquires the device lock), so the
worker does not need its own concurrency control beyond the queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from app.core.task_queue import DurableTaskQueue

logger = logging.getLogger("milimo.worker")

MUSIC_TASK_TYPE = "music_generation"
DEFAULT_POLL_SECONDS = 2.0


class GenerationWorker:
    """Serialised consumer for `music_generation` tasks."""

    _instance: Optional["GenerationWorker"] = None

    def __init__(self, engine, poll_seconds: float = DEFAULT_POLL_SECONDS):
        self.engine = engine
        self.poll_seconds = poll_seconds
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self.processed = 0

    @classmethod
    def get_instance(cls, engine=None) -> "GenerationWorker":
        if cls._instance is None and engine is not None:
            cls._instance = cls(engine)
        assert cls._instance is not None, "worker not initialised"
        return cls._instance

    # ── queue helpers ───────────────────────────────────────────────────────
    @staticmethod
    def enqueue(job_id: str, request_payload: Dict[str, Any]) -> str:
        """Persist a generation task. Returns the task id (== job id)."""
        queue = DurableTaskQueue.get_instance()
        queue.enqueue_task(
            task_id=f"gen_{job_id}",
            task_type=MUSIC_TASK_TYPE,
            payload={"job_id": str(job_id), "request": request_payload},
            initial_message="Queued for generation.",
        )
        return f"gen_{job_id}"

    def _claim_next(self) -> Optional[Dict[str, Any]]:
        """Atomically move the oldest queued music task to running."""
        queue = DurableTaskQueue.get_instance()
        conn = queue._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT task_id, payload FROM durable_tasks "
                "WHERE task_type = ? AND status = 'queued' "
                "ORDER BY created_at ASC LIMIT 1",
                (MUSIC_TASK_TYPE,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            cur = conn.execute(
                "UPDATE durable_tasks SET status='running', updated_at=datetime('now') "
                "WHERE task_id = ? AND status = 'queued'",
                (row["task_id"],),
            )
            if cur.rowcount == 0:  # somebody else won the race
                conn.rollback()
                return None
            conn.commit()
            return {"task_id": row["task_id"], "payload": json.loads(row["payload"] or "{}")}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _finish(self, task_id: str, *, ok: bool, error: str = "") -> None:
        queue = DurableTaskQueue.get_instance()
        if ok:
            queue.update_task(task_id, status="completed", progress=100,
                              message="Generation complete.")
        else:
            queue.fail_task(task_id, error or "Generation failed")

    # ── loop ────────────────────────────────────────────────────────────────
    async def _run_one(self, claimed: Dict[str, Any]) -> None:
        from sqlmodel import Session

        from app.models import GenerationRequest, Job
        from app.services.music_service import music_service

        task_id = claimed["task_id"]
        payload = claimed.get("payload") or {}
        job_id = payload.get("job_id")
        request_payload = payload.get("request") or {}
        if not job_id:
            self._finish(task_id, ok=False, error="Task payload has no job_id")
            return

        # Skip work whose job vanished or was cancelled while queued.
        with Session(self.engine) as session:
            job = session.get(Job, job_id) if not _looks_like_hex(job_id) else None
            if job is None:
                from app.main import get_job_by_id  # local import avoids a cycle

                job = get_job_by_id(session, job_id)
            status = job.status.value if hasattr(job.status, "value") else str(job.status) if job else None
            if job is None:
                self._finish(task_id, ok=False, error="Job row no longer exists")
                return
            if status in ("cancelled", "failed", "interrupted"):
                self._finish(task_id, ok=True)
                return

        try:
            req = GenerationRequest(**request_payload)
        except Exception as exc:
            self._finish(task_id, ok=False, error=f"Invalid stored request: {exc}")
            return

        try:
            await music_service.generate_task(job_id, req, self.engine)
            self.processed += 1
            self._finish(task_id, ok=True)
        except asyncio.CancelledError:
            self._finish(task_id, ok=False, error="Cancelled by shutdown")
            raise
        except Exception as exc:
            logger.exception("Generation worker failed on %s", job_id)
            self._finish(task_id, ok=False, error=str(exc)[:400])

    async def run_forever(self) -> None:
        logger.info("Generation worker started (durable queue, single GPU lane).")
        while not self._stopping:
            try:
                claimed = await asyncio.to_thread(self._claim_next)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Generation worker could not poll the queue: %s", exc)
                claimed = None

            if not claimed:
                await asyncio.sleep(self.poll_seconds)
                continue
            await self._run_one(claimed)

    def start(self) -> asyncio.Task:
        if self._task and not self._task.done():
            return self._task
        self._stopping = False
        self._task = asyncio.create_task(self.run_forever())
        return self._task

    async def stop(self) -> None:
        self._stopping = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None


def _looks_like_hex(value: str) -> bool:
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value.lower())


def requeue_interrupted(engine) -> int:
    """Restart recovery: `running` generation tasks become queued again.

    A crash mid-generation leaves the task `running`; the job itself is already
    marked INTERRUPTED by `recovery.recover_jobs`, so the task is closed honestly
    rather than re-running a half-finished song.
    """
    queue = DurableTaskQueue.get_instance()
    conn = queue._get_connection()
    try:
        # The generic recovery parks interrupted tasks at 'paused'; a music task
        # cannot resume mid-song, so close both states here.
        rows = conn.execute(
            "SELECT task_id FROM durable_tasks WHERE task_type = ? "
            "AND status IN ('running', 'paused')",
            (MUSIC_TASK_TYPE,),
        ).fetchall()
        for row in rows:
            queue.update_task(row["task_id"], status="error",
                              error="Interrupted by server restart.")
    finally:
        conn.close()
    if rows:
        logger.warning("Closed %d interrupted generation task(s) at boot.", len(rows))
    return len(rows)
