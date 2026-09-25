"""
Durable SQLite Persistent Task Queue & Checkpointed Project Recovery.

Replaces volatile in-memory task dictionaries with an ACID SQLite-backed persistent queue.
Provides automatic crash recovery for interrupted jobs, asset ownership isolation,
and asynchronous queue pre-enhancement.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("milimo.core.task_queue")


@dataclass
class PersistentTask:
    """A durable task record persisted in SQLite."""

    task_id: str
    task_type: str
    status: str  # "queued" | "running" | "paused" | "completed" | "failed" | "cancelled"
    progress: int = 0
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DurableTaskQueue:
    """ACID SQLite task queue supporting crash recovery and asset ownership."""

    _instance: Optional[DurableTaskQueue] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = ".milimo/durable_tasks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_base_dir = Path(".milimo/task_workspaces")
        self.workspace_base_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: str = ".milimo/durable_tasks.db") -> DurableTaskQueue:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path=db_path)
            return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS durable_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    message TEXT DEFAULT '',
                    payload TEXT,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON durable_tasks (status);")
            conn.commit()

    def enqueue_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        initial_status: str = "queued",
        initial_message: str = "Task queued for execution",
    ) -> PersistentTask:
        """Register a new task with ACID persistence."""
        now = datetime.now(timezone.utc).isoformat()
        task = PersistentTask(
            task_id=task_id,
            task_type=task_type,
            status=initial_status,
            progress=0,
            message=initial_message,
            payload=payload,
            result=None,
            error=None,
            created_at=now,
            updated_at=now,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO durable_tasks
                (task_id, task_type, status, progress, message, payload, result, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.task_type,
                    task.status,
                    task.progress,
                    task.message,
                    json.dumps(task.payload, ensure_ascii=False),
                    json.dumps(task.result, ensure_ascii=False) if task.result else None,
                    task.error,
                    task.created_at,
                    task.updated_at,
                ),
            )
            conn.commit()

        logger.info(f"Task {task_id} ({task_type}) persisted to SQLite queue")
        return task

    def get_task(self, task_id: str) -> Optional[PersistentTask]:
        """Fetch task from SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM durable_tasks WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return PersistentTask(
                task_id=row["task_id"],
                task_type=row["task_type"],
                status=row["status"],
                progress=row["progress"],
                message=row["message"],
                payload=json.loads(row["payload"]) if row["payload"] else {},
                result=json.loads(row["result"]) if row["result"] else None,
                error=row["error"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        payload_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[PersistentTask]:
        """Atomically update task progress, status, or results."""
        task = self.get_task(task_id)
        if not task:
            return None

        now = datetime.now(timezone.utc).isoformat()
        new_status = status if status is not None else task.status
        new_progress = progress if progress is not None else task.progress
        new_message = message if message is not None else task.message
        new_result = result if result is not None else task.result
        new_error = error if error is not None else task.error

        new_payload = dict(task.payload)
        if payload_updates:
            new_payload.update(payload_updates)

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE durable_tasks
                SET status = ?, progress = ?, message = ?, payload = ?, result = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    new_status,
                    new_progress,
                    new_message,
                    json.dumps(new_payload, ensure_ascii=False),
                    json.dumps(new_result, ensure_ascii=False) if new_result else None,
                    new_error,
                    now,
                    task_id,
                ),
            )
            conn.commit()

        task.status = new_status
        task.progress = new_progress
        task.message = new_message
        task.payload = new_payload
        task.result = new_result
        task.error = new_error
        task.updated_at = now
        return task

    def recover_interrupted_tasks(self) -> List[PersistentTask]:
        """
        Interrupted Job Auto-Recovery:
        Called at startup to transition any uncompleted 'running' or 'in-flight' tasks
        to 'paused' with an informative recovery message, preserving completed checkpoints.
        """
        recovered: List[PersistentTask] = []
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT task_id FROM durable_tasks WHERE status IN ('running', 'processing')"
            )
            rows = cursor.fetchall()
            for r in rows:
                tid = r["task_id"]
                conn.execute(
                    """
                    UPDATE durable_tasks
                    SET status = 'paused',
                        message = 'Interrupted by system restart. Completed clips preserved. Ready to resume.',
                        updated_at = ?
                    WHERE task_id = ?
                    """,
                    (now, tid),
                )
                logger.info(f"Auto-recovered interrupted task: {tid} -> status: paused")

            conn.commit()

        for r in rows:
            t = self.get_task(r["task_id"])
            if t:
                recovered.append(t)

        return recovered

    def create_task_workspace(
        self,
        task_id: str,
        asset_paths: List[str],
    ) -> Dict[str, str]:
        """
        Asset Ownership Isolation:
        Duplicates or links input media to a dedicated workspace folder so that
        external user deletions or movements do not corrupt running generative pipelines.
        """
        ws_dir = self.workspace_base_dir / task_id
        ws_dir.mkdir(parents=True, exist_ok=True)
        owned_assets: Dict[str, str] = {}

        for orig_path in asset_paths:
            p = Path(orig_path)
            if not p.exists():
                continue
            dest_file = ws_dir / p.name
            if not dest_file.exists():
                try:
                    # Attempt hardlink first to save space, fallback to copy
                    os.link(str(p), str(dest_file))
                except Exception:
                    shutil.copy2(str(p), str(dest_file))
            owned_assets[orig_path] = str(dest_file.resolve())

        return owned_assets

    def pre_enhance_task_prompts(
        self,
        task_id: str,
        enhancer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> bool:
        """
        Queue Pre-Enhancement:
        Executes prompt expansion / enhancement in background while GPU is busy with earlier jobs.
        """
        task = self.get_task(task_id)
        if not task:
            return False

        try:
            enhanced_payload = enhancer_fn(task.payload)
            self.update_task(
                task_id,
                payload_updates={"pre_enhanced": True, **enhanced_payload},
                message="Prompts pre-enhanced and staged for GPU execution",
            )
            return True
        except Exception as e:
            logger.warning(f"Pre-enhancement failed for task {task_id}: {e}")
            return False


# Global singleton instance
task_queue = DurableTaskQueue.get_instance()
