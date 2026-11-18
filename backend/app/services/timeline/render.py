"""
Real non-destructive timeline rendering.

`POST /api/editor/projects/{id}/compile` used to return the FFmpeg *command string*
it would have run plus an `output_path` that was never written — so "Export"
produced JSON and reported success. This module runs the compile for real:

    start_render(project)  -> task_id            (returns immediately)
    get_status(task_id)    -> {status, progress, video_url, error}
    cancel_render(task_id) -> bool

Renders run one at a time in a worker thread, stream progress from FFmpeg's
`-progress` pipe, and publish the finished file under `generated_audio/videos/`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("milimo.timeline.render")

VIDEO_OUT_DIR_NAME = "editor_renders"
MAX_CONCURRENT_RENDERS = 1


@dataclass
class RenderTask:
    task_id: str
    project_id: str
    status: str = "queued"          # queued | rendering | completed | failed | cancelled
    progress: int = 0
    message: str = ""
    output_path: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "video_url": self.video_url,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TimelineRenderer:
    """Serialised render queue for editor projects."""

    _instance: Optional["TimelineRenderer"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._tasks: Dict[str, RenderTask] = {}
        self._queue: List[str] = []
        self._worker: Optional[threading.Thread] = None
        self._active: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "TimelineRenderer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── public API ──────────────────────────────────────────────────────────
    def start_render(self, project) -> RenderTask:
        task = RenderTask(
            task_id=f"render_{uuid.uuid4().hex[:10]}",
            project_id=str(getattr(project, "project_id", "") or ""),
            status="queued",
            message="Queued for render.",
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._queue.append(task.task_id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_loop, daemon=True)
                self._worker.start()
        logger.info("Timeline render %s queued for project %s", task.task_id, task.project_id)
        return task

    def start_bounce(self, project, audio_format: str = "wav") -> RenderTask:
        """Queue an audio-only mixdown of the project (plan 7.25)."""
        task = RenderTask(
            task_id=f"bounce_{uuid.uuid4().hex[:10]}",
            project_id=str(getattr(project, "project_id", "") or ""),
            status="queued",
            message=f"Queued for {audio_format.upper()} bounce.",
        )
        task.audio_format = audio_format  # type: ignore[attr-defined]
        with self._lock:
            self._tasks[task.task_id] = task
            self._queue.append(task.task_id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_loop, daemon=True)
                self._worker.start()
        return task

    def get_status(self, task_id: str) -> Optional[RenderTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_render(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status in ("completed", "failed", "cancelled"):
                return False
            task._cancel.set()
            if task.status == "queued":
                task.status = "cancelled"
                task.message = "Cancelled before rendering."
                try:
                    self._queue.remove(task_id)
                except ValueError:
                    pass
            return True

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    # ── internals ───────────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                task_id = self._queue.pop(0)
                task = self._tasks[task_id]
                self._active = task_id
            try:
                # Bounce tasks target an audio file; render tasks target an mp4.
                if getattr(task, "audio_format", None):
                    self._bounce(task)
                else:
                    self._render(task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Timeline render %s crashed", task_id)
                task.status = "failed"
                task.error = str(exc)[:500]
                task.message = "Render failed."
            finally:
                with self._lock:
                    self._active = None
                task.updated_at = datetime.now(timezone.utc).isoformat()

    def _bounce(self, task: RenderTask) -> None:
        """Render the timeline to an audio file instead of a video."""
        from app.core.paths import get_generated_audio_dir
        from app.services.timeline.editor_projects import (
            EditorProject,
            EditorProjectManager,
            safe_load_json,
        )

        project_file = _project_file(task.project_id)
        if project_file is None or not project_file.exists():
            task.status = "failed"
            task.error = "Editor project not found."
            return

        project = EditorProject.from_dict(safe_load_json(project_file))
        fmt = getattr(task, "audio_format", "wav")
        out_dir = get_generated_audio_dir() / "bounces"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{task.project_id}_{task.task_id}.{fmt}"
        out_path = out_dir / out_name
        cmd = EditorProjectManager.compile_audio_bounce_command(project, str(out_path), fmt)

        task.status = "rendering"
        task.message = "Bouncing mixdown…"
        ok = self._run_ffmpeg(cmd, task, out_path, float(getattr(project, "duration", 0) or 0) or None)
        if ok and out_path.is_file() and out_path.stat().st_size > 0:
            task.status = "completed"
            task.progress = 100
            task.message = "Bounce complete."
            task.output_path = str(out_path)
            task.video_url = None
        else:
            task.status = "failed"
            task.error = task.error or "FFmpeg produced no output."

    def _render(self, task: RenderTask) -> None:
        from app.core.paths import get_generated_audio_dir
        from app.services.timeline.editor_projects import (
            EditorProjectManager,
            safe_load_json,
        )

        project_file = _project_file(task.project_id)
        if project_file is None or not project_file.exists():
            task.status = "failed"
            task.error = "Editor project not found."
            return

        from app.services.timeline.editor_projects import EditorProject

        project = EditorProject.from_dict(safe_load_json(project_file))

        out_dir = get_generated_audio_dir() / VIDEO_OUT_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{task.project_id}_{task.task_id}.mp4"
        out_path = out_dir / out_name

        cmd = EditorProjectManager.compile_editor_render_command(project, str(out_path))
        if task._cancel.is_set():
            task.status = "cancelled"
            task.message = "Cancelled."
            return

        task.status = "rendering"
        task.progress = 5
        task.message = "Rendering timeline…"

        total_duration = float(getattr(project, "duration", 0) or 0) or None
        ok = self._run_ffmpeg(cmd, task, out_path, total_duration)

        if task._cancel.is_set():
            task.status = "cancelled"
            task.message = "Cancelled."
            _safe_unlink(out_path)
            return

        if ok and out_path.is_file() and out_path.stat().st_size > 0:
            task.status = "completed"
            task.progress = 100
            task.message = "Render complete."
            task.output_path = str(out_path)
            task.video_url = f"/audio/videos/{VIDEO_OUT_DIR_NAME}/{out_name}"
        else:
            task.status = "failed"
            task.error = task.error or "FFmpeg produced no output."
            task.message = "Render failed."

    def _run_ffmpeg(self, cmd: List[str], task: RenderTask, out_path, total_duration) -> bool:
        """Run FFmpeg and translate its `-progress` stream into a percentage."""
        if not cmd or cmd[0] != "ffmpeg":
            task.error = "Invalid render command."
            return False
        if shutil.which("ffmpeg") is None:
            task.error = "FFmpeg is not installed."
            return False

        # Ask FFmpeg for machine-readable progress on stdout.
        cmd = list(cmd)
        insert_at = cmd.index("-y") + 1 if "-y" in cmd else 1
        cmd[insert_at:insert_at] = ["-progress", "pipe:1", "-nostats", "-loglevel", "error"]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
        except OSError as exc:
            task.error = f"Could not start FFmpeg: {exc}"
            return False

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if task._cancel.is_set():
                    proc.terminate()
                    break
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key == "out_time_ms" and total_duration:
                    try:
                        seconds = int(value) / 1_000_000
                        pct = min(99, max(5, int(5 + 90 * seconds / total_duration)))
                        task.progress = pct
                        task.message = f"Rendering timeline… {pct}%"
                    except ValueError:
                        continue
                elif key == "progress" and value == "end":
                    task.progress = 99
            stderr = (proc.stderr.read() if proc.stderr else "") or ""
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()
            task.error = "FFmpeg timed out."
            return False
        except Exception as exc:  # pragma: no cover - defensive
            proc.kill()
            task.error = str(exc)[:300]
            return False

        if proc.returncode != 0:
            task.error = stderr.strip()[:500] or f"FFmpeg exited with {proc.returncode}"
            return False
        return True


def _project_file(project_id: str):
    """Locate a saved editor project JSON (containment: id must be a plain name)."""
    from app.core.paths import get_data_dir

    if not project_id or any(part in project_id for part in ("/", "\\", "..")):
        return None
    for base in (
        get_data_dir() / "editor_projects",
        get_data_dir().parent / "data" / "editor_projects",
        __import__("pathlib").Path("data/editor_projects"),
    ):
        candidate = base / f"{project_id}.json"
        if candidate.is_file():
            return candidate
    return None


def _safe_unlink(path) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


timeline_renderer = TimelineRenderer.get_instance()
