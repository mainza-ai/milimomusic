"""Real timeline rendering (plan 4.4).

`POST /api/editor/projects/{id}/compile` used to return an FFmpeg command string
and an output path that was never written. These tests prove a project actually
renders to a playable file, that clip transforms reach the filter graph, and that
the task reaches a terminal state with an honest error when it cannot.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from app.services.timeline.editor_projects import (
    ClipTransform,
    EditorProject,
    EditorProjectManager,
    TimelineClip,
    TimelineTrack,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_asset(path: str, seconds: float = 2.0, size: str = "320x240") -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={size}:rate=25:duration={seconds}",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", path],
        check=True, capture_output=True, timeout=120,
    )
    return path


def _project(asset: str, transform: ClipTransform | None = None) -> EditorProject:
    clip = TimelineClip(
        clip_id="clip-1",
        asset_path=asset,
        start_time=0.0,
        duration=2.0,
        transform=transform or ClipTransform(),
    )
    track = TimelineTrack(track_id="v1", track_type="video", name="Video 1", clips=[clip])
    return EditorProject(
        project_id="proj-test",
        title="Render Test",
        duration=2.0,
        resolution=(640, 360),
        fps=25,
        tracks=[track],
    )


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_command_applies_clip_transform(tmp_path):
    """Scale / offset / opacity must change the filter graph, not be decorative."""
    asset = _make_asset(str(tmp_path / "a.mp4"))
    project = _project(asset, ClipTransform(scale=0.5, x_offset=40, y_offset=-20, opacity=0.6))

    cmd = EditorProjectManager.compile_editor_render_command(project, str(tmp_path / "out.mp4"))
    graph = cmd[cmd.index("-filter_complex") + 1]

    assert "iw*0.5000" in graph, "clip scale missing from the filter graph"
    assert "colorchannelmixer=aa=0.600" in graph, "opacity missing from the filter graph"
    assert "x=(W-w)/2+40.00" in graph, "x offset missing from the overlay"
    assert "y=(H-h)/2+-20.00" in graph, "y offset missing from the overlay"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_render_produces_a_playable_file(tmp_path, monkeypatch):
    from app.core import paths as paths_mod
    from app.services.timeline import render as render_mod

    asset = _make_asset(str(tmp_path / "a.mp4"))
    project = _project(asset)

    # Keep the render output inside the test sandbox.
    monkeypatch.setattr(paths_mod, "get_generated_audio_dir", lambda: tmp_path, raising=False)
    monkeypatch.setattr(render_mod, "_project_file", lambda pid: _write_project(tmp_path, project))

    renderer = render_mod.TimelineRenderer()
    task = renderer.start_render(project)

    deadline = time.time() + 180
    while task.status in ("queued", "rendering") and time.time() < deadline:
        time.sleep(0.3)

    assert task.status == "completed", f"render did not complete: {task.status} {task.error}"
    assert task.video_url and task.video_url.endswith(".mp4")
    out = tmp_path / render_mod.VIDEO_OUT_DIR_NAME / os.path.basename(task.output_path or "")
    assert out.is_file() and out.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) > 1.0


def _write_project(tmp_path, project):
    import json

    d = tmp_path / "editor_projects"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{project.project_id}.json"
    f.write_text(json.dumps(project.to_dict()))
    return f


def test_render_reports_missing_project(tmp_path, monkeypatch):
    from app.services.timeline import render as render_mod

    monkeypatch.setattr(render_mod, "_project_file", lambda pid: None)
    renderer = render_mod.TimelineRenderer()
    task = renderer.start_render(_project("/tmp/whatever.mp4"))

    deadline = time.time() + 30
    while task.status in ("queued", "rendering") and time.time() < deadline:
        time.sleep(0.1)

    assert task.status == "failed"
    assert "not found" in (task.error or "").lower()


def test_cancel_before_start_is_honest(tmp_path, monkeypatch):
    from app.services.timeline import render as render_mod

    renderer = render_mod.TimelineRenderer()
    # Occupy the worker so the second task stays queued.
    blocker = renderer.start_render(_project("/tmp/none.mp4"))
    second = renderer.start_render(_project("/tmp/none2.mp4"))
    assert renderer.cancel_render(second.task_id) is True
    assert second.status == "cancelled"

    deadline = time.time() + 30
    while blocker.status in ("queued", "rendering") and time.time() < deadline:
        time.sleep(0.1)
