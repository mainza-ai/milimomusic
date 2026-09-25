"""
Unit and integration tests for Phase 5: Durable Task Queue, Interrupted Recovery, and Asset Ownership.
"""

import tempfile
from pathlib import Path
import pytest

from app.core.task_queue import DurableTaskQueue, PersistentTask


def test_durable_task_queue_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_tasks.db"
        queue = DurableTaskQueue(db_path=str(db_file))

        # Enqueue task
        task = queue.enqueue_task(
            task_id="task_test_01",
            task_type="video_generation",
            payload={"prompt": "Cyberpunk stage", "clips": 5},
            initial_status="queued",
            initial_message="Task staged in queue",
        )
        assert task.task_id == "task_test_01"
        assert task.status == "queued"

        # Read back
        fetched = queue.get_task("task_test_01")
        assert fetched is not None
        assert fetched.payload["clips"] == 5

        # Update progress
        updated = queue.update_task(
            task_id="task_test_01",
            status="running",
            progress=40,
            message="Rendering clip 2/5",
        )
        assert updated is not None
        assert updated.status == "running"
        assert updated.progress == 40

        # Complete
        completed = queue.update_task(
            task_id="task_test_01",
            status="completed",
            progress=100,
            result={"video_path": "output.mp4"},
        )
        assert completed.status == "completed"
        assert completed.result["video_path"] == "output.mp4"


def test_interrupted_task_recovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_recovery.db"
        queue = DurableTaskQueue(db_path=str(db_file))

        # Simulate two tasks running when server crashes/restarts
        queue.enqueue_task("task_crash_01", "video_generation", {"clip": 3}, initial_status="running")
        queue.enqueue_task("task_crash_02", "audio_analysis", {}, initial_status="processing")
        queue.enqueue_task("task_done_01", "video_generation", {}, initial_status="completed")

        # Call recovery routine
        recovered = queue.recover_interrupted_tasks()
        assert len(recovered) == 2

        t1 = queue.get_task("task_crash_01")
        assert t1.status == "paused"
        assert "Interrupted" in t1.message

        t2 = queue.get_task("task_crash_02")
        assert t2.status == "paused"

        td = queue.get_task("task_done_01")
        assert td.status == "completed"


def test_asset_ownership_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_asset.db"
        queue = DurableTaskQueue(db_path=str(db_file))

        # Create dummy assets
        audio_src = Path(tmpdir) / "song.wav"
        audio_src.write_text("DUMMY_AUDIO_DATA")
        img_src = Path(tmpdir) / "ref.png"
        img_src.write_text("DUMMY_IMAGE_DATA")

        owned = queue.create_task_workspace("task_ws_01", [str(audio_src), str(img_src)])
        assert len(owned) == 2
        for orig, owned_path in owned.items():
            assert Path(owned_path).exists()
            assert "task_workspaces/task_ws_01" in owned_path.replace("\\", "/")


def test_queue_pre_enhancement():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_pre_enhance.db"
        queue = DurableTaskQueue(db_path=str(db_file))

        queue.enqueue_task(
            "task_enh_01",
            "video_generation",
            {"prompt": "sunset stage"},
            initial_status="queued",
        )

        def mock_enhancer(payload):
            return {"prompt": f"{payload['prompt']} | 8k cinematic lighting, volumetric haze"}

        success = queue.pre_enhance_task_prompts("task_enh_01", mock_enhancer)
        assert success

        task = queue.get_task("task_enh_01")
        assert task.payload.get("pre_enhanced") is True
        assert "volumetric haze" in task.payload.get("prompt")
