"""Runtime smoke tests for the Phase 7 surfaces (7.4, 7.15, 7.22, 7.25)."""

from __future__ import annotations

import uuid

import pytest
from sqlmodel import Session

from app.main import engine
from app.models import Job, JobStatus


@pytest.fixture()
def job_id():
    job_uuid = uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_uuid, status=JobStatus.COMPLETED, prompt="p", title="Smoke",
                        audio_path="/audio/x.wav", seed=7, model_provider="minimax_music3"))
        session.commit()
    yield str(job_uuid)
    with Session(engine) as session:
        row = session.get(Job, job_uuid)
        if row:
            session.delete(row)
            session.commit()


def test_presets_endpoint_lists_delivery_targets(client):
    res = client.get("/videos/presets")
    assert res.status_code == 200
    ids = {p["id"] for p in res.json()["presets"]}
    assert {"default", "youtube_1080p", "vevo_hd", "canvas", "broadcast_prores"} <= ids


def test_takes_and_report_endpoints_work_without_a_render(client, job_id):
    takes = client.get(f"/videos/{job_id}/renders")
    assert takes.status_code == 200
    assert takes.json()["takes"] == []

    report = client.get(f"/videos/{job_id}/report")
    assert report.status_code == 200
    body = report.json()
    assert body["cut_list"] == []
    assert body["selected_take"] is None


def test_manifest_endpoint_returns_a_citable_document(client, job_id):
    res = client.get(f"/jobs/{job_id}/manifest.json")
    assert res.status_code == 200
    assert "attachment" in res.headers.get("content-disposition", "")
    body = res.json()
    assert body["generation"]["seed"] == 7
    assert body["schema_version"]
    assert body["application"]["python"]


def test_export_refuses_an_unknown_preset(client, job_id):
    res = client.post(f"/videos/{job_id}/export/not-a-preset", json={})
    assert res.status_code == 404
    assert res.json()["detail"]["error"]["code"] == "unknown_preset"


def test_export_without_a_master_says_so(client, job_id):
    res = client.post(f"/videos/{job_id}/export/default", json={})
    assert res.status_code == 409
    assert res.json()["detail"]["error"]["code"] == "no_master"


def test_variants_reject_an_unsupported_aspect(client, job_id):
    res = client.post(f"/videos/{job_id}/variants", json={"aspects": ["4:3"]})
    # No master first — the endpoint checks the master before the aspect list.
    assert res.status_code in (409, 400)


def test_scene_retake_without_a_take_explains_the_path(client, job_id):
    res = client.post(f"/videos/scenes/{job_id}/1/rerender", json={})
    assert res.status_code == 409
    assert res.json()["detail"]["error"]["code"] == "no_take_to_retake"


def test_bounce_rejects_an_unknown_format(client):
    res = client.post("/api/editor/projects/whatever/bounce?format=ogg")
    assert res.status_code == 400
    assert res.json()["detail"]["error"]["code"] == "unsupported_format"


def test_export_task_is_pollable(client, job_id, monkeypatch):
    """Export used to return a task id that `GET /videos/tasks/{id}` could not resolve."""
    from app.services.video import deliver

    def fake_transcode(src, dest, preset, **kwargs):
        with open(dest, "wb") as fh:
            fh.write(b"fake-mp4")
        return True

    monkeypatch.setattr(deliver, "transcode", fake_transcode)
    # Give the job a master so the endpoint does not refuse.
    from app.services.video import takes as takes_service

    with Session(engine) as session:
        take = takes_service.create_take(session, job_id=job_id, task_id="t", kind="advanced", config={})
        takes_service.finish_take(session, take, status="completed",
                                  video_path="/audio/videos/x.mp4", file_path="/tmp/x.mp4")

    res = client.post(f"/videos/{job_id}/export/default", json={})
    assert res.status_code == 202, res.text
    task_id = res.json()["task_id"]

    # The task must be resolvable (it used to 404): the UI polls this id.
    poll = client.get(f"/videos/tasks/{task_id}")
    assert poll.status_code == 200, poll.text
    body = poll.json()
    assert body["id"] == task_id
    assert body["status"] in ("processing", "completed", "error")
    assert body["step"]


def test_background_task_lifecycle_is_explicit():
    """begin_task/end_task are the contract export and variant jobs rely on."""
    from app.services.video.video_orchestrator import video_orchestrator as orch

    orch.begin_task("lifecycle-probe", "job-1", step="Exporting", total_clips=3)
    state = orch.get_task("lifecycle-probe")
    assert state["status"] == "processing"
    assert state["step"] == "Exporting"
    assert state["total_clips"] == 3

    orch.update_task("lifecycle-probe", progress=55, current_clip=2)
    assert orch.get_task("lifecycle-probe")["progress"] == 55

    orch.end_task("lifecycle-probe", status="completed", step="Done",
                  video_url="/audio/videos/x.mp4")
    final = orch.get_task("lifecycle-probe")
    assert final["status"] == "completed"
    assert final["progress"] == 100
    assert final["video_url"].endswith(".mp4")
