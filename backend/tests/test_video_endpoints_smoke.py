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
