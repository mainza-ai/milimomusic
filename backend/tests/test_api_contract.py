"""API contract: one error envelope, correct status codes, idempotent generation.

The audit found three incompatible error shapes on the wire and no idempotency on
money/GPU-spending POSTs (a retried request created a second job and a second GPU
run). These tests pin the corrections.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.errors import STATUS_CODES, _normalise


class _Client:
    """Minimal sync wrapper (the suite has no shared client fixture)."""

    def __init__(self, app):
        import asyncio

        import httpx

        self._asyncio = asyncio
        self.transport = httpx.ASGITransport(app=app)

    def _run(self, coro):
        loop = self._asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def request(self, method, url, **kw):
        import httpx

        async def _go():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as c:
                return await c.request(method, url, **kw)

        return self._run(_go())

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)


@pytest.fixture()
def client(monkeypatch):
    """ASGI client with the generation pipeline stubbed.

    These tests assert API contract (status codes, envelope, idempotency); letting
    the background task run would start a real model inference.
    """
    from app.main import app
    from app.services import music_service as ms

    async def _no_pipeline(*args, **kwargs):
        return None

    monkeypatch.setattr(ms.music_service, "generate_task", _no_pipeline, raising=False)
    return _Client(app)


# ── error envelope ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "detail,status,expected_code",
    [
        ("Job not found", 404, "not_found"),
        ({"error": {"code": "active_run", "message": "busy"}}, 409, "active_run"),
        ([{"loc": ("body", "prompt"), "msg": "Field required"}], 422, "validation_error"),
        ("Rate limit exceeded", 429, "rate_limited"),
    ],
)
def test_normalise_produces_a_stable_code(detail, status, expected_code):
    payload = _normalise(detail, status)
    assert payload["code"] == expected_code
    assert payload["message"]


def test_every_status_has_a_code():
    for status in (400, 401, 403, 404, 405, 409, 413, 415, 422, 429, 500, 501, 502, 503, 507):
        assert status in STATUS_CODES


def test_http_error_carries_both_detail_and_error(client):
    res = client.get(f"/jobs/{uuid.uuid4()}")
    assert res.status_code == 404
    body = res.json()
    # Backwards compatible: old clients read `detail`.
    assert body["detail"] == "Job not found"
    # Machine readable: new clients read `error.code`.
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"] == "Job not found"


def test_validation_error_carries_both_shapes(client):
    res = client.post("/generate/music", json={})
    assert res.status_code == 422
    body = res.json()
    assert isinstance(body["detail"], list)
    assert body["error"]["code"] == "validation_error"
    assert "prompt" in body["error"]["message"]


def test_nested_error_dict_is_unwrapped(client):
    res = client.post(
        f"/tracks/{uuid.uuid4()}/commit-vocal",
        json={"vocal_path": "/etc/passwd"},
    )
    assert res.status_code in (400, 404)
    body = res.json()
    assert body["error"]["code"] == "invalid_media_path"
    assert "media directories" in body["error"]["message"]


# ── status codes ───────────────────────────────────────────────────────────

def test_queued_generation_answers_202(client):
    res = client.post("/generate/music", json={
        "prompt": "idempotency probe",
        "duration_ms": 5000,
        "model_provider": "minimax_music3",
    })
    assert res.status_code == 202, res.text
    assert res.json().get("status") in ("queued", "processing", "completed")


def test_project_creation_answers_201(client):
    res = client.post("/projects", json={"name": f"Status Probe {uuid.uuid4().hex[:6]}"})
    assert res.status_code == 201, res.text


# ── idempotency ────────────────────────────────────────────────────────────

def test_generation_honours_idempotency_key(client):
    """A retried request must not create a second job (and a second GPU run)."""
    key = f"probe-{uuid.uuid4().hex}"
    payload = {
        "prompt": "idempotent synthwave",
        "duration_ms": 5000,
        "model_provider": "minimax_music3",
    }
    first = client.post("/generate/music", json=payload, headers={"Idempotency-Key": key})
    second = client.post("/generate/music", json=payload, headers={"Idempotency-Key": key})

    assert first.status_code == 202 and second.status_code == 202, first.text
    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.json().get("idempotent_replay") is True


def test_different_keys_create_different_jobs(client):
    payload = {"prompt": "distinct takes", "duration_ms": 5000, "model_provider": "minimax_music3"}
    a = client.post("/generate/music", json=payload, headers={"Idempotency-Key": f"a-{uuid.uuid4().hex}"})
    b = client.post("/generate/music", json=payload, headers={"Idempotency-Key": f"b-{uuid.uuid4().hex}"})
    assert a.json()["job_id"] != b.json()["job_id"]


# ── SSE replay + filtering ─────────────────────────────────────────────────

def test_events_carry_monotonic_ids_and_replay():
    """A reconnect with Last-Event-ID must receive the events it missed."""
    from app.services.music_service import event_manager

    event_manager.publish("job_progress", {"job_id": "sse-probe-1", "progress": 10})
    event_manager.publish("job_progress", {"job_id": "sse-probe-1", "progress": 20})
    event_manager.publish("job_progress", {"job_id": "sse-probe-2", "progress": 30})

    ring = event_manager.replay_since(0)
    assert ring, "the replay ring is empty"
    ids = [seq for seq, _, _ in ring]
    assert ids == sorted(ids), "event ids must be monotonic"

    latest = ids[-1]
    replay = event_manager.replay_since(latest)
    assert replay == [], "replaying after the newest id must be empty"

    older = event_manager.replay_since(latest - 2)
    assert len(older) == 2
    for seq, evt_name, payload in older:
        frame = event_manager.format_event(seq, evt_name, payload)
        assert frame.startswith(f"id: {seq}\n")
        assert "event: job_progress" in frame


def test_replay_ring_is_bounded():
    from app.services.music_service import event_manager

    for i in range(event_manager.RING_SIZE + 50):
        event_manager.publish("job_progress", {"job_id": "ring-probe", "i": i})
    assert len(event_manager._ring) <= event_manager.RING_SIZE


# ── rate limiting ──────────────────────────────────────────────────────────

def test_rate_limit_keys_on_route_template(monkeypatch):
    """Varying the job id must not mint a fresh bucket per request."""
    from app.core import ratelimit

    ratelimit.reset()
    monkeypatch.setenv("MILIMO_RATE_LIMIT_PER_MIN", "3")

    class _Req:
        def __init__(self, path, route_path):
            self.url = type("U", (), {"path": path})()
            self.headers = {}
            self.client = type("C", (), {"host": "127.0.0.1"})()
            self.scope = {"route": type("R", (), {"path": route_path})()}

    template = "/jobs/{job_id}/voice-convert"
    results = [
        ratelimit.check(_Req(f"/jobs/job-{i}/voice-convert", template))
        for i in range(6)
    ]
    # Cost is 3 units per call, limit 3/min: the first call fills the window.
    assert results[0] is None
    assert results[1] is not None, "a different job id reused the same bucket"
    retry_after, message = results[1]
    assert retry_after > 0 and "voice-convert" in message


def test_rate_limit_covers_the_expensive_routes():
    from app.core.ratelimit import _is_protected

    for path in (
        "/tracks/abc/separate",
        "/jobs/abc/voice-convert",
        "/videos/render-advanced/abc",
        "/api/training/jobs",
        "/api/editor/projects/p/compile",
        "/transcribe/lead-sheet",
    ):
        assert _is_protected(path), f"{path} is expensive but unprotected"


def test_rate_limit_costs_are_graded():
    from app.core.ratelimit import cost_for

    assert cost_for("/generate/music") > cost_for("/tracks/{job_id}/peaks")


# ── MIDI import (plan 3.7) ─────────────────────────────────────────────────

def test_midi_import_roundtrip(client, tmp_path):
    """A Standard MIDI File must land in the piano roll as NoteEvent dicts."""
    import pretty_midi

    # Build a tiny 2-note MIDI file.
    midi = pretty_midi.PrettyMIDI(initial_tempo=96.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=1.0))
    midi.instruments.append(inst)
    midi_path = tmp_path / "probe.mid"
    midi.write(str(midi_path))

    from sqlmodel import Session

    from app.main import engine
    from app.models import Job
    import uuid as _uuid

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="midi import probe", status="completed"))
        session.commit()

    data = midi_path.read_bytes()
    res = client.post(
        f"/tracks/{job_id}/midi/import",
        files={"file": ("probe.mid", data, "audio/midi")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] == 2
    pitches = [n["pitch"] for n in body["notes"]]
    assert pitches == [60, 64]
    assert all("instrument" in n and "velocity" in n for n in body["notes"])
    assert abs(body["bpm"] - 96.0) < 0.01


def test_midi_import_rejects_garbage(client):
    from sqlmodel import Session

    from app.main import engine
    from app.models import Job
    import uuid as _uuid

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="midi import garbage", status="completed"))
        session.commit()

    res = client.post(
        f"/tracks/{job_id}/midi/import",
        files={"file": ("bad.mid", b"not a midi file at all", "audio/midi")},
    )
    assert res.status_code in (400, 413, 415), res.text
    assert res.json()["error"]["code"] in ("invalid_midi", "bad_content", "bad_type")


# ── DAW session state (plan 3.8) ───────────────────────────────────────────

def test_workspace_session_roundtrip(client):
    """Mix/loop/tempo live on the Job, not one browser's localStorage."""
    import uuid as _uuid
    from sqlmodel import Session

    from app.main import engine
    from app.models import Job

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="session probe", status="completed"))
        session.commit()

    empty = client.get(f"/workspace/{job_id}/session")
    assert empty.status_code == 200
    assert empty.json()["state"] is None

    state = {
        "version": 1,
        "masterVolume": 0.8,
        "stemSource": "neural",
        "loopStart": 4.0,
        "loopEnd": 12.0,
        "tempoMap": [{"t": 0, "bpm": 96, "sig": 4}, {"t": 32, "bpm": 120, "sig": 3}],
        "metronomeEnabled": True,
        "countInBars": 2,
    }
    saved = client.request("PUT", f"/workspace/{job_id}/session", json={"state": state})
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "saved"

    loaded = client.get(f"/workspace/{job_id}/session").json()["state"]
    assert loaded["tempoMap"][1]["bpm"] == 120
    assert loaded["metronomeEnabled"] is True
    assert loaded["loopEnd"] == 12.0


def test_workspace_session_rejects_non_object(client):
    import uuid as _uuid
    from sqlmodel import Session

    from app.main import engine
    from app.models import Job

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="session probe 2", status="completed"))
        session.commit()

    res = client.request("PUT", f"/workspace/{job_id}/session", json={"state": "not-an-object"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "invalid_session_state"


# ── video storyboard persistence (plan 4.3) ────────────────────────────────

def _video_job():
    import uuid as _uuid
    from sqlmodel import Session

    from app.main import engine
    from app.models import Job

    job_id = _uuid.uuid4()
    with Session(engine) as session:
        session.add(Job(id=job_id, prompt="storyboard probe", status="completed"))
        session.commit()
    return job_id


def test_storyboard_roundtrip_and_gap_report(client):
    job_id = _video_job()
    clips = [
        {"start_time": 0.0, "duration": 4.0, "prompt": "opening wide"},
        # Deliberate 2s gap: the API must report it rather than silently accept.
        {"start_time": 6.0, "duration": 3.0, "prompt": "chorus push-in"},
    ]
    saved = client.request("PATCH", f"/videos/{job_id}/storyboard", json={"clips": clips})
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["count"] == 2
    assert body["clips"][0]["clip_index"] == 1
    assert body["timeline_gaps"], "an obvious gap must be reported"

    loaded = client.get(f"/videos/{job_id}/storyboard").json()
    assert [c["prompt"] for c in loaded["clips"]] == ["opening wide", "chorus push-in"]


def test_storyboard_rejects_a_clip_without_duration(client):
    job_id = _video_job()
    res = client.request("PATCH", f"/videos/{job_id}/storyboard",
                         json={"clips": [{"start_time": 0, "duration": 0}]})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "invalid_storyboard"


def test_storyboard_rejects_empty_payload(client):
    job_id = _video_job()
    res = client.request("PATCH", f"/videos/{job_id}/storyboard", json={"clips": []})
    assert res.status_code == 422
