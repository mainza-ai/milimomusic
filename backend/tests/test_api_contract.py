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
