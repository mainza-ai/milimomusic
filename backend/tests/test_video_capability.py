"""Capability matrix: what this machine really renders (plan 8.10)."""

from __future__ import annotations

import pytest

from app.services.video import capability as cap


def test_mps_clamps_wan_class_engines(monkeypatch):
    from app.services.video.model_specs import VIDEO_MODEL_SPECS

    wan = VIDEO_MODEL_SPECS["wan_14b"]
    entry = cap._effective_for_spec(wan, "mps")
    assert entry["clamped"] is True
    assert entry["effective"]["width"] <= cap.MPS_MAX_W
    assert entry["effective"]["height"] <= cap.MPS_MAX_H
    assert entry["effective"]["max_frames"] <= cap.MPS_MAX_FRAMES
    assert any("clamps" in n for n in entry["notes"])


def test_cuda_keeps_the_model_native_limits():
    from app.services.video.model_specs import VIDEO_MODEL_SPECS

    wan = VIDEO_MODEL_SPECS["wan_14b"]
    entry = cap._effective_for_spec(wan, "cuda")
    assert entry["clamped"] is False
    assert entry["effective"]["width"] == entry["native"]["width"]


def test_unimplemented_engines_say_so():
    from app.services.video.model_specs import VIDEO_MODEL_SPECS

    entry = cap._effective_for_spec(VIDEO_MODEL_SPECS["cogvideox"], "cuda")
    assert entry["implemented"] is False
    assert entry["notes"], "an unimplemented engine must explain itself"


def test_interpolation_backend_is_detected_not_assumed(monkeypatch):
    monkeypatch.delenv("MILIMO_RIFE_CMD", raising=False)
    info = cap.interpolation_backend()
    assert info["active"] == "minterpolate"
    assert info["rife_available"] is False
    assert "RIFE" in info["note"]

    monkeypatch.setenv("MILIMO_RIFE_CMD", "/usr/bin/true")
    assert cap.interpolation_backend()["active"] == "rife"


def test_matrix_covers_every_spec_and_reports_the_device():
    report = cap.matrix()
    from app.services.video.model_specs import VIDEO_MODEL_SPECS

    assert report["device"] in ("mps", "cuda", "cpu")
    assert len(report["engines"]) == len(VIDEO_MODEL_SPECS)
    assert report["interpolation"]["active"] in ("rife", "minterpolate")


def test_describe_returns_one_engine_or_none():
    assert cap.describe("wan_14b")["engine"] == "wan_14b"
    assert cap.describe("does-not-exist") is None


def test_endpoint_serves_the_matrix(client):
    res = client.get("/videos/capability-matrix")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["device"]
    assert any(e["engine"] == "wan_14b" for e in body["engines"])
