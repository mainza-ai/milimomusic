"""Encoder choice by measurement, not by name (plan 9.5).

The bug this pins: `detect_hardware_encoder()` preferred VideoToolbox because it
*exists*. Measured, it is slower than x264 on the reference machine — so the choice
must come from numbers, be cached per machine, and be overridable.
"""

from __future__ import annotations

import json
import os

import pytest

from app.services.video import encoder_bench as eb


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "encoder_choice.json"
    monkeypatch.setattr(eb, "_cache_path", lambda: str(cache))
    monkeypatch.delenv("MILIMO_VIDEO_ENCODER", raising=False)
    return cache


def _fake_timings(monkeypatch, timings, available=("x264", "videotoolbox")):
    monkeypatch.setattr(eb, "_encoder_available", lambda eid, binary="ffmpeg": eid in available)
    monkeypatch.setattr(
        eb, "_time_encode",
        lambda eid, workdir, binary="ffmpeg": timings.get(eid),
    )
    monkeypatch.setattr(eb, "machine_fingerprint", lambda: "test-machine|arm64|10|ffmpeg 9")


def test_the_faster_encoder_wins(monkeypatch):
    _fake_timings(monkeypatch, {"x264": 0.21, "videotoolbox": 0.31})
    record = eb.benchmark(force=True)
    assert record["winner"] == "x264"
    assert record["results"]["videotoolbox"] == 0.31


def test_a_faster_hardware_encoder_would_win(monkeypatch):
    _fake_timings(monkeypatch, {"x264": 0.40, "videotoolbox": 0.12})
    assert eb.benchmark(force=True)["winner"] == "videotoolbox"


def test_the_choice_is_cached_and_not_re_measured(monkeypatch):
    _fake_timings(monkeypatch, {"x264": 0.10, "videotoolbox": 0.20})
    first = eb.benchmark(force=True)
    assert first["winner"] == "x264"

    calls = {"n": 0}
    monkeypatch.setattr(eb, "_time_encode", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or 9.9)
    second = eb.benchmark()
    assert calls["n"] == 0, "a valid cache must not re-run the benchmark"
    assert second["winner"] == "x264"


def test_a_stale_cache_for_another_machine_is_re_measured(monkeypatch):
    _fake_timings(monkeypatch, {"x264": 0.10, "videotoolbox": 0.20})
    eb.benchmark(force=True)

    monkeypatch.setattr(eb, "machine_fingerprint", lambda: "different-machine|arm64|10|ffmpeg 9")
    monkeypatch.setattr(eb, "_time_encode", lambda eid, workdir, binary="ffmpeg": {"x264": 0.5, "videotoolbox": 0.1}[eid])
    assert eb.benchmark()["winner"] == "videotoolbox"


def test_unavailable_encoders_are_skipped(monkeypatch):
    _fake_timings(monkeypatch, {"x264": 0.2}, available=("x264",))
    record = eb.benchmark(force=True)
    assert record["winner"] == "x264"
    assert "videotoolbox" not in record["results"]


def test_no_measurements_still_yields_x264(monkeypatch):
    _fake_timings(monkeypatch, {}, available=())
    assert eb.benchmark(force=True)["winner"] == "x264"


def test_the_override_skips_measurement(monkeypatch):
    monkeypatch.setenv("MILIMO_VIDEO_ENCODER", "videotoolbox")
    choice = eb.chosen_encoder()
    assert choice["encoder"] == "videotoolbox"
    assert "override" in choice["reason"].lower()


def test_an_unknown_override_is_ignored(monkeypatch):
    monkeypatch.setenv("MILIMO_VIDEO_ENCODER", "quantum-encoder")
    assert eb.override() is None


def test_detection_uses_the_measured_choice(monkeypatch):
    from app.services.video import subtitle_styles as ss

    monkeypatch.setattr(ss, "_DETECTED_ENCODER", None)
    monkeypatch.setattr(eb, "chosen_encoder", lambda force=False: {
        "encoder": "videotoolbox", "reason": "measured", "results": {"videotoolbox": 0.1},
    })
    encoder, flags = ss.detect_hardware_encoder()
    assert encoder == "h264_videotoolbox"
    # `-q:v 65` is the quality-matched setting, not a fixed bitrate.
    assert "-q:v" in flags


def test_the_benchmark_record_is_json_serialisable(monkeypatch, isolated_cache):
    _fake_timings(monkeypatch, {"x264": 0.2, "videotoolbox": 0.3})
    eb.benchmark(force=True)
    data = json.loads(isolated_cache.read_text())
    assert data["winner"] == "x264"
    assert data["bench_size"] == eb.BENCH_SIZE
    assert os.path.isfile(str(isolated_cache))
