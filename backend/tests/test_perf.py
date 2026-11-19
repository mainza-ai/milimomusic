"""Performance layer: response caching, ETags and the model-tree memo (perf phase).

The numbers that motivated these tests: `get_model_tree()` costs ~1.1 s cold and
three read endpoints call it; the API was entirely uncompressed (which turned out to
be *correct* on loopback) and had no conditional-GET support at all.
"""

from __future__ import annotations

import time

import pytest

from app.core import perf


# ── ETag / 304 ────────────────────────────────────────────────────────────

def test_cached_json_sets_a_private_etag():
    resp = perf.cached_json(None, {"a": 1}, token="tok")
    assert resp.headers["etag"] == '"tok"'
    assert "private" in resp.headers["cache-control"]


def test_matching_if_none_match_returns_304(client):
    first = client.get("/videos/presets")
    assert first.status_code == 200
    etag = first.headers["etag"]

    second = client.get("/videos/presets", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.headers["etag"] == etag
    assert second.content == b""


def test_a_stale_if_none_match_gets_the_body(client):
    resp = client.get("/videos/presets", headers={"If-None-Match": '"not-the-tag"'})
    assert resp.status_code == 200
    assert resp.json()["presets"]


def test_expensively_derived_payloads_are_cacheable(client):
    for route in ("/models/tree", "/videos/models", "/system/licences",
                  "/videos/capability-matrix"):
        first = client.get(route)
        assert first.status_code == 200, route
        etag = first.headers.get("etag")
        assert etag, f"{route} must expose an ETag"
        again = client.get(route, headers={"If-None-Match": etag})
        assert again.status_code == 304, route


# ── model tree memo ───────────────────────────────────────────────────────

def test_the_tree_is_memoised_and_invalidated():
    from app.services import model_manager as mm

    mm.invalidate_model_tree()
    first = mm.model_manager.get_model_tree()
    t0 = time.perf_counter()
    second = mm.model_manager.get_model_tree()
    warm_ms = (time.perf_counter() - t0) * 1000
    assert second is first, "the memoised tree must be returned by identity"
    assert warm_ms < 50, f"warm read took {warm_ms:.0f} ms"

    before = mm.model_tree_generation()
    mm.invalidate_model_tree()
    assert mm.model_tree_generation() > before
    assert mm.model_manager.get_model_tree() is not first


def test_the_ttl_is_configurable(monkeypatch):
    from app.services import model_manager as mm

    monkeypatch.setenv("MILIMO_MODEL_TREE_TTL_SECONDS", "0")
    mm.invalidate_model_tree()
    first = mm.model_manager.get_model_tree()
    # TTL 0 means the next call rebuilds (fresh object, same content).
    second = mm.model_manager.get_model_tree()
    assert second is not first
    assert [m["id"] for m in second] == [m["id"] for m in first]


# ── compression is opt-in ─────────────────────────────────────────────────

def test_compression_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MILIMO_COMPRESS", raising=False)
    assert perf.compression_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_compression_can_be_enabled_for_lan_clients(monkeypatch, value):
    monkeypatch.setenv("MILIMO_COMPRESS", value)
    assert perf.compression_enabled() is True


def test_streaming_paths_are_excluded_from_compression():
    assert "/events" in perf.STREAMING_PREFIXES
    assert "/audio/" in perf.STREAMING_PREFIXES


def test_the_running_app_does_not_compress_by_default(client):
    """On loopback compression is a net loss; the default must be no encoding."""
    resp = client.get("/models/tree", headers={"Accept-Encoding": "gzip, br"})
    assert resp.headers.get("content-encoding") is None


# ── FFmpeg thread budget ──────────────────────────────────────────────────

def test_encodes_are_thread_capped_by_default():
    """x264 defaults to ~1.5× cores and starves the editor during a render."""
    from app.services.video.export_presets import encoder_threads, thread_args

    assert encoder_threads() == 4
    assert thread_args() == ["-threads", "4"]


def test_the_thread_cap_is_configurable_and_validated(monkeypatch):
    from app.services.video.export_presets import encoder_threads, thread_args

    monkeypatch.setenv("MILIMO_FFMPEG_THREADS", "2")
    assert thread_args() == ["-threads", "2"]

    monkeypatch.setenv("MILIMO_FFMPEG_THREADS", "0")
    assert thread_args() == [], "0 disables the cap"

    monkeypatch.setenv("MILIMO_FFMPEG_THREADS", "999")
    assert encoder_threads() == 32, "absurd values are clamped"
    monkeypatch.setenv("MILIMO_FFMPEG_THREADS", "nonsense")
    assert encoder_threads() == 4


def test_delivery_and_normalise_commands_carry_the_cap(tmp_path, monkeypatch):
    import subprocess

    from app.services.video import deliver
    from app.services.video.clip_normalise import normalise_clip
    from app.services.video.export_presets import get_preset

    src = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24:duration=1",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True, timeout=120,
    )
    monkeypatch.setenv("MILIMO_FFMPEG_THREADS", "2")

    # Both paths must succeed with the cap in place.
    assert normalise_clip(str(src), str(tmp_path / "norm.mp4"),
                          duration_s=1.0, width=160, height=120, fps=24) is True
    assert deliver.transcode(str(src), str(tmp_path / "out.mp4"),
                             get_preset("default")) is True
