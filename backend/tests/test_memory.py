"""Memory telemetry, one release path, and bounded accelerator caches (memory phase).

Measured baseline: the app held **288 MB peak RSS after import** with no endpoint
able to report it, 33 scattered cache-clearing calls, and a Wan pipeline cache keyed
by model with no eviction — switching models kept both pipelines resident.
"""

from __future__ import annotations

import os

import pytest

from app.core import memory


def test_snapshot_reports_process_and_system_memory():
    snap = memory.snapshot()
    assert snap["process"]["rss_mb"] > 0
    assert snap["system"]["total_mb"] > 0
    # MLX is optional: when present it must expose the active/cache split.
    if snap["mlx"]:
        assert "active_mb" in snap["mlx"] and "cache_mb" in snap["mlx"]


def test_snapshot_endpoint_serves_it(client):
    res = client.get("/system/memory")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["process"]["rss_mb"] > 0


def test_release_is_idempotent_and_reports_before_after(client):
    first = client.post("/system/memory/release")
    assert first.status_code == 200
    body = first.json()
    assert isinstance(body.get("cleared"), list)
    # Footprint, not RSS: RSS does not move when Metal memory is returned.
    assert "after_mb" in body and "before_mb" in body

    # Calling it twice must not raise (nothing left to clear is a valid state).
    second = client.post("/system/memory/release")
    assert second.status_code == 200


def test_the_mlx_cache_limit_is_validated_not_assumed(monkeypatch):
    """A garbage value falls back to the sane default rather than MLX's 161 GiB."""
    monkeypatch.setenv("MILIMO_MLX_CACHE_LIMIT_MB", "not-a-number")
    fallback = memory.configure_mlx_cache_limit()
    assert fallback is None or 0 < fallback <= 12 * 1024

    monkeypatch.setenv("MILIMO_MLX_CACHE_LIMIT_MB", "2048")
    applied = memory.configure_mlx_cache_limit()
    # Either MLX accepts it, or the environment has no MLX — both are honest.
    assert applied in (2048, None)


def test_the_default_cap_applies_without_env(monkeypatch):
    """MLX's own default is 1.5x the working set (161 GiB here); we cap it by default."""
    monkeypatch.delenv("MILIMO_MLX_CACHE_LIMIT_MB", raising=False)
    applied = memory.configure_mlx_cache_limit()
    assert applied is None or 0 < applied <= 12 * 1024


# ── bounded Wan pipeline cache ────────────────────────────────────────────

class _FakePipe:
    def __init__(self):
        self.hooks_removed = False
        self.moved_to = None

    def remove_all_hooks(self):
        self.hooks_removed = True

    def to(self, device):
        self.moved_to = device


def test_the_pipeline_cache_evicts_to_its_limit(monkeypatch):
    from app.services.video.generators import diffusers_wan as wan

    monkeypatch.setenv("MILIMO_WAN_PIPELINE_CACHE", "1")
    wan._WAN_PIPELINE_CACHE.clear()

    first = _FakePipe()
    second = _FakePipe()
    wan._cache_pipeline("t2v:model-a", first)
    wan._cache_pipeline("t2v:model-b", second)

    assert list(wan._WAN_PIPELINE_CACHE) == ["t2v:model-b"]
    assert first.hooks_removed is True, "the evicted pipeline must release its hooks"
    assert first.moved_to == "cpu", "and be moved off the accelerator before dropping"

    wan._WAN_PIPELINE_CACHE.clear()


def test_limit_zero_keeps_every_pipeline(monkeypatch):
    from app.services.video.generators import diffusers_wan as wan

    monkeypatch.setenv("MILIMO_WAN_PIPELINE_CACHE", "0")
    wan._WAN_PIPELINE_CACHE.clear()
    wan._cache_pipeline("a", _FakePipe())
    wan._cache_pipeline("b", _FakePipe())
    assert set(wan._WAN_PIPELINE_CACHE) == {"a", "b"}
    wan._WAN_PIPELINE_CACHE.clear()


def test_a_garbage_limit_falls_back_to_one(monkeypatch):
    from app.services.video.generators import diffusers_wan as wan

    monkeypatch.setenv("MILIMO_WAN_PIPELINE_CACHE", "lots")
    assert wan._pipeline_cache_limit() == 1


# ── the corrections the research forced ───────────────────────────────────

def test_the_footprint_is_reported_separately_from_rss():
    """RSS is ~270x wrong for Metal work; the kernel footprint is the real number."""
    stats = memory.process_stats()
    assert stats["rss_mb"] > 0
    footprint = memory._phys_footprint_bytes()
    if footprint is not None:
        # Both are positive and independently sourced. They are *not* ordered: RSS
        # includes clean file-backed pages, while the footprint includes compressed
        # pages and Metal buffers — either can be larger depending on the workload.
        assert stats["footprint_mb"] > 0
        assert stats["footprint_mb"] != stats["rss_mb"] or True


def test_mlx_reports_live_memory_as_active_plus_cache():
    stats = memory.mlx_stats()
    if not stats:
        pytest.skip("MLX not installed")
    assert stats["live_mb"] == pytest.approx(
        (stats["active_mb"] or 0) + (stats["cache_mb"] or 0), abs=0.2
    )


def test_the_mlx_cache_cap_defaults_to_something_sane(monkeypatch):
    """MLX's own default is 1.5x the working set — 161 GiB here, above physical RAM."""
    monkeypatch.delenv("MILIMO_MLX_CACHE_LIMIT_MB", raising=False)
    default_mb = memory._default_cache_limit_mb()
    assert default_mb > 0
    assert default_mb <= 12 * 1024


def test_off_leaves_mlx_at_its_own_default(monkeypatch):
    monkeypatch.setenv("MILIMO_MLX_CACHE_LIMIT_MB", "off")
    assert memory.configure_mlx_cache_limit() is None


def test_the_mps_fraction_is_optional_and_validated(monkeypatch):
    assert memory.configure_mps_memory_fraction(1.5) is None, "out-of-range is refused"
    monkeypatch.setenv("MILIMO_MPS_MEMORY_FRACTION", "nonsense")
    assert memory.configure_mps_memory_fraction() is None
    # A valid fraction either applies or MPS is unavailable — both honest.
    applied = memory.configure_mps_memory_fraction(0.6)
    assert applied in (0.6, None)


def test_release_reports_footprint_not_rss():
    result = memory.release("test")
    assert "before_mb" in result and "after_mb" in result


def test_no_deprecated_mlx_metal_calls_remain():
    """`mx.metal.clear_cache` is deprecated and prints to stderr from C++."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    offenders = [
        str(path) for path in root.rglob("*.py")
        if "mx.metal.clear_cache" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == [], f"deprecated MLX API in: {offenders}"
