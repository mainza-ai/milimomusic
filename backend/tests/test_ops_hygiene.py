"""Ops hygiene: health counters, WAL checkpoint, memory trend, budget, export cancel.

These are the Phase 9/10 remainders that need no model weights (plans 10.7-10.12,
8.18, 8.19).
"""

from __future__ import annotations

import threading

import pytest


# ── 10.10 process hygiene ─────────────────────────────────────────────────

def test_health_reports_process_and_memory(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    proc = body.get("process") or {}
    assert proc.get("threads", 0) >= 1
    assert proc.get("open_fds", 0) >= 1
    assert "zombie_children" in proc
    assert body.get("memory", {}).get("rss_mb", 0) > 0


def test_hygiene_counts_zombies_without_raising():
    from app.core import memory

    stats = memory.process_hygiene()
    assert stats["zombie_children"] >= 0


# ── 10.7 WAL checkpoint ───────────────────────────────────────────────────

def test_the_wal_checkpoint_runs_and_reports(client):
    from app.core.storage_gc import checkpoint_wal

    result = checkpoint_wal()
    # Either the pragma answered (busy/log_frames/checkpointed) or the DB is absent.
    assert isinstance(result, dict)


# ── 10.8 memory trend + leak verdict ──────────────────────────────────────

def test_samples_accumulate_and_the_slope_is_computed():
    from app.core import memory_guard as guard

    guard._SAMPLES.clear()
    base = 1000.0
    for i in range(4):
        guard._SAMPLES.append({
            "at": base + i * 60, "footprint_mb": 100 + i * 40,
            "rss_mb": 100, "mlx_live_mb": 0, "mps_driver_mb": 0, "swap_used_mb": 0,
        })
    report = guard.report()
    assert report["samples"] == 4
    assert report["slope_per_min"]["footprint_mb"] == pytest.approx(40.0, rel=0.05)
    assert report["leak_suspected"] is True


def test_flat_memory_is_not_a_leak():
    from app.core import memory_guard as guard

    guard._SAMPLES.clear()
    for i in range(4):
        guard._SAMPLES.append({
            "at": 1000.0 + i * 60, "footprint_mb": 100,
            "rss_mb": 100, "mlx_live_mb": 0, "mps_driver_mb": 0, "swap_used_mb": 0,
        })
    assert guard.report()["leak_suspected"] is False


def test_history_endpoint_serves_the_report(client):
    res = client.get("/system/memory/history")
    assert res.status_code == 200
    body = res.json()
    assert "report" in body and "samples" in body


# ── 10.12 model budget guard ──────────────────────────────────────────────

def test_the_budget_refuses_a_model_larger_than_the_machine():
    from app.core.memory import fits

    verdict = fits(10_000_000)          # 10 TB
    assert verdict["ok"] is False
    assert "swap" in (verdict["reason"] or "").lower()


def test_the_budget_allows_a_small_model():
    from app.core.memory import fits

    assert fits(512)["ok"] is True


# ── 10.11 peak reset is best-effort ───────────────────────────────────────

def test_peak_reset_never_raises():
    from app.core.memory import reset_mlx_peak

    assert reset_mlx_peak() in (True, False)


# ── 8.18 export cancellation ──────────────────────────────────────────────

def test_an_export_cancel_is_registered_and_observed():
    from app.services.video import deliver

    event = deliver.register_cancel("task-abc")
    try:
        assert deliver.cancel_task("task-abc") is True
        assert event.is_set() is True
    finally:
        deliver.release_cancel("task-abc")
    assert deliver.cancel_task("task-abc") is False, "released tasks cannot be cancelled"


def test_a_cancelled_transcode_stops_and_leaves_no_file(tmp_path, monkeypatch):
    import subprocess

    from app.services.video import deliver
    from app.services.video.export_presets import get_preset

    src = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True, timeout=120,
    )
    dest = tmp_path / "out.mp4"
    cancel = threading.Event()
    cancel.set()                       # cancel before it starts

    ok = deliver.transcode(str(src), str(dest), get_preset("youtube_1080p"),
                           cancel_event=cancel)
    assert ok is False
    assert not dest.exists(), "a cancelled export must not leave a take behind"


# ── 8.19 dependency licence audit ─────────────────────────────────────────

def test_dependency_audit_reports_flagged_packages(monkeypatch):
    from app.core import licensing as lic

    monkeypatch.setitem(lic._RESTRICTED_DISTRIBUTIONS, "pytest", "test-only restriction")
    report = lic.audit_dependencies()
    assert report["checked"] > 0
    assert any(entry["package"] == "pytest" for entry in report["flagged"])
    assert report["ok"] is False


def test_the_licence_endpoint_includes_dependencies(client):
    body = client.get("/system/licences").json()
    assert "dependencies" in body
    assert "checked" in body["dependencies"]
