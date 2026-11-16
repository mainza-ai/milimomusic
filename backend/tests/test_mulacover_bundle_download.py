"""Regression tests for the MuLaCover bundle downloader (GitHub issue #5).

Issue #5 ("Error on downloading models") showed the Models card failing at 61%
with `name 'time' is not defined`. `bundle_downloader.py` called
`time.monotonic()` inside its progress-monitor thread and
`time.sleep()` in its retry loop without importing `time`, so the download died
mid-flight — after several files had already transferred.

The import is fixed; these tests pin the behaviour so it cannot regress and so
the resume semantics are proven (the crash was survivable only if retry resumes).
"""

from __future__ import annotations

import json
import time

import pytest

from app.services.mulacover import bundle_downloader as bd


def _manifest(root, entries):
    """Build the dict `get_bundle_manifest` returns, for two fake files."""
    files = []
    total = 0
    for name, payload in entries:
        target_dir = root / "MuLaCover"
        files.append({
            "kind": "hf_model",
            "repo_id": "HeartMuLa/MuLaCover",
            "repo_type": "model",
            "rfilename": name,
            "target_dir": str(target_dir),
            "target_file": str(target_dir / name),
            "size": len(payload),
            "label": f"MuLaCover: {name}",
        })
        total += len(payload)
    return {"manifest": files, "total_bytes": total, "found_local_codec": None, "total_files": len(files)}


@pytest.fixture
def fake_bundle(tmp_path, monkeypatch):
    """Two-file bundle with a slow download so the monitor thread actually ticks."""
    entries = [
        ("config.json", b'{"arch": "mulacover"}'),
        ("model.safetensors", b"X" * 4096),
    ]
    manifest = _manifest(tmp_path, entries)
    payloads = dict(entries)

    monkeypatch.setattr(bd, "get_bundle_manifest", lambda base_dir=None: manifest)
    # Never touch the real model registry during a test.
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"},
        raising=False,
    )

    calls = {"n": 0}

    def fake_download(repo_id, filename, repo_type=None, local_dir=None):
        calls["n"] += 1
        path = tmp_path / "MuLaCover" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = payloads[filename]
        # Write half, wait past one monitor tick (0.5 s), then finish: this is the
        # exact window in which `time.monotonic()` used to raise NameError.
        path.write_bytes(data[: len(data) // 2])
        time.sleep(0.6)
        path.write_bytes(data)
        return str(path)

    monkeypatch.setattr(bd, "hf_hub_download", fake_download)
    return {"root": tmp_path, "manifest": manifest, "calls": calls, "entries": entries}


def test_bundle_download_completes_without_name_error(fake_bundle):
    """The reported failure: progress monitor + retry loop must not raise NameError."""
    rec = {
        "id": "probe", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    import threading

    bd.download_mulacover_bundle("probe", rec, threading.Event(), base_dir=fake_bundle["root"])

    assert rec["status"] == "completed", f"download failed: {rec.get('error')}"
    assert "error" not in rec or not rec["error"]
    assert rec["files_done"] == 2
    assert fake_bundle["calls"]["n"] == 2
    # Both files landed with their full contents.
    for name, payload in fake_bundle["entries"]:
        path = fake_bundle["root"] / "MuLaCover" / name
        assert path.is_file()
        assert path.read_bytes() == payload


def test_bundle_download_reports_progress_and_speed(fake_bundle):
    rec = {
        "id": "probe-progress", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    import threading

    bd.download_mulacover_bundle("probe-progress", rec, threading.Event(), base_dir=fake_bundle["root"])

    assert rec["total_bytes"] > 0
    assert rec["total_files"] == 2
    assert rec["received_bytes"] == rec["total_bytes"]


def test_bundle_download_resumes_and_skips_completed_files(fake_bundle):
    """A failed attempt must be recoverable: finished files are not re-fetched."""
    root = fake_bundle["root"]
    done_name, done_payload = fake_bundle["entries"][0]
    existing = root / "MuLaCover" / done_name
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(done_payload)

    rec = {
        "id": "probe-resume", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    import threading

    bd.download_mulacover_bundle("probe-resume", rec, threading.Event(), base_dir=root)

    assert rec["status"] == "completed", f"resume failed: {rec.get('error')}"
    assert fake_bundle["calls"]["n"] == 1, "the already-present file must not be re-downloaded"
    assert rec["files_done"] == 2


def test_bundle_download_honours_cancel(fake_bundle):
    """Cancel must stop between files and report 'cancelled', not 'error'."""
    import threading

    cancel = threading.Event()
    cancel.set()
    rec = {
        "id": "probe-cancel", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    bd.download_mulacover_bundle("probe-cancel", rec, cancel, base_dir=fake_bundle["root"])
    assert rec["status"] == "cancelled"
    assert fake_bundle["calls"]["n"] == 0


def test_bundle_download_surfaces_a_real_failure(fake_bundle, monkeypatch):
    """A genuine download error must land on the record, not be swallowed."""
    import threading

    def boom(*a, **kw):
        raise OSError("network unreachable")

    monkeypatch.setattr(bd, "hf_hub_download", boom)
    # Keep the retry backoff fast so the test does not sleep 1+2 seconds.
    monkeypatch.setattr(bd.time, "sleep", lambda *_: None)

    rec = {
        "id": "probe-fail", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    bd.download_mulacover_bundle("probe-fail", rec, threading.Event(), base_dir=fake_bundle["root"])

    assert rec["status"] == "error"
    assert "network unreachable" in rec["error"]


# ── the plain (non-bundle) model downloader ────────────────────────────────

def test_model_download_worker_reports_byte_progress(tmp_path, monkeypatch):
    """A single large shard must show movement, not a frozen percentage.

    This is the second half of issue #5's symptom: the card sat at 61% because
    progress only advanced between files. The worker now samples the partial file
    while `hf_hub_download` runs.
    """
    import threading

    from app import main as app_main

    payload = b"Z" * 8192
    calls = {"n": 0}

    def slow_download(repo_id, filename, local_dir=None, **kw):
        calls["n"] += 1
        dest = tmp_path / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload[: len(payload) // 2])
        time.sleep(0.7)  # let the monitor sample at least once
        dest.write_bytes(payload)
        return str(dest)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", slow_download)
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )

    rec = {
        "id": "plain-probe", "repo_id": "org/model", "status": "queued",
        "total_files": 1, "files_done": 0, "current_file": "",
        "received_bytes": 0, "total_bytes": len(payload),
        "local_dir": str(tmp_path), "category": "audio",
        "category_source": "user", "error": "", "speed_mbps": 0.0,
    }
    app_main._model_downloads["plain-probe"] = rec
    try:
        app_main._model_download_worker(
            "plain-probe", "org/model", str(tmp_path),
            [("shard.safetensors", len(payload))], threading.Event(), "audio",
        )
        assert rec["status"] == "completed", rec.get("error")
        assert rec["files_done"] == 1
        assert rec["received_bytes"] == len(payload)
        assert calls["n"] == 1
    finally:
        app_main._model_downloads.pop("plain-probe", None)


def test_model_download_skips_complete_shards_on_retry(tmp_path, monkeypatch):
    """Retry must resume: a fully-transferred shard is not fetched again."""
    import threading

    from app import main as app_main

    payload = b"Y" * 1024
    (tmp_path / "done.bin").write_bytes(payload)
    calls = {"n": 0}

    def counting_download(repo_id, filename, local_dir=None, **kw):
        calls["n"] += 1
        raise AssertionError("should not download an already-complete shard")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", counting_download)
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )

    rec = {
        "id": "resume-probe", "repo_id": "org/model", "status": "queued",
        "total_files": 1, "files_done": 0, "current_file": "",
        "received_bytes": 0, "total_bytes": len(payload),
        "local_dir": str(tmp_path), "category": "audio",
        "category_source": "user", "error": "", "speed_mbps": 0.0,
    }
    app_main._model_downloads["resume-probe"] = rec
    try:
        app_main._model_download_worker(
            "resume-probe", "org/model", str(tmp_path),
            [("done.bin", len(payload))], threading.Event(), "audio",
        )
        assert rec["status"] == "completed", rec.get("error")
        assert calls["n"] == 0
    finally:
        app_main._model_downloads.pop("resume-probe", None)
