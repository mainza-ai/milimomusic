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


# ── cross-platform: symlinks are refused on Windows (and some filesystems) ──

class _Sibling:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


class _Info:
    def __init__(self, siblings):
        self.siblings = siblings


def _make_installed_tree(root):
    """Pre-create everything `is_mulacover_installed` checks, except the codec."""
    gen = root / "MuLaCover"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "config.json").write_bytes(b"{}")
    for i in range(5):
        (gen / f"model-0000{i}-of-00005.safetensors").write_bytes(b"S" * 16)
    qwen = root / "Qwen3-Embedding-0.6B"
    qwen.mkdir(parents=True, exist_ok=True)
    (qwen / "config.json").write_bytes(b"{}")
    st = root / "SymbolicTranscriptor"
    (st / "yourmt3").mkdir(parents=True, exist_ok=True)
    (st / "yourmt3" / "last.ckpt").write_bytes(b"C" * 16)
    (st / "chord").mkdir(parents=True, exist_ok=True)
    for c_file in bd.CHORD_FILES:
        (st / "chord" / c_file).write_bytes(b"D" * 16)


def test_bundle_downloads_codec_when_symlink_is_refused(tmp_path, monkeypatch):
    """Windows without Developer Mode/admin rejects symlinks with OSError.

    `get_bundle_manifest` omits HeartCodec when a *local* copy exists, so a
    swallowed symlink error used to end with status "completed", no codec on
    disk, and `is_mulacover_installed()` False — a download that cannot run.
    The bundle must fall back to downloading the codec instead.
    """
    import threading

    _make_installed_tree(tmp_path)
    local_codec = tmp_path / "heartlib" / "ckpt" / "HeartCodec-oss"
    local_codec.mkdir(parents=True, exist_ok=True)
    (local_codec / "config.json").write_bytes(b"{}")

    manifest = [{
        "kind": "hf_model", "repo_id": "HeartMuLa/MuLaCover", "repo_type": "model",
        "rfilename": "extra/metadata.json",
        "target_dir": str(tmp_path / "MuLaCover"),
        "target_file": str(tmp_path / "MuLaCover" / "extra" / "metadata.json"),
        "size": 8, "label": "MuLaCover: metadata.json",
    }]
    monkeypatch.setattr(bd, "get_bundle_manifest", lambda base_dir=None: {
        "manifest": manifest, "total_files": len(manifest), "total_bytes": 8,
        "found_local_codec": str(local_codec),
    })

    codec_payloads = {"config.json": b"codec-cfg", "model.safetensors": b"K" * 2048}
    monkeypatch.setattr(
        bd, "HfApi",
        lambda *a, **kw: type("Api", (), {
            "model_info": lambda self, repo_id, files_metadata=False: _Info(
                [_Sibling(n, len(p)) for n, p in codec_payloads.items()]
            )
        })(),
    )

    def download(repo_id, filename, repo_type=None, local_dir=None):
        payload = codec_payloads.get(filename, b"x" * 8)
        dest = tmp_path / "HeartCodec-oss" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return str(dest)

    monkeypatch.setattr(bd, "hf_hub_download", download)
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )

    def refuse(*a, **kw):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr("pathlib.Path.symlink_to", refuse)

    rec = {
        "id": "win-probe", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    bd.download_mulacover_bundle("win-probe", rec, threading.Event(), base_dir=tmp_path)

    assert rec["status"] == "completed", rec.get("error")
    # The codec was added to the manifest and fetched, not silently skipped.
    assert (tmp_path / "HeartCodec-oss" / "model.safetensors").read_bytes() == b"K" * 2048
    assert rec["total_files"] == 3, "codec files must join the manifest"
    assert rec["files_done"] == 3
    assert rec["total_bytes"] == 8 + len(b"codec-cfg") + 2048
    assert bd.is_mulacover_installed(tmp_path) is True


def test_installed_check_rejects_a_dangling_codec_symlink(tmp_path):
    """A symlink whose target vanished must not count as an installed bundle."""
    _make_installed_tree(tmp_path)
    target = tmp_path / "moved-away-codec"
    target.mkdir()
    (target / "config.json").write_bytes(b"{}")
    (tmp_path / "HeartCodec-oss").symlink_to(target, target_is_directory=True)
    assert bd.is_mulacover_installed(tmp_path) is True

    for f in target.iterdir():
        f.unlink()
    target.rmdir()
    assert bd.is_mulacover_installed(tmp_path) is False


def test_bundle_download_failure_is_reported_as_error_not_failed(tmp_path, monkeypatch):
    """The UI polls for completed/cancelled/error; any other status polls forever."""
    import threading

    _make_installed_tree(tmp_path)
    monkeypatch.setattr(bd, "get_bundle_manifest", lambda base_dir=None: {
        "manifest": [], "total_files": 0, "total_bytes": 8,
        "found_local_codec": None,
    })
    monkeypatch.setattr(
        "app.services.mulacover.bundle_downloader.shutil.disk_usage",
        lambda p: type("U", (), {"free": 1024, "total": 4096, "used": 3072})(),
    )

    rec = {
        "id": "disk-probe", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    bd.download_mulacover_bundle("disk-probe", rec, threading.Event(), base_dir=tmp_path)

    assert rec["status"] == "error"
    assert "disk space" in rec["error"].lower()


def test_bundle_links_existing_codec_instead_of_downloading_it(tmp_path, monkeypatch):
    """The happy path on Linux/macOS: a local codec is linked, not re-fetched."""
    import threading

    _make_installed_tree(tmp_path)
    local_codec = tmp_path / "heartlib" / "ckpt" / "HeartCodec-oss"
    local_codec.mkdir(parents=True, exist_ok=True)
    (local_codec / "config.json").write_bytes(b"{}")

    manifest = [{
        "kind": "hf_model", "repo_id": "HeartMuLa/MuLaCover", "repo_type": "model",
        "rfilename": "metadata.json", "target_dir": str(tmp_path / "MuLaCover"),
        "target_file": str(tmp_path / "MuLaCover" / "metadata.json"),
        "size": 8, "label": "MuLaCover: metadata.json",
    }]
    monkeypatch.setattr(bd, "get_bundle_manifest", lambda base_dir=None: {
        "manifest": list(manifest), "total_files": 1, "total_bytes": 8,
        "found_local_codec": str(local_codec),
    })

    fetched = []

    def download(repo_id, filename, repo_type=None, local_dir=None):
        fetched.append(filename)
        dest = tmp_path / "MuLaCover" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"m" * 8)
        return str(dest)

    # HfApi is only used to list the codec; if it is consulted, la link was missed.
    monkeypatch.setattr(bd, "HfApi", lambda *a, **kw: type("Api", (), {
        "model_info": lambda self, repo_id, files_metadata=False: _Info(
            [_Sibling("model.safetensors", 2048)]
        )
    })())
    monkeypatch.setattr(bd, "hf_hub_download", download)
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )

    rec = {
        "id": "link-probe", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    bd.download_mulacover_bundle("link-probe", rec, threading.Event(), base_dir=tmp_path)

    assert rec["status"] == "completed", rec.get("error")
    assert rec["total_files"] == 1, "a reachable local codec must not join the manifest"
    assert fetched == ["metadata.json"], "only the missing file may be downloaded"
    assert (tmp_path / "HeartCodec-oss" / "config.json").read_bytes() == b"{}"
    assert bd.is_mulacover_installed(tmp_path) is True
