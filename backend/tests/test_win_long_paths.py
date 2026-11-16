r"""Windows MAX_PATH guard for model downloads.

Windows refuses paths longer than 260 characters unless the machine opts into
long-path support (registry / Developer Mode). A model directory plus a safetensors
shard, plus Hugging Face's own `.cache/huggingface/download/<name>.incomplete`
bookkeeping, gets close to that on ordinary installs. The extended-length prefix
(`\\?\`) makes the OS accept the path without any machine-level change.

These tests pin the prefix logic itself and that the download paths actually use
it — the failure mode is silent only if the guard is wired in nowhere.
"""

from __future__ import annotations

import os

import pytest

from app.core import paths as paths_mod
from app.core.paths import (
    WIN_LONG_PATH_LIMIT,
    _extended_prefix,
    describe_fs_error,
    strip_win_long_prefix,
    win_long_path,
    win_path_length_warning,
)


# ── the prefix logic ───────────────────────────────────────────────────────

def test_short_paths_are_untouched():
    """Every non-Windows call, and every short path, must pass through unchanged."""
    assert win_long_path("/tmp/models/audio") == "/tmp/models/audio"
    assert win_long_path("relative/path") == "relative/path"


def test_windows_leaves_short_paths_alone():
    assert win_long_path("/tmp/short", windows=True) == "/tmp/short"


def test_windows_prefixes_long_paths():
    long_path = "/" + "a" * (WIN_LONG_PATH_LIMIT + 20)
    guarded = win_long_path(long_path, windows=True)
    assert guarded.startswith("\\\\?\\")
    assert guarded.endswith(long_path)


def test_guard_is_idempotent():
    long_path = "/" + "b" * (WIN_LONG_PATH_LIMIT + 20)
    once = win_long_path(long_path, windows=True)
    assert win_long_path(once, windows=True) == once


def test_boundary_is_exclusive_at_the_limit():
    """A path of exactly WIN_LONG_PATH_LIMIT characters is already guarded."""
    base = "/" + "c" * (WIN_LONG_PATH_LIMIT - 1)
    assert len(os.path.abspath(base)) == WIN_LONG_PATH_LIMIT
    assert win_long_path(base, windows=True).startswith("\\\\?\\")
    shorter = "/" + "c" * (WIN_LONG_PATH_LIMIT - 2)
    assert win_long_path(shorter, windows=True) == shorter


def test_drive_letter_path_gets_a_plain_prefix():
    assert _extended_prefix("C:\\models\\audio\\shard.bin") == "\\\\?\\C:\\models\\audio\\shard.bin"


def test_unc_path_uses_the_unc_form():
    """\\\\server\\share is not a drive, so it needs \\\\?\\UNC\\, not \\\\?\\."""
    assert _extended_prefix("\\\\server\\share\\models") == "\\\\?\\UNC\\server\\share\\models"


def test_prefix_is_not_doubled():
    already = "\\\\?\\C:\\models"
    assert _extended_prefix(already) == already


@pytest.mark.parametrize("path", [
    "\\\\?\\C:\\models\\audio",
    "\\\\?\\UNC\\server\\share\\models",
    "C:\\models\\audio",
])
def test_strip_round_trips_for_display(path):
    """Error messages and logs must not leak the \\\\?\\ marker."""
    stripped = strip_win_long_prefix(path)
    assert not stripped.startswith("\\\\?\\")
    if path.startswith("\\\\?\\UNC\\"):
        assert stripped.startswith("\\\\server")
    elif path.startswith("\\\\?\\"):
        assert stripped == path[4:]
    else:
        assert stripped == path


# ── the actionable warning / error text ────────────────────────────────────

def test_no_length_warning_off_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    long_path = "/" + "d" * 400
    assert win_path_length_warning([long_path]) is None


def test_length_warning_names_the_env_var(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    long_path = "/" + "e" * (WIN_LONG_PATH_LIMIT + 40)
    warning = win_path_length_warning([long_path])
    assert warning is not None
    assert "MODELS_DIRECTORY" in warning
    assert "260" in warning


def test_no_warning_when_all_paths_are_short(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert win_path_length_warning(["/tmp/models/audio/shard.bin"]) is None


def test_describe_fs_error_translates_winerror_206():
    err = OSError(206, "The filename or extension is too long")
    err.winerror = 206
    text = describe_fs_error(err)
    assert "MODELS_DIRECTORY" in text
    assert len(text) <= 500


def test_describe_fs_error_passes_ordinary_errors_through():
    text = describe_fs_error(OSError("network unreachable"))
    assert text == "network unreachable"


def test_describe_fs_error_is_truncated():
    assert len(describe_fs_error(OSError("x" * 5000))) == 500


# ── wiring: the download paths must actually call the guard ────────────────

def test_plain_worker_resolves_paths_through_the_guard(tmp_path, monkeypatch):
    import threading

    from app import main as app_main

    seen = []
    real = app_main.win_long_path

    def record(path, *a, **kw):
        seen.append(os.fspath(path))
        return real(path, *a, **kw)

    monkeypatch.setattr(app_main, "win_long_path", record)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, local_dir=None, **kw: str(tmp_path / filename),
    )
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )

    rec = {
        "id": "guard-probe", "repo_id": "org/model", "status": "queued",
        "total_files": 1, "files_done": 0, "current_file": "",
        "received_bytes": 0, "total_bytes": 10, "local_dir": str(tmp_path),
        "category": "audio", "category_source": "user", "error": "", "speed_mbps": 0.0,
    }
    app_main._model_downloads["guard-probe"] = rec
    try:
        app_main._model_download_worker(
            "guard-probe", "org/model", str(tmp_path),
            [("a.bin", 10)], threading.Event(), "audio",
        )
        assert rec["status"] == "completed", rec.get("error")
    finally:
        app_main._model_downloads.pop("guard-probe", None)

    # The worker guards the directory once; os.path.join carries the prefix into
    # every per-file path (see test_joined_paths_keep_the_prefix).
    assert str(tmp_path) in seen, "the worker must resolve local_dir through win_long_path"
    assert seen.count(str(tmp_path)) == 1, "guarding each file separately is redundant"


def test_joined_paths_keep_the_prefix():
    """Why guarding the directory once is enough."""
    long_dir = win_long_path("/" + "a" * (WIN_LONG_PATH_LIMIT + 20), windows=True)
    assert os.path.join(long_dir, "shard.bin").startswith("\\\\?\\")


def test_bundle_downloader_resolves_targets_through_the_guard(tmp_path, monkeypatch):
    import threading

    from app.services.mulacover import bundle_downloader as bd

    seen = []
    real = bd.win_long_path

    def record(path, *a, **kw):
        seen.append(os.fspath(path))
        return real(path, *a, **kw)

    monkeypatch.setattr(bd, "win_long_path", record)
    manifest = [{
        "kind": "hf_model", "repo_id": "HeartMuLa/MuLaCover", "repo_type": "model",
        "rfilename": "config.json", "target_dir": str(tmp_path / "MuLaCover"),
        "target_file": str(tmp_path / "MuLaCover" / "config.json"),
        "size": 2, "label": "MuLaCover: config.json",
    }]
    monkeypatch.setattr(bd, "get_bundle_manifest", lambda base_dir=None: {
        "manifest": list(manifest), "total_files": 1, "total_bytes": 2,
        "found_local_codec": None,
    })

    def download(repo_id, filename, repo_type=None, local_dir=None):
        dest = tmp_path / "MuLaCover" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"{}")
        return str(dest)

    monkeypatch.setattr(bd, "hf_hub_download", download)
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )

    rec = {
        "id": "guard-bundle", "status": "queued", "files_done": 0,
        "received_bytes": 0, "total_bytes": 0, "speed_mbps": 0.0,
        "current_file": "", "total_files": 0,
    }
    bd.download_mulacover_bundle("guard-bundle", rec, threading.Event(), base_dir=tmp_path)

    assert rec["status"] == "completed", rec.get("error")
    # Bundle targets are guarded per file (per-file checks and os.replace targets).
    assert str(tmp_path / "MuLaCover" / "config.json") in seen


def test_huggingface_hub_still_receives_the_plain_local_dir(tmp_path, monkeypatch):
    """The hub builds its own paths and cannot take the \\\\?\\ marker."""
    import threading

    from app import main as app_main

    received = {}

    def download(repo_id, filename, local_dir=None, **kw):
        received["local_dir"] = local_dir
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(
        "app.services.model_manager.model_manager.register_custom_model",
        lambda *a, **kw: {"id": "fake"}, raising=False,
    )
    # Force the guarded form to differ from the plain form.
    monkeypatch.setattr(app_main, "win_long_path", lambda p, *a, **kw: "\\\\?\\" + os.fspath(p))

    rec = {
        "id": "hub-probe", "repo_id": "org/model", "status": "queued",
        "total_files": 1, "files_done": 0, "current_file": "",
        "received_bytes": 0, "total_bytes": 10, "local_dir": str(tmp_path),
        "category": "audio", "category_source": "user", "error": "", "speed_mbps": 0.0,
    }
    app_main._model_downloads["hub-probe"] = rec
    try:
        app_main._model_download_worker(
            "hub-probe", "org/model", str(tmp_path),
            [("a.bin", 10)], threading.Event(), "audio",
        )
    finally:
        app_main._model_downloads.pop("hub-probe", None)

    assert received["local_dir"] == str(tmp_path)
    assert not received["local_dir"].startswith("\\\\?\\")


def test_models_dir_honours_a_short_override(monkeypatch, tmp_path):
    """The advice the warning gives must actually work."""
    monkeypatch.setenv("MODELS_DIRECTORY", str(tmp_path / "models"))
    assert paths_mod.get_models_dir() == (tmp_path / "models").resolve()
