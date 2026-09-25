"""
Cross-Platform Safe File Operations & Non-Locking Media Streaming.

Solves Windows file-locking issues (WinError 32) when deleting/renaming media
while video playback is actively streaming, and provides atomic configuration persistence.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

from starlette.responses import FileResponse, Response, StreamingResponse

logger = logging.getLogger("milimo.win_safe_files")

IS_WINDOWS = platform.system() == "Windows"


def open_shared_delete(file_path: Union[str, Path], mode: str = "rb") -> Any:
    """
    Open file with FILE_SHARE_DELETE on Windows so that files can be deleted
    or renamed even while an active read stream is open.
    On macOS/Linux (POSIX), uses standard python open() which natively supports unlink.
    """
    path_str = str(file_path)
    if not IS_WINDOWS or "b" not in mode:
        return open(path_str, mode)

    # Windows specific FILE_SHARE_DELETE implementation via ctypes
    try:
        import ctypes
        from ctypes import wintypes

        GENERIC_READ = 0x80000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        FILE_SHARE_DELETE = 0x00000004
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x00000080

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CreateFileW = kernel32.CreateFileW
        CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        CreateFileW.restype = wintypes.HANDLE

        handle = CreateFileW(
            path_str,
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )

        if handle == wintypes.HANDLE(-1).value or handle == 0:
            # Fallback to standard open if CreateFileW fails
            return open(path_str, mode)

        import msvcrt

        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        return os.fdopen(fd, mode)
    except Exception as e:
        logger.debug(f"Could not open with shared-delete flags on Windows: {e}, falling back to standard open")
        return open(path_str, mode)


def share_delete_file_response(
    file_path: Union[str, Path],
    media_type: Optional[str] = None,
    filename: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """
    Return a Starlette response for media files that does not hard-lock the file on disk.
    Allows gallery deletion or renaming to succeed during video playback.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        return Response(status_code=404, content="File not found")

    file_size = path_obj.stat().st_size
    resp_headers = headers or {}
    resp_headers.setdefault("Accept-Ranges", "bytes")
    resp_headers.setdefault("Content-Length", str(file_size))

    if not IS_WINDOWS:
        return FileResponse(
            path=str(path_obj),
            media_type=media_type,
            filename=filename,
            headers=resp_headers,
        )

    # On Windows, stream using shared-delete descriptor generator
    def iter_file() -> AsyncIterator[bytes]:
        f = open_shared_delete(path_obj, "rb")
        try:
            chunk_size = 64 * 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            f.close()

    return StreamingResponse(
        iter_file(),
        media_type=media_type or "application/octet-stream",
        headers=resp_headers,
    )


def atomic_write_json(
    target_path: Union[str, Path],
    data: Any,
    make_backup: bool = True,
    indent: int = 2,
) -> bool:
    """
    Atomically writes JSON configuration to disk via tempfile and atomic rename.
    Maintains a .bak rollback copy if make_backup is True.
    """
    dest = Path(target_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    backup_path = dest.with_suffix(dest.suffix + ".bak")

    # Write to temporary file in the same filesystem directory to guarantee atomic rename
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=dest.parent,
        prefix=f".{dest.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Create backup if original exists
        if make_backup and dest.exists():
            shutil.copy2(dest, backup_path)

        # Atomic rename replacing destination
        os.replace(tmp_path, dest)
        return True
    except Exception as e:
        logger.error(f"Atomic write to {dest} failed: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        # If destination was corrupted and backup exists, restore it
        if make_backup and backup_path.exists() and not dest.exists():
            shutil.copy2(backup_path, dest)
        raise


def safe_load_json(target_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON configuration. If corrupt, attempts to restore from .bak file.
    """
    dest = Path(target_path).resolve()
    backup = dest.with_suffix(dest.suffix + ".bak")

    if not dest.exists():
        if backup.exists():
            logger.warning(f"Configuration file {dest} missing, recovering from {backup}")
            try:
                shutil.copy2(backup, dest)
            except Exception:
                pass
        else:
            return default

    try:
        with open(dest, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as primary_err:
        logger.warning(f"Error parsing {dest}: {primary_err}")
        if backup.exists():
            logger.info(f"Attempting to restore from backup {backup}")
            try:
                with open(backup, "r", encoding="utf-8") as f:
                    restored = json.load(f)
                shutil.copy2(backup, dest)
                return restored
            except Exception as backup_err:
                logger.error(f"Backup {backup} also unreadable: {backup_err}")

        return default
