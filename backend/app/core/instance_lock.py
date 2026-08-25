"""Single-instance guard: prevents two backend processes from fighting over
the GPU, the SQLite WAL, and each other's reconcile passes.

The failure class this kills: a second boot (manual start, --reload spawn,
IDE run button) marks a live generation 'interrupted' via boot reconciliation
while the first process is still burning GPU on it — orphaned work.
"""
from __future__ import annotations

import atexit
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("milimo.instance")

LOCK_PATH = Path(os.environ.get("MILIMO_LOCK_FILE", ".milimo.lock"))
_stale_grace_s = 30.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def acquire_instance_lock() -> bool:
    """Returns True when we own the lock. Refuses to boot otherwise unless
    MILIMO_ALLOW_MULTI_INSTANCE=1 (power users accept the risks)."""
    if os.environ.get("MILIMO_ALLOW_MULTI_INSTANCE") == "1":
        logger.warning("Multi-instance override active — GPU/DB contention possible.")
        return True

    if LOCK_PATH.exists():
        try:
            raw = LOCK_PATH.read_text().strip()
            pid_str, _, ts_str = raw.partition(":")
            pid = int(pid_str)
            booted = float(ts_str) if ts_str else 0.0
        except ValueError:
            pid, booted = -1, 0.0

        if _pid_alive(pid):
            logger.error(
                f"Another backend instance is running (pid {pid}, lock {LOCK_PATH}). "
                "Refusing to start to protect GPU + DB integrity. "
                "Stop it first or set MILIMO_ALLOW_MULTI_INSTANCE=1."
            )
            return False

        # Stale lock from a crashed process: only steal after grace period so
        # two simultaneous boots don't both conclude the other is dead.
        age = time.time() - booted
        if booted and age < _stale_grace_s:
            logger.error(
                f"Stale-looking lock (pid {pid} dead, but only {age:.0f}s old). "
                f"Retry in a few seconds or delete {LOCK_PATH}."
            )
            return False
        logger.warning(f"Stealing stale instance lock from dead pid {pid}.")

    LOCK_PATH.write_text(f"{os.getpid()}:{time.time()}")
    atexit.register(release_instance_lock)
    return True


def release_instance_lock() -> None:
    try:
        if LOCK_PATH.exists():
            raw = LOCK_PATH.read_text().strip().partition(":")[0]
            if int(raw) == os.getpid():
                LOCK_PATH.unlink()
    except (ValueError, OSError):
        pass
