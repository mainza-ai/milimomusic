"""Memory telemetry and one release path (memory phase).

The app had 33 scattered `clear_cache()`/`gc.collect()` calls and **no way to see
memory at all**: `/health` reported disk, DB and queue depth but not RSS, not MLX's
active/cache split, and not MPS allocations. That makes a leak invisible until the
machine swaps.

`snapshot()` answers "what is this process holding right now"; `release()` is the one
place that returns memory to the OS. Both are cheap enough to call from an endpoint.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("milimo.memory")

_MB = 1024 * 1024


def _mb(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value) / _MB, 1)


def _phys_footprint_bytes() -> Optional[int]:
    """macOS's jetsam-relevant memory for this process.

    Measured on this machine: with 8 GiB of MLX live, RSS read 30 MB while the
    footprint read 8 370 MB — RSS is ~270x wrong for accelerator work because Metal
    buffers are IOSurface-backed and never enter `resident_size`. psutil cannot report
    footprint on Darwin, so this calls `proc_pid_rusage` directly (~0.5 us).

    The kernel writes a `rusage_info_v4` larger than the fields we read, so the buffer
    is deliberately oversized and the values are read at their documented offsets
    (`ri_resident_size` = 64, `ri_phys_footprint` = 72). Declaring a struct with only
    the leading fields corrupts memory — it hung a later torch call when this was
    first written as a ctypes.Structure.
    """
    if os.uname().sysname != "Darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        buffer = ctypes.create_string_buffer(512)   # larger than rusage_info_v4
        if libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(4), buffer) != 0:
            return None
        return int(ctypes.c_uint64.from_buffer(buffer, 72).value) or None
    except Exception:
        return None


def process_stats() -> Dict[str, Any]:
    """RSS/VMS, the macOS footprint, and handle counts for this process."""
    stats: Dict[str, Any] = {}
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        stats.update({
            "rss_mb": round(info.rss / _MB, 1),
            "vms_mb": round(getattr(info, "vms", 0) / _MB, 1),
            # The kernel's per-process ledger: Metal/MLX buffers, compressed pages,
            # page tables. This is the number Activity Monitor and jetsam use.
            "footprint_mb": _mb(_phys_footprint_bytes() or info.rss),
            "open_fds": proc.num_fds() if hasattr(proc, "num_fds") else len(proc.open_files()),
            "threads": proc.num_threads(),
            "cpu_percent": proc.cpu_percent(interval=None),
        })
    except Exception as exc:  # pragma: no cover - psutil is optional at runtime
        logger.debug("psutil unavailable for process stats: %s", exc)
        try:
            import resource

            stats["rss_mb"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1
            )
        except Exception:
            pass
    return stats


def system_stats() -> Dict[str, Any]:
    """Machine-wide memory and swap, which is what a unified-memory box runs out of."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_mb": round(vm.total / _MB, 1),
            "available_mb": round(vm.available / _MB, 1),
            "used_percent": round(vm.percent, 1),
            "swap_used_mb": round(swap.used / _MB, 1),
        }
    except Exception as exc:
        logger.debug("System memory unavailable: %s", exc)
        return {}


def mlx_stats() -> Dict[str, Any]:
    """MLX's own split: active (live arrays) vs cache (freed but retained)."""
    try:
        import mlx.core as mx  # type: ignore

        active = mx.get_active_memory()
        cache = mx.get_cache_memory()
        return {
            "active_mb": _mb(active),
            "cache_mb": _mb(cache),
            # Neither figure alone is "what MLX holds": live = active + cache.
            "live_mb": _mb(active + cache),
            "peak_mb": _mb(mx.get_peak_memory()),
        }
    except Exception:
        return {}


def torch_stats() -> Dict[str, Any]:
    """PyTorch MPS allocations when the video path has been used."""
    try:
        import torch

        if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
            return {}
        stats: Dict[str, Any] = {}
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["allocated_mb"] = _mb(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["driver_allocated_mb"] = _mb(torch.mps.driver_allocated_memory())
        return stats
    except Exception:
        return {}


def snapshot() -> Dict[str, Any]:
    """Everything the app can honestly report about its memory right now."""
    return {
        "process": process_stats(),
        "system": system_stats(),
        "mlx": mlx_stats(),
        "torch": torch_stats(),
        "limits": {
            "mlx_cache_limit_env_mb": os.environ.get("MILIMO_MLX_CACHE_LIMIT_MB"),
        },
    }


def _machine_ram_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _default_cache_limit_mb() -> int:
    """Small reuse headroom: 12 GiB or a tenth of RAM, whichever is smaller.

    MLX's own default is **1.5x the recommended working set** — measured at 161 GiB on
    this 128 GiB machine, i.e. larger than physical RAM, so an unloaded model's buffers
    can sit in the cache and push the machine into swap.
    """
    ram = _machine_ram_bytes()
    if not ram:
        return 12 * 1024
    return int(min(12 * 1024, ram / _MB / 10))


def configure_mlx_cache_limit() -> Optional[int]:
    """Cap MLX's buffer cache once at boot. Defaults to a sane bound.

    `MILIMO_MLX_CACHE_LIMIT_MB`: a value caps it, `0` disables the cache entirely,
    `off` leaves MLX's own (very large) default in place.
    """
    raw = os.environ.get("MILIMO_MLX_CACHE_LIMIT_MB")
    if raw is not None and str(raw).strip().lower() in ("off", "default"):
        logger.info("MLX cache limit left at MLX's default")
        return None
    if raw is None:
        limit_mb = _default_cache_limit_mb()
    else:
        try:
            limit_mb = max(0, int(raw))
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid MILIMO_MLX_CACHE_LIMIT_MB=%r", raw)
            limit_mb = _default_cache_limit_mb()
    try:
        import mlx.core as mx  # type: ignore

        if not hasattr(mx, "set_cache_limit"):
            logger.info("MLX has no set_cache_limit; skipping cache cap")
            return None
        mx.set_cache_limit(limit_mb * _MB)
        logger.info("MLX buffer cache capped at %d MB", limit_mb)
        return limit_mb
    except Exception as exc:
        logger.debug("Could not set the MLX cache limit: %s", exc)
        return None


def configure_mps_memory_fraction(fraction: Optional[float] = None) -> Optional[float]:
    """Cap PyTorch's MPS allocator before the first video tensor.

    PyTorch's default high watermark is **1.7x** `recommended_max_memory()`, which is
    182 GiB on this 128 GiB machine — unreachable, so MPS never raises OOM and the box
    swaps instead. `set_per_process_memory_fraction` is enforced at runtime (0 disables),
    and it must be set before the first MPS allocation.
    """
    raw = fraction if fraction is not None else os.environ.get("MILIMO_MPS_MEMORY_FRACTION", "0.6")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= value <= 1.0):
        logger.warning("Ignoring invalid MPS memory fraction %r", raw)
        return None
    try:
        import torch

        if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
            return None
        if not hasattr(torch.mps, "set_per_process_memory_fraction"):
            return None
        torch.mps.set_per_process_memory_fraction(value)
        logger.info("MPS memory fraction capped at %.2f", value)
        return value
    except Exception as exc:
        logger.debug("Could not cap MPS memory: %s", exc)
        return None


def release(reason: str = "", *, collect_cycles: Optional[bool] = None) -> Dict[str, Any]:
    """Return cached accelerator memory to the OS, in one place.

    Order matters: drop Python references first (the caller does that), then clear the
    framework pools, then *optionally* collect cycles that closures still hold.

    Cycle collection is **opt-in** (`MILIMO_MEMORY_GC=1`). Measured: `gc.collect()` on a
    long-lived server heap costs seconds — it walks every object — while tensors are
    refcounted and freed by the framework cache clear. Paying seconds on a telemetry
    endpoint to find no cycles is the wrong trade; the ops path can opt in.
    """
    if collect_cycles is None:
        collect_cycles = os.environ.get("MILIMO_MEMORY_GC", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
    before = snapshot()
    result: Dict[str, Any] = {"reason": reason, "cleared": []}

    try:
        import mlx.core as mx  # type: ignore

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
            result["cleared"].append("mlx")
    except Exception as exc:
        logger.debug("MLX cache clear skipped: %s", exc)

    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            result["cleared"].append("torch-mps")
    except Exception as exc:
        logger.debug("torch MPS cache clear skipped: %s", exc)

    if collect_cycles:
        try:
            collected = gc.collect()
            result["gc_collected"] = collected
        except Exception:
            pass

    # Report the kernel footprint: RSS does not move when Metal memory is released.
    result["before_mb"] = before.get("process", {}).get("footprint_mb")
    result["after_mb"] = process_stats().get("footprint_mb")
    return result
