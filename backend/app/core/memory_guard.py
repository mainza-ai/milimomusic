"""Memory trend sampling and leak detection (plan 10.8).

A point-in-time snapshot cannot distinguish "large" from "leaking". A leak is a
*positive slope at idle*: memory that keeps growing while nothing is running. This
module keeps a short ring of samples and answers that question.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("milimo.memory.guard")

_SAMPLES: Deque[Dict[str, Any]] = deque(maxlen=180)   # ~3 h at one/minute
_LOCK = threading.Lock()


def sample() -> Dict[str, Any]:
    """Take one memory sample and append it to the ring."""
    from app.core import memory

    snap = memory.snapshot()
    point = {
        "at": time.time(),
        "footprint_mb": snap.get("process", {}).get("footprint_mb"),
        "rss_mb": snap.get("process", {}).get("rss_mb"),
        "mlx_live_mb": snap.get("mlx", {}).get("live_mb"),
        "mps_driver_mb": snap.get("torch", {}).get("driver_allocated_mb"),
        "swap_used_mb": snap.get("system", {}).get("swap_used_mb"),
    }
    with _LOCK:
        _SAMPLES.append(point)
    return point


def history(limit: int = 60) -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_SAMPLES)[-max(1, limit):]


def _slope_per_minute(points: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [(p["at"], p.get(key)) for p in points if p.get(key) is not None]
    if len(values) < 3:
        return None
    first_t, first_v = values[0]
    last_t, last_v = values[-1]
    minutes = (last_t - first_t) / 60.0
    if minutes <= 0:
        return None
    return round((last_v - first_v) / minutes, 2)


def report() -> Dict[str, Any]:
    """Growth per minute per channel, and whether anything looks like a leak.

    The threshold is deliberately blunt (a sustained +25 MB/min at idle) because a
    false alarm is cheap and a missed leak takes the machine down.
    """
    try:
        threshold = float(os.environ.get("MILIMO_MEMORY_LEAK_MB_PER_MIN", "25"))
    except (TypeError, ValueError):
        threshold = 25.0

    points = history(limit=60)
    slopes = {
        key: _slope_per_minute(points, key)
        for key in ("footprint_mb", "mlx_live_mb", "mps_driver_mb", "swap_used_mb")
    }
    suspicious = {
        key: value for key, value in slopes.items()
        if value is not None and value >= threshold
    }
    return {
        "samples": len(points),
        "window_minutes": round(
            (points[-1]["at"] - points[0]["at"]) / 60.0, 1
        ) if len(points) > 1 else 0.0,
        "slope_per_min": slopes,
        "threshold_mb_per_min": threshold,
        "leak_suspected": bool(suspicious),
        "suspicious": suspicious,
        "latest": points[-1] if points else None,
    }
