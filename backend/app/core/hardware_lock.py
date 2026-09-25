"""Global Hardware Coordinator and Accelerator Resource Manager for Milimo Music.

Serializes VRAM access between heavy generative neural backbones (MiniMax Music 3,
MuLaCover 3B, BS-Roformer, Wan 2.1 Video, and Neural Voice Conversion) to prevent
GPU OOM collisions on CUDA and MPS unified memory architectures.
"""

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("milimo.hardware_lock")

# Try importing torch safely
try:
    import torch
except ImportError:
    torch = None


class GlobalHardwareCoordinator:
    """Manages lock serialization, VRAM monitoring, and cache flushing across neural engines."""

    _lock = asyncio.Lock()
    _active_consumer: str = "idle"
    _consumer_acquired_at: float = 0.0

    @classmethod
    async def acquire_device(cls, consumer: str, timeout: Optional[float] = 600.0) -> bool:
        """Acquire exclusive accelerator device access for a heavy neural workload."""
        logger.info(f"Hardware lock requested by: {consumer}")
        try:
            if timeout:
                await asyncio.wait_for(cls._lock.acquire(), timeout=timeout)
            else:
                await cls._lock.acquire()
            cls._active_consumer = consumer
            cls._consumer_acquired_at = time.time()
            logger.info(f"Hardware lock ACQUIRED by: {consumer}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Hardware lock acquisition timed out for {consumer} (held by {cls._active_consumer})")
            raise RuntimeError(f"GPU device is busy with {cls._active_consumer}. Please retry shortly.")

    @classmethod
    def release_device(cls, consumer: str) -> None:
        """Release accelerator device and perform aggressive memory reclamation."""
        cls._active_consumer = "idle"
        cls._consumer_acquired_at = 0.0
        cls.flush_memory()
        if cls._lock.locked():
            try:
                cls._lock.release()
            except RuntimeError:
                pass
        logger.info(f"Hardware lock RELEASED by: {consumer}")

    @classmethod
    @asynccontextmanager
    async def scoped_device(cls, consumer: str, timeout: Optional[float] = 600.0):
        """Async context manager to safely acquire and release the accelerator device."""
        await cls.acquire_device(consumer, timeout=timeout)
        try:
            yield
        finally:
            cls.release_device(consumer)

    @classmethod
    def flush_memory(cls) -> Dict[str, Any]:
        """Evict cached tensors and call garbage collection across CUDA and MPS."""
        reclaimed_mb = 0.0
        before_mb = 0.0
        after_mb = 0.0

        gc.collect()

        if torch is not None:
            if torch.cuda.is_available():
                before_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                after_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reclaimed_mb = max(0.0, before_mb - after_mb)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "current_allocated_memory"):
                    before_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, "current_allocated_memory"):
                    after_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                reclaimed_mb = max(0.0, before_mb - after_mb)

        logger.info(f"VRAM / Device cache flushed (freed ~{reclaimed_mb:.1f} MB).")
        return {
            "reclaimed_mb": round(reclaimed_mb, 1),
            "current_allocated_mb": round(after_mb, 1),
            "status": "flushed",
        }

    @classmethod
    def get_telemetry(cls) -> Dict[str, Any]:
        """Fetch real-time accelerator and system memory metrics."""
        device_type = "cpu"
        device_name = "CPU Standard Architecture"
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        vram_total_mb = 0.0

        if torch is not None:
            if torch.cuda.is_available():
                device_type = "cuda"
                device_name = torch.cuda.get_device_name(0)
                vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
                vram_reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
                vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_type = "mps"
                device_name = "Apple Silicon Unified Memory (MPS)"
                if hasattr(torch.mps, "current_allocated_memory"):
                    vram_allocated_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                # Approximation from total system RAM via psutil
                try:
                    import psutil
                    vm = psutil.virtual_memory()
                    vram_total_mb = vm.total / (1024 * 1024)
                    vram_reserved_mb = (vm.total - vm.available) / (1024 * 1024)
                except Exception:
                    vram_total_mb = 16384.0

        # Percentage calculation
        usage_pct = round((vram_allocated_mb / vram_total_mb * 100), 1) if vram_total_mb > 0 else 0.0

        return {
            "device_type": device_type,
            "device_name": device_name,
            "active_consumer": cls._active_consumer,
            "vram_allocated_mb": round(vram_allocated_mb, 1),
            "vram_reserved_mb": round(vram_reserved_mb, 1),
            "vram_total_mb": round(vram_total_mb, 1),
            "usage_percent": usage_pct,
            "lock_held": cls._lock.locked(),
        }
