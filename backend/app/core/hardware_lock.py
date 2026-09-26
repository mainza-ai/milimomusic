"""Global Hardware Coordinator and Accelerator Resource Manager for Milimo Music.

Serializes VRAM access between heavy generative neural backbones (MiniMax Music 3,
MuLaCover 3B, BS-Roformer, Wan 2.1 Video, and Neural Voice Conversion) to prevent
GPU OOM collisions on CUDA and MPS unified memory architectures.
"""

import asyncio
import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger("milimo.hardware_lock")

# Try importing torch safely
try:
    import torch
except ImportError:
    torch = None


class GlobalHardwareCoordinator:
    """Manages lock serialization, cross-modal eviction, VRAM monitoring, and cache flushing."""

    _lock = asyncio.Lock()
    _active_consumer: str = "idle"
    _consumer_acquired_at: float = 0.0
    _eviction_hooks: Dict[str, list] = {}
    _memory_policy: str = os.environ.get("MILIMO_MEMORY_POLICY", "eager").lower()
    _ttl_seconds: float = float(os.environ.get("MILIMO_WARM_CACHE_TTL", "180"))
    _ttl_tasks: Dict[str, Any] = {}

    @classmethod
    def register_eviction_hook(cls, modality: str, hook: Callable[[], None]) -> None:
        """Register a model unloading callback for a modality domain.

        Modalities: 'audio_gen', 'audio_sep', 'audio_trans', 'audio_voice',
        'image_gen', 'video_gen'.
        Supports multiple hooks per modality without overwriting existing registrations.
        """
        if modality not in cls._eviction_hooks:
            cls._eviction_hooks[modality] = []
        if hook not in cls._eviction_hooks[modality]:
            cls._eviction_hooks[modality].append(hook)
        logger.debug(f"Hardware coordinator registered eviction hook for: {modality} (total: {len(cls._eviction_hooks[modality])})")

    @classmethod
    def get_active_consumer(cls) -> str:
        """Return the current active consumer holding the accelerator device, or 'idle'."""
        return cls._active_consumer

    @classmethod
    def is_locked(cls) -> bool:
        """Check if accelerator device is currently acquired."""
        return cls._lock.locked()

    @classmethod
    def get_memory_policy(cls) -> Dict[str, Any]:
        """Return the active memory lifecycle policy and TTL settings."""
        return {
            "policy": cls._memory_policy,
            "ttl_seconds": cls._ttl_seconds,
            "registered_modalities": list(cls._eviction_hooks.keys()),
            "active_ttl_timers": list(cls._ttl_tasks.keys()),
        }

    @classmethod
    def set_memory_policy(cls, policy: str, ttl_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Update runtime memory policy ('eager' or 'warm_ttl').

        If switched to 'eager', all idle cached models are immediately evicted.
        """
        clean_policy = policy.lower().strip()
        if clean_policy not in ("eager", "warm_ttl"):
            raise ValueError(f"Invalid memory policy '{policy}'. Must be 'eager' or 'warm_ttl'.")
        cls._memory_policy = clean_policy
        if ttl_seconds is not None and ttl_seconds > 0:
            cls._ttl_seconds = float(ttl_seconds)

        if clean_policy == "eager":
            # Cancel all TTL timers and immediately purge foreign models
            for timer in cls._ttl_tasks.values():
                try:
                    timer.cancel()
                except Exception:
                    pass
            cls._ttl_tasks.clear()
            cls.evict_all_except(active_modality=None)

        logger.info(f"GlobalHardwareCoordinator: Memory policy updated to '{cls._memory_policy}' (TTL={cls._ttl_seconds}s)")
        return cls.get_memory_policy()

    @classmethod
    def cancel_modality_ttl(cls, modality: str) -> None:
        """Cancel any pending TTL eviction timer for a modality."""
        timer = cls._ttl_tasks.pop(modality, None)
        if timer:
            try:
                timer.cancel()
            except Exception:
                pass

    @classmethod
    def schedule_modality_ttl_eviction(cls, modality: str) -> None:
        """Schedule delayed auto-eviction of warm models after TTL seconds."""
        if cls._memory_policy != "warm_ttl":
            return
        cls.cancel_modality_ttl(modality)

        try:
            loop = asyncio.get_running_loop()
            handle = loop.call_later(
                cls._ttl_seconds,
                lambda mod=modality: asyncio.create_task(cls._ttl_evict_worker(mod))
            )
            cls._ttl_tasks[modality] = handle
            logger.debug(f"Scheduled TTL eviction for {modality} in {cls._ttl_seconds}s.")
        except RuntimeError:
            pass

    @classmethod
    async def _ttl_evict_worker(cls, modality: str) -> None:
        """Execute delayed eviction when TTL timer fires and device is not in use by that modality."""
        cls._ttl_tasks.pop(modality, None)
        if cls._active_consumer != "idle":
            logger.debug(f"Skipping TTL eviction for {modality}; device busy with {cls._active_consumer}.")
            return
        logger.info(f"TTL expired ({cls._ttl_seconds}s) for {modality}; evicting cached models to free RAM.")
        cls.evict_modality(modality)

    @classmethod
    def evict_modality(cls, modality: str) -> None:
        """Evict all registered models for a specific modality."""
        cls.cancel_modality_ttl(modality)
        hooks = cls._eviction_hooks.get(modality, [])
        for hook in hooks:
            try:
                hook()
            except Exception as e:
                logger.warning(f"Eviction hook failed for {modality}: {e}")
        if hooks:
            logger.info(f"Evicted models for modality: {modality} ({len(hooks)} hooks ran)")
        cls.flush_memory()

    @classmethod
    def evict_all_except(cls, active_modality: Optional[str] = None) -> int:
        """Evict cached models across all other modalities before running a new workload."""
        evicted_count = 0
        for mod, hooks in list(cls._eviction_hooks.items()):
            if active_modality is None or mod != active_modality:
                cls.cancel_modality_ttl(mod)
                for hook in hooks:
                    try:
                        hook()
                        evicted_count += 1
                    except Exception as e:
                        logger.warning(f"Eviction hook failed for {mod}: {e}")
        if evicted_count > 0 or active_modality is None:
            cls.flush_memory()
        return evicted_count

    @classmethod
    async def acquire_device(
        cls,
        consumer: str,
        modality: Optional[str] = None,
        timeout: Optional[float] = 600.0
    ) -> bool:
        """Acquire exclusive accelerator device access with automatic inter-modality eviction."""
        logger.info(f"Hardware lock requested by: {consumer} (modality={modality})")
        target_modality = modality or consumer
        # Cancel any pending TTL auto-eviction because this modality is now active again
        if target_modality:
            cls.cancel_modality_ttl(target_modality)

        try:
            if timeout:
                await asyncio.wait_for(cls._lock.acquire(), timeout=timeout)
            else:
                await cls._lock.acquire()
            cls._active_consumer = consumer
            cls._consumer_acquired_at = time.time()
            # Automatically evict models from other modalities to maximize VRAM headroom
            if target_modality:
                cls.evict_all_except(active_modality=target_modality)
            logger.info(f"Hardware lock ACQUIRED by: {consumer}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Hardware lock acquisition timed out for {consumer} (held by {cls._active_consumer})")
            raise RuntimeError(f"GPU device is busy with {cls._active_consumer}. Please retry shortly.")

    @classmethod
    def release_device(cls, consumer: str, modality: Optional[str] = None) -> None:
        """Release accelerator device and perform memory management according to active policy."""
        target_modality = modality or consumer
        cls._active_consumer = "idle"
        cls._consumer_acquired_at = 0.0

        if cls._memory_policy == "eager" and target_modality and target_modality != "idle":
            cls.evict_modality(target_modality)
        elif cls._memory_policy == "warm_ttl" and target_modality and target_modality != "idle":
            cls.schedule_modality_ttl_eviction(target_modality)

        cls.flush_memory()
        if cls._lock.locked():
            try:
                cls._lock.release()
            except RuntimeError:
                pass
        logger.info(f"Hardware lock RELEASED by: {consumer} (policy={cls._memory_policy})")

    @classmethod
    @asynccontextmanager
    async def scoped_device(
        cls,
        consumer: str,
        modality: Optional[str] = None,
        timeout: Optional[float] = 600.0
    ):
        """Async context manager to safely acquire and release the accelerator device."""
        target_modality = modality or consumer
        await cls.acquire_device(consumer, modality=target_modality, timeout=timeout)
        try:
            yield
        finally:
            cls.release_device(consumer, modality=target_modality)

    @classmethod
    def flush_memory(cls) -> Dict[str, Any]:
        """Evict cached tensors and call garbage collection across MLX Metal, CUDA, MPS, and glibc."""
        reclaimed_mb = 0.0
        before_mb = 0.0
        after_mb = 0.0

        gc.collect()

        # 1. Apple Silicon MLX Metal Buffer Cache Purge
        try:
            import mlx.core as mx
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

        # 2. PyTorch CUDA & MPS
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

        # 3. Linux glibc arena trimming
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

        logger.info(f"VRAM / Device cache flushed (freed ~{reclaimed_mb:.1f} MB).")
        return {
            "reclaimed_mb": round(reclaimed_mb, 1),
            "current_allocated_mb": round(after_mb, 1),
            "after_mb": round(after_mb, 1),
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
