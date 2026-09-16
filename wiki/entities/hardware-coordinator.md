---
title: Global Hardware Coordinator
type: entity
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [hardware, gpu, vram, coordinator, mps, cuda, telemetry, architecture]
aliases: [Hardware Coordinator, GlobalHardwareCoordinator, VRAM Manager]
---

# Global Hardware Coordinator

The **Global Hardware Coordinator** (`backend/app/core/hardware_lock.py`) provides centralized hardware resource management and execution serialization across all heavy neural backbones in [Milimo Music](../overview.md).

## Architectural Purpose

Milimo Music runs multiple generative AI and neural audio/video models locally on Apple Silicon (MPS Unified Memory) and NVIDIA (CUDA):
- **MiniMax Music 3**: MLX mxfp4 / DiT flow-matching generation (~14 GB).
- **MuLaCover 3B**: Autoregressive symbolic cover generation (~6.2 GB).
- **BS-Roformer & HTDemucs**: 6-stem neural source separation (~3.2 GB).
- **MuScriptor / MT3**: Note-level multi-instrument transcription (~2.1 GB).
- **Wan 2.1 DiT (14B / 1.3B) & LivePortrait**: Diffusion music video rendering (~12–16 GB).
- **Neural SVC**: Zero-shot pitch-adaptive vocal timbre transfer (~1.8 GB).

Concurrent execution of these models without serialization causes catastrophic out-of-memory (OOM) errors and MPS unified memory thrashing. The `GlobalHardwareCoordinator` acts as an authoritative async device mutex ensuring only one heavy neural workload commands the accelerator at any given instant.

## Core API & Implementation

```python
class GlobalHardwareCoordinator:
    _lock = asyncio.Lock()
    _active_consumer: str = "idle"
    _consumer_acquired_at: float = 0.0

    @classmethod
    async def acquire_device(cls, consumer: str, timeout: Optional[float] = 600.0) -> bool:
        """Acquire exclusive accelerator device access with automatic timeout guard."""
        ...

    @classmethod
    def release_device(cls, consumer: str) -> None:
        """Release accelerator device and perform aggressive memory reclamation."""
        ...

    @classmethod
    def flush_memory(cls) -> Dict[str, Any]:
        """Evict cached tensors and call garbage collection across CUDA and MPS."""
        ...

    @classmethod
    def get_telemetry(cls) -> Dict[str, Any]:
        """Fetch real-time accelerator and system memory metrics."""
        ...
```

### Multi-Backend Memory Reclamation

When `release_device()` or `flush_memory()` is triggered:
1. **Garbage Collection**: Invokes `gc.collect()` to purge unreferenced Python tensor wrappers.
2. **CUDA Empty Cache**: Invokes `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`.
3. **MPS Empty Cache**: Invokes `torch.mps.empty_cache()` and queries Apple Silicon unified memory allocation via `psutil`.

## Endpoints

- `GET /system/telemetry`: Returns real-time VRAM allocation, total capacity, percentage usage, and current active consumer:
  ```json
  {
    "device_type": "mps",
    "device_name": "Apple Silicon Unified Memory (MPS)",
    "active_consumer": "idle",
    "vram_allocated_mb": 0.0,
    "vram_reserved_mb": 44423.1,
    "vram_total_mb": 131072.0,
    "usage_percent": 0.0,
    "lock_held": false
  }
  ```
- `POST /system/flush`: Immediately flushes dormant accelerator tensor caches and returns reclaimed memory metrics.

## Frontend Telemetry Bar & `Ctrl+E` Switcher

- **`HardwareTelemetryBar.tsx`**: Top-bar pill displaying active device type (`MPS` / `CUDA` / `CPU`), interactive usage progress bar, allocated/total memory, active consumer tag, and a one-click cache purge button.
- **`EngineSwitcherModal.tsx`**: Modal summoned via `Ctrl+E` or clicking the telemetry pill. Catalogues all active neural engines, memory footprints, and provides granular hardware monitoring.

## Related Pages

- [Architecture](../architecture.md)
- [Video Studio](video-studio.md)
- [Generation Pipeline](../concepts/generation-pipeline.md)
- [Neural SVC](neural-svc.md)
- [MuLaCover](mulacover.md)
