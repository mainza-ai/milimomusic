---
title: Global Hardware Coordinator & Auto-Tune Manager
type: entity
created: 2026-09-15
updated: 2026-09-16
sources: [sources/v2-refactor-plan.md, sources/readme.md, sources/maestro-creative-studio.md]
tags: [hardware, gpu, vram, coordinator, mps, cuda, telemetry, architecture, autotune, profiles, oom]
aliases: [Hardware Coordinator, GlobalHardwareCoordinator, VRAM Manager, HardwareAutoTune]
---

# Global Hardware Coordinator & Auto-Tune Manager

The **Global Hardware Coordinator** (`backend/app/core/hardware_lock.py` and `hardware_autotune.py`) provides centralized hardware resource management, lock serialization, empirical profile auto-tuning, and OOM self-healing across all heavy neural backbones in [Milimo Music](../overview.md).

## Architectural Purpose

Milimo Music runs multiple generative AI and neural audio/video models locally on Apple Silicon (MPS Unified Memory) and NVIDIA (CUDA):
- **MiniMax Music 3**: MLX mxfp4 / DiT flow-matching generation (~14 GB).
- **YuE2 3B**: 48 kHz stereo autoregressive + diffusion audio synthesis (~12–16 GB).
- **MuLaCover 3B**: Autoregressive symbolic cover generation (~6.2 GB).
- **BS-Roformer & HTDemucs**: 6-stem neural source separation (~3.2 GB).
- **MuScriptor / MT3**: Note-level multi-instrument transcription (~2.1 GB).
- **Wan 2.1 DiT (14B / 1.3B) & MiniMax H3 (33B)**: Diffusion music video rendering (~12–24 GB).
- **Neural SVC**: Zero-shot pitch-adaptive vocal timbre transfer (~1.8 GB).

Concurrent execution of these models without serialization causes catastrophic out-of-memory (OOM) errors and MPS unified memory thrashing. The `GlobalHardwareCoordinator` acts as an authoritative async device mutex ensuring only one heavy neural workload commands the accelerator at any given instant.

## Hardware Auto-Tune & Empirical Profiles

Modeled after [Maestro](../sources/maestro-creative-studio.md), the system auto-tunes hardware on startup:

| Profile | Tier | Configuration | Strategy |
|---|---|---|---|
| **1.0** | High VRAM + High RAM | VRAM $\ge 24\text{ GB}$, RAM $\ge 60\text{ GB}$ | 100% VRAM resident models. Zero streaming latency. |
| **2.0** | Balanced Streaming | VRAM $12-24\text{ GB}$, RAM $\ge 31\text{ GB}$ | Transformer weights pinned in host RAM, streamed per block. |
| **3.0 / 3.5** | High VRAM + Low RAM | VRAM $\ge 24\text{ GB}$, RAM $< 28\text{ GB}$ | Prioritizes VRAM residency, limits host memory caching. |
| **4.0 / 4.5** | Consumer Production | VRAM $12-16\text{ GB}$, RAM $16-31\text{ GB}$ | Aggressive memory cleanup after each inference stage. |
| **5.0** | Max Layer Offload | VRAM $< 12\text{ GB}$ or CPU | Quantized INT8 ConvRot / NVFP4 execution with CPU offload. |

See [Hardware Auto-Tune, Memory Profiles & OOM Self-Healing](../concepts/hardware-autotune-memory-profiles.md) for detailed mathematical criteria and benchmarks.

## Safety Coefficient & Headroom Reservation

To prevent sudden spikes in PyTorch activation caching and VAE decoding from crashing the server, memory allocations are bounded by an empirical safety coefficient:
$$\text{MaxUsableVRAM} = \text{TotalPhysicalVRAM} \times C_{\text{safety}}$$
- $C_{\text{safety}} = 0.80$ for cards with $\ge 12\text{ GB}$ VRAM.
- $C_{\text{safety}} = 0.70$ for cards with $< 12\text{ GB}$ VRAM.
- $C_{\text{safety}} = 0.75$ on Apple Silicon MPS Unified Memory.

## Scoped CPU Execution (`cpu_scoped()`)

Audio loading and format conversion (via Librosa, torchaudio, or soundfile) are executed inside `cpu_scoped()` context managers to ensure helper allocations never contaminate CUDA/MPS memory pools or trigger heap fragmentation.

## OOM Interception & Self-Healing Telemetry

If an out-of-memory exception occurs:
1. Catches the exception before process termination.
2. Dumps memory diagnostics (process RSS, system RAM, PyTorch allocated/reserved/peak).
3. Evicts dormant tensor caches via `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`.
4. Lowers the active safety coefficient by $0.10$ and surfaces an in-app banner for one-click retry with reduced batch size.

## Cross-Modal Eviction Bus
To prevent cross-modal memory collisions (e.g. resident LLM inference engines like oMLX, Ollama, LM Studio colliding with FLUX.2 image diffusion or Wan 2.1 video diffusion or MiniMax Music 3), the coordinator acts as a central **Eviction Bus**:
- Workloads register eviction callbacks: `register_eviction_hook(modality, callback)`.
- Modalities registered include: `"llm"` (oMLX, Ollama, LM Studio), `"image_gen"` (FLUX.2 MLX / diffusers), `"audio_gen"` (MiniMax Music 3, Stable Audio Open, MusicGen), `"separation"` (HTDemucs, BS-Roformer), `"transcription"` (MuScriptor), `"video_gen"` (Wan 2.1, LTX-Video), and `"voice_conversion"` (Neural SVC).
- When entering `scoped_device(consumer, modality=...)`, the coordinator triggers `evict_all_except(modality)` to flush all warm models from other modalities before granting device access.
- See [Cross-Modal Model Lifecycle & Immediate Eviction Architecture](../concepts/cross-modal-model-lifecycle.md) for full protocol details.

## Core API & Implementation

```python
class GlobalHardwareCoordinator:
    _lock = asyncio.Lock()
    _active_consumer: str = "idle"
    _consumer_acquired_at: float = 0.0
    _eviction_hooks: Dict[str, Callable[[], None]] = {}

    @classmethod
    def register_eviction_hook(cls, modality: str, hook: Callable[[], None]) -> None:
        """Register model unloading callback for a modality domain."""
        cls._eviction_hooks[modality] = hook

    @classmethod
    def evict_all_except(cls, active_modality: Optional[str] = None) -> None:
        """Evict cached models across other modalities before running a new workload."""
        for mod, hook in cls._eviction_hooks.items():
            if mod != active_modality:
                try: hook()
                except Exception: pass
        cls.flush_memory()

    @classmethod
    async def acquire_device(cls, consumer: str, modality: Optional[str] = None, timeout: Optional[float] = 600.0) -> bool:
        """Acquire exclusive accelerator device access with automatic inter-modality eviction."""
        ...

    @classmethod
    def release_device(cls, consumer: str) -> None:
        """Release accelerator device and perform aggressive memory reclamation."""
        ...

    @classmethod
    def flush_memory(cls) -> Dict[str, Any]:
        """Evict cached tensors and call garbage collection across MLX Metal, CUDA, MPS, and glibc."""
        ...

    @classmethod
    def get_telemetry(cls) -> Dict[str, Any]:
        """Fetch real-time accelerator and system memory metrics."""
        ...
```

## Endpoints

- `GET /system/telemetry`: Returns real-time VRAM allocation, total capacity, percentage usage, active consumer, and active performance profile:
  ```json
  {
    "device_type": "cuda",
    "device_name": "NVIDIA GeForce RTX 4090",
    "active_consumer": "idle",
    "vram_allocated_mb": 0.0,
    "vram_reserved_mb": 1240.0,
    "vram_total_mb": 24564.0,
    "usage_percent": 5.05,
    "profile": 1.0,
    "profile_name": "Profile 1 — Maximum Performance",
    "safety_coefficient": 0.80,
    "lock_held": false
  }
  ```
- `POST /system/flush`: Immediately executes cross-modal eviction, flushes dormant MLX Metal, PyTorch CUDA/MPS caches, and trims host heap pages.

## Frontend Telemetry Bar & `Ctrl+E` Switcher

- **`HardwareTelemetryBar.tsx`**: Top-bar pill displaying active device type (`MPS` / `CUDA` / `CPU`), interactive usage progress bar, allocated/total memory, active consumer tag, and a one-click cache purge button.
- **`EngineSwitcherModal.tsx`**: Modal summoned via `Ctrl+E` or clicking the telemetry pill. Catalogues all active neural engines, memory footprints, and provides granular hardware monitoring.

## Related Pages

- [Architecture](../architecture.md)
- [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md)
- [Video Studio](video-studio.md)
- [Hardware Auto-Tune](../concepts/hardware-autotune-memory-profiles.md)
- [Stable Audio Open](stable-audio-open.md)
- [Meta MusicGen](musicgen.md)
- [YuE2 Music](yue2-music.md)
- [Durable Task Queue](durable-task-queue.md)
- [Generation Pipeline](../concepts/generation-pipeline.md)
- [Neural SVC](neural-svc.md)
- [MuLaCover](mulacover.md)

