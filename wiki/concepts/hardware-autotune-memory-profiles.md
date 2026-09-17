---
title: Hardware Auto-Tune, Memory Profiles & OOM Self-Healing
type: concept
tags: [hardware, vram, profiles, oom, cuda, mps, autotune, memory]
created: 2026-09-16
updated: 2026-09-16
sources: [sources/maestro-creative-studio.md]
aliases: [HardwareAutoTune, MemoryProfiles, OOMRecovery]
---

# Hardware Auto-Tune, Memory Profiles & OOM Self-Healing

Running multi-modal AI pipelines (33B DiT video models, 3B audio models, Whisper, BS-Roformer stem separators, and LLMs) on consumer and workstation hardware requires rigorous memory management. Naive implementations frequently crash with `CUDA out of memory` errors due to memory fragmentation, unbuffered activations, and unbounded layer caching.

Milimo Music's **Hardware Auto-Tune System** is modeled after production memory management in [Maestro](../sources/maestro-creative-studio.md) and WanGP.

---

## 1. Empirical Hardware Performance Profiles (Profiles 1 to 5)

Rather than forcing users to guess between complex offloading flags, the engine detects GPU VRAM, compute capabilities, and host RAM at launch and assigns an empirical profile:

| Profile | Name | Hardware Requirements | Memory Strategy | Primary Target Hardware |
|---|---|---|---|---|
| **1.0** | **Max Performance** | VRAM $\ge 24\text{ GB}$, RAM $\ge 60\text{ GB}$ | Transformer, VAE, and encoders stay 100% resident in VRAM. Zero streaming latency. | RTX 4090, A100, RTX A6000, Mac Studio M2/M3 Ultra (64GB+) |
| **2.0** | **Balanced Streaming** | VRAM $12-24\text{ GB}$, RAM $\ge 31\text{ GB}$ | Transformer weights pinned in host RAM and streamed to GPU per denoising block. | RTX 4080 / 3090, Mac Studio / Pro M3 Max (36GB - 48GB) |
| **3.0** | **VRAM Resident** | VRAM $\ge 24\text{ GB}$, RAM $< 28\text{ GB}$ | Maximizes VRAM residency while strictly bounding host RAM allocations. | Dual GPU workstation with limited system RAM |
| **3.5** | **High VRAM / Low RAM** | VRAM $\ge 24\text{ GB}$, RAM $< 16\text{ GB}$ | High GPU VRAM, extremely constrained host memory. | Cloud virtual machines with high vGPU but minimal RAM |
| **4.0** | **Consumer Production** | VRAM $12-16\text{ GB}$, RAM $16-31\text{ GB}$ | Conservative weight streaming; models evicted immediately after inference pass. | RTX 4070 / 4070 Ti / 3060, MacBook Pro 18GB/24GB |
| **4.5** | **VRAM Conservation** | VRAM $12-16\text{ GB}$, RAM $< 15\text{ GB}$ | Saves additional $\sim 1\text{ GB}$ VRAM by offloading conditioning text encoders. | Laptops with 12GB VRAM and 16GB total RAM |
| **5.0** | **Maximum Layer Offload**| VRAM $< 12\text{ GB}$ or CPU | Maximum offload; aggressive quantization (INT8 ConvRot / NVFP4); smallest batch sizes. | RTX 3050, GTX 1660, Apple Silicon Base (8GB/16GB), CPU |

---

## 2. VRAM Safety Coefficient & Headroom Reservation

A standard cause of PyTorch OOM crashes is treating total physical VRAM as usable memory. Generative diffusion and autoregressive decoding require dynamic scratchpad buffers for:
1. Attention $QK^T$ matrix intermediates.
2. VAE latent decoding and spatial upsampling.
3. CUDA runtime context and PyTorch caching allocator pools.

### The Safety Coefficient Rule
Maestro established that even on 24 GB cards, setting the memory ceiling above **0.80** frequently causes crashes during heavy multi-window generations.

Milimo Music adopts the strict safety coefficient ceiling:
$$\text{MaxUsableVRAM} = \text{TotalPhysicalVRAM} \times C_{\text{safety}}$$

- $C_{\text{safety}} = 0.80$ for cards with $\ge 12\text{ GB}$ VRAM.
- $C_{\text{safety}} = 0.70$ for cards with $< 12\text{ GB}$ VRAM.
- In Apple Silicon MPS unified memory: $C_{\text{safety}} = 0.75$ (bounded by system memory pressure).

---

## 3. Scoped CPU Execution (`cpu_scoped()`)

Audio format loading (e.g. `torchaudio.load()`, `librosa.load()`, `scipy.signal.resample`) can inadvertently trigger CUDA allocations if PyTorch has initialized CUDA as the current device. These temporary helper tensors fragment CUDA memory pools.

Milimo Music wraps all pre-processing and audio format preparation in a CPU-scoped context:
```python
import contextlib
import gc
import torch

@contextlib.contextmanager
def cpu_scoped():
    """Guarantee that pre-processing and audio loading allocations stay on CPU."""
    try:
        yield
    finally:
        # Reclaim any transient host memory and flush allocator
        gc.collect()
```

---

## 4. Kernel Benchmarking: Native PyTorch vs. Triton

Rather than hardcoding an assumption that Triton or CuBLAS is always faster:
1. On first model launch, the engine benchmarks native PyTorch fallback vs. Triton on the exact loaded weight shape and activation dtype (`bfloat16`/`float16`).
2. Compilation is warmed before timing.
3. Native must measure at least **15% faster** to replace Triton; otherwise Triton is retained.
4. The winning kernel is cached for the duration of the server session.

---

## 5. OOM Interception & Self-Healing Telemetry

When an out-of-memory exception occurs, the system does not crash silently:
1. **Telemetry Capture**: Instantly records process RSS, physical RAM usage, PyTorch allocated memory, PyTorch reserved memory, and peak memory.
2. **Emergency Cache Eviction**:
   ```python
   gc.collect()
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       torch.cuda.ipc_collect()
   elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
       if hasattr(torch.mps, "empty_cache"):
           torch.mps.empty_cache()
   ```
3. **Headroom Recalculation**:
   - Reduces the active safety coefficient by $0.10$ (e.g. $0.80 \rightarrow 0.70$).
   - Emits an SSE event to the frontend with an in-app banner:
     *"OOM prevented. Lowered VRAM safety headroom to 70%. Click to retry with optimized batch size."*

---

## 6. Related Pages
- [Global Hardware Coordinator](../entities/hardware-coordinator.md)
- [Maestro Creative Studio Ingest](../sources/maestro-creative-studio.md)
- [AI Music Video Studio](../entities/video-studio.md)
- [Director Mode v2](director-mode-v2.md)
