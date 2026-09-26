---
title: Generation Provider Abstraction
type: entity
created: 2026-08-20
updated: 2026-09-25
tags: [generation, provider, abstraction, architecture, backend, stable-audio, musicgen, lifecycle]
aliases: [GenerationProvider, ProviderRegistry, provider abstraction]
---

# Generation Provider Abstraction

The **generation-provider layer** decouples Milimo's generation flow from any single model.
It is the §3.1 "capabilities, not model names" design from the [v2 plan](../roadmap.md),
now implemented. Backend lives in `backend/app/providers/`.

## Interface (`providers/base.py`)
`GenerationProvider` (ABC) defines:

- `get_capabilities()` → a **capability manifest** (`GenerationCapabilities`) that drives the UI.
- `initialize()`, `is_ready()` — load/connect model weights.
- `generate()`, `extend()`, `repair_segment()` — the core generation operations.
- `unload() -> bool` — releases model weights, unregisters acceleration hooks, and triggers accelerator cache reclamation conforming to the [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md).

`GenerationCapabilities` fields: `provider_id`, `display_name`, `version`,
`max_duration_sec`, `supports_structured_caption`, `supports_section_tags`, `supports_lora`,
`supports_voice_conversion`, `supports_track_extension`, `supports_segment_repair`,
`recommended_hardware`, `license_class`, `default_sample_rate`.

`HardwareTier` enum: `entry_cpu`, `mid_single_gpu`, `high_dual_gpu`.

## Registry (`providers/registry.py`)
`ProviderRegistry` (singleton) registers providers and tracks the **active provider**:

- **Defaults**: `minimax_music3` (Apple Silicon MLX default), `stable_audio_open` (CUDA/MPS cross-platform default), `musicgen` (lightweight/CPU default), and `heartmula` (legacy/local).
- `register_provider()`, `get_provider()`, `set_active_provider()`, `list_capabilities()`,
  `get_active_capabilities()`.
- Falls back to MiniMax on Apple Silicon, or Stable Audio / MusicGen on Linux/CUDA/CPU if the requested provider is unknown.
- A module-level `provider_registry` singleton is used throughout the backend.

## Concrete providers
- [MiniMax Music 3](minimax-music3.md) — Apple Silicon MLX default; structured captions; up to 300s.
- [Stable Audio Open 1.0](stable-audio-open.md) — Cross-platform DiT default (CUDA/MPS/CPU); native 44.1 kHz stereo.
- [Meta MusicGen](musicgen.md) — Cross-platform autoregressive model; melody-guided conditioning; CPU-friendly.
- [YuE2 48kHz Stereo](yue2-music.md) — High-VRAM dual-stage full-song generator.
- [HeartMuLa](heartmula.md) — Legacy/local offline model; 44.1 kHz.

## In the pipeline
The active provider is resolved inside the [orchestration pipeline](../concepts/generation-pipeline.md)
based on `GenerationRequest.model_provider`. On completion of Step 1, the pipeline immediately calls `provider.unload()` to release memory before Step 2 (stem separation) begins.

## Related pages
- [Orchestration pipeline](../concepts/generation-pipeline.md) | [MiniMax Music 3](minimax-music3.md)
- [Stable Audio Open](stable-audio-open.md) | [Meta MusicGen](musicgen.md) | [YuE2](yue2-music.md)
- [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md) | [Model Manager](model-manager.md) | [Architecture](../architecture.md)

