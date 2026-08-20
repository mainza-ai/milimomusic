---
title: Generation Provider Abstraction
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [generation, provider, abstraction, architecture, backend]
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

`GenerationCapabilities` fields: `provider_id`, `display_name`, `version`,
`max_duration_sec`, `supports_structured_caption`, `supports_section_tags`, `supports_lora`,
`supports_voice_conversion`, `supports_track_extension`, `supports_segment_repair`,
`recommended_hardware`, `license_class`, `default_sample_rate`.

`HardwareTier` enum: `entry_cpu`, `mid_single_gpu`, `high_dual_gpu`.

## Registry (`providers/registry.py`)
`ProviderRegistry` (singleton) registers providers and tracks the **active provider**:

- **Defaults**: `minimax_music3` (active by default) and `heartmula` (legacy/local).
- `register_provider()`, `get_provider()`, `set_active_provider()`, `list_capabilities()`,
  `get_active_capabilities()`.
- Falls back to MiniMax if the requested provider is unknown.
- A module-level `provider_registry` singleton is used throughout the backend.

## Concrete providers
- [MiniMax Music 3](minimax-music3.md) — default; structured captions; up to 300s.
- [HeartMuLa](heartmula.md) — legacy/local; non-structured tags; 48kHz.

## In the pipeline
The active provider is resolved inside the [orchestration pipeline](../concepts/generation-pipeline.md)
based on `GenerationRequest.model_provider`, and its capabilities are surfaced via the
`/models/*` API endpoints (see [Backend & API](backend-api.md)).

## Related pages
- [Orchestration pipeline](../concepts/generation-pipeline.md) | [MiniMax Music 3](minimax-music3.md)
- [HeartMuLa](heartmula.md) | [Model Manager](model-manager.md) | [Architecture](../architecture.md)
