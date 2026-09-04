---
title: Model Manager
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [model, hardware, tree, variants, download]
aliases: [ModelManager, model tree, hardware profile]
---

# Model Manager

The **Model Manager** (`services/model_manager.py` + frontend `ModelsManagerModal`) manages
Milimo's **model tree** and hardware tiers. It surfaces which weights are installed vs.
need downloading, and detects the local hardware profile.

## Hardware detection
`detect_hardware()` → `HardwareProfile`:
- Detects `has_cuda` / `has_mps` (via `torch`; on Apple Silicon Darwin/arm64 assumes MPS).
- Computes a `HardwareTier` (`entry_cpu`, `mid_single_gpu`, `high_dual_gpu`) and flags
  `can_run_minimax_full`, `can_run_heartmula`.

## Model tree
`get_model_tree()` → `ModelVariant[]`:
- **MiniMax Music 3 (bfloat16 base)** — `minimax_music3_bf16`, Qwen3+RVQ8+Flow-Matching DiT,
  ~28.5 GB, BF16, **default**, installed state from the local snapshot.
- **MiniMax Music 3 (8-bit quantized)** — `minimax_music3_int8`, ~14.2 GB, INT8, not installed.
- **HeartMuLa-3B** — `heartmula_3b`, autoregressive + HeartCodec, ~6.2 GB, Apache-2.0,
  installed from `heartlib/ckpt`.
- Each carries `license`, `recommended_hardware`, `local_path`, `is_installed`.

## Missing-dependency checking
`check_missing_dependencies(model_id)` → `{missing, size_gb, local_path, message}`. The
frontend shows an **Installed & Ready** vs **Download** state (download UI is simulated)
so generation refuses-or-prompts before running rather than failing mid-job
(v2 plan §3.4).

## API surface
- `GET /models/tree`, `GET /models/capabilities`, `GET /models/hardware`,
  `GET /models/check/{model_id}`, `POST /models/active/{provider_id}`.
- See [Backend & API](backend-api.md).

## Related pages
- [Generation provider](generation-provider.md) | [MiniMax Music 3](minimax-music3.md)
- [Backend & API](backend-api.md) | [Roadmap (v2)](../roadmap.md)
