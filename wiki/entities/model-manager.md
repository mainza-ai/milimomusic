---
title: Model Manager
type: entity
created: 2026-08-20
updated: 2026-09-07
tags: [model, hardware, tree, variants, download, huggingface, multi-modal, modality]
aliases: [ModelManager, model tree, hardware profile, Hugging Face Hub]
---

# Model Manager

The **Model Manager** (`backend/app/services/model_manager.py` and frontend `ModelsManagerModal.tsx`) manages Milimo's **multi-modal model catalog**, active provider switching, real background weight downloads, hardware profile detection, and live Hugging Face Hub discovery.

## 1. Multi-Modal Catalog (Audio, Image, Video)

The model tree encompasses 23 official and open-weights variants across three modalities:

### A. Audio Generation (MiniMax Music 3 & HeartMuLa)
- **Apple Silicon MLX Suite**: `mlx-community/MiniMax-Music3-mxfp4` (8.28 GB, 4-bit block-quantized), `4bit`, `6bit`, `8bit`, and `bf16` (26.55 GB).
- **Official PyTorch**: `MiniMaxAI/MiniMax-Music3` (28.5 GB, BF16 reference weights for CUDA/MPS).
- **ComfyUI / CUDA INT8**: `Comfy-Org/MiniMax-Music-3` (11.3 GB, INT8 quantized for 16GB consumer GPUs).
- **Universal GGUF Q4**: `molbal/Minimax-Music3-GGUF` (7.7 GB, quantized for Windows/Linux/CPU execution).
- **Legacy HeartMuLa**: `heartmula_3b` (6.2 GB, autoregressive HeartCodec pipeline).

### B. Image Diffusion & Album Cover Art
- **Black Forest Labs FLUX.2**:
  - `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0 sub-second 4B distilled generator).
  - `mlx-community/FLUX.2-Klein-4B-4bit` (2.8 GB, 4-bit Apple Silicon MLX).
  - `black-forest-labs/FLUX.2-klein-9B` (18.0 GB, 4-step distilled Qwen3 embedder).
  - `black-forest-labs/FLUX.2-dev` (64.0 GB, 32B flagship rectified flow transformer).
- **FLUX.1 Reference**: `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell` (and MLX 4-bit).
- **SDXL Turbo**: `stabilityai/sdxl-turbo` (single-step real-time synthesis).

### C. Generative Video Models
- **MiniMax Hailuo H3**: `MiniMaxAI/MiniMax-H3` (24.0 GB 33B Omni-Modal DiT, up to 15.0s clips), GGUF Q4 (`unsloth/MiniMax-H3-GGUF`, 14.2 GB), and **MLX 8-bit** (`pipenetwork/MiniMax-H3-MLX-8bit`, 35.3 GB on disk / ~21.5 GB resident, Apple Silicon 32 GB+; transformer weights only — needs upstream VAEs + Qwen3-VL-32B encoder + `minimax-h3-mlx` pipeline).
- **Wan 2.1**: `Wan-AI/Wan2.1-T2V-1.3B` (Lightweight 1.3B DiT) and `Wan-AI/Wan2.1-T2V-14B` (Flagship 14B DiT, 5.0s clips).
- **THUDM CogVideoX 1.5**: `THUDM/CogVideoX1.5-5B` (5B causal 3D VAE, up to 10.0s clips).
- **Tencent HunyuanVideo**: `tencent/HunyuanVideo` (13B visual DiT, up to 15.0s clips).

> [!WARNING] H3 weights are governed by the **MiniMax H3 Community License**
> (not open weights: territorial exclusion, separate authorization above $20M
> yearly revenue). The catalog reflects this; older "MiniMax Open Weights"
> labels on H3 entries were corrected on 2026-09-07.

## 2. Modality Taxonomy & Download Routing

Modality decisions (`audio` | `image` | `video`) have a single source of truth:
`backend/app/services/modality.py`. Full precedence chain, the 2026-09-07
H3→`models/audio/` root-cause case study, and the relocate/delete/guard rules
live in [Modality Taxonomy](../concepts/modality-taxonomy.md). Summary of the
behavioral contract:

- `POST /models/download { repo_id, model_id?, category? }` — explicit
  `category` (the UI dropdown) wins and is recorded as
  `category_source: "user"`; otherwise inference uses `pipeline_tag` + HF
  tags + filenames (never filenames alone). Duplicate in-flight downloads to
  the same target dir are rejected (409).
- Download worker: resume-aware (complete shards skipped and credited),
  3-attempt backoff per file, `.lock` cleanup, cancel-handle release.
- Recategorize relocates bytes; delete frees bytes across all modality roots;
  unknown repos resolve to `audio` + `needs_review` (never `"custom"`).
- `HuggingFaceAudioProvider` refuses video-diffusion weights with an explicit
  error.

> [!WARNING] Correctly placed video weights still have **no DiT inference
> backend** (`VIDEO_MODEL_PATH` is written but never read; `video_service.py`
> is assembly-only). Download/registry is production-grade; execution is open.

## 3. Live Hugging Face Hub Search & Custom Registry

- **Hub Search Engine**: Uses `huggingface_hub.HfApi().list_models(search=query, sort="downloads", limit=limit)` with pipeline filter chips (`text-to-audio`, `text-to-image`, `text-to-video`).
- **Direct Repository Downloader**: Allows users to input any Hugging Face model ID (e.g. `facebook/musicgen-small` or `stabilityai/stable-audio-open-1.0`) and select target modality — the selection is sent as `category` and honored over inference.
- **Custom Model Registry**: Stores user-downloaded models in the canonical `data/models/custom_models.json` (single resolved path; a legacy `backend/data/` copy may exist as backup but is never read once the canonical file exists). Custom models are dynamically merged into `get_model_tree()` and registered into `ProviderRegistry` via `HuggingFaceAudioProvider`.
- **Deletion**: Unregisters and deletes downloaded custom model snapshot files across all modality roots.

## 4. Strict Download Policy & Hardware Tiers

- **Empty System Auto-Download**: On fresh installations with zero audio models installed, the system automatically downloads only the single smallest audio model (`mxfp4` on macOS, `GGUF Q4` on Linux/Windows).
- **On-Demand**: Image and video models are 100% on-demand and never auto-downloaded without explicit user consent.
- **Hardware Profile Detection**: Detects CUDA and Apple Silicon MPS hardware tiers, setting active defaults appropriately.

## 5. API Surface
- `GET /models/tree` — Full multi-modal catalog and installed states.
- `GET /models/capabilities` — Manifest for active generation engines.
- `GET /models/hardware` — Local hardware detection profile.
- `GET /models/search` — Live query search against Hugging Face Hub.
- `POST /models/download` — body `{ repo_id?, model_id?, category? }`; background worker with per-file progress, `category_source` in the download record.
- `DELETE /models/custom/{model_id}` — Remove custom model registry entry and snapshot bytes.
- `PATCH /models/custom/{model_id}` — Update metadata; a `category` change relocates bytes on disk.
- `POST /models/active/{provider_id}` — Switch active generation engine.

## Related pages
- [Generation Provider](generation-provider.md) | [MiniMax Music 3](minimax-music3.md) | [Video Studio](video-studio.md) | [Backend & API](backend-api.md) | [Modality Taxonomy](../concepts/modality-taxonomy.md)
