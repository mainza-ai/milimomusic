---
title: Source — Training Studio Guide
type: source
created: 2026-08-19
updated: 2026-08-19
tags: [source, training, studio]
aliases: [TRAINING_STUDIO]
---

# Source — Training Studio Guide

The **Training Studio Guide** documents the fine-tuning UI and API.
**Raw location:** `docs/TRAINING_STUDIO.md` (immutable source).

## Key contents (summary)
- **What it is**: fine-tune HeartMuLa models on custom audio for custom styles.
- **Glassmorphism UI note** — refresh browser if old solid styling shows.
- **Quick start**: create dataset → upload ≥5 audio files → configure (LoRA/Full) →
  monitor Jobs → Activate in Models.
- **Data storage layout**: `backend/data/{datasets,jobs,checkpoints}/{id}/{...}`.
- **API endpoints**: datasets, audio upload, training jobs, checkpoints, activate.
- **Metrics**: Loss (lower better), status badges; tip: lower LR if loss flat.

## Entities it feeds
- [Training Studio](../entities/training-studio.md),
  [LoRA fine-tuning](../concepts/lora-finetuning.md),
  [Backend & API](../entities/backend-api.md), [HeartMuLa](../entities/heartmula.md).

## Related pages
- [Training Studio](../entities/training-studio.md) | [Index](../index.md)
