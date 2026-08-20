---
title: LoRA Fine-Tuning
type: concept
created: 2026-08-19
updated: 2026-08-19
sources: [sources/training-studio-guide.md, sources/readme.md]
tags: [lora, finetuning, training, adapter]
aliases: [LoRA, LoRA training]
---

# LoRA Fine-Tuning

**LoRA** (Low-Rank Adaptation) is an efficient fine-tuning method used by the
[Training Studio](../entities/training-studio.md) to teach [HeartMuLa](../entities/heartmula.md)
new *styles* (genre or artist-like biases) from a small custom dataset.

## Concept
Instead of updating all model weights, LoRA trains small low-rank adapter matrices that
steer the base model. Result: fast, lightweight (~100MB) adapters versus full fine-tuning
(~6GB, best quality, heavier resources).

## In the Training Studio
- **Method**: `LoRA` (fast) vs `Full` (best quality).
- **LoRA Rank**: adapter complexity, 8–32 (default 8); higher = more expressive.
- Train over **Epochs** (default 3) at **Learning Rate** (default 0.0001).
- Requires **≥5 audio files** per dataset; matching `.txt` captions add lyrics/captions.
- Checkpoints stored under `backend/data/checkpoints/{id}/adapter_model.safetensors`;
  **Activating** one loads the custom weights into the generation engine.
- Loss in the Jobs tab — **lower is better**; if flat/rising, lower the learning rate.

> [!NOTE] LoRA in the Training Studio teaches a *style* bias, not a specific person's vocal
> identity. Voice *identity* cloning is a separate v2 concern (RVC-based SVC) — see [roadmap](../roadmap.md).

## Related pages
- [Training Studio](../entities/training-studio.md) | [HeartMuLa](../entities/heartmula.md)
- [Roadmap (v2)](../roadmap.md) | [Training Studio Guide source](../sources/training-studio-guide.md)
