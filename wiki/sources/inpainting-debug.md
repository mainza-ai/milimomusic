---
title: Source — Inpainting & Glitch Repair Debug Log
type: source
created: 2026-08-19
updated: 2026-08-19
tags: [source, inpainting, repair]
aliases: [INPAINTING_DEBUG]
---

# Source — Inpainting & Glitch Repair Debug Log

The **In-Painting & Glitch Repair Debug Log** records how the repair feature was
engineered. **Raw location:** `docs/INPAINTING_DEBUG.md` (immutable source).

## Key contents (summary)
- **Status (2026-01-24)**: LM-Guided Repair fully implemented and stable;
  two-stage generation; near-deterministic params (`temp=0.2`, `topk=30`, `cfg_scale=1.0`);
  vocal-consistency verification pending.
- **The final solution**: LM-guided repair — Stage 1 HeartMuLa semantic generation of NEW
  tokens; Stage 2 HeartCodec acoustic reconstruction with 100ms crossfade.
- **Failure history**: blind masking → silence; copy tokens → repetition; latent blur →
  phase smear; gradient masking → silence.
- **Fix history**: infra/stability (InpaintingService, codec inpaint/encode, dimension
  fixes, SQLite seed overflow), MPS/Mac compat (CPU offload, RNG offload, device checks).

## Entities it feeds
- [Repair Segment](../entities/inpainting.md),
  [LM-guided inpainting](../concepts/lm-guided-inpainting.md),
  [HeartCodec](../entities/heartcodec.md), [HeartMuLa](../entities/heartmula.md).

## Related pages
- [Repair Segment](../entities/inpainting.md) | [Index](../index.md)
