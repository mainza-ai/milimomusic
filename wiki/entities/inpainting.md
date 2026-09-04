---
title: Repair Segment (Inpainting Service)
type: entity
created: 2026-08-19
updated: 2026-08-19
sources: [sources/inpainting-debug.md, sources/readme.md]
tags: [inpainting, repair, repair-segment, heartcodec, heartmula]
aliases: [Repair Segment, InpaintingService, Inpainting]
---

# Repair Segment (Inpainting Service)

**Repair Segment** (Beta) lets the user fix a specific part of a generated track without
regenerating the entire song: select a time range and the AI rewrites just that segment
while preserving the surrounding context. It is implemented by the backend
`InpaintingService` using **LM-guided repair**.

## Current status (2026-01-24)
- **LM-Guided Repair fully implemented and stable.**
- Two-stage generation: [HeartMuLa](heartmula.md) (semantic tokens) +
  [HeartCodec](heartcodec.md) (acoustic reconstruction).
- Parameters: near-deterministic (`temp=0.2`, `topk=30`, `cfg_scale=1.0`).
- Pending: user verification of vocal consistency after latest tuning.

## Technique
See [LM-guided inpainting](../concepts/lm-guided-inpainting.md) for the full mechanism and
the history of failed approaches (blind masking → silence, copy tokens → repetition, etc.).

## Key codec capabilities used
- `HeartCodec.inpaint` — masking support in the codec.
- `HeartCodec.encode` — re-tokenization support.

## Platform notes (debug history)
- Mono/stereo dimension fixes (`[1, 1, T]` constraint).
- SQLite 32-bit integer overflow fix (`torch.seed()` clamping).
- MPS/Mac compat: CPU offload for `ScalarModel.decode/encode`, RNG offload
  (MPS placeholder storage bug), device-type safety checks.

## Related pages
- [LM-guided inpainting](../concepts/lm-guided-inpainting.md) | [HeartCodec](heartcodec.md)
- [HeartMuLa](heartmula.md) | [Backend & API](backend-api.md)
- [Inpainting Debug Log source](../sources/inpainting-debug.md)
