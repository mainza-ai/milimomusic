---
title: LM-Guided Inpainting
type: concept
created: 2026-08-19
updated: 2026-08-19
sources: [sources/inpainting-debug.md]
tags: [inpainting, repair, heartmula, heartcodec, lm-guided]
aliases: [LM-guided repair, Inpainting pipeline]
---

# LM-Guided Inpainting

**LM-Guided Inpainting** is the repair strategy behind Milimo's
[Repair Segment](../entities/inpainting.md). It replaces tokens in a gap/region using a
language model rather than copying or blind-masking, then re-synthesizes audio.

## Why prior approaches failed
| Approach | Result | Failure reason |
|----------|--------|----------------|
| `mask=0` (Blind) | Silence | Codec treats null tokens as silence |
| `mask=2` + Copy tokens | Repetition | Copied tokens = repeated content |
| Latent Blur | Phase smear | Destructive post-processing |
| Gradient Masking | Silence | Gradient values < 1.0 kill signal |

## Two-stage architecture
```
Stage 1: SEMANTIC GENERATION (HeartMuLa)
  Input:  history_tokens (8s before gap) + lyrics + style tags
  Output: NEW tokens for the gap (not copied!)

Stage 2: ACOUSTIC RECONSTRUCTION (HeartCodec)
  Input:  [8s context] + [new tokens] + [8s context]
  Output: Phase-aligned audio with 100ms crossfade
```

- Near-deterministic sampling for stage 1: `temp=0.2`, `topk=30`, `cfg_scale=1.0`.

## Why it works
The key insight is that the gap is filled with **newly generated** semantic tokens (from
HeartMuLa, conditioned on the history + lyrics + tags), and **not copied** from elsewhere;
HeartCodec then reconstructs acoustically consistent, phase-aligned audio around them.

## Related pages
- [Repair Segment](../entities/inpainting.md) | [HeartMuLa](../entities/heartmula.md) | [HeartCodec](../entities/heartcodec.md)
- [Inpainting Debug Log source](../sources/inpainting-debug.md)
