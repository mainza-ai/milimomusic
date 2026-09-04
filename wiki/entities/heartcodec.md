---
title: HeartCodec
type: entity
created: 2026-08-19
updated: 2026-08-19
sources: [sources/heartlib-bible.md, sources/inpainting-debug.md]
tags: [heartcodec, codec, audio, neural]
aliases: [HeartCodec-oss]
---

# HeartCodec

**HeartCodec** is a low-frame-rate, high-fidelity **neural audio codec** used in Milimo's
generation pipeline. It discretizes continuous audio into semantic tokens and reconstructs
a waveform from predicted tokens.

## Key numbers
- **Frame rate**: 12.5 Hz (ultra-low, for efficient long-sequence modeling).
- **Quantization**: 8 codebooks (RVQ — residual vector quantization).

## Architecture
- **Encoder**: semantic-rich (Whisper + WavLM + MuEncoder).
- **Compressor**: downsamples to 12.5 Hz.
- **Decoder**: **Flow Matching**-based high-fidelity reconstruction.

## In Milimo Music
- During generation, HeartMuLa predicts tokens and HeartCodec decodes them to audio
  (see [architecture](../architecture.md) and [HeartMuLaGenPipeline](heartmulagenpipeline.md)).
- **Repair / inpainting**: HeartCodec is the acoustic-reconstruction stage (Stage 2) of
  LM-guided repair, doing phase-aligned reconstruction with a 100ms crossfade over an
  `[8s context] + [new tokens] + [8s context]` input — see
  [LM-guided inpainting](../concepts/lm-guided-inpainting.md).
- Exposes `inpaint` and `encode` (re-tokenization) methods used by the
  [Repair Segment](inpainting.md) service.

## Related pages
- [Heartlib](heartlib.md) | [HeartMuLa](heartmula.md)
- [LM-guided inpainting](../concepts/lm-guided-inpainting.md)
- [Heartlib Bible source](../sources/heartlib-bible.md)
