---
title: Heartlib
type: entity
created: 2026-08-19
updated: 2026-09-07
sources: [sources/heartlib-bible.md, sources/readme.md]
tags: [heartlib, framework, audio, generation]
aliases: [Heartlib framework]
---

# Heartlib

**Heartlib** is the specialized modular audio-generation framework at the heart of
Milimo Music. It bridges LLMs and neural audio codecs to enable text-conditional music
generation, infinite track extension, and lyrics alignment.

## Core philosophy
Heartlib treats audio generation as a **language modeling task**: it discretizes
continuous audio waveforms into semantic tokens using a neural codec, then a Transformer
backbone ([HeartMuLa](heartmula.md)) predicts audio tokens autoregressively, conditioned on
text (lyrics) and style tags.

## Four components
The framework is built from four primary components:

- **HeartCLAP** — contrastive language–audio component ([HeartCLAP](heartclap.md)).
- **HeartTranscriptor** — transcription component ([HeartTranscriptor](hearttranscriptor.md)).
- **HeartCodec** — low-frame-rate neural audio codec ([HeartCodec](heartcodec.md)).
- **HeartMuLa** — the music language model backbone ([HeartMuLa](heartmula.md)).

## Pipeline
The [`HeartMuLaGenPipeline`](heartmulagenpipeline.md) orchestrates generation:
preprocessing builds the prompt `C = [Tags, Reference, Lyrics]`, the generation loop
predicts audio tokens, and decoding passes them to HeartCodec to produce a waveform.

## In Milimo Music
- Lives under `heartlib/` in the repo; checkpoints are *not* committed and must be
  downloaded manually into `heartlib/ckpt/` (HeartMuLa-oss-3B, HeartCodec-oss, tokenizer,
  gen_config) — see [sources/readme.md](../sources/readme.md).
- > [!NOTE]
  > `heartlib/ckpt/` is **strictly legacy** and reserved solely for HeartMuLa and HeartCodec weights.
  > All multi-modal foundation models (MiniMax Music 3, FLUX cover art diffusion, Wan2.1 video)
  > and audio separation models are managed under the canonical `models/` directory framework
  > (`models/audio/`, `models/image/`, `models/video/`, `models/audio_separator/`).
- Wrapped by the backend's `MusicService` for generation, extension, and repair.

## Related pages
- [HeartMuLa](heartmula.md) | [HeartCodec](heartcodec.md) | [HeartMuLaGenPipeline](heartmulagenpipeline.md)
- [Architecture](../architecture.md) | [Heartlib Bible source](../sources/heartlib-bible.md)
