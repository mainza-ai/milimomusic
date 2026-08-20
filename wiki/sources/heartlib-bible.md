---
title: Source — Heartlib Bible
type: source
created: 2026-08-19
updated: 2026-08-19
tags: [source, heartlib, framework]
aliases: [HEARTLIB_BIBLE]
---

# Source — Heartlib Bible

**"Heartlib Bible: The Definitive Guide to the Heartlib Audio Generation Framework"** (v1.0).
**Raw location:** `docs/HEARTLIB_BIBLE.md` (immutable source).

## Key contents (summary)
- **Intro/philosophy**: Heartlib treats audio generation as language modeling —
  discretize waveform into semantic tokens via a neural codec, then a Transformer
  ([HeartMuLa](../entities/heartmula.md)) predicts audio tokens autoregressively.
- **Four components**: [HeartCLAP](../entities/heartclap.md),
  [HeartTranscriptor](../entities/hearttranscriptor.md),
  [HeartCodec](../entities/heartcodec.md), HeartMuLa.
- **HeartMuLa architecture**: Global (+Local) transformers; Llama 3.2 3B + 300M;
  text vocab 128,256; audio vocab 8,197.
- **HeartCodec**: 12.5 Hz, 8 codebooks RVQ, encoder (Whisper+WavLM+MuEncoder) →
  compressor → Flow-Matching decoder.
- **Pipeline**: `HeartMuLaGenPipeline` — preprocess, generation loop, decoding.
- **Prompt structure**: `[BOS] <tag>... </tag> [EOS] [MUQ_EMBED] [Lyrics...] [EOS]` +
  supported tag list (Beta).
- **Advanced generation techniques** (section 6, not fully detailed here).

## Entities it feeds
- [Heartlib](../entities/heartlib.md), [HeartMuLa](../entities/heartmula.md),
  [HeartCodec](../entities/heartcodec.md), [HeartCLAP](../entities/heartclap.md),
  [HeartTranscriptor](../entities/hearttranscriptor.md),
  [HeartMuLaGenPipeline](../entities/heartmulagenpipeline.md).
- [Prompt structure & style tags](../concepts/prompt-structure.md),
  [Lyrics conditioning](../concepts/lyrics-conditioning.md).

## Related pages
- [Heartlib](../entities/heartlib.md) | [Index](../index.md)
