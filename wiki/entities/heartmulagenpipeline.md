---
title: HeartMuLaGenPipeline
type: entity
created: 2026-08-19
updated: 2026-08-19
sources: [sources/heartlib-bible.md]
tags: [heartmula, pipeline, generation]
aliases: [GenerationPipeline]
---

# HeartMuLaGenPipeline

**HeartMuLaGenPipeline** is the glue that orchestrates generation in [Heartlib](heartlib.md).
It coordinates the [HeartMuLa](heartmula.md) language model with the
[HeartCodec](heartcodec.md) audio codec.

## The three stages
1. **Preprocessing** — constructs the prompt `C = [Tags, Reference, Lyrics]`
   (see [Prompt structure & style tags](../concepts/prompt-structure.md)).
2. **Generation loop** — autoregressively predicts audio tokens with HeartMuLa.
3. **Decoding** — passes predicted tokens to HeartCodec to generate the waveform.

## Related pages
- [Heartlib](heartlib.md) | [HeartMuLa](heartmula.md) | [HeartCodec](heartcodec.md)
- [Architecture](../architecture.md)
