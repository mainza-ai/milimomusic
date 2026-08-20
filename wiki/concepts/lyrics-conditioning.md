---
title: Lyrics Conditioning
type: concept
created: 2026-08-19
updated: 2026-08-19
sources: [sources/readme.md, sources/heartlib-bible.md]
tags: [lyrics, conditioning, synthesis, prosody]
---

# Lyrics Conditioning

**Lyrics conditioning** is the mechanism by which Milimo Music aligns generated audio with
provided lyrics, respecting **prosody and structure** (Verse/Chorus/Bridge markers).

## How it works (HeartMuLa)
The generation prompt is built as a continuous sequence:

`[BOS] <tag> {Style Tags} </tag> [EOS] [MUQ_EMBED] [Lyrics...] [EOS]`

The lyrics section carries structure markers (`[Verse]`, `[Chorus]`, etc.), and the model
conditions its autoregressive audio-token predictions on both the style tags and the
lyric text ([HeartMuLa](../entities/heartmula.md), [Prompt structure & style tags](prompt-structure.md)).

## Supporting pieces
- The [AI Co-Writer](../entities/ai-cowriter.md) generates/produces structured lyrics as
  Pydantic schemas so they always fit the engine's expected format.
- In the [Training Studio](../entities/training-studio.md), uploaded lyrics/captions
  (`.txt` files) condition fine-tuning.

## Structured Captions (v2)
When [MiniMax Music 3](../entities/minimax-music3.md) becomes the default, conditioning uses
**Structured Captions** (Global Metadata / Vocal Details / Arrangement) with explicit
section tags instead of the HeartMuLa tag-list format — see [roadmap](../roadmap.md).

## Related pages
- [Prompt structure & style tags](prompt-structure.md) | [HeartMuLa](../entities/heartmula.md)
- [AI Co-Writer](../entities/ai-cowriter.md) | [Roadmap (v2)](../roadmap.md)
