---
title: Prompt Structure & Style Tags
type: concept
created: 2026-08-19
updated: 2026-08-19
sources: [sources/heartlib-bible.md, sources/readme.md]
tags: [prompt, style-tags, heartmula, conditioning]
aliases: [Style tags, Prompt structure]
---

# Prompt Structure & Style Tags

To ensure style adherence, the HeartMuLa generation prompt must follow a strict structural
format. Style is expressed via a curated, finite set of **supported tags**.

## Prompt structure
A continuous sequence without internal resets:

```
[BOS] <tag> {Style Tags} </tag> [EOS] [MUQ_EMBED] [Lyrics...] [EOS]
```

- **Tags**: comma-separated style descriptors.
- **`[MUQ]`**: placeholder for MuQ-MuLan embeddings of reference audio.
- **Lyrics**: structured lyrics with markers like `[Verse]`, `[Chorus]`.

## Supported HeartMuLa-3B tags (Beta)
Music style integration is in beta; for best results use only these fine-tuned tags:

`Warm, Reflection, Pop, Cafe, R&B, Keyboard, Regret, Drum machine, Electric guitar,
Synthesizer, Soft, Energetic, Electronic, Self-discovery, Sad, Ballad, Longing, Meditation,
Faith, Acoustic, Peaceful, Wedding, Piano, Strings, Acoustic guitar, Romantic, Drums,
Emotional, Walking, Hope, Hopeful, Powerful, Epic, Driving, Rock`

## In the codebase
- `backend/app/services/style_registry.py` — `StyleRegistry` and `OFFICIAL_STYLES`,
  consistent tag normalization/handling across requests.
- Tags are normalized to comma-separated strings in the backend
  (`GenerationRequest.tags` validator, see [Backend & API](../entities/backend-api.md)).

> [!NOTE] In v2, when [MiniMax Music 3](../entities/minimax-music3.md) is active, the Lyricist
> emits **Structured Captions** (Global Metadata / Vocal Details / Arrangement + section
> tags) instead of this tag-list — see [roadmap](../roadmap.md).

## Related pages
- [Lyrics conditioning](lyrics-conditioning.md) | [HeartMuLa](../entities/heartmula.md)
- [AI Co-Writer](../entities/ai-cowriter.md) | [Backend & API](../entities/backend-api.md)
