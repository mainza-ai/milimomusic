---
title: HeartCLAP
type: entity
created: 2026-08-19
updated: 2026-08-19
sources: [sources/heartlib-bible.md]
tags: [heartclap, clap, audio, embeddings]
aliases: [HeartCLAP]
---

# HeartCLAP

**HeartCLAP** is one of the four primary components of [Heartlib](heartlib.md). It is a
**contrastive language–audio pretraining** component — the CLAP-style module that learns a
joint embedding space between text and audio.

## Role
CLAP-style components learn aligned text↔audio representations. In the Heartlib stack,
HeartCLAP contributes semantic understanding between natural language descriptions and
audio, complementing the token-prediction role of [HeartMuLa](heartmula.md) and the
codec role of [HeartCodec](heartcodec.md).

> [!NOTE] This page is intentionally light. The Heartlib Bible names HeartCLAP as one of
> the four components but does not detail it further. Enriching this page (and
> [HeartTranscriptor](hearttranscriptor.md)) remains a candidate lint task — the Heartlib
> technical report that once covered these was removed; a web search on the Heartlib/HeartMuLa
> components would be the fallback path.

## Related pages
- [Heartlib](heartlib.md) | [HeartMuLa](heartmula.md) | [HeartCodec](heartcodec.md)
- [HeartTranscriptor](hearttranscriptor.md)
