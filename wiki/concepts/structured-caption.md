---
title: Structured Captions (MiniMax)
type: concept
created: 2026-08-20
updated: 2026-08-20
tags: [structured-caption, minimax, prompt, section-tags]
aliases: [Structured Caption, Structured Captions]
---

# Structured Captions (MiniMax)

**Structured Captions** are the prompt format used by [MiniMax Music 3](../entities/minimax-music3.md)
generation: instead of a flat tag list, the prompt is organized into semantic sections —
**Global Metadata / Vocal Details / Arrangement** — plus explicit **section tags**
(`[Intro]`, `[Verse]`, `[Chorus]`, `[Solo]`…).

## The three sections
- **Global Metadata** — genre, tempo, mood.
- **Vocal Details** — voice / vocal style description.
- **Arrangement** — instrumentation, production, structure.

## In the codebase
- `MiniMaxMusic3Provider.parse_structured_caption(prompt, tags)`:
  - If prompt already contains `[Global Metadata]` / `[Vocal Details]` / `[Arrangement]`
    headers, parses them into a dict.
  - Otherwise **constructs** them from `tags` + free text (first tag → genre, second → mood,
    rest → instruments).
- `format_full_caption(...)` reassembles the `[Global Metadata]`/`[Vocal Details]`/
  `[Arrangement]`/`[Description]` blocks.
- `sanitize_section_tags(lyrics)` normalizes loose section labels into bracket tags
  (e.g. "Verse 1:" → `[Verse 1]`).
- The Composer exposes a **Structured Caption Spec** expander (three fields) in the
  Sound & Style tab; the backend validates via `GenerationRequest.structured_caption`.

## Relationship to HeartMuLa tags
[HeartMuLa](../entities/heartmula.md) uses a flat [tag-list prompt](prompt-structure.md) and
`supports_structured_caption=false`. Structured Captions are the MiniMax-native format and
the two are interchangeable at the UI layer because generation is driven by
[capability manifests](../entities/generation-provider.md) rather than hardcoded per-model forms.

## Related pages
- [MiniMax Music 3](../entities/minimax-music3.md) | [Prompt structure & style tags](prompt-structure.md)
- [Generation provider](../entities/generation-provider.md) | [Orchestration pipeline](generation-pipeline.md)
