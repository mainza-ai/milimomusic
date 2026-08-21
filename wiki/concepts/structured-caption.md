---
title: Structured Captions (MiniMax)
type: concept
created: 2026-08-20
updated: 2026-08-21
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
  - Otherwise **constructs** them from `tags` + free text following the official
    prompting guide's three-heading skeleton (Basic Attributes / Emotional Progression /
    Imagery / Sonics; Vocal Gender & Timbre / Style / Harmony / FX; Instrument Lifecycle /
    Groove / Embellishments), with vocals always stated explicitly.
- `format_full_caption(...)` reassembles exactly the three sections (no `[Description]`
  block — the raw prompt lives inside Global Metadata imagery instead).
- `sanitize_section_tags(lyrics)` normalizes loose section labels into bracket tags
  (e.g. "Verse 1:" → `[Verse 1]`) **and forces every tag onto its own line** — the model
  silently drops lyric text that shares a line with a leading tag.
- **Honoring user captions:** `provider.generate(..., structured_caption=...)` merges the
  caller's non-empty sections over the auto-constructed ones, so the composer's three
  fields and the Ask Producer flow reach the model (this was previously dead UI — see
  [Caption Rewriter](caption-rewriter.md)).
- The Composer exposes a **Structured Caption Spec** expander (three fields) in the
  Sound & Style tab; its "Enhance" button now fills those fields via `POST
  /generate/rewrite_caption` ([Caption Rewriter](caption-rewriter.md)).

## Relationship to HeartMuLa tags
[HeartMuLa](../entities/heartmula.md) uses a flat [tag-list prompt](prompt-structure.md) and
`supports_structured_caption=false`. Structured Captions are the MiniMax-native format and
the two are interchangeable at the UI layer because generation is driven by
[capability manifests](../entities/generation-provider.md) rather than hardcoded per-model forms.

## Related pages
- [MiniMax Music 3](../entities/minimax-music3.md) | [Prompt structure & style tags](prompt-structure.md)
- [Generation provider](../entities/generation-provider.md) | [Orchestration pipeline](generation-pipeline.md)
