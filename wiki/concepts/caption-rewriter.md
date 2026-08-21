---
title: Caption Rewriter (MiniMax music-caption-rewriter port)
type: concept
created: 2026-08-21
updated: 2026-08-21
sources: [sources/v2-refactor-plan.md]
tags: [caption, minimax, prompt, structured-caption, llm, enhance]
aliases: [caption rewriter, music-caption-rewriter, rewrite_caption]
---

# Caption Rewriter

The **caption rewriter** turns a brief music description (plus optional lyrics and style
tags) into a professional three-heading [Structured Caption](structured-caption.md)
(**Global Metadata / Vocal Details / Arrangement**) for
[MiniMax Music 3](../entities/minimax-music3.md). It is the production port of the official
`music-caption-rewriter` agent skill from `MiniMax-AI/MiniMax-Music3` (the same content the
prompting guide publishes), moved into the app's own LLM layer so the in-app "Enhance" flow
produces the same quality a human prompt-writer would.

## How it works

1. **Vendored reference library** — `backend/data/caption-library/`: the official genre
   router (`references/genre-router.md`), 18 family indexes, and ~1,000 caption templates,
   copied verbatim from the upstream repo (see [LICENSES.md](../../LICENSES.md) §5).
2. **Routing** — `LLMService._rank_caption_families()` scores the brief against family
   slugs and router cue words, keeping 1–2 style families.
3. **Template selection** — `_pick_caption_templates()` scores template filenames against
   the brief tokens; the top 3 become few-shot references for the LLM.
4. **LLM synthesis** — the real configured LLM provider
   ([LLM Service](../entities/llm-service.md), e.g. OpenCode `minimax-m3`) receives the
   official write contract: exactly three sections, explicit vocal gender/instrumental,
   lyric tags preserved on their own lines, lyric text never quoted, no copied template
   sentences, no fabricated precision, 250–450 words total.
5. **Validation & fallback** — the response must contain all three non-empty sections;
   otherwise `rewrite_caption` returns an honest `fallback_reason` plus a complete
   constructed caption. It **never raises and never blocks generation**.

## API

`POST /generate/rewrite_caption` — body `{concept, lyrics?, tags?, model_name?}` returns
`{global_metadata, vocal_details, arrangement, rewritten, fallback_reason, families, templates}`.

## Where it plugs in

- **Composer "Enhance" button** (`ComposerSidebar.tsx`): rewrites the current concept into
  a full caption and fills the three caption fields (sparks a topic first only when empty).
- **Ask Producer** (`LLMService.produce_full_track`): the produced track's structured
  caption now comes from the rewriter instead of a hardcoded template.
- **Generation**: the filled fields flow through `GenerationRequest.structured_caption` →
  [orchestration pipeline](generation-pipeline.md) → `provider.generate()`, which honors
  them and auto-fills any missing section (previously the UI caption fields never reached
  the model).

## Related pages
- [Structured Captions](structured-caption.md) | [MiniMax Music 3](../entities/minimax-music3.md)
- [LLM Service & Providers](../entities/llm-service.md) | [Producer Service](../entities/producer-service.md)
- [Orchestration pipeline](generation-pipeline.md) | [LICENSES.md](../../LICENSES.md)
