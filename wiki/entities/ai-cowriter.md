---
title: AI Co-Writer
type: entity
created: 2026-08-19
updated: 2026-08-19
sources: [sources/readme.md, sources/v2-refactor-plan.md]
tags: [co-writer, agent, lyrics, pydantic]
aliases: [Co-Writer, AICoWriter]
---

# AI Co-Writer

The **AI Co-Writer** is Milimo Music's multi-agent lyrics engine. It is *not* a simple
chatbot — it uses a **graph of specialized Pydantic Agents** working in tandem
(implemented with `pydantic-graph`).

## The agent graph
See [Co-Writer graph](../concepts/co-writer-graph.md) for the full workflow. The three roles:

- **Coordinator Agent** — analyzes the request and routes it to the correct workflow
  (Creation vs. Editing).
- **Lyricist Agent** — the creative engine that drafts content and executes complex editing
  operations (Update, Insert, Append).
- **StructureGuard Agent** — a dedicated QA agent that validates every output against strict
  schemas. If the Lyricist makes a mistake, the Guard catches it and forces an automatic retry.

## Why Pydantic-native
By treating lyrics as **code artifacts (Schemas)**, the Co-Writer eliminates hallucinated
formatting and ensures lyrics always fit the music-generation engine (`lyrics_schemas.py`,
`LyricsResponse`, `LyricsEditOp`). StructureGuard validates schema-per-engine.

## Backend implementation
- `backend/app/services/lyrics_graph.py` — the pydantic-graph nodes
  (`CoordinatorNode`, `LyricistNode`, `StructureGuardNode`).
- `lyrics_engine.py` (`StructuredLyricsEngine`), `lyrics_utils.py` (`LyricsDOM`),
  `style_registry.py` (`StyleRegistry`), `config_manager.py` — supporting services.
- `ai_debug.log` — debug logging used by the graph.

## Reasoning & Think Tag Sanitization
To support reasoning/thinking models (e.g. DeepSeek-R1, Qwen-QwQ, Gemini Flash Thinking), the Co-Writer enforces strict sanitization via `_strip_thinking()` and `sanitize_lyrics()`:
- **Thinking Envelopes**: Strips all matched and orphaned `<think>`, `<thinking>`, `<reasoning>`, `<thought>`, `<reflection>`, `<deliberation>`, `<scratchpad>` tags.
- **Section Anchoring**: Automatically strips scratchpad brainstorm notes, rhyme bullet lists, and conversational preambles that occur prior to the first valid song section header (`[Intro]`, `[Verse]`, etc.).
- **Code Fence & Postamble Stripping**: Removes markdown blocks (```` ``` ````) and model chatter (`Final output ready.`).

## Related pages
- [Co-Writer graph](../concepts/co-writer-graph.md) | [LLM Service](llm-service.md)
- [Backend & API](backend-api.md) | [Prompt structure & style tags](../concepts/prompt-structure.md)
