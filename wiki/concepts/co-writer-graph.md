---
title: AI Co-Writer Graph
type: concept
created: 2026-08-19
updated: 2026-08-19
sources: [sources/readme.md]
tags: [co-writer, pydantic-graph, agents, lyrics]
aliases: [Co-Writer graph, pydantic-graph workflow]
---

# AI Co-Writer Graph

The [AI Co-Writer](../entities/ai-cowriter.md) is implemented as a **graph-based workflow**
using `pydantic-graph` (`backend/app/services/lyrics_graph.py`). It routes requests through
specialized nodes that can loop (retry) when validation fails.

## Nodes
- **CoordinatorNode** — analyzes the request and routes to **CREATION** or **EDIT** mode.
- **LyricistNode** — the creative agent that drafts lyrics and executes editing operations
  (Update, Insert, Append).
- **StructureGuardNode** — QA agent that validates every output against strict schemas
  (`lyrics_schemas.py`: `LyricsResponse`, `LyricsEditOp`). If the Lyricist errs, the Guard
  catches it and forces an **automatic retry**.

## Why a graph (not a chatbot)
- **Agentic workflow**: routing + validation + retry produce reliable, schema-conformant
  lyrics instead of freeform text.
- **Pydantic-native**: lyrics are treated as code artifacts (schemas), eliminating
  hallucinated formatting and guaranteeing the output fits the music engine.

## Supporting services
`StructuredLyricsEngine` (`lyrics_engine.py`), `LyricsDOM` (`lyrics_utils.py`),
`StyleRegistry` (`style_registry.py`), `ConfigManager` (`config_manager.py`), plus debug
logging to `ai_debug.log`.

## Related pages
- [AI Co-Writer](../entities/ai-cowriter.md) | [Backend & API](../entities/backend-api.md)
- [Lyrics conditioning](lyrics-conditioning.md)
