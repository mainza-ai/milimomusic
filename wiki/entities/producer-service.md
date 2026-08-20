---
title: Producer Service
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [producer, llm, co-writer, lyrics, prompt, enhancement, generation]
aliases: [ProducerService, producer, self-healing producer]
---

# Producer Service

The **Producer Service** (`services/producer_service.py`) is Milimo's LLM-driven creative
layer that guarantees a generation request is *always* well-conditioned for
[MiniMax Music 3](minimax-music3.md) real inference — it **thinks and creates**, never fakes.

## Why it exists
A real producer is what the app should behave like. If a user clicks **Generate** with only a
bare idea ("A smash hit pop song") and no lyrics, real MiniMax inference used to throw
`ValueError: Lyrics are required` and silently fall back to the synthetic waveform — so every
track sounded like the same fake tone. The Producer prevents that by enhancing the input first.

## How it works
`ProducerService.enhance_for_generation(prompt, lyrics, tags, model)` decides whether the inputs
are inadequate:

- **weak prompt** (too few words, or a bare genre/title with no musical detail) → nothing is
  fabricated; the real LLM producer (`LLMService.enhance_prompt`) expands the concept into a
  detailed musical direction (`{topic, tags}`).
- **missing/short lyrics** → the AI Co-Writer (`LLMService.generate_lyrics_async`, the
  pydantic-graph lyrics engine) writes genuine, structured sections (`[Intro]`…`[Verse]`…
  `[Chorus]`…`[Outro]`).

Enhanced inputs are returned so the engine is conditioned properly. This is wired into
`MiniMaxMusic3Provider.generate()` (lazy import, no circular dependency), and the pipeline
persists the enhanced prompt / lyrics / tags onto the `Job` so the UI shows what the producer
actually wrote.

## Reasoning-stripping
LLM lyric writers often emit internal thinking ("Let me check the meter… let me finalize")
interleaved with a draft. `extract_final_lyrics(raw)` keeps only the **last, clean, structured
song** and drops all reasoning/preamble — so the lyrics that reach the field, the Job, and the
model are the final song, never the thinking.

## Production guarantees
- **No template logic:** when the LLM is reachable, real creative output is always used.
- **No silent fake:** if the LLM producer is unreachable, the user's own inputs are preserved and
  real inference still enforces a valid lyric — generation never blocks and never fabricates
  content the user didn't ask for.

## Related pages
- [MiniMax Music 3](minimax-music3.md) | [AI Co-Writer](ai-cowriter.md) | [LLM Service](llm-service.md)
- [Orchestration pipeline](../concepts/generation-pipeline.md) | [Generation Provider](generation-provider.md)
