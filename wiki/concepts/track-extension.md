---
title: Track Extension
type: concept
created: 2026-08-19
updated: 2026-09-03
sources: [sources/readme.md, production-readiness-plan.md]
tags: [extension, track, generation, continuation, minimax, phase-5]
aliases: [Extend, Track continuation]
---

# Track Extension

**Track Extension** lets Milimo continue generating from where a previous track left off,
allowing the creation of longer compositions **segment by segment**.

## How it works
- Generation continues from a prior track's context (the tail audio/history becomes the new
  prompt's reference/context), rather than starting fresh.
- The backend links jobs via `parent_job_id` on the `Job` model, so an extension is recorded
  as a child of the original generation (see [Backend & API](../entities/backend-api.md)).
- The output is a longer, continuous composition built incrementally.

## Phase 5 redesign — analysis-conditioned extension (locked 2026-09-03)

> [!WARNING] **Current state:** `MiniMaxProvider.extend()` (`providers/minimax_provider.py:588`)
> just calls `generate()` with `extend_ms` — zero parent conditioning. The legacy
> HeartMuLa token-inpainting extension is dead (HeartMuLa is legacy-isolated). The open
> MLX MiniMax hook is caption+lyrics+duration conditioned and cannot consume reference
> audio, so **true tail-embedding conditioning is not implementable today**.

Locked design — **analysis-conditioned**, honestly labeled (no fabricated
"audio embedding" claims):

1. **Load parent context**: `beat_grid_json` (BPM), `structured_caption_json`, `tags`,
   `lyrics`, plus the tail seconds of the parent master.
2. **Tail analysis (DSP)**: RMS decay curve for the ending character; key estimate from
   `notes_json` pitch-class histogram (Krumhansl profile). Stored in a `conditioning`
   metadata block.
3. **Lyric continuation**: user-supplied text, or the producer LLM writes the next
   section seeded with parent lyrics + BPM/key/caption — the analysis is the
   "audio-domain context" delivered through the caption.
4. **Generate continuation** via the real MLX path (steps budget scaled to `extend_ms`).
5. **Equal-power crossfade mixdown**: scan the child's head for the lowest-energy
   overlap window (1–4 s), apply √-power fades, join → extended master.
6. **Re-finalize**: transcription, karaoke sync and stems re-run on the extended master
   (the post-generation block of `orchestration/pipeline.py` is extracted into a shared
   `finalize_track_assets()` helper used by both generate and extend).

Planned surface: `POST /jobs/{job_id}/extend` body `{extend_ms, lyrics_continuation?}`
→ child Job (`parent_job_id`, `queued`) + GPU-lane [task](../entities/task-queue.md)
→ `202 Accepted`, SSE progress. Strict-inference mode applies to the continuation.

## Related pages
- [HeartMuLaGenPipeline](../entities/heartmulagenpipeline.md) | [Backend & API](../entities/backend-api.md)
- [Lyrics conditioning](lyrics-conditioning.md) | [MiniMax Music 3](../entities/minimax-music3.md)
- [Task Queue](../entities/task-queue.md) | [Singing Voice Conversion](singing-voice-conversion.md) (Phase 5 sibling)
