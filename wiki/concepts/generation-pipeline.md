---
title: Orchestration Pipeline (Generate → Transcribe)
type: concept
created: 2026-08-20
updated: 2026-08-21
tags: [pipeline, orchestration, generation, transcription, stems, demucs, voice, karaoke]
aliases: [generation pipeline, orchestration pipeline, GenerateAndTranscribePipeline]
---

# Orchestration Pipeline (Generate → Transcribe)

The **orchestration pipeline** (`orchestration/pipeline.py`,
`GenerateAndTranscribePipeline`) is the end-to-end flow that turns a generation request
into a fully transcribable, editable production asset. It is the core of the v2 "generate →
auto-transcribe → edit" thesis and currently runs **real** model inference and **real**
neural source separation.

## The 4 steps
```
Step 1  GENERATION       provider.generate()                   → real audio (MiniMax Music 3 MLX)
Step 2  STEM SEPARATION  real_separator.separate_sources()     → HTDemucs: Vocals/Drums/Bass/Other
Step 3  (optional) VOICE conversion  voice_service.convert_vocals(vocal stem, voice_profile_id)
Step 4  TRANSCRIPTION    muscriptor_provider.transcribe()       → MIDI + MusicXML + notes + beat grid
        + per-instrument stems  instrument_stems.render_instrument_parts(notes) → parts + GM
        + lyric sync     lyric_sync_engine.align_lyrics()       → timed lyrics (karaoke)
```

Progress is published across all 4 stages via the SSE `job_progress` event
(surfaces in the frontend's [FloatingStatusWidget](../entities/frontend.md)).

## Dual-engine stems
Two stem sets are written onto `Job.stems_json` (both real, user-selectable in the DAW):

1. **HTDemucs** — real neural source separation of the actual master into
   `vocals/drums/bass/other` (see [Stem Separation](../entities/stem-separator.md)).
   Runs off the event loop; if it ever fails, the pipeline **degrades gracefully** to the
   per-instrument parts instead of failing the whole job.
2. **MuScriptor per-instrument** — one stem per distinct instrument in the transcription,
   with its General MIDI program (via `instrument_stems.py`). Distinctly namespaced
   (`_part_` prefix) so it never collides with the HTDemucs master stems.

The [Session workspace (DAW)](../entities/session-workspace.md) exposes a **stem-source
toggle** (4 Master Stems ↔ Per-Instrument) so the user chooses which to view/hear.

## Outputs on the `Job`
`audio_path`, `midi_path`, `musicxml_path`, `notes_json`, `stems_json`, `beat_grid_json`,
`timed_lyrics_json`, `structured_caption_json` — all consumed by the
[Session workspace (DAW)](../entities/session-workspace.md).

## Generation provenance (never silent)
- `GenerationRequest.structured_caption` (composer / Ask Producer) is passed through to
  `provider.generate(..., structured_caption=...)` and honored, so user-authored captions
  actually reach [MiniMax Music 3](../entities/minimax-music3.md) (see
  [Caption Rewriter](caption-rewriter.md)).
- The pipeline persists `Job.used_fallback_synth` + `Job.fallback_reason` from the provider
  result. When real MiniMax inference is skipped or throws, the UI shows a visible
  "Fallback synthesis" badge (hero + AI Provenance tab) instead of silently playing the
  procedural synth.

## Entry points
- Driven by `POST /generate/music` (background task).
- `POST /transcribe/upload` runs separation + transcription + per-instrument stems on user
  audio (import path).

## Fidelity notes (current, accurate as of 2026-08-20)
- **Step 1** runs **real MiniMax Music 3 MLX weight inference** (`mlx_audio.music.generate`)
  on Apple Silicon, conditioned on prompt + structured caption + lyrics + section tags;
  inference `steps` are clamped to the model's 1–30 range (a prior clamp of 32 made every
  ≥62s song silently fall back to the synth). On Windows/Linux (no mlx) it falls back to
  the conditioned placeholder path, now surfaced to the user
  ([MiniMax Music 3](../entities/minimax-music3.md)).
- **Step 2** is real **HTDemucs** neural separation ([Stem Separation](../entities/stem-separator.md)).
- **Step 3** is optional voice-conversion ([Voice Studio](../entities/voice-service.md)).
- **Step 4** uses real `MuScriptor` inference ([MuScriptor](../entities/muscriptor.md));
  per-instrument stems are a faithful note-rendering, not a neural separation of the
  original; lyric sync is partition-based ([Karaoke](../entities/karaoke-lyricsync.md)).

## Related pages
- [Generation provider](../entities/generation-provider.md) | [Backend & API](../entities/backend-api.md)
- [Stem Separation](../entities/stem-separator.md) | [MuScriptor](../entities/muscriptor.md)
- [Caption Rewriter](caption-rewriter.md) | [Structured Captions](structured-caption.md)
- [Voice Studio](../entities/voice-service.md) | [Session workspace](../entities/session-workspace.md)
- [Architecture](../architecture.md)
