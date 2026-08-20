---
title: Milimo Music — Architecture
type: overview
created: 2026-08-19
updated: 2026-08-20
sources: [sources/heartlib-bible.md, sources/readme.md, sources/v2-refactor-plan.md]
tags: [architecture, system, backend, frontend, minimax, muscriptor, daw]
---

# Milimo Music — Architecture (v2 AI Production DAW)

Milimo Music is a full-featured open-source AI music generation and production DAW platform (FastAPI backend + React 19 / Vite frontend).

## System layers (v2)

```
┌────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (React 19 / Vite / Tailwind)  :5173                          │
│  Explore & Producer Landing · 5-Mode Session Workspace (Listen,       │
│  Arrange, Piano Roll, Notation, Mix) · Voice Identity Studio · Model   │
│  Manager · Floating Task Monitor · LoRA Training Studio                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP + SSE + Audio Streaming
┌───────────────────────────────────┴────────────────────────────────────┐
│  BACKEND (FastAPI / SQLModel / SQLite WAL)  :8000                      │
│  ProviderRegistry · GenerateAndTranscribePipeline · MuScriptorProvider │
│  StemSeparator · MatcheringEngine · LyricSyncEngine · VoiceService     │
└─────────────┬───────────────────────────┬──────────────────────────────┘
              │                           │
  ┌───────────▼────────────┐  ┌───────────▼────────────┐  ┌──────────────▼────────────┐
  │  GENERATION PROVIDERS  │  │  TRANSCRIPTION ENGINE  │  │  LLM PROVIDERS            │
  │  MiniMax Music 3 (Def) │  │  MuScriptor (MT3)      │  │  Ollama / OpenAI / Gemini │
  │  HeartMuLa-3B (Legacy) │  │  MIDI + MusicXML +     │  │  DeepSeek / Claude        │
  │  Capability manifests  │  │  Note events + Stems   │  │  (Lyrics, Co-Writer graph)│
  └────────────────────────┘  └────────────────────────┘  └───────────────────────────┘
```

## Generation & Transcription Pipeline

The full flow is the [orchestration pipeline](concepts/generation-pipeline.md)
(`orchestration/pipeline.py`):

1. **Generation (MiniMax Music 3 default / HeartMuLa)**: [Structured Caption](concepts/structured-caption.md)
   embeddings conditioning flow-matching DiT with section tags (`[Intro]`, `[Verse]`, `[Chorus]`, etc.).
2. **Stem Separation**: [Stem Separator](entities/stem-separator.md) — filter-bank extraction of
   4 preview clips (Vocals, Drums, Bass, Instruments) + combined Instrumental. The **DAW's
   playback channels**, however, source from **dynamic per-instrument parts derived from the
   transcription** (see step 4) so Solo/Mute truly isolates each instrument, not a fixed 4-set.
3. **Vocal Identity Cloning (SVC)**: Optional local SVC inference on vocal stem using
   consent-verified [Voice Profiles](entities/voice-service.md).
4. **MuScriptor Transcription**: Note-level multi-instrument transcription into Standard
   MIDI, MusicXML score, and interactive JSON note events ([MuScriptor](entities/muscriptor.md)).
5. **Mastering & Export**: [Matchering](entities/matchering-mastering.md) reference mastering
   (-14 LUFS) and multi-format export (MIDI, MusicXML, LRC, SRT).

All outputs feed the [Session Workspace (DAW)](entities/session-workspace.md).

## Related pages

- [Overview](overview.md) | [Backend & API](entities/backend-api.md) | [Frontend](entities/frontend.md)
- [Generation Provider](entities/generation-provider.md) | [Model Manager](entities/model-manager.md)
- [MiniMax Music 3](entities/minimax-music3.md) | [MuScriptor](entities/muscriptor.md)
- [Session Workspace](entities/session-workspace.md) | [Index](index.md)
