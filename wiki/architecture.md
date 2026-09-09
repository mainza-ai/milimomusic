---
title: Milimo Music — Architecture
type: overview
created: 2026-08-19
updated: 2026-09-09
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
│ AgentRuntime (4 crew agents) · Album Orchestrator · Release Lifecycle  │
└─────────────┬───────────────────────────┬──────────────────────────────┘
              │                           │
  ┌───────────▼────────────┐  ┌───────────▼────────────┐  ┌──────────────▼────────────┐
  │  GENERATION PROVIDERS  │  │  TRANSCRIPTION ENGINE  │  │  LLM PROVIDERS            │
  │  MiniMax Music 3 (Def) │  │  MuScriptor (MT3)      │  │  Ollama / OpenAI / Gemini │
  │  HeartMuLa-3B (Legacy) │  │  MIDI + MusicXML +     │  │  DeepSeek / Claude        │
  │  Capability manifests  │  │  Note events + Stems   │  │  (Lyrics, Co-Writer graph,│
│   artist crew + critic)   │
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

## Acoustic Calibration & Procedural DSP Synthesis
The platform features dual-engine stem auditioning: real neural source separation (HTDemucs) alongside MuScriptor note-level procedural synthesis. The synthesis engine enforces strict psychoacoustic calibration (see [Audio Synthesis Standards](concepts/audio-synthesis-standards.md)):
- **Psychoacoustic Target RMS**: Fletcher-Munson staged loudness curve (Drums: -13.1 dBFS / 0.22 RMS, Piano: -14.0 dBFS / 0.20 RMS, Clean Electric Guitar: -16.5 dBFS / 0.15–0.18 RMS, Clarinet: -18.4 dBFS / 0.12 RMS).
- **Physical Acoustic Modeling**: Exponential pick transient clicks ($\exp(-220t)$) and pickup harmonics ($f, 2f, 3f, 4f, 5f$) for electric guitar; cylindrical stopped-pipe odd harmonics ($f, 3f, 5f, 7f$) with suppressed even harmonics for clarinet; sub-frequency sweeps ($140 \to 48\text{ Hz}$) for acoustic drums.
- **Physical Signal Verification**: Crest Factor ($>6.0$ for pluck transients, $<3.0$ for pipe resonance) and Spectral Centroid profiling.

## Data Integrity, UUID Standards & Relational Lifecycle
Tracks are managed under an atomic lifecycle architecture (see [Database Integrity Lifecycle](concepts/database-integrity-lifecycle.md)):
- **Universal Multi-Format Lookup**: Parameterized text SQL condition (`id = :c OR id = :h OR id = :hyp`) resolving both 32-hex and 36-hyphenated UUID representations, bypassing SQLite dialect hex stripping.
- **Startup Self-Healing Migration**: Boot-time migration in `init_db()` normalizing non-canonical UUIDs across `job`, `session`, `sessionmessage`, `playlisttrack`, and `release`.
- **Atomic Cascade Deletion**: `DELETE /jobs/{id}` nullifies session and message references, deletes playlist tracks, expunges ORM tracking to eliminate duplicate-delete warnings, and executes an exhaustive multi-directory filesystem sweep.

## System Performance Standards & Benchmarks
Milimo Music enforces measurable production performance benchmarks across the stack:
- **Web Audio Clock Jitter**: **0.00ms** clock skew across all stems via sample-locked `AudioContext.currentTime` transport.
- **Web Audio Playback Latency**: **< 20ms** start-to-sound latency; **< 250ms** multitrack stem decode latency.
- **Database Query Latency**: Universal `get_job_by_id()` resolves in **< 1.5ms**; full cascading delete in **< 25ms**.
- **Frontend Production Build**: Client bundle compiles via Vite (`tsc -b && vite build`) in **< 1.6s** (gzip size: < 261 kB JS, < 16 kB CSS).
- **Automated Test Integrity**: Full test suite comprises **213 tests** executing in **< 6.0s** with a 100% pass rate.

## Related pages

- [Overview](overview.md) | [Backend & API](entities/backend-api.md) | [Frontend](entities/frontend.md)
- [Audio Synthesis Standards](concepts/audio-synthesis-standards.md) | [Database Integrity Lifecycle](concepts/database-integrity-lifecycle.md)
- [Generation Provider](entities/generation-provider.md) | [Model Manager](entities/model-manager.md)
- [MiniMax Music 3](entities/minimax-music3.md) | [MuScriptor](entities/muscriptor.md)
- [Session Workspace](entities/session-workspace.md) | [Index](index.md)
