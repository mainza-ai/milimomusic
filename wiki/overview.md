---
title: Milimo Music — Overview
type: overview
created: 2026-08-19
updated: 2026-08-20
sources: [sources/readme.md, sources/v2-refactor-plan.md]
tags: [milimo, product, music-generation, daw, transcription]
aliases: [Milimo]
---

# Milimo Music — Overview

> **"Give the silence something worth remembering."**
> *Speak it into being. Shape it until it's yours.*

**Milimo Music** is an open-source, non-commercial **AI music generation, neural
transcription, and multitrack production platform** ("DAW") created by [Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295).
It pairs state-of-the-art generation models with DAW editing, note-level transcription
(MIDI + MusicXML), stem separation, and offline voice cloning — a genuinely different
category from black-box generators like Suno.

## What it does today

- **Pluggable generation** — [MiniMax Music 3](entities/minimax-music3.md) (default,
  structured captions, up to 5 min) and [HeartMuLa-3B](entities/heartmula.md) via a
  [generation-provider](entities/generation-provider.md) layer.
- **Inpainting & continuation** — extend tracks or re-generate specific measures.
- **Precision controls** — duration (5–300 s), CFG, temperature, top-k/top-p, DiT steps,
  seed locking.
- **MuScriptor transcription** — multi-instrument polyphonic transcription to multi-track
  MIDI + W3C MusicXML 3.1 (Grand Staff engraving), beat/tempo tracking
  ([MuScriptor](entities/muscriptor.md)).
- **Web Audio DAW workspace** — [Arrange timeline, Piano Roll, Score Notation, Mix Console]
  ([Session Workspace](entities/session-workspace.md)); **dynamic per-instrument parts** (not a
  fixed 4-set) with Solo/Mute that truly isolates each transcribed instrument.
- **Matchering reference mastering** — -14.0 LUFS broadcast target
  ([Matchering](entities/matchering-mastering.md)).
- **Studio workflow & AI Co-Writer** — project folders (BPM/key/tags), multi-provider LLM,
  agentic lyrics engine ([AI Co-Writer](entities/ai-cowriter.md)).
- **Voice Training Studio** — offline SVC + vocal timbre tuning with consent enforcement
  ([Voice Studio](entities/voice-service.md)).

## Core technology

- Generation: [MiniMax Music 3](entities/minimax-music3.md) / [HeartMuLa](entities/heartmula.md)
  + [HeartCodec](entities/heartcodec.md) (12.5 Hz codec) via provider abstraction.
- Transcription/editing: [MuScriptor](entities/muscriptor.md) (git submodule) + stem
  separation + karaoke lyric sync, orchestrated by the
  [4-step pipeline](concepts/generation-pipeline.md).
- LLM: [OpenCode, OMLX (local Apple Silicon), Ollama, OpenAI, Gemini, OpenRouter, DeepSeek, LM Studio]
  ([LLM Service](entities/llm-service.md)).
- Multilingual lyrics (English, Chinese, Japanese, Korean, Spanish).

## Repos & structure

- `backend/` — FastAPI/SQLModel Python API (:8000) ([Backend & API](entities/backend-api.md)).
- `frontend/` — React 19 + Vite + Tailwind UI (:5173) ([Frontend](entities/frontend.md)).
- `heartlib/` — HeartMuLa-3B + HeartCodec wrapper.
- `muscriptor/` — transcription engine (**git submodule**).
- `LICENSES.md` — licensing matrix ([v2 references](entities/v2-references.md)).
- `docs/`, `devs/`, `assets/` — raw source material.

## Licensing

Open source, **non-commercial by intent** (Apache-2.0 `LICENSE` + `LICENSES.md` matrix):
MuScriptor weights CC BY-NC 4.0, MiniMax "MiniMax Open Weights", Matchering GPL-3.0,
WhisperX BSD-2-Clause, RVC MIT — see [v2 references](entities/v2-references.md).

## Implementation status

The v2 plan is substantially implemented, with the engines largely wired rather than
placeholder: **MiniMax Music 3 real MLX inference** (Apple Silicon) with a graceful
cross-platform fallback, **MuScriptor real transcription**, **dynamic per-instrument stems**
from the transcription, and a healthy integrated pipeline. Some optional/lower-priority paths
remain scaffold-level (DSP stem-preview, Matchering mastering stub, SVC not fully wired;
HeartMuLa/Heartlib is optional/legacy only). The [v2 references](entities/v2-references.md)
page tracks 🔵/🟡/⚪ status per component. The app is **cross-platform** (macOS Apple Silicon,
Windows, Linux).

## Related pages

- [Architecture](architecture.md) | [Roadmap (v2)](roadmap.md) | [Index](index.md)
