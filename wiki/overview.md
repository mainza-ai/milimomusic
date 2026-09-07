---
title: Milimo Music — Overview
type: overview
created: 2026-08-19
updated: 2026-08-30
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
- **Artist section** — persistent artist identities with assigned AI crews ([Artist Crew
  Agents](entities/artist-crew-agents.md)): World-Builder lore canon, Experiencer visions,
  Stylist tag curation, Critic quality gates, per-artist model overrides and singing voice,
  and gated album production with resume/budget/retry ([Artist Domain](concepts/artist-domain.md)).
- **AI Music Video Studio** — multi-scene video generation, musical bar-aligned cuts respecting model duration limits (Hailuo H3, Wan 2.1, CogVideoX, HunyuanVideo), isolated vocal stem facial viseme lip-syncing with OpenCV landmark tracking, ASS karaoke subtitles, and procedural B-roll ([Video Studio](entities/video-studio.md)).
- **Multi-Modal Model Manager & Hugging Face Hub** — 23-variant catalog across audio, image, and video; live Hugging Face search and on-demand model downloader ([Model Manager](entities/model-manager.md)).
- **Cover Art Studio** — real neural diffusion via Diffusers/MFlux (Black Forest Labs FLUX.2, SDXL Turbo) with studio-grade 1024x1024 raster PNG generation.
- **Voice Training Studio** — offline SVC + RVC `.pth` neural checkpoint loader + profile-specific acoustic formant/presence equalization chains ([Voice Studio](entities/voice-service.md)).

## Core technology

- Generation: [MiniMax Music 3](entities/minimax-music3.md) / [HeartMuLa](entities/heartmula.md)
  + [HeartCodec](entities/heartcodec.md) (12.5 Hz codec) via provider abstraction, plus dynamic [HuggingFaceAudioProvider](entities/generation-provider.md).
- Transcription/editing: [MuScriptor](entities/muscriptor.md) (git submodule) + stem
  separation + karaoke lyric sync, orchestrated by the
  [4-step pipeline](concepts/generation-pipeline.md).
- Video & Visuals: OpenCV facial viseme deformation + Wan 2.1 / Hailuo H3 / CogVideoX integration + Diffusers FLUX.2 cover art.
- LLM: [OpenCode, OMLX (local Apple Silicon), Ollama, OpenAI, Gemini, OpenRouter, DeepSeek, LM Studio]
  ([LLM Service](entities/llm-service.md)).
- Multilingual lyrics (English, Chinese, Japanese, Korean, Spanish).

## Repos & structure

- `backend/` — FastAPI/SQLModel Python API (:8000) ([Backend & API](entities/backend-api.md)).
- `frontend/` — React 19 + Vite + Tailwind UI (:5173 / compiled into single-process `:8000`) ([Frontend](entities/frontend.md)).
- `heartlib/` — HeartMuLa-3B + HeartCodec wrapper.
- `muscriptor/` — transcription engine (**git submodule**).
- `LICENSES.md` — licensing matrix ([v2 references](entities/v2-references.md)).
- `docs/`, `devs/`, `assets/` — raw source material.

## Licensing

Open source, **non-commercial by intent** (Apache-2.0 `LICENSE` + `LICENSES.md` matrix):
MuScriptor weights CC BY-NC 4.0, MiniMax "MiniMax Open Weights", Matchering GPL-3.0,
WhisperX BSD-2-Clause, RVC MIT, FLUX.2 Apache 2.0 / non-commercial dev — see [v2 references](entities/v2-references.md).

## Implementation status

The platform is fully implemented at production grade with zero placeholder shortcuts:
- **MiniMax Music 3 MLX & Sampling Controls**: Autoregressive Qwen3 token generation and DiT flow denoiser wired with `temperature`, `cfg_scale`, and `topk`.
- **MuScriptor Real Transcription**: Note-level polyphonic MIDI and MusicXML with dynamic per-instrument parts.
- **AI Music Video Studio**: Fully operational with vocal-stem viseme lip-syncing, duration constraint clamping, and stylized karaoke burning.
- **Singing Voice Conversion**: Real RVC neural checkpoints and acoustic formant EQ shaping.
- **Cover Art Studio**: FLUX.2 / SDXL Turbo neural diffusion and high-res studio raster PNG synthesis.
- **Deployment**: Single-process unified serving with React SPA fallback and multi-stage Docker packaging.

The app is **cross-platform** (macOS Apple Silicon, Linux, Windows).

## Related pages

- [Architecture](architecture.md) | [Roadmap (v2)](roadmap.md) | [Index](index.md)
