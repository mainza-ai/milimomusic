---
title: Source — README
type: source
created: 2026-08-19
updated: 2026-08-20
tags: [source, readme, overview]
aliases: [README]
---

# Source — README

The project `README.md` — canonical product overview, core capabilities, and concise setup guide.
**Raw location:** repo root `README.md`.

## Key contents (summary)
- **Positioning**: Open-source AI music generation, neural transcription, and multitrack production platform.
- **Core Technology**:
  - MiniMax Music 3 default engine (MLX Apple Silicon / CUDA / CPU) with Structured Caption Rewriter.
  - BS-Roformer neural 6-stem source separation (`vocals`, `drums`, `bass`, `guitar`, `piano`, `other`).
  - MuScriptor note-level polyphonic transcription (MIDI, W3C MusicXML 3.1) and dynamic instrument parts.
  - Neural acoustic lyric synchronization (RMS energy envelope, VAD, syllable-weighted timing, `.lrc` and `.srt` export).
  - 6-Mode Web Audio DAW Workspace (Listen, Arrange, Piano Roll, Notation, Console Mixer, Lyrics).
  - Matchering Reference Mastering (-14.0 LUFS broadcast target).
  - Multi-Provider LLM Integration (NVIDIA NIM, OpenCode Go, DeepSeek, local OMLX, Ollama, OpenAI, Gemini).
  - Voice Training Studio & singing voice conversion (SVC).
- **Prerequisites & Setup**: Python 3.12 (Conda), Node 18+, Backend Uvicorn on `:8000`, Frontend Vite on `:5173`.
- **License**: Apache-2.0. See `LICENSES.md` at repo root for the model/weight licensing matrix (summarized in [v2 references](../entities/v2-references.md)).
- **Creator**: Mainza Kangombe.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Index](../index.md)
