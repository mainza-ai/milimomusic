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
  - MiniMax Music 3 & HeartMuLa-3B / HeartCodec generation engines.
  - MuScriptor note-level polyphonic transcription (MIDI, W3C MusicXML 3.1).
  - 5-Mode Web Audio DAW Workspace (Listen, Arrange, Piano Roll, Notation, Console Mixer).
  - Matchering Reference Mastering (-14.0 LUFS) & Fast 4-Stem Separation.
  - Multi-Provider LLM Integration (OpenCode Go, local OMLX, Ollama, OpenAI, Gemini, DeepSeek).
  - Voice Training Studio & singing voice conversion (SVC).
- **Prerequisites & Setup**: Python 3.12 (Conda), Node 18+, Backend Uvicorn on `:8000`, Frontend Vite on `:5173`.
- **License**: Open-source and non-commercial. See `LICENSES.md` at repo root for the
  model/weight licensing matrix (summarized in [v2 references](../entities/v2-references.md)).
- **Creator**: Mainza Kangombe.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Index](../index.md)
