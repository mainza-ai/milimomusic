---
title: v2 Reference Projects
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/v2-refactor-plan.md]
tags: [v2, references, minimax, muscriptor, tools, dependencies]
aliases: [v2 references]
---

# v2 Reference Projects

This page catalogs the external projects the v2 plan pulls in to transform Milimo from a
generator into a production DAW (see [roadmap](../roadmap.md)). It is organized around a
**dependency-currency policy**: every third-party dependency is a swappable, versioned
reference, not a hardcoded choice.

**Implementation status legend:** 🔵 integrated (code path exists) · 🟡 scaffolded
(endpoint/UI exists but real engine not wired) · ⚪ still planned.

## Generation models
- 🔵 **MiniMax Music 3** — **default** provider; MLX snapshot, structured captions, 300 s
  ([MiniMax Music 3](minimax-music3.md)). Note: current `generate()` uses procedural DSP
  waveform synthesis, not real weights.
- 🔵 **HeartMuLa-3B** — legacy/local provider ([HeartMuLa](heartmula.md)).

## Transcription / producer-edit
- 🔵 **MuScriptor** (Kyutai × Mirelo) — **git submodule**, fully integrated
  `MuScriptorProvider.transcribe()` → MIDI + MusicXML + notes ([MuScriptor](muscriptor.md)).
  Code MIT; **weights CC BY-NC 4.0** (non-commercial only).

## Audio-processing tools
- 🔵 **BS-Roformer / MelBand-Roformer** — SOTA 6-stem separation (`vocals`, `drums`, `bass`, `guitar`, `piano`, `other`) via `audio-separator` and native neural pipeline across CUDA/MPS/CPU ([Stem Separator](stem-separator.md)).
- 🟡 **Matchering** — reference mastering. Endpoint + Mix-tab UI exist, but the
  implementation is a stub ([Matchering mastering](matchering-mastering.md)).
- 🔵 **Neural Acoustic & Syllable VAD Sync** — dynamic RMS vocal energy envelope extraction, VAD, and syllable-weighted karaoke timing ([Karaoke & Lyric Sync](karaoke-lyricsync.md)).
- ⚪ **Demucs** — legacy reference superseded by BS-Roformer (the "dep currency" lesson).

## Voice cloning (SVC)
- 🟡 **RVC v2** (via forks e.g. **Applio**, MIT) / **So-VITS-SVC** — profile management +
  consent + a `convert_vocals()` scaffold exist ([Voice Studio](voice-service.md)); real SVC
  model conversion not yet wired.

## Licensing summary
Core platform code is licensed under **Apache-2.0**. **`LICENSES.md`** at repo root details the full matrix
(platform, models, tools, user-rights notice). Upstream model weights operate under their respective licenses:
MuScriptor neural weights CC BY-NC 4.0; MiniMax weights "MiniMax Open Weights"; Matchering GPL-3.0; WhisperX BSD-2-Clause; RVC MIT.

## Related pages
- [Roadmap (v2)](../roadmap.md) | [MiniMax Music 3](minimax-music3.md) | [MuScriptor](muscriptor.md)
- [Stem separator](stem-separator.md) | [Voice Studio](voice-service.md)
