---
title: Milimo Music v2 — Refactor & Upgrade Roadmap
type: overview
created: 2026-08-19
updated: 2026-09-16
sources: [sources/v2-refactor-plan.md, sources/maestro-creative-studio.md]
tags: [roadmap, v2, daw, minimax, muscriptor, yue2, director, editor, autotune, queue]
aliases: [v2 plan, refactor plan]
---

# Milimo Music v2 — Refactor & Upgrade Roadmap

Synthesis of `devs/milimo-music-v2-refactor-plan.md` and production architecture ingest from [Maestro Creative Studio](sources/maestro-creative-studio.md). The goal: evolve Milimo from a generative music tool into a **complete open-source AI production DAW and creative multimedia studio**.

> [!IMPORTANT] **Implementation status (2026-09-16).**
> Core v2 architecture (MiniMax Music 3 default, MuScriptor transcription, dynamic per-instrument stems, Apple theme, DAW workspace, project folders, and agent crew) is active.
> Following the comprehensive investigation of Maestro, the roadmap now incorporates **Director Mode v2**, the **Non-Destructive Multitrack Timeline Editor**, **YuE2 48kHz Stereo Provider**, **Hardware Auto-Tune (Profiles 1–5)**, and the **Durable Task Queue**.

## The Core Insight

- **MiniMax Music 3 & YuE2 3B** make Milimo *generate at studio quality* (full songs, structured captions, 48 kHz stereo, ABC notation guidance).
- **MuScriptor** makes Milimo *editable* — turning opaque audio back into structured, note-level, per-instrument data.
- **Director Mode v2 & Multitrack Timeline Editor** make Milimo a *complete audio-visual production suite* — aligning cinematic scene cuts with musical downbeats and providing a non-destructive multi-track editor with AI round-trip retakes.

## Key Plan Elements

### 3.1 Generation Provider Abstraction
Pluggable `GenerationProvider` interface (`generate()`, `extend()`, `repair_segment()`, `capabilities()`). Capabilities drive UI options:
- **MiniMax Music 3**: Default DiT flow-matching with Structured Captions.
- **YuE2 3B**: Open-weight foundation model delivering native **48 kHz stereo music**, symbolic ABC notation guidance, and personal style adapter fine-tuning.
- **HeartMuLa**: Legacy 44.1kHz provider.
- **MuLaCover 3B**: Symbolic cover generation.

### 3.2 MuScriptor Neural Transcription Engine
Generated WAV $\rightarrow$ neural transcription $\rightarrow$ per-instrument MIDI + MusicXML + note JSON.
Users can edit notes directly in the 5-mode DAW workspace (Piano Roll, Notation, Arrange, Mixer, Lyrics).

### 3.3 Director Mode v2 (Beat-Aware & Performer Directing)
Upgrades video generation into a multi-signal directing engine:
- **Hierarchical Accent Snapping**: Scored cut boundaries landing on beats ($+0.5$), downbeats ($+1.8$), lyric phrase boundaries ($+2.5$), and percussion entrances.
- **Cut Speed Slider ($-2$ to $+2$)**: Lets creators select between sweeping long takes and rapid beat-matched montage cuts.
- **Performer Role Ownership**: Mandates `mouth_movement: closed` during instrumental breaks and guitar/drum solos, reserving lip-sync exclusively for the active singer.
- **Discrete Frame Lattice Snapping**: Snaps clips to discrete video model frame increments ($F_{\text{min}} + k \cdot F_{\text{step}}$) with sample-accurate FFmpeg sub-second trimming (`music_output_trim`) to eliminate cumulative drift.

### 3.4 Non-Destructive Multitrack Timeline Editor
- Multitrack composition workspace layering video tracks, isolated stem channels (`vocals`, `drums`, `bass`, `other`), and animated subtitle lanes.
- Single-pass FFmpeg filter graph compilation using hardware encoders (NVIDIA NVENC, Apple Silicon VideoToolbox, Linux VAAPI).
- **AI Round-Trip Take**: Select any timeline clip $\rightarrow$ send to AI for a retake or variation $\rightarrow$ drops back into the timeline slot without disturbing cut boundaries or soundtrack sync.

### 3.5 Hardware Auto-Tune & Memory Profiles (Profiles 1 to 5)
Zero-config startup profiling:
- Profiles 1–5 automatically configured from GPU VRAM and host RAM.
- Strict $\le 0.80$ VRAM safety coefficient to prevent activation spikes from causing OOM errors.
- `cpu_scoped()` execution for Librosa, torchaudio, and audio pre-processing.
- OOM interception and self-healing headroom adjustment.

### 3.6 Durable Task Queue & Job Recovery
- Persistent SQLite queue replacing volatile in-memory task dictionaries.
- Input asset vaulting ensuring running jobs cannot be corrupted by external file movements.
- 1-Click restart recovery restoring interrupted jobs to `PAUSED` state without losing previously rendered clips.
- Queue pre-enhancement executing prompt expansion in the background while the GPU is busy.

## Status

Core generative and transcription pipelines are integrated. Current priority focus is deploying Director Mode v2, the Multitrack Timeline Editor, Hardware Auto-Tune, and YuE2 48kHz audio support.

## Related pages
- [Overview](overview.md) | [Architecture](architecture.md)
- [Director Mode v2](concepts/director-mode-v2.md) | [Multitrack Timeline Editor](entities/multitrack-editor.md)
- [Non-Destructive Multitrack Timeline](concepts/non-destructive-multitrack-timeline.md) | [YuE2 Music](entities/yue2-music.md)
- [Hardware Auto-Tune](concepts/hardware-autotune-memory-profiles.md) | [Durable Task Queue](entities/durable-task-queue.md)
- [v2 reference projects](entities/v2-references.md) | [Index](index.md)
