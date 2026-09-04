---
title: Milimo Music v2 — Refactor & Upgrade Roadmap
type: overview
created: 2026-08-19
updated: 2026-08-20
sources: [sources/v2-refactor-plan.md]
tags: [roadmap, v2, daw, minimax, muscriptor]
aliases: [v2 plan, refactor plan]
---

# Milimo Music v2 — Refactor & Upgrade Roadmap

Synthesis of `devs/milimo-music-v2-refactor-plan.md`. The goal: evolve Milimo from a
HeartMuLa music generator into a **full open-source AI production DAW**.

> [!IMPORTANT] **Implementation status (2026-08-20).** Most of this plan is now *built and
> wired*: the provider abstraction, **MiniMax default with real MLX inference on Apple** (and
> cross-platform fallback), MuScriptor integration, **dynamic per-instrument stems**,
> mastering, lyric sync, voice studio, Suno-class IA, 5-mode DAW workspace, Project folders,
> and OpenCode/OMLX providers are implemented and verified end-to-end. Lower-priority paths
> remain scaffold-level (DSP stem preview, Matchering mastering stub, SVC not fully wired;
> HeartMuLa/Heartlib is optional/legacy only). See [v2 reference projects](entities/v2-references.md)
> for a 🔵/🟡/⚪ status legend per component, and [overview](overview.md) for the current product.

## The core insight

- **MiniMax Music 3** makes Milimo *generate better* (full songs, structured captions).
- **MuScriptor** makes Milimo *editable* — it's the only piece that turns opaque audio
  back into structured, note-level, per-instrument data you can touch.

> Target state: generate with any model → auto-transcribe every stem into MIDI + notation
> via MuScriptor → drop into a multitrack editor (piano roll, mixer, per-instrument
> regeneration) → export stems/MIDI/MusicXML/master. This is a different product category
> from Suno, which stays a black box.

## Key plan elements

### 3.1 Generation Provider abstraction
Replace the hardcoded HeartMuLa path with a **`GenerationProvider` interface**
(`generate()`, `extend()`, `repair_segment()`, `capabilities()`). **Capabilities, not model
names**, drive the UI (e.g. `max_duration`, `supports_structured_caption`,
`supports_section_tags`, `supports_lora`).

### 3.2 MiniMax Music 3 as default, HeartMuLa as option
Settings gains a **Model** section. MiniMax is a heavier deployment (2-GPU split or
SGLang-Omni); docs need a real hardware-tiers table. The Co-Writer's Lyricist learns to
emit **Structured Captions** when MiniMax is active.

### 3.3 MuScriptor as the producer-edit engine
Biggest structural addition: generated WAV → transcription → per-instrument MIDI +
MusicXML + note-JSON. A "song" becomes `{ audio, midi, notation, stems, metadata }`.
User note-level edits become an editable production layer over the immutable AI master.

### 3.4 Model & adapter management
MiniMax's HF repo is a **model tree** (2B base + Adapters/Finetunes/Quantizations).
Backend needs a model-manifest fetcher + download-on-demand UI, applied to HeartMuLa too.

### 3.5 Dependency currency policy
Every third-party dependency gets a swappable, versioned reference (lesson from Demucs
being archived). Provider abstraction applies to audio-processing tools too.

### 3.6 Voice Training & Vocal Identity Cloning
Neither model does vocal-identity cloning natively. Path: **post-generation SVC** using
**RVC v2** (via maintained forks like Applio) or So-VITS-SVC, layered on the stem
separation pipeline. A "Sing as…" voice-profile selector in Compose. Requires a consent gate.

### 3.7 New backend capabilities
Fast stem separation (**BS-Roformer/MelBand-Roformer** 6-stem via `audio-separator` and native neural pipeline), reference mastering
(**Matchering**), acoustic & syllable-weighted karaoke sync, stem/MIDI/MusicXML export,
piano roll/notation editing, instrument re-assignment, remix/rearrange, import &
transcribe user audio, DAW-native (Ableton) export.

### 4. UI: session grows a workspace
Keep the Apple-grade DAW workspace (left rail, chat-first Compose, Explore feed) with a finished
session mode switcher: **Listen → Arrange → Piano Roll → Notation → Mix → Lyrics**.

### 5. Licensing
Non-commercial open source; produce an accurate `LICENSES.md` matrix
(model → weight license → code license). MuScriptor weights are CC BY-NC 4.0 (MIT code);
usage terms prohibit transcribing audio you don't hold rights to.

## Status
Core v2 capabilities (MiniMax Music 3 provider, MuScriptor neural transcription, BS-Roformer 6-stem separation, dynamic DAW workspace, acoustic karaoke synchronization, voice profile management) are actively shipped and integrated. Ongoing work focuses on fine-tuning extensions and external mastering plugins.

## Related pages
- [Overview](overview.md) | [Architecture](architecture.md)
- [MiniMax Music 3](entities/minimax-music3.md) | [MuScriptor](entities/muscriptor.md)
- [v2 reference projects](entities/v2-references.md) | [Index](index.md)
