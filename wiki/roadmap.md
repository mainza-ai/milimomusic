---
title: Milimo Music v2 — Refactor & Upgrade Roadmap
type: overview
created: 2026-08-19
updated: 2026-09-24
sources: [sources/v2-refactor-plan.md, sources/maestro-creative-studio.md]
tags: [roadmap, v2, daw, minimax, muscriptor, yue2, director, editor, autotune, queue, qwen21, singularity]
aliases: [v2 plan, refactor plan]
---

# Milimo Music v2 — Refactor & Upgrade Roadmap

Synthesis of `devs/milimo-music-v2-refactor-plan.md` and production architecture ingest from [Maestro Creative Studio](sources/maestro-creative-studio.md) (v2.4.0). The goal: evolve Milimo from a generative music tool into a **complete open-source AI production DAW and creative multimedia studio**.

> [!IMPORTANT] **Implementation status (2026-09-24).**
> Core v2 architecture (MiniMax Music 3 default, MuScriptor transcription, dynamic per-instrument stems, Apple theme, DAW workspace, project folders, and agent crew) is active.
> Following the v2.4.0 Maestro architecture update, the roadmap incorporates **Director Mode v2 with 0–5 Fidelity Repair Retries**, **Music Timeline Vocal Bypass**, **Auto + Guided My Music Training**, **H3 Singularity References + 4-Step Turbo**, the **Non-Destructive Multitrack Timeline Editor**, **Immersive Gallery & Media Bridge**, **Hardware Auto-Tune (Profiles 1–5)**, and **Single-Frame Bounded Memory Execution**.

## The Core Insight

- **MiniMax Music 3 & YuE2 3B** make Milimo *generate at studio quality* (full songs, structured captions, 48 kHz stereo, ABC notation guidance, personal style LoRA adapters).
- **MuScriptor** makes Milimo *editable* — turning opaque audio back into structured, note-level, per-instrument data.
- **Director Mode v2 & Multitrack Timeline Editor** make Milimo a *complete audio-visual production suite* — aligning cinematic scene cuts with musical downbeats, scoping performance directions to the visible cast, and providing a non-destructive multi-track editor with AI round-trip takes.
- **Durable Task Queue & Bounded Memory Coordinator** make Milimo *production-reliable* — eliminating volatile in-memory queues, bounding memory on consumer GPUs (single-frame mask composition, VRAM-sized reference caches), and surviving crashes.

---

## Key Plan Elements

### 3.1 Generation Provider Abstraction & Music Stack
Pluggable `GenerationProvider` interface (`generate()`, `extend()`, `repair_segment()`, `capabilities()`). Capabilities drive UI options:
- **MiniMax Music 3**: Default DiT flow-matching with Structured Captions.
- **YuE2 3B Foundation Provider**: Open-weight model delivering native **48 kHz stereo music**, symbolic ABC notation guidance, and source song covers via SheetSage2/MERT2.
- **Automatic Instrumental LoRA (v2.3.0 pattern)**: Automatically routing instrumental generations to Mothersuperior's Instrumental AR LoRA at strength 1.0, pausing active artist LoRAs, and logging recipes in song metadata.
- **Multi-LoRA Mixing**: Experimental multi-LoRA mixes with independent weights and trigger injection.
- **"My Music" Personal Training Studio**:
  - *Auto Mode*: Full-song preparation $\rightarrow$ vocal separation $\rightarrow$ timed lyrics $\rightarrow$ phrase excerpting $\rightarrow$ paired audio tokenizer/decoder training (100 steps) $\rightarrow$ AR song style training (200 steps) in one queued run.
  - *Guided Mode*: Explicit 4-stage pipeline (Recordings $\rightarrow$ Matched Voice/Sound Adaptation $\rightarrow$ AR Song Style $\rightarrow$ Test Song Auditions at 100/200 steps).

### 3.2 MuScriptor Neural Transcription Engine
Generated WAV $\rightarrow$ neural transcription $\rightarrow$ per-instrument MIDI + MusicXML + note JSON.
Users can edit notes directly in the 5-mode DAW workspace (Piano Roll, Notation, Arrange, Mixer, Lyrics).

### 3.3 Director Mode v2 (Beat-Aware & Performer Directing)
Upgrades video generation into a multi-signal directing engine:
- **Hierarchical Accent Snapping**: Scored cut boundaries landing on beats ($+0.5$), downbeats ($+1.8$), lyric phrase boundaries ($+2.5$), and percussion entrances.
- **Cut Speed Slider ($-2$ to $+2$)**: Lets creators select between sweeping long takes and rapid beat-matched montage cuts.
- **Performer Role Ownership & Visible-Cast Scoping (v2.4.0)**: Mandates `mouth_movement: closed` during instrumental breaks and solos. Performance instructions are strictly scoped to the people actually shown in each shot, preventing narrative or scenery shots from inheriting boilerplate lists of musicians.
- **Music Timeline Vocal Bypass (v2.4.0)**: In music video mode, the supplied song audio is treated as the authoritative source of vocals and timing. Dialogue writing and word-count gates are suppressed, preventing spurious spoken dialogue from corrupting music video prompts.
- **0–5 Fidelity Repair Retries & Auto-Continuation**: User-configurable fidelity repair retries (0–5, default 1) with an optional *"Generate even if fidelity checks fail"* switch that proceeds with the saved draft rather than stalling the queue. Localized card repairs avoid invalidating neighboring windows.
- **Discrete Frame Lattice Snapping**: Snaps clips to discrete video model frame increments ($F_{\text{min}} + k \cdot F_{\text{step}}$) with sample-accurate FFmpeg sub-second trimming (`music_output_trim`) to eliminate cumulative drift.
- **H3 Singularity References + LightX2V 4-Step Turbo (v2.4.0)**: Support for the 21 GB pruned INT8 ConvRot checkpoint with 4-step Euler Turbo LoRA for rapid multi-reference video generation.

### 3.4 Non-Destructive Multitrack Timeline Editor
- Multitrack composition workspace layering video tracks, isolated stem channels (`vocals`, `drums`, `bass`, `other`), and animated subtitle lanes.
- Single-pass FFmpeg filter graph compilation using hardware encoders (NVIDIA NVENC, Apple Silicon VideoToolbox, Linux VAAPI).
- **AI Round-Trip Take**: Select any timeline clip $\rightarrow$ send to AI for a retake or variation $\rightarrow$ drops back into the timeline slot without disturbing cut boundaries or soundtrack sync.

### 3.5 Immersive Gallery & Media Bridge (v2.4.0)
- **1-Click Gallery-to-Input Routing**: Interactive media menu to route any gallery image, captured frame, or full video directly into active Studio or Director inputs (References, Frames, Animate, Edit, Upscale).
- **Before/After Image Comparison**: Interactive divider slider comparing source and generated results.
- **First-Frame Video Posters**: Cached first-frame JPEG endpoint (`/api/v1/thumbnail/{filename}`) enabling smooth mobile browsing without decoding heavy video containers.
- **Fullscreen Swipe Navigation**: Mobile-responsive viewer with persistent volume/mute state across clips.

### 3.6 Hardware Auto-Tune & Bounded Memory Execution
Zero-config startup profiling:
- Profiles 1–5 automatically configured from GPU VRAM and host RAM.
- Strict $\le 0.80$ VRAM safety coefficient to prevent activation spikes from causing OOM errors.
- `cpu_scoped()` execution for Librosa, torchaudio, and audio pre-processing.
- **Single-Frame Bounded Mask Memory**: Frame-by-frame processing for character masks and Recast operations, eliminating multi-gigabyte memory allocations.
- **VRAM-Proportional Reference Caching**: Sizing attention reference caches against available VRAM and releasing prior to VAE decoding (Qwen 2.1 / Wan patterns).
- Kernel benchmarking (PyTorch vs Triton) and self-healing OOM recovery.

### 3.7 Durable Task Queue & Job Recovery
- Persistent SQLite queue replacing volatile in-memory task dictionaries.
- Input asset vaulting ensuring running jobs cannot be corrupted by external file movements.
- 1-Click restart recovery restoring interrupted jobs to `PAUSED` state without losing previously rendered clips.
- Queue pre-enhancement executing prompt expansion in the background while the GPU is busy.

---

## Status & Phased Rollout Plan

1. **Phase 1: Foundation (Active)**: MiniMax Music 3, MuScriptor transcription, dynamic per-instrument stems, Apple glass theme, DAW workspace.
2. **Phase 2: Director Mode v2 & Video Studio (Next Up)**: Beat-grid accent scoring, cut-speed pacing, visible-cast performer scoping, music timeline vocal bypass, 0–5 repair retries, and discrete frame trimming.
3. **Phase 3: Multitrack Timeline & Gallery Bridge**: Non-destructive multi-track editor, single-pass FFmpeg hardware rendering, 1-click media routing, before/after comparison, and video poster caching.
4. **Phase 4: YuE2 48kHz Audio & My Music Studio**: 48 kHz stereo generation, ABC notation guidance, Mothersuperior instrumental LoRA auto-routing, and Auto/Guided My Music training.
5. **Phase 5: Auto-Tune, Memory Bounding & Universal Queue**: Profiles 1–5, single-frame mask bounding, reference cache sizing, and persistent SQLite task queue.
6. **Phase 6: Cross-Modal Model Lifecycle & Cross-Platform Audio (Planned)**: Immediate model eviction across Audio (MiniMax, Stable Audio Open, MusicGen), Image (FLUX.2 Klein, SDXL), and Video (Wan 2.1, LTX); phase-decoupled Director execution (Keyframes $\to$ Purge $\to$ Video Diffusion $\to$ Purge); native Stable Audio Open 1.0 (44.1 kHz stereo DiT) and Meta MusicGen (melody conditioning) to bridge the Apple Silicon MLX architecture gap.

---

## Related pages
- [Overview](overview.md) | [Architecture](architecture.md)
- [Cross-Modal Model Lifecycle](concepts/cross-modal-model-lifecycle.md) | [Generation Provider](entities/generation-provider.md)
- [Stable Audio Open](entities/stable-audio-open.md) | [Meta MusicGen](entities/musicgen.md)
- [Maestro Creative Studio Ingest](sources/maestro-creative-studio.md)
- [Director Mode v2](concepts/director-mode-v2.md) | [Multitrack Timeline Editor](entities/multitrack-editor.md)
- [Non-Destructive Multitrack Timeline](concepts/non-destructive-multitrack-timeline.md) | [YuE2 Music](entities/yue2-music.md)
- [Hardware Auto-Tune](concepts/hardware-autotune-memory-profiles.md) | [Durable Task Queue](entities/durable-task-queue.md)
- [AI Music Video Studio](entities/video-studio.md) | [Stem Audio-Reactive Video](concepts/stem-audio-reactive-video.md)
- [v2 reference projects](entities/v2-references.md) | [Index](index.md)

