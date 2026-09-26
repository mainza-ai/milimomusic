---
title: Milimo Music — Architecture
type: overview
created: 2026-08-19
updated: 2026-09-16
sources: [sources/heartlib-bible.md, sources/readme.md, sources/v2-refactor-plan.md, sources/maestro-creative-studio.md]
tags: [architecture, system, backend, frontend, minimax, yue2, mulacover, muscriptor, daw, director, timeline, autotune, queue]
---

# Milimo Music — Architecture (v2 AI Production DAW)

Milimo Music is a full-featured open-source AI music generation and production DAW platform (FastAPI backend + React 19 / Vite frontend).

## System layers (v2)

```
┌────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (React 19 / Vite / Tailwind)  :5173                          │
│  Explore & Producer Landing · 5-Mode Session Workspace (Listen,       │
│  Arrange, Piano Roll, Notation, Mix) · Multitrack Timeline Editor ·    │
│  Music Video Director Studio · Voice Identity Studio · Model Manager · │
│  Hardware Telemetry Bar · Global Modal Store (Zustand)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP + SSE + Audio Streaming
┌───────────────────────────────────┴────────────────────────────────────┐
│  BACKEND (FastAPI / SQLModel / SQLite WAL)  :8000                      │
│  GlobalHardwareCoordinator (Auto-Tune) · DurableTaskQueue · Pipeline   │
│  ProviderRegistry (MiniMax, YuE2, MuLaCover, HeartMuLa) · VideoDirector│
│  MultitrackTimelineCompiler · MuScriptorProvider · MuLaCoverEngine     │
│  DrumTracker · SymbolicHub · NeuralSVCService · StemSeparator          │
│  MatcheringEngine · LyricSyncEngine · SidecarEngineManager             │
└─────────────┬───────────────────────────┬──────────────────────────────┘
              │                           │
  ┌───────────▼────────────┐  ┌───────────▼────────────┐  ┌──────────────▼────────────┐
  │  GENERATION PROVIDERS  │  │  TRANSCRIPTION ENGINE  │  │  LLM PROVIDERS            │
  │  MiniMax Music 3 (MLX) │  │  MuScriptor (MT3)      │  │  Ollama / OpenAI / Gemini │
  │  Stable Audio Open 1.0 │  │  Dual SymbolicHub      │  │  DeepSeek / Claude        │
  │  Meta MusicGen (Melody)│  │  Drum Tracker (MIDI)   │  │  (Lyrics, Co-Writer graph,│
  │  YuE2 3B (48kHz Stereo)│  │  Note events + Stems   │  │   director shot planner)  │
  │  MuLaCover-3B (Remix)  │  └────────────────────────┘  └───────────────────────────┘
  │  HeartMuLa-3B (Legacy) │
  │  Capability manifests  │
  └────────────────────────┘
```

## Generation & Transcription Pipeline

The full flow is the [orchestration pipeline](concepts/generation-pipeline.md)
(`orchestration/pipeline.py`) operating under strict [Cross-Modal Model Lifecycle](concepts/cross-modal-model-lifecycle.md) immediate eviction:

1. **Generation (MiniMax / Stable Audio / MusicGen / YuE2 / HeartMuLa)**: [Structured Caption](concepts/structured-caption.md)
   embeddings conditioning flow-matching DiT with section tags (`[Intro]`, `[Verse]`, `[Chorus]`, etc.), or [Stable Audio Open](entities/stable-audio-open.md) 44.1 kHz stereo diffusion, or [MusicGen](entities/musicgen.md) melody conditioning, or [YuE2](entities/yue2-music.md) 48 kHz stereo music generation.
   *Immediate Eviction*: On generation completion, `provider.unload()` instantly purges model weights and clears framework allocators.
2. **Stem Separation**: [Stem Separator](entities/stem-separator.md) — filter-bank extraction of
   4 preview clips (Vocals, Drums, Bass, Instruments) + combined Instrumental. The **DAW's
   playback channels**, however, source from **dynamic per-instrument parts derived from the
   transcription** (see step 4) so Solo/Mute truly isolates each instrument, not a fixed 4-set.
   *Immediate Eviction*: BS-Roformer unloads and flushes device memory in `finally:` block.
3. **Vocal Identity Cloning (SVC)**: Optional local SVC inference on vocal stem using
   consent-verified [Voice Profiles](entities/voice-service.md).
4. **MuScriptor Transcription**: Note-level multi-instrument transcription into Standard
   MIDI, MusicXML score, and interactive JSON note events ([MuScriptor](entities/muscriptor.md)).
5. **Mastering & Export**: [Matchering](entities/matchering-mastering.md) reference mastering
   (-14 LUFS) and multi-format export (MIDI, MusicXML, LRC, SRT).

All outputs feed the [Session Workspace (DAW)](entities/session-workspace.md) and the [Multitrack Timeline Editor](entities/multitrack-editor.md).

## Director Mode v2 & AI Music Video Studio
Directs synchronized cinematic video clips:
- **Hierarchical Musical Accent Snapping**: Upgraded to [Director Mode v2](concepts/director-mode-v2.md) in `video_director.py`, scoring beats, downbeats, lyric phrase boundaries, and percussion entrances.
- **Phase-Decoupled Execution**: Enforces strict phase separation under the [Cross-Modal Model Lifecycle](concepts/cross-modal-model-lifecycle.md): all keyframe stills are generated via FLUX.2/SDXL and saved to disk, followed by eager image model unloading before Wan 2.1 / LTX video diffusion commences.
- **Pacing Control**: User-selectable Cut Speed bias slider ($-2$ to $+2$).
- **Performer Role Ownership**: Assigns visual and vocal roles to performers, ensuring `mouth_movement: closed` during instrumental breaks and solos.
- **Discrete Frame Lattice Snapping**: Snaps clips to discrete video model frame lattices ($F_{\text{min}} + k \cdot F_{\text{step}}$) and applies sample-accurate sub-second trimming (`music_output_trim`) to eliminate cumulative audio-video drift.

## Non-Destructive Multitrack Timeline Editor
- **Atomic Project Schema**: Described in [Non-Destructive Multitrack Timeline](concepts/non-destructive-multitrack-timeline.md), supporting layered video, isolated audio stems, and animated subtitle text.
- **Single-Pass Hardware-Accelerated Export**: `compile_editor_render()` compiles the multi-track timeline directly into a single FFmpeg `-filter_complex` command via NVENC or Apple Silicon VideoToolbox with zero intermediate generational loss.
- **AI Round-Trip Take**: Select any timeline clip $\rightarrow$ send to AI for a retake or variation $\rightarrow$ drops back into the timeline slot without disturbing cut boundaries or soundtrack sync.

## Hardware Auto-Tune & Resilient Memory Lifecycle
Milimo Music orchestrates concurrent audio and video generative backbones using the [Global Hardware Coordinator](entities/hardware-coordinator.md) and [Hardware Auto-Tune](concepts/hardware-autotune-memory-profiles.md):
- **Cross-Modal Eviction Bus**: Coordinates mutual exclusion across `audio`, `image`, and `video` modalities, auto-evicting warm models when preempted by another media pipeline.
- **Empirical Performance Profiles (1 to 5)**: Automatically detects GPU VRAM, compute capability, and host RAM at startup and selects optimal memory offloading (Profile 1: Max Performance, Profile 2: Balanced Streaming, Profile 4: Consumer Standard, Profile 5: Max Layer Offload).
- **VRAM Safety Coefficient**: Enforces a strict $\le 0.80$ memory ceiling ($0.70$ for $< 12\text{ GB}$ VRAM) to prevent activation spikes and VAE decoding from crashing the GPU.
- **Scoped CPU Execution (`cpu_scoped()`)**: Pre-processing, format loading (Librosa/torchaudio), and audio decoders are strictly scoped to CPU memory, preventing CUDA memory heap fragmentation.
- **OOM Interception & Self-Healing**: Catches allocation failures, flushes PyTorch caches, lowers safety coefficients by $0.10$, and emits self-healing telemetry.
- **Sidecar Virtualenv Isolation**: [Sidecar Engine Manager](entities/sidecar-engine-manager.md) isolates conflicting neural dependencies under `backend/engines/<id>/.venv`.

## Durable Task Queue & Job Recovery
- **Persistent SQLite Store**: Upgraded from transient memory dictionaries to [Durable Task Queue](entities/durable-task-queue.md).
- **Asset Ownership Vault**: Copies input assets to dedicated job workspaces so external file moves cannot corrupt active jobs.
- **Restart & Crash Recovery**: Interrupted jobs transition to `PAUSED` on boot, allowing 1-click resumption without re-rendering completed scenes.
- **Queue Pre-Enhancement**: Asynchronously expands prompts in the background while the GPU is executing previous tasks.

## Related pages

- [Overview](overview.md) | [Backend & API](entities/backend-api.md) | [Frontend](entities/frontend.md)
- [Cross-Modal Model Lifecycle](concepts/cross-modal-model-lifecycle.md) | [Generation Provider](entities/generation-provider.md)
- [Stable Audio Open](entities/stable-audio-open.md) | [Meta MusicGen](entities/musicgen.md)
- [Director Mode v2](concepts/director-mode-v2.md) | [AI Music Video Studio](entities/video-studio.md)
- [Multitrack Timeline Editor](entities/multitrack-editor.md) | [Non-Destructive Multitrack Timeline](concepts/non-destructive-multitrack-timeline.md)
- [YuE2 48kHz Stereo](entities/yue2-music.md) | [Hardware Auto-Tune](concepts/hardware-autotune-memory-profiles.md)
- [Durable Task Queue](entities/durable-task-queue.md) | [Global Hardware Coordinator](entities/hardware-coordinator.md)
- [Audio Synthesis Standards](concepts/audio-synthesis-standards.md) | [Database Integrity Lifecycle](concepts/database-integrity-lifecycle.md)

