---
title: Maestro Creative Studio Architecture & Ingest
type: source
tags: [maestro, wangp, ltx-video, minimax-h3, yue2, director, editor, autotune, queue]
created: 2026-09-16
updated: 2026-09-16
sources: [https://github.com/Blizaine/Maestro]
aliases: [Maestro, Blizaine/Maestro]
---

# Maestro Creative Studio Architecture & Ingest

Source repository: [https://github.com/Blizaine/Maestro](https://github.com/Blizaine/Maestro) (Release v2.2.1, 2026-09-16)
Authors/Maintainers: Blizaine

## 1. Overview & System Scope
Maestro is a 100% local, all-in-one AI creative studio combining image, video, and audio generation with an LLM-directed production workflow, a non-destructive multi-track editor, hardware auto-tuning, and a universal persistent queue. Built primarily on top of the WanGP pipeline, diffusers, PyTorch, and llama.cpp (`llama-server`), it targets local workstations and consumer GPUs (RTX 30/40/50 series, Apple Silicon, Linux/Windows).

The architecture is divided into five core pillars:
1. **Director Mode v2**: Automated screenplay and beat-aware music video director driven by local LLMs (Gemma 4, Qwen 3.8).
2. **Editor Mode**: Non-destructive multi-track timeline video and audio editor with single-pass FFmpeg hardware-accelerated rendering and AI round-trip clip replacement.
3. **Audio & Music Stack**: Multi-engine generation featuring **YuE2 3B** (48 kHz stereo, ABC score conditioning, personal style LoRA training), MiniMax Music 3, ACE-Step v1.5, and H3 Voice Audio.
4. **Performance Auto-Tune & Memory Coordinator**: Zero-config GPU/VRAM/RAM detection mapping to empirical performance profiles (1–5), VRAM safety coefficients, scoped CPU memory execution, runtime kernel benchmarking, and OOM recovery.
5. **Universal Queue & State Checkpointing**: Persistent task queue with asset ownership copies, independent clip re-rendering, pre-enhancement, and crash/restart recovery.

---

## 2. Director Mode v2: Music Video Architecture

### 2.1 Multi-Signal Audio Analysis (`audio_analysis.py`)
Rather than slicing audio into uniform or arbitrary time segments, Maestro runs a deep multi-signal analysis pass:
- **Beat & Downbeat Detection**: Uses Librosa spectral onsets and beat-tracking networks to extract tempo (BPM), beat grids, and bar start timestamps (downbeats).
- **Section Segmentation**: Segments tracks into structural passages (`[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`) using RMS energy envelopes combined with Whisper transcription timestamps.
- **Speaker Diarization & Vocal Intervals (`vocal_activity.py`)**: Identifies active singing spans and assigns speaker IDs to each vocal line, distinguishing between lead vocalists, guest vocalists, and background harmonies.
- **Percussion Cues (`music_cues.py`)**: Analyzes transient energy on CPU to detect drum entrances and high-impact rhythmic crashes.

### 2.2 Musical Anchor Snapping & Pacing (`director_music_timing.py`)
- **Scored Accent System**: Assigns hierarchical weights to musical events:
  $$\text{Score}(t) = \text{Beat}(t) \times 0.5 + \text{Downbeat}(t) \times 1.8 + \text{LyricBoundary}(t) \times 2.5 + \text{AnchorPoints}(t)$$
- **Cut Speed / Pacing Bias Slider ($-2$ to $+2$)**:
  - `0`: Normal musical pacing following natural section and bar boundaries.
  - `+1 / +2`: Favors shorter, rapid montage cuts, splitting long sections on downbeats and lyric phrases.
  - `-1 / -2`: Favors sweeping long shots; at `-2`, it uses the absolute minimum number of clips that fit the model ceiling, snapping cuts near musical transitions.
- **Model-Native Discrete Frame Snapping & Trimming**:
  - Frontier video models require exact discrete frame increments ($F_{\text{min}} + k \cdot F_{\text{step}}$, e.g. MiniMax H3 requires $1 + 8k$ frames at 24fps; Wan2.1 requires multiples of 4).
  - Maestro calculates the exact frame count, generates the clip at model-native duration, and computes `music_output_trim` to trim any sub-second excess video. This eliminates cumulative audio-video drift across long songs.

### 2.3 Performer Ownership & Lip-Sync Isolation (`music_performance.py`)
- Solves the unnatural "flapping lips" problem during instrumental breaks and guitar/drum solos.
- The planner maintains a strict performer map: when a cutaway occurs to the drummer or guitarist, the prompt mandates `mouth_movement: closed` and directs camera focus onto the instrument.
- Lip-sync generation is strictly bound to the active singing speaker, preserving visual and vocal consistency across multi-performer productions.

---

## 3. Editor Mode: Non-Destructive Multitrack Timeline (`editor_projects.py`)

### 3.1 Project Schema & Data Architecture
- Multi-track timeline supporting parallel video tracks, audio tracks (stems, sound effects, master), and text/subtitle tracks.
- Every clip tracks: start time, duration, source in/out trim points, asset ID, canvas transformations (scale, crop, position, aspect ratios 16:9, 9:16, 21:9), volume automation, opacity, and transitions (crossfade, dissolve).
- Atomic JSON serialization ensures projects survive process interruptions without corruption.

### 3.2 Single-Pass FFmpeg Filter Graph Compiler
- `compile_editor_render()` compiles the complete multi-track composition into a single FFmpeg `-filter_complex` command.
- Avoids lossy intermediate rendering. Audio tracks are synchronized with `atempo` chains and mixed cleanly via `amix`/`amerge`.
- Automatic hardware encoder detection: checks for NVIDIA NVENC (`h264_nvenc`, `hevc_nvenc`, `av1_nvenc`), Apple Silicon VideoToolbox (`h264_videotoolbox`, `hevc_videotoolbox`), VAAPI, AMF, QSV, and falls back to CPU (`libx264`, `libsvtav1`).

### 3.3 AI Round-Trip Integration
- Creators can select any clip on the timeline, click "Send to AI", generate a retake, variation, or stem-reactive re-render, and return the newly generated take directly into the clip's slot on the timeline without altering cut boundaries or soundtrack sync.

---

## 4. YuE2 48kHz Stereo Generation & Style LoRA Studio (`YuE2-music.md`)

- **48 kHz Stereo Generation**: Native high-fidelity audio conditioned on lyrics and music style tags.
- **Three Conditioning Modes**:
  1. *Direct Generation*: Fast synthesis from lyrics and genre tags.
  2. *Melody and Chords*: Guides melodic progression and harmonic changes using ABC notation scores.
  3. *Source Song Covers*: Ingests audio, transcribes note and chord progressions via SheetSage/MERT, and renders a new stylistic performance.
- **"My Music" Training Studio**:
  - Resumable training of personal Autoregressive (AR) and Non-Autoregressive (NAR) style adapters.
  - Held-out validation sets (excluded from gradient updates).
  - Automatic checkpoint auditions: renders a standard test song at steps 100/200 to verify human listening quality over raw validation loss.
  - Diagnostic token reconstruction: resynthesizes tokenized audio with adapter on/off to isolate whether quality issues stem from tokenization or generation conditioning.

---

## 5. Performance Auto-Tune & Memory Management (`Performance-auto-tune.md`)

### 5.1 Empirical Hardware Profiles (Profiles 1–5)
- **Profile 1**: High VRAM ($\ge 24\text{ GB}$), High RAM ($\ge 64\text{ GB}$) $\rightarrow$ Full model residency in VRAM.
- **Profile 2**: High RAM ($\ge 32\text{ GB}$), Moderate VRAM ($12-23\text{ GB}$) $\rightarrow$ Balanced transformer weight pinning in host RAM, streamed to GPU.
- **Profile 3 / 3.5**: High VRAM ($\ge 24\text{ GB}$), Low Host RAM $\rightarrow$ Weights in VRAM, minimal host memory allocation.
- **Profile 4 / 4.5**: Low VRAM ($12-16\text{ GB}$), Moderate Host RAM $\rightarrow$ Consumer standard (RTX 3060/4070/4080); aggressive memory cleanup.
- **Profile 5**: Very Low Hardware ($< 12\text{ GB}$ VRAM, or CPU) $\rightarrow$ Maximum layer offloading, INT8 ConvRot / NVFP4 quantization.

### 5.2 VRAM Safety Coefficient & Headroom
- Fixed at $\le 0.80$ on cards with $\ge 12\text{ GB}$ VRAM, and $0.70$ on cards with $< 12\text{ GB}$ VRAM.
- CPU-scoped memory execution: helper tasks (torchaudio, audio loading, resamplers) are strictly executed in CPU space to prevent CUDA heap fragmentation.

### 5.3 Kernel Benchmarking & OOM Self-Healing
- **INT8 ConvRot vs. Triton**: Benchmarks native PyTorch fallback against Triton on actual tensor shapes during first inference, caching the faster path for the session.
- **OOM Recovery**: Intercepts PyTorch out-of-memory errors, dumps diagnostic telemetry (process RSS, system RAM, PyTorch allocated/reserved), releases temporary inference tensors, and auto-suggests a 1-click VRAM headroom adjustment.

---

## 6. Universal Queue & State Recovery (`job_lifecycle.py`)

- **Persistent Render Queue**: Survives application crashes and restarts; interrupted jobs remain paused with their completed clips and inputs intact.
- **Asset Ownership**: When a job is submitted, input assets are copied or symlinked into the job workspace so subsequent user deletions or edits cannot corrupt active pipelines.
- **Queue Pre-Enhancement**: Asynchronously develops and adapts prompts while the GPU is busy, ensuring models receive optimized prompts immediately when GPU turns become available.
