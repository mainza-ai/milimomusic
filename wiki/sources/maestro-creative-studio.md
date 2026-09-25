---
title: Maestro Creative Studio Architecture & Ingest
type: source
tags: [maestro, wangp, ltx-video, minimax-h3, yue2, director, editor, autotune, queue, qwen21, singularity]
created: 2026-09-16
updated: 2026-09-24
sources: [https://github.com/Blizaine/Maestro]
aliases: [Maestro, Blizaine/Maestro]
---

# Maestro Creative Studio Architecture & Ingest

Source repository: [https://github.com/Blizaine/Maestro](https://github.com/Blizaine/Maestro) (Release v2.4.0, 2026-09-24; comprehensive update covering v2.2.2 through v2.4.0)
Authors/Maintainers: Blizaine

## 1. Overview & System Scope
Maestro is a 100% local, all-in-one AI creative studio combining image, video, and audio generation with an LLM-directed production workflow, a non-destructive multi-track editor, hardware auto-tuning, an immersive gallery, and a universal persistent queue. Built primarily on top of the WanGP pipeline, diffusers, PyTorch, and llama.cpp (`llama-server`), it targets local workstations and consumer GPUs (RTX 30/40/50 series, Apple Silicon, Linux/Windows).

The architecture is organized around six production pillars:
1. **Director Mode v2**: Automated beat-aware music video and screenplay director driven by local/remote LLMs, featuring musical accent snapping, visible-cast performance scoping, music-timeline vocal bypass, and targeted multi-window fidelity repairs (0–5 retries).
2. **Editor Mode & Immersive Gallery**: Non-destructive multi-track timeline video and audio editor with single-pass FFmpeg hardware-accelerated rendering, AI round-trip clip replacement, full-screen vertical swipe gallery browsing, before/after image comparison, and 1-click gallery-to-active-input routing.
3. **Audio & Music Stack**: Multi-engine generation featuring **YuE2 3B** (48 kHz stereo, ABC score conditioning, personal style LoRA training), MiniMax Music 3, ACE-Step v1.5, H3 Voice Audio, auto-applied instrumental LoRAs, and multi-LoRA mixes.
4. **Frontier Visual Diffusion Models**: Integration of Alibaba Wan 2.1 (T2V/I2V), MiniMax H3 (33B Omni DiT with Context-IR), **H3 Singularity v1.3 References** (21 GB pruned INT8 ConvRot with 4-step LightX2V Turbo), and **Qwen Image 2.1 7B** (text-to-image, reference editing up to 10 images, and transparent RGBA PNGs).
5. **Performance Auto-Tune & Bounded Memory Coordinator**: Zero-config GPU/VRAM/RAM detection mapping to empirical performance profiles (1–5), VRAM safety coefficients ($\le 0.80$), scoped CPU execution, single-frame bounded mask memory (preventing multi-gigabyte index arrays), VRAM-proportional reference attention caching, runtime kernel benchmarking, and OOM recovery.
6. **Universal Queue & State Checkpointing**: Persistent SQLite task queue with asset ownership copies, independent clip re-rendering, queue pre-enhancement, and crash/restart recovery.

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

### 2.3 Performer Ownership & Visible-Cast Scoping (`music_performance.py`)
- **Solves the "Flapping Lips" Problem**: During instrumental breaks and guitar/drum solos, the planner strictly mandates `mouth_movement: closed` and directs camera focus onto the instrument.
- **Visible-Cast Scoping (v2.4.0)**: Music-performance instructions are strictly scoped to the people actually depicted in each specific shot. Narrative, dance, and scenery shots no longer inherit boilerplate lists of vocalists and instrumentalists. Visible performers retain their assigned vocal roles and instrumental gaps, and old injected boilerplate is removed when recompiling saved prompts.
- **Lip-Sync Isolation**: Lip-sync generation is strictly bound to the active singing speaker, preserving visual and vocal consistency across multi-performer productions.

### 2.4 Prompt Enhancement & Fidelity Repair Controls (v2.2.2–v2.4.0)
- **Fidelity Repair Controls (0–5 Attempts)**: `Settings → Integrations → Prompt Enhancement` lets users configure between 0 and 5 fidelity repair attempts (default 1).
- **Auto-Continue Option**: An optional *"Generate even if fidelity checks fail"* setting allows Enhance on generation to proceed with the saved draft after retry attempts are exhausted, avoiding stalling the queue.
- **Localized Window Repairs**: When multi-window prompts trigger fidelity warnings, Maestro retries only the flagged event card/window prompt, preserving the overall story schedule, dialogue, continuity boundaries, and all accepted neighboring window prompts.
- **Music Timeline Vocal Bypass**: When a music/performance timeline is active, prompt enhancement treats the supplied audio as the authoritative source of vocals and timing. It skips dialogue writing and word-count gating while retaining visual action checks, preventing spurious spoken dialogue or silence directives from corrupting music video prompts.
- **Speech Duration Pre-checks**: Validates whether exact dialogue fits within the clip duration *before* loading the LLM writer, preventing repeated cycles trying to compress words that cannot be changed.
- **Nonverbal Sound Isolation**: Preserves authored sound effects (e.g. clicks, clangs, sweeps) in audio directions without requiring them to be falsely duplicated into visual action descriptions.

---

## 3. Editor Mode & Immersive Gallery

### 3.1 Non-Destructive Multitrack Timeline (`editor_projects.py`)
- Multi-track timeline supporting parallel video tracks, audio tracks (stems, sound effects, master), and text/subtitle tracks.
- Every clip tracks: start time, duration, source in/out trim points, asset ID, canvas transformations (scale, crop, position, aspect ratios 16:9, 9:16, 21:9), volume automation, opacity, and transitions (crossfade, dissolve).
- Atomic JSON serialization ensures projects survive process interruptions without corruption.

### 3.2 Single-Pass FFmpeg Filter Graph Compiler
- `compile_editor_render()` compiles the complete multi-track composition into a single FFmpeg `-filter_complex` command.
- Avoids lossy intermediate rendering. Audio tracks are synchronized with `atempo` chains and mixed cleanly via `amix`/`amerge`.
- Automatic hardware encoder detection: checks for NVIDIA NVENC (`h264_nvenc`, `hevc_nvenc`, `av1_nvenc`), Apple Silicon VideoToolbox (`h264_videotoolbox`, `hevc_videotoolbox`), VAAPI, AMF, QSV, and falls back to CPU (`libx264`, `libsvtav1`).

### 3.3 AI Round-Trip Take Workflow
- Creators select any clip on the timeline, click "Send to AI", generate a retake, variation, or stem-reactive re-render, and return the newly generated take directly into the clip's slot on the timeline without altering cut boundaries or soundtrack sync.

### 3.4 Immersive Gallery & Mobile UX (v2.4.0)
- **Fullscreen Vertical Swipe Viewer**: Enlarge images and browse media with vertical touch swipes and keyboard shortcuts; enters native browser fullscreen where supported.
- **Video Playback & Sound Persistence**: Single tap to pause/resume with auto-fading controls. Audio playback preference (mute/unmute) persists across clips during swipe browsing.
- **Cached First-Frame Posters**: Dedicated endpoint (`GET /api/v1/thumbnail/{filename}`) serves on-demand and cached JPEG first frames, allowing mobile browsers to render smooth gallery feeds without decoding full video files.
- **Before/After Image Comparison**: Interactive divider slider comparing source and generated results. Defaults to the active sidecar image or allows selecting arbitrary gallery/local images.
- **Pinch-to-Zoom & Drag**: Full-screen image zoom and pan with touch gesture reset.
- **Gallery-to-Active-Input Routing**: One-click media menu to dispatch any gallery image, captured video frame, or full video directly into the active Studio or Director input (References, Frames, Animate, Retake, Repaint, Recast, Upscale).

---

## 4. YuE2 48kHz Stereo Generation & "My Music" Training Studio

### 4.1 48 kHz Stereo Generation (`YuE2-music.md`)
- **Native High-Fidelity Audio**: 48 kHz stereo music generation conditioned on lyrics and style tags.
- **Three Conditioning Modes**:
  1. *Direct Generation*: Synthesis from lyrics with section tags and descriptive genre/instrument prompts.
  2. *Melody and Chords*: Guides melodic progression and harmonic changes using ABC notation scores.
  3. *Source Song Covers*: Ingests audio, transcribes note and chord progressions via SheetSage/MERT, and renders a new stylistic performance.
- **Automatic Instrumental LoRA (v2.3.0)**: Automatically routes instrumental requests to Mothersuperior's Instrumental AR LoRA at strength 1.0, pauses active artist LoRAs, and records the recipe in song metadata.
- **Multi-LoRA Mixing**: Supports experimental multi-LoRA mixes with independent weights, triggers, and weighted companion models.

### 4.2 "My Music" Training Studio: Auto & Guided Modes (v2.3.0)
Maestro v2.3.0 completely redesigned music fine-tuning into two streamlined workflows:
1. **Auto Mode**:
   - End-to-end queued workflow: uploads full songs $\rightarrow$ automatic vocal separation $\rightarrow$ timed lyrics $\rightarrow$ phrase excerpting $\rightarrow$ trains voice/sound (default 100 steps) $\rightarrow$ trains song style (default 200 steps) $\rightarrow$ saves matched LoRA.
   - Advances automatically in the background even if the browser is closed.
2. **Guided Mode (4 Explicit Stages)**:
   - *Stage 1: Recordings & Preparation*: Ingests full songs without requiring pre-written lyrics. Automatically separates vocal stems, aligns timed words, previews detected voices, and suggests phrase-based excerpts. Check-only recordings are excluded from training.
   - *Stage 2: Matched Voice/Sound Adaptation*: Jointly trains the real-audio tokenizer and decoder against recordings with waveform supervision (recommending the v9 tokenizer). Allows auditory comparison of `original` vs `before` vs `after` reconstruction.
   - *Stage 3: Autoregressive (AR) Song Style Training*: Trains the AR style adapter using the freshly prepared tokens from Stage 2.
   - *Stage 4: Test Song Audition*: Evaluates checkpoints at steps 100, 200, and completion by rendering a standardized test song through the model.
3. **Diagnostic Token Reconstruction**: Resynthesizes tokenized audio with adapter on/off to isolate whether artifacts stem from tokenization or generation conditioning.

---

## 5. Frontier Visual Models: H3 Singularity & Qwen Image 2.1

### 5.1 MiniMax H3 Singularity v1.3 References (v2.4.0)
- **Experimental Pruned INT8 Checkpoint**: Added as a dedicated reference model (`minimax_h3_ref2va_singularity`) using a 21 GB pruned INT8 ConvRot checkpoint.
- **LightX2V Ref2VA Turbo4 Preset**: Includes the recommended 4-step Turbo adapter (1.96 GB, strength 1.0, Euler sampler, CFG 1.0, video/audio shift 12/3). Turning Turbo off restores the 20-step default.
- **Quantization & Layout Parsing (`convrot_layout.py`)**: Automatically parses Comfy quantization descriptors (`.comfy_quant`) from safetensors to extract ConvRot format and group size (256) while handling grouped QKV row transformations.

### 5.2 Qwen Image 2.1 7B (v2.3.0–v2.4.0)
- **Unified Text-to-Image & Multi-Reference Editing**: Generates 1024×1024 to 2K images, performs multi-image reference editing (up to 10 references via `<image1>`, `<image2>`), and outputs transparent RGBA PNGs.
- **LoRA Architecture Compatibility**: Dedicated CivitAI filter and `qwen21` LoRA storage. Supports fused feed-forward layers (`gate_up`) exported by AI Toolkit / ComfyUI.
- **Reference Attention Memory Management**: Avoids large FP32 attention allocations on systems without native FlashAttention. Sizes the reference cache dynamically against available VRAM, recomputes when needed, and releases the cache prior to VAE decoding. Allows cancellation between encoder layers.

---

## 6. Performance Auto-Tune, Memory Bounding & Reliability

### 6.1 Empirical Hardware Profiles (Profiles 1–5)
- **Profile 1**: High VRAM ($\ge 24\text{ GB}$), High RAM ($\ge 64\text{ GB}$) $\rightarrow$ Full model residency in VRAM.
- **Profile 2**: High RAM ($\ge 32\text{ GB}$), Moderate VRAM ($12-23\text{ GB}$) $\rightarrow$ Balanced transformer weight pinning in host RAM, streamed to GPU.
- **Profile 3 / 3.5**: High VRAM ($\ge 24\text{ GB}$), Low Host RAM $\rightarrow$ Weights in VRAM, minimal host memory allocation.
- **Profile 4 / 4.5**: Low VRAM ($12-16\text{ GB}$), Moderate Host RAM $\rightarrow$ Consumer standard (RTX 3060/4070/4080); aggressive memory cleanup.
- **Profile 5**: Very Low Hardware ($< 12\text{ GB}$ VRAM, or CPU) $\rightarrow$ Maximum layer offloading, INT8 ConvRot / NVFP4 quantization.
- **Per-Job MMGP Allowance**: Passes per-job transformer VRAM allowance to MMGP, minimizing repeated weight streaming on lower profiles.

### 6.2 Bounded Memory Execution
- **Single-Frame Bounded Mask Memory (`_compose_recast_character_masks`)**: Processes video character masks one frame at a time rather than allocating whole-video boolean index and occupancy arrays. Eliminates tens of gigabytes of temporary memory spikes during Recast operations.
- **VRAM Safety Coefficient**: Enforces $\le 0.80$ headroom cap on $\ge 12\text{ GB}$ GPUs and $0.70$ on $< 12\text{ GB}$ GPUs.
- **CPU-Scoped Execution**: Resampling, torchaudio allocations, and audio loading strictly execute on CPU to prevent CUDA memory fragmentation.

### 6.3 System Reliability & File Locking
- **Windows Safe File Deletion (`share_delete_file_response`)**: Uses shared-delete file handles so files can be deleted or renamed in the gallery while video playback is actively streaming.
- **Atomic Settings Persistence**: Writes configuration files atomically with fallback restoration to prevent corruption on sudden shutdowns.

---

## 7. Universal Queue & State Recovery (`job_lifecycle.py`)

- **Persistent Render Queue**: Survives application crashes and restarts; interrupted jobs remain paused with their completed clips and inputs intact.
- **Asset Ownership**: When a job is submitted, input assets are copied or symlinked into the job workspace so subsequent user deletions or edits cannot corrupt active pipelines.
- **Queue Pre-Enhancement**: Asynchronously develops and adapts prompts while the GPU is busy, ensuring models receive optimized prompts immediately when GPU turns become available.
