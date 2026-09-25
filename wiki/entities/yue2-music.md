---
title: YuE2 48kHz Stereo Music Provider
type: entity
tags: [yue2, music, 48khz, stereo, abc-notation, style-lora, provider, my-music]
created: 2026-09-16
updated: 2026-09-24
sources: [sources/maestro-creative-studio.md]
aliases: [YuE2, YuE2Provider, 48kHzMusicEngine]
---

# YuE2 48kHz Stereo Music Provider

**YuE2 3B** is an open-weight foundation model for full-length **48 kHz stereo music generation**, integrated into Milimo Music as a frontier generative provider alongside MiniMax Music 3 and legacy HeartMuLa.

Originally ported and production-hardened in [Maestro](../sources/maestro-creative-studio.md) (v2.4.0), YuE2 delivers studio-grade 48 kHz stereo audio, symbolic ABC notation conditioning, source song covers, Mothersuperior instrumental AR LoRA auto-routing, and a two-track personal style training studio (Auto & Guided modes).

---

## 1. Core Architectural Capabilities

| Feature | YuE2 3B | MiniMax Music 3 | HeartMuLa (Legacy) |
|---|---|---|---|
| **Audio Quality** | **48 kHz Full Stereo** | 32 kHz / 44.1 kHz Stereo | 44.1 kHz Stereo |
| **Model Size** | 3B Autoregressive + Diffusion Vocoder | Large DiT (Cloud/MLX) | 3B (RL-tuned / base) |
| **Symbolic Conditioning**| **ABC Notation / Chords / Melody** | Section tags & Text Captions | Lyrics tags only |
| **Source Song Covers** | **Yes** (SheetSage/MERT2 symbolic extraction) | Text prompt adaptation | MulaCover latent alignment |
| **Instrumental Routing** | **Yes** (Mothersuperior Instrumental AR LoRA auto-applied) | Text prompt ("Instrumental") | Negative prompt |
| **Multi-LoRA Mixes** | **Yes** (Independent weights & triggers) | None | None |
| **Personal Style Training** | **Yes** (Auto & Guided 4-Stage Studio) | No local training | Experimental Heartlib LoRA |
| **Deployment Mode** | 100% Local (CUDA / MPS) | Local MLX or Cloud API | Local Torch / MPS |

---

## 2. Generation Modes

### 2.1 Direct Generation (Default)
- Conditioned on lyrics formatted with standard section tags (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`) and a descriptive style prompt detailing genre, tempo, instruments, vocal timbre, and production style.
- Uses 32 denoising steps with the v4 neural audio tokenizer.

### 2.2 Melody & Chord Planning (Symbolic Guidance)
- Accepts standard **ABC notation** musical scores defining lead melodies and chord progressions (e.g. `|: "C" C2 E2 "G" G4 :|`).
- Enables musicians and composers to enforce specific chord changes, hook melodies, and modal tonalities rather than relying purely on random generation.

### 2.3 Source Song Covers
- Ingests an existing audio file (WAV/MP3).
- Uses SheetSage2 / MERT2 neural pitch and harmony transcribers to extract symbolic musical notes and harmony contours.
- Renders a brand-new musical arrangement and performance conditioned on the extracted score and newly supplied lyrics/style tags.

### 2.4 Automatic Instrumental LoRA (v2.3.0)
- When generating instrumental music, the provider automatically engages **Mothersuperior's Instrumental AR LoRA** at strength 1.0.
- Temporarily pauses active artist LoRAs for that specific job to prevent vocal artifacts from leaking into instrumental tracks.
- Preserves the recipe transparently in the generated track's metadata.

### 2.5 Multi-LoRA Mixing
- Supports concurrent multi-LoRA mixes with independent weights and trigger injections in advanced settings.
- Rejects incompatible tokenizer pairs (e.g. mixing v4 and v9 tokenizers) while supporting weighted sound companions.

---

## 3. "My Music" Personal Training Studio (Auto & Guided)

Maestro v2.3.0 established a streamlined, production-grade fine-tuning system offering two operational modes:

### 3.1 Auto Mode
- Completely automated, background-advancing queued workflow.
- Ingests full song recordings $\rightarrow$ performs vocal separation $\rightarrow$ extracts timed lyrics and phrase excerpts $\rightarrow$ trains matched real-audio tokenizer/decoder (100 steps) $\rightarrow$ trains AR song-style adapter (200 steps) $\rightarrow$ saves verified LoRA.
- Runs without requiring the browser to stay open.

### 3.2 Guided Mode (4 Explicit Stages)
1. **Recordings & Dataset Preparation**:
   - Ingests full songs without requiring manual lyrics.
   - Automatically separates vocals, transcribes timed words, previews detected voices, and suggests phrase-based excerpts.
   - Check-only recordings are excluded from gradient updates; manual and reviewed excerpts survive rescanning.
2. **Matched Voice/Sound Adaptation**:
   - Jointly trains the real-audio tokenizer and decoder against recordings with waveform supervision (recommending the v9 tokenizer architecture).
   - Allows auditory comparison of `original` vs `before` vs `after` reconstruction to verify acoustic transparency.
3. **Autoregressive (AR) Song Style Training**:
   - Trains the AR style adapter using the freshly prepared tokens from Stage 2.
   - Saves paired optimizer checkpoints allowing seamless resumption.
4. **Test Song Audition**:
   - At steps 100, 200, and final completion, automatically renders a standardized test song through YuE2.
   - Creators evaluate checkpoint quality by listening to actual musical audio rather than relying on abstract validation loss metrics.

### 3.3 Diagnostic Token Reconstruction
- A developer diagnostic endpoint (`/reconstruct`) that resynthesizes source audio with adapter ON vs OFF:
  - `original`: raw excerpt.
  - `adapter-off`: decoded with base model.
  - `adapter-on`: decoded with personal style adapter.
- Accurately isolates whether vocal or stylistic issues stem from audio tokenization or downstream generation conditioning.

---

## 4. Hardware Requirements
- **Inference**: Requires $\ge 12\text{ GB}$ VRAM on CUDA or 18GB Unified Memory on Apple Silicon.
- **Training Studio**: Requires $\ge 20\text{ GB}$ VRAM (exercised on RTX 4090 24GB or A100).
- Runs under [Global Hardware Coordinator](hardware-coordinator.md) device locks to guarantee mutual exclusion with video diffusion models.

---

## 5. Related Pages
- [Generation Provider](generation-provider.md)
- [MiniMax Music 3](minimax-music3.md)
- [HeartMuLa](heartmula.md)
- [Maestro Creative Studio Ingest](../sources/maestro-creative-studio.md)
- [Hardware Auto-Tune](../concepts/hardware-autotune-memory-profiles.md)
