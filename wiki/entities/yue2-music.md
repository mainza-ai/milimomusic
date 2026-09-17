---
title: YuE2 48kHz Stereo Music Provider
type: entity
tags: [yue2, music, 48khz, stereo, abc-notation, style-lora, provider]
created: 2026-09-16
updated: 2026-09-16
sources: [sources/maestro-creative-studio.md]
aliases: [YuE2, YuE2Provider, 48kHzMusicEngine]
---

# YuE2 48kHz Stereo Music Provider

**YuE2 3B** is an open-weight foundation model for full-length **48 kHz stereo music generation**, integrated into Milimo Music as a frontier generative provider alongside MiniMax Music 3 and legacy HeartMuLa.

Originally ported and production-hardened in [Maestro](../sources/maestro-creative-studio.md), YuE2 delivers studio-grade 48 kHz stereo audio, symbolic ABC notation conditioning, and a full personal style adapter fine-tuning studio.

---

## 1. Core Architectural Capabilities

| Feature | YuE2 3B | MiniMax Music 3 | HeartMuLa (Legacy) |
|---|---|---|---|
| **Audio Quality** | **48 kHz Full Stereo** | 32 kHz / 44.1 kHz Stereo | 44.1 kHz Stereo |
| **Model Size** | 3B Autoregressive + Diffusion Vocoder | Large DiT (Cloud/MLX) | 3B (RL-tuned / base) |
| **Symbolic Conditioning**| **ABC Notation / Chords / Melody** | Section tags & Text Captions | Lyrics tags only |
| **Source Song Covers** | **Yes** (SheetSage/MERT2 symbolic extraction) | Text prompt adaptation | MulaCover latent alignment |
| **Personal Style LoRA** | **Yes** (Resumable AR/NAR adapters) | No local training | Experimental Heartlib LoRA |
| **Deployment Mode** | 100% Local (CUDA / MPS) | Local MLX or Cloud API | Local Torch / MPS |

---

## 2. Three Generation Modes

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

---

## 3. "My Music" Style Adapter Training Studio

YuE2 includes an experimental, production-hardened style fine-tuning pipeline:

1. **Dataset Ingestion**:
   - Ingests user recordings with exact timestamped lyrics and style descriptions.
   - Enforces **held-out validation tracks** (at least one recording is excluded from gradient updates to prevent overfitting).
2. **Resumable Checkpoints**:
   - Trains low-rank Autoregressive (AR) and Non-Autoregressive (NAR) adapters.
   - Checkpoints save optimizer states, random seeds, and tokenizer states, enabling 1-click resumption.
3. **Automatic Checkpoint Auditions**:
   - At steps 100, 200, and final, training temporarily releases models and renders a standardized test song through YuE2.
   - Creators evaluate checkpoints by listening to real audio samples rather than relying on abstract validation loss.
4. **Source Token Reconstruction Diagnostic**:
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
