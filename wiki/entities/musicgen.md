---
title: Meta MusicGen Provider
type: entity
tags: [model, audio, provider, musicgen, audiocraft, melody, cpu-friendly, cross-platform]
created: 2026-09-25
updated: 2026-09-25
sources: [entities/generation-provider.md, concepts/audio-synthesis-standards.md, concepts/cross-modal-model-lifecycle.md]
aliases: [MusicGen, Meta MusicGen, MusicGenProvider, musicgen_small, musicgen_melody]
---

# Meta MusicGen Provider

**Meta MusicGen** (from Meta AI's AudioCraft team) is a robust autoregressive transformer model for controllable music generation. In Milimo Music, MusicGen provides a **lightweight, CPU-capable, and melody-guided** generative alternative to [MiniMax Music 3](minimax-music3.md) and [Stable Audio Open](stable-audio-open.md).

---

## 1. Supported Model Variants

Milimo Music integrates three primary MusicGen variants via Hugging Face `transformers`:

| Variant | Parameters | VRAM (FP16) | Conditioning Capabilities | Recommended Hardware |
|---|---|---|---|---|
| **`musicgen-small`** | 300M | ~1.5 GB | Text prompt only | Entry CPU / 4GB VRAM Laptops |
| **`musicgen-melody`** | 1.5B | ~3.5 GB | Text prompt + reference melody audio | Mid Single GPU (6GB+ VRAM) / MPS |
| **`musicgen-large`** | 3.3B | ~6.8 GB | Text prompt only (higher acoustic detail) | 8GB–12GB GPU / Apple Silicon 16GB |

---

## 2. Key Features in the Milimo DAW

### 2.1 Melody & Whistle Conditioning (`musicgen-melody`)
Users in the [Session Workspace](session-workspace.md) can record a rough vocal melody, humming, whistle, or acoustic instrument riff into an audio track. `musicgen-melody` extracts the pitch contour and harmonic structure, generating a fully orchestrated track adhering to that melodic guide.

### 2.2 Low-Resource & CPU Execution
For users running on older Intel Macs, Windows laptops without discrete GPUs, or Dockerized CPU microservices, `musicgen-small` executes reliably without triggering memory pressure or requiring Apple Silicon MLX.

---

## 3. Acoustic Resampling & Loudness Calibration

MusicGen models output native **32.0 kHz** mono or stereo audio. To ensure seamless playback within the 44.1 kHz DAW timeline and zero volume jumps:

1. **Polyphase Sinc Resampling**: High-quality resampling from 32.0 kHz to 44.1 kHz (`torchaudio.transforms.Resample` or `librosa.resample` with Kaiser best window).
2. **Channel Expansion**: Mono outputs are expanded to stereo via subtle decorrelation or dual-channel duplication.
3. **Psychoacoustic Loudness Calibration**: Staged to **-14.0 LUFS** with a **-1.0 dBFS** true peak ceiling according to [Audio Synthesis Standards](../concepts/audio-synthesis-standards.md).

---

## 4. Lifecycle & Immediate Eviction

To conform to the [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md):
- On completion of generation, `MusicGenProvider.unload()` immediately deletes model references, runs `gc.collect()`, and flushes `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`.
- Memory returns from ~3.5 GB to 1.1 GB before downstream stem separation begins.

---

## Related Pages
- [Generation Provider Abstraction](generation-provider.md)
- [Stable Audio Open](stable-audio-open.md)
- [MiniMax Music 3](minimax-music3.md)
- [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md)
- [Audio Synthesis Standards](../concepts/audio-synthesis-standards.md)
