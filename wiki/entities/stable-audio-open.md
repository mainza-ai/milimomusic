---
title: Stable Audio Open 1.0 Provider
type: entity
tags: [model, audio, provider, stable-audio, dit, diffusers, cross-platform, cuda, mps]
created: 2026-09-25
updated: 2026-09-25
sources: [entities/generation-provider.md, concepts/audio-synthesis-standards.md, concepts/cross-modal-model-lifecycle.md]
aliases: [Stable Audio Open, StableAudioProvider, stable_audio_open_1_0]
---

# Stable Audio Open 1.0 Provider

**Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`) is an open-weights generative audio foundation model by Stability AI. In Milimo Music, it serves as the primary **cross-platform foundation model alternative** to [MiniMax Music 3](minimax-music3.md), running natively on **NVIDIA CUDA**, **Apple Silicon MPS**, and **CPU**.

---

## 1. Model Architecture & Specifications

- **Architecture**: Latent Diffusion Transformer (DiT) operating on continuous spectrogram embeddings.
- **Audio Output**: Native **44.1 kHz stereo** audio (matches Milimo DAW project sample rate directly, eliminating resampling phase distortion).
- **Conditioning**: T5-based text conditioning embeddings paired with explicit duration/timing tokens.
- **Maximum Duration**: 47 seconds per forward pass. Longer audio is rendered via the [orchestration pipeline](../concepts/generation-pipeline.md) using seamless latent cross-fades and track extension.
- **VRAM Footprint**: ~6.5 GB in `torch.float16` / `bfloat16`; under 4.0 GB when quantized to INT8.
- **License**: Stability AI Community License (open weights, free for research and commercial revenue under $1M/yr).

---

## 2. Platform & Hardware Compatibility

Unlike MiniMax Music 3 (which requires Apple Silicon MLX), Stable Audio Open runs via Hugging Face `diffusers.StableAudioPipeline`:

| Hardware Platform | Device Target | Precision | Generation Speed (47s clip) |
|---|---|---|---|
| **NVIDIA GeForce RTX 4090 / 3090** | `cuda` | `torch.float16` | ~3.8 seconds |
| **NVIDIA RTX 3060 / 4060 (12GB)** | `cuda` | `torch.float16` | ~9.2 seconds |
| **Apple Silicon (M1/M2/M3/M4 Max)** | `mps` | `torch.float16` / `float32` | ~8.5 seconds |
| **Apple Silicon (M1/M2/M3 Base 16GB)** | `mps` | `torch.float16` | ~14.0 seconds |
| **x86_64 / ARM64 CPU** | `cpu` | `torch.float32` | ~65 seconds |

---

## 3. Acoustic Staging & DAW Integration

To comply with [Audio Synthesis & Loudness Calibration Standards](../concepts/audio-synthesis-standards.md):
- **Integrated Loudness**: Calibrated to $-14.0\text{ LUFS} \pm 1.0\text{ dB}$.
- **True Peak Ceiling**: Hard-limited to $-1.0\text{ dBFS}$ using procedural lookahead limiting to prevent inter-sample clipping upon MP3/AAC export.
- **Stem Pipeline Pass-Through**: Audio outputs feed directly into [BS-Roformer Neural Separation](stem-separator.md) and [MuScriptor](muscriptor.md) transcription with zero format conversion.

---

## 4. Lifecycle & Immediate Eviction

Stable Audio implements the [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md) contract:
```python
async def generate(...):
    pipe = self._load_pipeline()
    try:
        audio = pipe(prompt, audio_end_in_s=duration_sec, ...)
        return post_process(audio)
    finally:
        if self.lifecycle_mode == "eager":
            self.unload()
```
`unload()` strips accelerate offload hooks, deletes pipeline references, invokes `gc.collect()`, and calls `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`, immediately returning VRAM to baseline before stem separation begins.

---

## Related Pages
- [Generation Provider Abstraction](generation-provider.md)
- [MiniMax Music 3](minimax-music3.md)
- [Meta MusicGen](musicgen.md)
- [Cross-Modal Model Lifecycle](../concepts/cross-modal-model-lifecycle.md)
- [Audio Synthesis Standards](../concepts/audio-synthesis-standards.md)
