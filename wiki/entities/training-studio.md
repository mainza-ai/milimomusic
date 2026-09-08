---
title: Training Studio (Deprecated)
type: entity
created: 2026-08-19
updated: 2026-09-08
sources: [sources/training-studio-guide.md, sources/readme.md]
tags: [training, lora, finetune, studio, deprecated, minimax, heartmula]
aliases: [LoRA Training Studio]
---

# Training Studio (Deprecated & Retired)

> [!WARNING]
> **Status: Deprecated & Retired (2026-09-08)**  
> The Training Studio has been formally decommissioned and removed from the active codebase. Milimo Music v2 has standardized on **MiniMax Music 3** as its primary production music generation engine and **RVC v2** for vocal cloning. Local LoRA training for MiniMax Music 3 is technically infeasible for consumer workstations.

## Architectural Decision Record (Why Training Studio Was Retired)

A comprehensive architectural and empirical investigation across the model architecture, Hugging Face discussions, and hardware constraints concluded that local fine-tuning / LoRA training cannot be delivered in production for the following reasons:

### 1. Missing Neural Audio Encoder (Architectural Impossibility)
MiniMax Music 3 is a hybrid Diffusion Transformer (DiT) and Autoregressive (AR) pipeline. To train or fine-tune on user audio files (e.g. 5 uploaded MP3/WAV stems), raw audio must be converted into the model's semantic latent space using an encoder.
- MiniMax open-sourced **only the decoder / generation pipeline** (`dit.py`, `ar.py`, `fusion.py`, `depth.py`, `vocoder.py`).
- **MiniMax never open-sourced the neural audio encoder / RVQ tokenizer**.
- Without this proprietary encoder, arbitrary user audio cannot be converted into the latent representations required to compute flow-matching loss or cross-entropy. Any attempt to train without it produces meaningless noise.

### 2. VRAM & Hardware Memory Physics (Hardware Impossibility)
- MiniMax Music 3 base model weights in `bfloat16` occupy **27 Gigabytes** of memory.
- Backpropagation through a 14B multi-stage diffusion pipeline across 20–30 second stereo audio requires holding activation graphs and optimizer states (AdamW), demanding **50 GB to 64 GB+ of dedicated GPU memory**.
- Consumer hardware (Apple Silicon Macs with 16GB–36GB Unified Memory, or NVIDIA RTX GPUs with 8GB–24GB VRAM) immediately encounters fatal Out-Of-Memory (OOM) crashes during backprop.

### 3. HeartMuLa Legacy Separation
The earlier prototype scripts in `backend/app/services/training/` targeted the legacy [HeartMuLa](heartmula.md) 3B model. Because HeartMuLa was superseded by MiniMax Music 3 in Milimo Music v2 due to fidelity and musicality advantages, maintaining training stubs for an obsolete model created confusion and technical debt.

### 4. "LoRAs" in the Ecosystem
Files labeled "MiniMax Music 3 LoRA" circulating in the open-source community (e.g., ComfyUI) are **step-distillation LoRAs** (trained on supercomputer clusters to reduce sampling steps from 24 to 8 for faster inference), **not** user-trained style adapters.

---

## Active Alternatives in Milimo Music

Instead of maintaining a non-functional training studio, Milimo Music provides genuine production capabilities:
1. **Prompt & Tag Conditioning**: Steering MiniMax Music 3 via rich structural prompts, genre tags, BPM, and mood descriptors.
2. **Reference Audio Inpainting & Continuity**: Providing reference audio frames for stylistic priming.
3. **Voice Studio (RVC v2)**: Real, local vocal cloning using Retrieval-based Voice Conversion (~100MB VRAM, trains in minutes on Apple Silicon or consumer GPUs).

---

## Related pages
- [MiniMax Music 3](../overview.md) | [System Architecture](../architecture.md)
- [HeartMuLa](heartmula.md) | [Roadmap (v2)](../roadmap.md)
