# Heartlib Bible
**The Definitive Guide to the Heartlib Audio Generation Framework**

> **Version**: 1.0
> **Target Audience**: AI Engineers, Backend Developers, and Researchers.

---

## 1. Introduction

**Heartlib** is a specialized modular framework for high-fidelity music generation, bridging the gap between Large Language Models (LLMs) and Neural Audio Codecs. It enables text-conditional music generation, infinite track extension, and lyrics alignment.

### Core Philosophy
Heartlib treats audio generation as a **language modeling task**. It discretizes continuous audio waveforms into semantic tokens using a neural codec, allowing a Transformer backbone (HeartMuLa) to predict audio tokens autoregressively, conditioned on text (lyrics) and style tags.

---

## 2. System Architecture

Heartlib consists of four primary components as detailed in the technical report: **HeartCLAP**, **HeartTranscriptor**, **HeartCodec**, and **HeartMuLa**.

### 2.1 HeartMuLa (Music Language Model)
The backbone responsible for semantic understanding and token prediction.
*   **Architecture**: **Hierarchical Factorization** (Global + Local Transformers).
    *   **Global Transformer**: Predicts the coarse semantic tokens (Layer 0) conditioned on history.
    *   **Local Transformer**: Predicts fine-grained acoustic details (Layers 1-7) conditioned on the global token.
*   **Backbone**: Modified `Llama 3.2` 3B (Global) + 300M (Local).
*   **Vocabulary**:
    *   **Text**: 128,256 tokens (Llama 3 Tokenizer).
    *   **Audio**: 8,197 tokens.

### 2.2 HeartCodec (Neural Audio Codec)
A low-frame-rate, high-fidelity neural audio codec.
*   **Frame Rate**: **12.5 Hz** (Ultra-low for efficient long-sequence modeling).
*   **Quantization**: 8 Codebooks (RVQ).
*   **Architecture**:
    *   **Encoder**: Semantic-rich (Whisper + WavLM + MuEncoder).
    *   **Compressor**: Downsamples to 12.5 Hz.
    *   **Decoder**: Flow Matching based high-fidelity reconstruction.

### 2.3 The Pipeline (`HeartMuLaGenPipeline`)
The glue that orchestrates generation:
1.  **Preprocessing**: Constructs the prompt `C = [Tags, Reference, Lyrics]`.
2.  **Generation Loop**: Autoregressively predicts audio tokens.
3.  **Decoding**: Passes predicted tokens to `HeartCodec` to generate waveform.

---

## 3. Conditioning Mechanism (Crucial)

To ensure style adherence, the input prompt must follow a strict structural format.

### 3.1 Prompt Structure
The model expects a continuous sequence without internal resets. The required structure is:

`[BOS] <tag> {Style Tags} </tag> [EOS] [MUQ_EMBED] [Lyrics...] [EOS]`

*   **Tags**: Comma-separated style descriptors.
*   **[MUQ]**: A placeholder for the MuQ-MuLan embeddings of reference audio.
*   **Lyrics**: Structured lyrics with markers like `[Verse]`, `[Chorus]`.

### 3.2 Supported Tags (HeartMuLa-3B Beta)
For optimal results, stick to the following supported tags. The model is fine-tuned to respond specifically to these concepts:

*   **Warm, Reflection, Pop, Cafe, R&B, Keyboard, Regret**
*   **Drum machine, Electric guitar, Synthesizer, Soft, Energetic, Electronic**
*   **Self-discovery, Sad, Ballad, Longing, Meditation, Faith**
*   **Acoustic, Peaceful, Wedding, Piano, Strings, Acoustic guitar**
*   **Romantic, Drums, Emotional, Walking, Hope, Hopeful**
*   **Powerful, Epic, Driving, Rock**

---

## 4. Installation & Setup
*(See README.md for latest instructions)*

## 5. Production Usage Patterns
*(Unchanged - Follow MusicService implementation)*

## 6. Advanced Generation Techniques

### 6.1 Structural Lyrics
Use explicit structure markers to guide the generation flow:
```text
[Intro]
(Instrumental build up)
[Verse]
Hello darkness my old friend...
[Chorus]
I've come to talk with you again...
[Outro]
(Fade out)
```

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **"NotImplementedError: The operator 'aten::conv1d'..."** | MPS (Mac) missing backend support for specific op. | Set `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`. |
| **FastAPI server hangs during generation** | Pipeline running on main thread. | Wrap pipeline call in `await loop.run_in_executor(...)`. |
| **Generation continues after client disconnect** | No cancellation logic connected. | Implement `abort_event` passing and trigger it on disconnect/API call. |

---

## 8. Internal Mechanics & Secrets

### 8.1 The Tokenizer
*   **Double BOS Trap**: The Llama 3 tokenizer automatically prepends a `[BOS]` token. When concatenating [Tags] and [Lyrics], you must ensure you do not accidentally insert a second `[BOS]` before the lyrics, or the model will ignore the tags.
*   **Correct Sequence**: `[128000 (BOS), ...tags..., 128001 (EOS), ...lyrics..., 128001 (EOS)]`

### 8.2 The Generation Loop
*   **Latency**: The model generates at approx 12.5 Hz (frames), but decodes to 48kHz audio.
*   **Context**: The `HeartMuLaGenPipeline` handles `muq_embed` injection implicitly at the junction between tags and lyrics.

---

## 9. API Reference (`HeartMuLaGenPipeline`)

**`__call__` Parameters:**

*   `model_inputs` (Dict):
    *   `lyrics` (str): Full lyrics.
    *   `tags` (str): Comma-separated style tags (e.g., "pop, female vocals").
*   `max_audio_length_ms` (int): Duration in milliseconds (Default: 120,000).
*   `temperature` (float): Sampling randomness (0.1 = deterministic, 1.5 = chaotic). Default 1.0.
*   `topk` (int): Vocabulary truncation. Lower (e.g., 50) = safer, Higher (e.g., 200) = expressive.
*   `cfg_scale` (float): Guidance scale.
    *   `1.0`: Unconditional (ignore text).
    *   `>1.0`: Conditional (follow text). Default 1.5.
*   `history_tokens` (Tensor): Preceding audio tokens `[B, S, codes]`.
*   `abort_event` (Event): Signal to stop generation.
*   `callback` (Callable): `func(progress_int, msg_str)`.

---

## 10. Architecture & Extension Deep Dive (Critical Engineering Notes)

### 10.1 Parallel Codebooks (The "8+1" Rule)
*   **Concept**: HeartMuLa operates on **9 parallel channels**:
    *   **Channel 0-7**: Audio Codebooks (8 total).
    *   **Channel 8**: Text/Control (1 total).
*   **The Trap**: Some versions of `HeartCodec` configuration file report `num_quantizers = 7`. Trusting this config blindly leads to a pipeline configured for 8 channels (7+1), which causes `Tensor Size Mismatch [2, 7] vs [2, 8]`.
*   **The Fix**: Always hardcode `parallel_number = 8 + 1` in the pipeline initialization to match the actual model architecture, regardless of the codec config file.

### 10.2 Extension Logic (The "Double Injection" Stutter)
When extending a track, **History Injection** must be handled with precision to avoid phase discontinuity (stutter/noise).
*   **Incorrect Approach**: Injecting history frames, then taking the *last history frame* and re-feeding it as the *first generation seed*.
    *   *Result*: The model sees the same frame twice (once at `t`, once at `t+1`). This breaks the audio waveform phase.
*   **Correct Approach**:
    1.  Inject history frames to warm up the KV Cache.
    2.  Capture the model's **PREDICTION** from the final history step.
    3.  Use this **PREDICTION** as the seed for the generation loop.
    *   *Result*: Seamless continuation based on the model's autoregressive flow.

### 10.3 Model Output Slicing
*   **Inputs**: The model accepts [B, S, 9] (8 Audio + 1 Text).
*   **Outputs (`generate_frame`)**: The model returns **[B, 8]** (8 Audio Channels ONLY).
*   **The Trap**: Attempting to slice the output (e.g., `[:, :-1]`) thinking it contains a text column will destroy the last audio codebook (Channel 7), causing invalid tensor shapes.
*   **Rule**: Use the model prediction **as-is**. It is already pure audio.

### 10.4 Apple Silicon (MPS) Specifics
*   **Precision**: MPS on macOS struggles with `bfloat16` and `autocast`.
*   **Configuration**:
    *   **Dtype**: Use `torch.float16`.
    *   **Autocast**: **DISABLE** it. Use `contextlib.nullcontext()` instead.
    *   Enable `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` for missing operators (like `aten::conv1d` on some versions).

### 10.5 Audio Reference & Conditioning (Status Report 2026)
HeartMuLa technically supports two forms of audio conditioning, but only one is fully functional in the OSS release:

1.  **Continuation (Audio-to-Audio Extension)**: **Fully Supported**.
    *   **Mechanism**: You can extend an existing audio track by feeding its encoded tokens as `history_tokens`.
    *   **Uploads**: To extend an uploaded external file (MP3/WAV), you must first run it through `HeartCodec.encode()` to generate the compatible 8-codebook token sequence.
    
2.  **Style Conditioning (Audio-to-Style)**: **Not Supported (Missing Encoder)**.
    *   **Mechanism**: The model has a `muq_embed` projection layer intended to take embeddings from a separate audio encoder (like MuLan or CLAP) to guide style.
    *   **Status**: The necessary `muq_mulan` encoder weights and preprocessing code are **not included** in the open-source release. Providing `ref_audio` to the pipeline will raise a `NotImplementedError`. Style must be controlled via text tags.

---

## 11. Additional Capabilities (Unused Gems)
Audit 2026 revealed valuable components included in the library but not currently utilized in the standard pipeline.

### 11.1 Lyrics Transcription (`HeartTranscriptor`)
A complete automatic speech recognition (ASR) pipeline based on Whisper.
*   **Pipeline**: `heartlib.pipelines.HeartTranscriptorPipeline`.
*   **Capabilities**:
    *   High-accuracy lyrics transcription from raw audio.
    *   Timestamp generation (word-level or segment-level).
*   **Potential Use Cases**:
    *   "Lyrics from Audio": Auto-generate lyrics for uploaded files.
    *   "Karaoke Mode": Synchronize lyrics display with playback.
    *   "Style Extraction": Analyze lyrical themes from existing songs to auto-prompt generation.

### 11.2 Standalone Neural Compression (`HeartCodec`)
The codec can be used independently of the generative model (`HeartMuLa`) as a powerful compression tool.
*   **Model**: `heartlib.heartcodec.HeartCodec.detokenize(codes, ...)`
*   **Performance**: Compresses 48kHz audio into 8 discrete codebooks at 12.5 Hz.
*   **Capabilities**:
    *   **Extreme Compression**: Store hours of audio in mere megabytes of token data (~1.5 kbps effective bitrate).
    *   **Neural Remixing**: By manipulating the token sequences directly (e.g., shuffling codebooks, repeating token segments) before decoding, you can achieve unique "glitch" or "granular" synthesis effects impossible with traditional DSP.

---
---

## 12. Advanced Architecture Internals (Deep Dive 2026)

### 12.1 Audio In-Painting (LM-Guided Repair)

> [!IMPORTANT]
> `HeartCodec.inpaint()` is for **acoustic reconstruction only**. It cannot generate new semantic content. For true in-painting (filling gaps with fresh lyrics/music), you must use **LM-Guided Repair** (two-stage pipeline).

#### The Problem with Codec-Only In-Painting
| Mask Mode | Result | Why |
|-----------|--------|-----|
| `mask=0` | Silence | Codec treats null tokens as "no signal" |
| `mask=2` + copy tokens | Repetition | Copied tokens = repeated audio |

#### LM-Guided Repair Architecture
```
Stage 1: HeartMuLa (Semantic Generation)
  → Generate NEW tokens for the gap
  → Input: history_tokens (8s context) + lyrics + tags

Stage 2: HeartCodec (Acoustic Reconstruction)  
  → Decode [context + new tokens + context]
  → Output: Phase-aligned audio with crossfade
```

#### Possible Optimal Parameters for Style Consistency (still testing)
When generating repair tokens, use "Audio Dominance Mode" to force the model to continue based on what it hears rather than text guidance:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | **0.2** | Near-deterministic sampling (stable vocals) |
| `topk` | **30** | Narrow beam search |
| `cfg_scale` | **1.0** | Disable text guidance; rely on audio history |

#### Tensor Shape Gotchas
`HeartMuLa` returns tokens in inconsistent shapes. Always normalize:
```python
# Normalize to [1, C, T] before splicing
if generated_tokens.dim() == 2:
    generated_tokens = generated_tokens.unsqueeze(0)  # [C,T] → [1,C,T]
elif generated_tokens.dim() == 3 and generated_tokens.shape[-1] == 8:
    generated_tokens = generated_tokens.transpose(1, 2)  # [B,T,C] → [B,C,T]
```

*   **Implementation**: See `backend/app/services/inpainting_service.py`
*   **Debug Log**: See `INPAINTING_DEBUG.md` for full history

### 12.2 Fine-Tuning Capabilities (Training Studio 2026 Update)
The HeartMuLa framework now includes a dedicated **Training Studio** for user-friendly fine-tuning.

#### 12.2.1 Core Training Architecture
We support two primary methods for adapting the model to new styles:

1.  **LoRA (Low-Rank Adaptation)**:
    *   **Mechanism**: Injects low-rank decomposition matrices (`A` and `B`) into the attention projections (`q_proj`, `v_proj`, etc.) of the backbone.
    *   **Efficiency**: Trains only ~0.1% of parameters. Requires ~16GB VRAM for the 3B model.
    *   **Device Handling (Crucial)**: LoRA adapters must be initialized on the **same device** as the model to avoid CPU/GPU conflicts. This is now handled automatically in `HeartMuLaLoRATrainer`.
    *   **Use Case**: User-specific style cloning, quick experiments.

2.  **Full Fine-Tuning**:
    *   **Mechanism**: Updates all weights of the global and local transformers.
    *   **Efficiency**: Extremely resource intensive. Requires ~24GB+ VRAM with gradient checkpointing and mixed precision.
    *   **Use Case**: Fundamental genre expansion, distinct language adaptability.

#### 12.2.2 Style Expansion Workflow (How to Add New Genres)
Since generation is a language modeling task, "styles" are just tokens. You can add new styles (e.g., "Samba", "Chiptune") without code changes:
1.  **Dataset Prep**: Collect audio files for the target genre.
2.  **Labeling**: Create text captions containing the new tag: `<tag> Samba, Rhythmic </tag>`.
3.  **Training**: Fine-tune the model (via Training Studio) to map the *Samba text tokens* to the *Samba audio tokens*.
4.  **Result**: The model learns the statistical correlation. When prompted with "Samba", it will now produce the associated audio patterns.

### 12.3 The "9th Channel" (Time-Aligned Control)
The pipeline hardcodes `parallel_number = 8 + 1`. This 9th channel is theoretically capable of holding **Time-Aligned Text Tokens**.
*   **Current State**: Currently used mostly for global conditioning padding.
*   **Future Potential**: By injecting specific text tokens at specific timestamps in Channel 8, it is theoretically possible to force the model to sing a specific word at a specific second, enabling precise "Lyrics Timing Control" rather than the current autoregressive "free flow."

---
---

## 13. UI Architecture & Design System (2026 Audit)

### 13.1 Glassmorphism Standard
The Milimo UI (including **Style Manager** and **Training Studio**) now follows a strict **Glassmorphism** design system to ensure visual premium quality:
*   **Panels**: `bg-white/80 backdrop-blur-2xl border-white/50`. Never use solid opaque backgrounds for modals.
*   **Gradients**: Use **Cyan-to-Fuchsia** (`from-cyan-500 to-fuchsia-500`) for primary actions and accents. Avoid generic primary blue.
*   **Typography**: Use generic sans-serif for UI, `font-mono` for metrics (Loss, Epochs).
*   **Feedback**: Real-time metrics (like Training Loss) must be displayed directly in the card UI, not hidden in logs.

### 13.2 Real-Time Progress Architecture
*   **State Tracking**: Training progress is dual-tracked by `current_epoch` (rough) and `percent` (precise step-based).
*   **Metric Stream**: The backend parses stdout JSON lines from training subprocesses to extract `loss` and `lr` in real-time, feeding them to the frontend via polling endpoints.

---

## 14. Low-Level Optimizations & Inductive Biases (2026 Audit)

### 14.1 The "Snake" Activation (Periodic Inductive Bias)
The codec uses a specialized activation function called **Snake1d** (`x + sin^2(x)/a`).
*   **Purpose**: Induces a periodic inductive bias, making the model exceptionally good at capturing waveform frequencies (pitch, harmonics) compared to standard ReLU/SiLU.
*   **Performance**: The implementation supports `torch.jit.script`. Enabling JIT for the Snake activation can yield a **~1.4x speedup** in decoding.

### 14.2 Multi-Band Decomposition (Neural EQ)
The `ScalarModel` component of `HeartCodec` processes audio in multiple frequency bands (`num_bands`).
*   **Implication**: Latent dimensions are likely spatially mapped to frequency sub-bands.
*   **Potential**: This structure hints at the possibility of **Neural EQ**—chemically altering the mix (e.g., bass boost, treble cut) by scaling specific codebook channels before decoding, without standard DSP filters.

### 14.3 Transformer Acceleration
The `HeartMuLa` backbone includes modern Transformer optimizations:
*   **RoPE (Rotary Embeddings)**: improving long-context coherence.
*   **FlashAttention**: The `LlamaAttention` module explicitly checks for `F.scaled_dot_product_attention`. Running on PyTorch 2.0+ (and compatible hardware) will automatically trigger optimized kernels for significantly faster generation.

---
*Maintained by the Milimo Music core engineering team.*
