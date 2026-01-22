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

### 3.2 Tag Weighting
During training, tag categories are sampled with different probabilities. Prioritize these categories in your prompts for better control:

| Category | Probability | Examples |
| :--- | :--- | :--- |
| **Genre** | 0.95 | Pop, Rock, Hip Hop |
| **Timbre** | 0.50 | Female Vocals, Raspy |
| **Gender** | 0.375 | Male, Female |
| **Mood** | 0.325 | Sad, Energetic, Uplifting |
| **Instrumentation**| 0.25 | Guitar, Piano, Synth |
| **Scene/Topic** | 0.2/0.1 | Workout, Love, Party |

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
*Maintained by the Milimo Music core engineering team.*
