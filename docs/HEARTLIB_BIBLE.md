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

Heartlib consists of two primary component models and a pipeline.

### 2.1 HeartMuLa (Music Language Model)
The backbone responsible for semantic understanding and token prediction.
*   **Architecture**: Modified **Llama 3.2** (Transformer Decoder).
*   **Backbone**: `torchtune.models.llama3_2`.
*   **Vocabulary**:
    *   **Text**: 128,256 tokens (Llama 3 Tokenizer).
    *   **Audio**: 8,197 tokens (8192 codebook entries + special tokens).
*   **Model Variants**:
    *   `3B`: 28 Layers, 24 Heads, 3072 dim (Most common).
    *   `300M`, `400M`, `7B` variants defined in configuration.
*   **Prediction Head**:
    *   **Dual-Head mechanism**:
        *   `codebook0_head`: Predicts the first (most significant) codebook index.
        *   `audio_head`: Jointly predicts codebooks 1-7 for fine acoustic detail.

### 2.2 HeartCodec (Neural Audio Codec)
The tokenizer/detokenizer responsible for compressing audio into tokens and reconstructing waveforms.
*   **Architecture**: **Flow Matching** Transformer + RVQ (Residual Vector Quantization).
*   **Resolution**: 48kHz Sample Rate.
*   **Quantization**: 8 Codebooks (Quantizers).
*   **Components**:
    *   `FlowMatching`: Diffuser-based latent generation.
    *   `ScalarModel`: Multi-band signal reconstruction.

### 2.3 The Pipeline (`HeartMuLaGenPipeline`)
The glue that orchestrates generation:
1.  **Preprocessing**: Tokenizes Text/Lyrics → Adds Special Tokens (`<tag>`, `[BOS]`, `[EOS]`).
2.  **Generation Loop**: Autoregressively predicts audio tokens frame-by-frame.
3.  **Decoding**: Passes predicted tokens to `HeartCodec` to generate the `.mp3`/`.wav` waveform.

---

## 3. Installation & Setup

### Requirements
*   **Python**: 3.10+ recommended.
*   **PyTorch**: 2.1+ (CUDA 12.1+ for NVIDIA, MPS for Mac Silicon).
*   **FFmpeg**: Required for audio IO.

### Directory Structure
Ensure your checkpoints follow this exact structure for `pipeline.from_pretrained` to work:
```
/path/to/checkpoints/
├── HeartMuLa-oss-3B/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── HeartCodec-oss/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── tokenizer.json       # Llama 3 tokenizer
└── gen_config.json      # Generation config
```

### Environment Variables
**For Mac Users (Apple Silicon/MPS):**
Heartlib uses some Conv1D operations that may not be fully supported on MPS.
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

---

## 4. Production Usage Patterns

This section details how to integrate Heartlib into a production backend (like FastAPI), based on the `MusicService` implementation.

### 4.1 Asynchronous Execution
The pipeline is **synchronous** (blocking). In an async web server, you **must** wrap it in an executor to prevent freezing the event loop.

```python
import asyncio

# Inside an async handler
loop = asyncio.get_running_loop()
output = await loop.run_in_executor(
    None, 
    lambda: pipeline(model_inputs, **params)
)
```

### 4.2 Handling Cancellation
Heartlib supports graceful cancellation using `threading.Event`. The pipeline checks this event every ~10 frames.

```python
import threading

# 1. Create event
abort_event = threading.Event()

# 2. Pass to pipeline
pipeline(..., abort_event=abort_event)

# 3. Trigger cancellation from another thread/request
abort_event.set()
```

### 4.3 Seeding (Reproducibility)
For consistent results, you must seed `torch`, `random`, and `numpy` immediately before generation.

```python
seed = 42

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
```

### 4.4 Managing Memory (OOM)
*   **Unload Codec**: The pipeline automatically moves `HeartCodec` to CPU during generation and back to GPU for decoding.
*   **History Tokens**: When extending tracks, `history_tokens` can grow large. Ensure you only load what is needed.
*   **MPS Garbage Collection**: On Mac, occasionally call `torch.mps.empty_cache()` if facing memory fragmentation.

---

## 5. Advanced Generation Techniques

### 5.1 Infinite Extension (Context Injection)
Heartlib allows continuing a song indefinitely.
1.  **Save Tokens**: After Generation A, save `output["tokens"]` (Tensor `[1, seq_len, 9]`).
2.  **Load Tokens**: Load this tensor for Generation B.
3.  **Inject**: Pass as `history_tokens`.

```python
# Pass the full tensor. The pipeline handles caching and warming up the context.
output = pipeline(..., history_tokens=previous_tokens)
```

### 5.2 Auto-Titling
Heartlib does not generate titles. Use a strictly text-based LLM (like Llama 3 8B or GPT-4) to summarize the lyrics/prompt into a title *before* starting generation.

---

## 6. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **"NotImplementedError: The operator 'aten::conv1d'..."** | MPS (Mac) missing backend support for specific op. | Set `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`. |
| **FastAPI server hangs during generation** | Pipeline running on main thread. | Wrap pipeline call in `await loop.run_in_executor(...)`. |
| **Generation continues after client disconnect** | No cancellation logic connected. | Implement `abort_event` passing and trigger it on disconnect/API call. |

---

## 7. Internal Mechanics & Secrets

### 7.1 The Tokenizer (`tokenizer.json`)
*   **Base**: Standard Llama 3 tokenizer.
*   **Special Audio Tokens**:
    *   `audio_eos_id`: **8193** (The "Stop Generating" signal).
    *   `empty_id`: **0** (Padding/Empty).
    *   `text_bos_id`: **128000** (Beginning of Text).
    *   `text_eos_id`: **128001** (End of Text).
*   **Style Tokens**:
    *   The model recognizes BPE-merged style words.
    *   Example: `Ġinstrumental` (ID **42045**) is a distinct token, validating its use in tags.

### 7.2 The Generation Loop (`music_generation.py`)
*   **Mechanism**: Frame-by-frame autoregressive generation.
*   **Latency**: Approx. 80 frames per second of audio (variable).
*   **Context Injection**:
    *   Text (Lyrics/Tags) is encoded and prepended to the context.
    *   `muq_embed` vectors guide the continuous generation.
*   **Stopping Condition**:
    *   The loop breaks if **ANY** parallel codebook token in the first frame predicts `audio_eos_id` (>= 8193).


### 7.3 Context Extension Logic
*   **History Tokens**: To extend a track, previous audio tokens are fed back into the model to warm up the Key-Value (KV) cache.
*   **Pattern**: `[History Tokens] -> [Model] -> [New Tokens]`.
*   **Implementation Note**: The history must be properly batched and match the `cfg_scale` dimensions (tiled if `cfg_scale > 1.0`).

## 8. API Reference (`HeartMuLaGenPipeline`)

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
