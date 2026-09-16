---
title: MuLaCover
type: entity
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md]
tags: [mulacover, cover, remix, symbolic, midi, provider, model]
aliases: [MuLaCover-3B, HeartMuLa-Cover, Remix Engine]
---

# MuLaCover

**MuLaCover** (`HeartMuLa/MuLaCover`) is an open-source, controllable **AI music cover and remixing engine** integrated into Milimo Music. It conditions an autoregressive music language model on symbolic musical representations (melody, chords, drums) and style prompts to synthesize high-fidelity stereo remixes and cover songs.

MuLaCover is registered as a first-class `GenerationProvider` in [Generation Provider Abstraction](generation-provider.md) with ID `mulacover`.

---

## 1. Capabilities & Specifications

From `MuLaCoverProvider.get_capabilities()`:
- **provider_id**: `mulacover` — version `MuLaCover-3B`.
- **max_duration_sec**: 240 (4 min).
- **default_sample_rate**: 48,000 Hz (48 kHz high-fidelity stereo).
- **supports_structured_caption**: yes (`topic:[...]; genre:[...]; instrument:[...]; mood:[...]`).
- **supports_section_tags**: yes (`[Intro]`, `[Verse]`, `[Chorus]`, `[Outro]`).
- **recommended_hardware**: Apple Silicon (MPS `float16`), NVIDIA CUDA (`bfloat16`), or CPU fallback (`float32`).
- **license_class**: CC BY-NC 4.0.

---

## 2. Checkpoint Architecture & Bundle Downloader

Unlike single-repository models, MuLaCover relies on a unified ensemble of 4 distinct upstream components organized under `models/audio/HeartMuLa__MuLaCover/`:

1. **`MuLaCover/`** (`HeartMuLa/MuLaCover`):
   - 3B Autoregressive Transformer LM weights (5 safetensors shards, ~6.2 GB).
   - Tokenizer and model generation configuration.
2. **`Qwen3-Embedding-0.6B/`** (`Qwen/Qwen3-Embedding-0.6B`):
   - Dense text embedding model (~1.2 GB) encoding structured style descriptions and section lyrics.
3. **`SymbolicTranscriptor/`**:
   - `yourmt3/last.ckpt` (536 MB): YourMT3 multi-instrument symbolic transcription checkpoint from Hugging Face Space `mimbres/YourMT3`.
   - `chord/fold_0.best.sdict` to `fold_4.best.sdict`: 5-fold ChordNet harmonic ensemble checkpoints.
4. **`HeartCodec-oss/`** (`HeartMuLa/HeartCodec-oss-20260123`):
   - High-fidelity neural audio codec (8-codebook RVQ) symlinked from `heartlib/ckpt/HeartCodec-oss` or downloaded automatically.

### Composite Bundle Downloader (`app.services.mulacover.bundle_downloader`)
- Handles sequential download of all 4 sub-models with resume support, disk space checks, and cancellation.
- Exposes `is_mulacover_installed()` to inspect the presence of all subdirectories and weight files.
- Surface live progress (percentage and active file) to the UI.

---

## 3. Dual Symbolic Transcription Hub (`SymbolicHub`)

MuLaCover accepts both reference audio files and direct MIDI lead sheets. When provided with audio, Milimo Music offers dual neural transcription:

- **Milimo Neural Mode (SOTA)**:
  - Uses BS-Roformer to split the source audio into isolated vocal, accompaniment, and drum stems.
  - Runs [MuScriptor](muscriptor.md) note-tracking on the vocal stem to yield clean melody events.
  - Performs neural pitch and chord estimation on accompaniment stems.
  - Runs [Drum Tracker](drum-tracker.md) (`drum_tracker.py`) with spectral flux onset envelope detection and sub-band filtering (Kick Note 36, Snare Note 38, Hi-Hat Note 42) into `SymbolicCondition.drums`.
- **Upstream Classic Mode**:
  - Direct YourMT3 multi-instrument transcription + 5-fold ChordNet ensemble inference directly on audio.

### Quantization & MIDI Interchange
- Quantizes note onsets and durations to discrete 16th-note grids at the detected BPM.
- Generates standard `.mid` files:
  - `melody.mid`: Isolated melodic lead line.
  - `chord.mid`: Quantized harmonic chord progression.
  - `drum.mid`: Percussion rhythm pattern from [Drum Tracker](drum-tracker.md).
  - `leadsheet_summary_midi`: Combined lead sheet for DAW and PianoRoll import.

---

## 4. Workflows & User Interface Integration

1. **Cover & Remix Studio (`CoverStudioModal.tsx`)**:
   - Dedicated modal accessible from the left navigation bar, Command Palette (`⌘K`), and composer header, unified under [Modal Store Architecture](../concepts/modal-store-architecture.md).
   - Dual input modes: **Reference Audio Mode** (with real-time audio player, BPM detection, and lead sheet extraction) and **Symbolic MIDI Lead Sheet Mode**.
   - First-run onboarding card with one-click bundle downloading and real-time progress bar when checkpoints are uninstalled.
   - MIDI lead sheet download chips with direct **Edit in PianoRoll** navigation.
2. **Track Studio (`TrackDetailView.tsx`)**:
   - **MuLaCover Remix** toolbar button to branch any track into a new cover.
   - **Cover Song** badge on remix versions.
   - **MuLaCover Symbolic Lead Sheet Tracks** section in the Score & MIDI tab.
3. **Songs View (`SongsView.tsx`)**:
   - **Remix** quick-action buttons on every track in both Table and Grid views.
   - **Cover** badges indicating remix lineage.
4. **Composer Sidebar (`ComposerSidebar.tsx`)**:
   - MuLaCover conditioning notice and shortcut to open the Cover Studio when MuLaCover is selected.

---

## 5. Endpoints & Database Schema

- `POST /generate/cover`: Enqueues cover song jobs with upfront validation of checkpoints and symbolic inputs.
- `POST /transcribe/lead-sheet`: Extracts multi-track MIDI lead sheets from audio. Features robust audio path resolution (`_resolve_audio_file`), stripping dev server URLs (`http://localhost:5173/audio/...`) and resolving static mount aliases (`/audio/...` -> `generated_audio/...`) to prevent 404 extraction failures.
- `GET /jobs/{job_id}/symbolic`: Fetches extracted MIDI paths for completed jobs.
- `GET /models/check/mulacover`: Validates checkpoint integrity.
- **Database Schema (`Job` model)**:
  - `is_cover: bool`
  - `cover_mode: str` (`audio_reference` or `symbolic_midi`)
  - `ref_audio_path: str`
  - `melody_midi_path: str`
  - `chord_midi_path: str`
  - `drum_midi_path: str`
  - `bpm: float`

---

## 6. Hardware Memory Management & Apple Silicon MPS Optimization

MuLaCover's 3B autoregressive parameter scale and multi-component pipeline require strict unified memory isolation:

1. **Autoregressive Activation Freeing (`@torch.inference_mode()`)**:
   - Autoregressive token generation runs strictly under `@torch.inference_mode()`. Without explicit inference mode, PyTorch's autograd engine builds dynamic computational graphs across all 28 backbone layers, cross-attention projections, and codebook heads, leaking **~10.85 GB of activation tensors per step** (triggering MPS OOM at step 14 / 163.2 GB).
   - Under `torch.inference_mode()`, activation memory is 100% flat (exact 0.0 MB leaked per step, constant 13.9 GB footprint).
2. **Torchtune Causal Mask Rank Preservation**:
   - `_index_causal_mask(self.backbone_causal_mask, input_pos)` ensures indexed slice shapes preserve the 3D tensor contract expected by Torchtune's `TransformerDecoder` (`[batch, s_q, s_kv]`), preventing Metal Performance Shaders graph dynamic broadcasting failures.
3. **HeartCodec Vocoder Metal Dimension Guardrail**:
   - In `HeartCodec.detokenize()`, the final `scalar_model` 1D convolution decode stage (operating on full-length waveform audio > 65,536 samples) routes through CPU execution on MPS devices. This bypasses Apple Silicon Metal's 16-bit 65,536 spatial sample grid dispatch limit (`NotImplementedError: Output channels > 65536 not supported at the MPS device`), completing in 280 ms with zero VRAM pressure.
4. **Inter-Stage Cache Flushing**:
   - Pre-synthesis and post-synthesis pipeline boundaries invoke `GlobalHardwareCoordinator.flush_memory()` to evict BS-Roformer, MuScriptor, and vocoder caches before subsequent heavy operations.

---

## Related Pages

- [Generation Provider](generation-provider.md)
- [Drum Tracker](drum-tracker.md)
- [MuScriptor](muscriptor.md)
- [Global Hardware Coordinator](hardware-coordinator.md)
- [Modal Store Architecture](../concepts/modal-store-architecture.md)

