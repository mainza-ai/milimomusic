---
title: AI Music Video Studio — Issue Catalog & Troubleshooting Handoff
type: concept
tags: [video, troubleshooting, handoff, wan, minimax-h3, ltx-video, flux, keyframes, metal, mps, milimovideo]
created: 2026-09-26
updated: 2026-10-07
sources: [sources/maestro-creative-studio.md]
aliases: [VideoTroubleshooting, VideoHandoff, VideoGenerationIssues, MilimoVideoInsights]
---

# AI Music Video Studio — Issue Catalog & Troubleshooting Handoff

> [!IMPORTANT]
> **AI-to-AI Handoff Document**: This page is the authoritative diagnostic catalog, engineering handoff guide, and architectural reference for the AI Music Video Studio in Milimo Music (`backend/app/services/video/` and `frontend/src/components/views/MusicVideosView.tsx`). It synthesizes field diagnostics from local testing on Apple Silicon (M3 Max) alongside proven engineering solutions extracted from the sister project [**Milimo Video**](https://github.com/mainza-ai/milimovideo).

---

## 1. Executive Summary & Diagnostic Matrix

Milimo Music is a **local-first** audio/video production platform. Video generation combines automated scene planning ([Director Mode v2](director-mode-v2.md)), FLUX.2 keyframe stills pre-rendering, and spatio-temporal video diffusion (Wan 2.1, LTX-Video, MiniMax H3).

During local generation testing on Apple Silicon (M3 Max), several bottlenecks and architectural mismatches were discovered. The table below summarizes these issues with standardized tracking labels for incoming engineers and AI assistants.

| Issue ID | Subsystem / Model | Severity | Observable Symptom | Technical Root Cause | Status |
|---|---|---|---|---|---|
| **`[ISSUE-VID-01]`** | Wan 2.1 1.3B | **High** | UI reports "Active & Ready", but generation fails or falls back to animatics | Downloaded weights are raw standalone PyTorch/safetensors, whereas `DiffusersWanGenerator` requires HuggingFace Diffusers directory format (`model_index.json`). | **Resolved**: Converted Diffusers weights downloaded and verified at `models/video/Wan-AI__Wan2.1-T2V-1.3B-Diffusers` |
| **`[ISSUE-VID-02]`** | Keyframe Stills UI | **Medium** | Pre-rendering stills shows spinning button, but no step-by-step progress or clip count in HUD | React `activeTask` state is never updated during keyframe polling in `MusicVideosView.tsx`. | Root Cause Identified; 1-Line State Fix Needed |
| **`[ISSUE-VID-03]`** | Metal Memory / MPS | **Critical** (Mitigated) | Un-fused attention crash: `Invalid buffer size: 558.11 GB` | At $1280 \times 720 \times 65$ frames, un-fused $S = 61,200$ attention tensor scales quadratically ($O(S^2)$). | Fixed via attention slicing, VAE tiling, and MPS dimension clamping |
| **`[ISSUE-VID-04]`** | MiniMax Hailuo H3 | **Medium** | Local generation creates 24 fps animatic preview instead of full diffusion | Local 35.3 GB MLX weights contain only DiT weights without text-encoder/VAE MLX runner; 33B dense attention requires ~1.2 hrs/clip. | Architectural Constraint; Documented & Animatic Fallback Active |
| **`[ISSUE-VID-05]`** | Keyframe Diffusion Engine | **Low** | Still generation takes ~5 min/scene with zero intermediate step updates | `mflux` runs synchronously in `image_service.py` without diffusion step callbacks to the task queue. | Improvement Opportunity; Expose Step Callbacks |
| **`[ISSUE-VID-06]`** | LTX-Video 0.9B | **Low** | Diffusion fails or produces distorted aspect ratio | Requires width/height divisible by 32 and frame lattice $F \equiv 1 \pmod 8$, differing from Wan's modulo 16 / modulo 4 rule. | Addressed in `model_specs.py` & `diffusers_ltx.py` |
| **`[ISSUE-VID-07]`** | Memory Collisions | **High** (Mitigated) | Memory exhaustion during transition from Director planning to Still diffusion | 35B Local LLM stays resident in unified memory (22.7 GB) while FLUX.2 (9B) attempts to allocate. | Fixed via eager LLM unloads in `video_orchestrator.py` |
| **`[ISSUE-VID-08]`** | Audio Reactivity | **Medium** | "Audio-Reactive Video" produces static motion with no pulse or beat-synced camera work | `extract_stem_reactive_modulation` extracts envelopes but variables are discarded; generators and FFmpeg filters ignore them. | Uncovered in Audit; Pipeline Wiring Needed |
| **`[ISSUE-VID-09]`** | FFmpeg Stitching | **Medium** | Transitions (Crossfade, Whip Pan) selected in settings are ignored, resulting in hard cuts | `_stitch_video_segments` relies exclusively on FFmpeg `concat` demuxer and ignores `transition_style` configuration. | Uncovered in Audit; `xfade` Filter Graph Needed |


---

## 2. Detailed Issue Catalog

---

### `[ISSUE-VID-01]` Wan 2.1 1.3B Weight Format Incompatibility (Diffusers vs Raw Checkpoint)

- **Target Files**:
  - `backend/app/services/video/generators/diffusers_wan.py` (`_resolve_model_path`, lines 51–85)
  - `backend/app/services/video/generator_registry.py` (`_resolve_wan_13b`, lines 55–78)
  - `backend/app/services/model_manager.py` (model registration tree)
  - Local Directory: `models/video/Wan-AI__Wan2.1-T2V-1.3B/`
- **Severity**: High (Blocks native 1.3B diffusion)
- **Observable Symptoms**:
  1. Frontend Model Manager displays `Wan 2.1 1.3B Fast` as **Active & Ready** because the folder exists on disk and exceeds 10 GB.
  2. When the user initiates video generation with `wan_1.3b` selected, the server logs:
     ```text
     DiffusersWanGenerator: Wan 2.1 T2V (1.3b) local weights not installed; skipping background download and falling back to procedural animatic.
     ```
  3. The studio produces a procedural storyboard animatic instead of a neural video clip.
- **Deep Technical Root Cause**:
  - The repository downloaded to `models/video/Wan-AI__Wan2.1-T2V-1.3B` is the **original upstream Wan-AI checkpoint repository**:
    ```text
    models/video/Wan-AI__Wan2.1-T2V-1.3B/
    ├── Wan2.1_VAE.pth                       (507 MB)
    ├── config.json                          (249 bytes)
    ├── diffusion_pytorch_model.safetensors  (5.67 GB)
    └── models_t5_umt5-xxl-enc-bf16.pth      (11.36 GB)
    ```
  - However, `DiffusersWanGenerator` relies on HuggingFace `diffusers.WanPipeline`, which expects the **converted Diffusers repository structure** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`), containing:
    ```text
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/
    ├── transformer/
    └── vae/
    ```
  - In `diffusers_wan.py`:
    ```python
    for c in local_cand:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "model_index.json")):
            return os.path.abspath(c)
    ```
    Because `model_index.json` is missing in `models/video/Wan-AI__Wan2.1-T2V-1.3B`, `_resolve_model_path` returns `None`.
- **Status & Resolution**:
  - **Resolved via Option A**: Downloaded the complete official Diffusers snapshot from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (~27 GB across all 31 files) into `models/video/Wan-AI__Wan2.1-T2V-1.3B-Diffusers`.
  - Verified: `DiffusersWanGenerator._resolve_model_path("t2v")` correctly resolves to the directory and `is_available` returns `True`. Native 1.3B diffusion pipeline is now active and ready.

---

### `[ISSUE-VID-02]` Keyframe Still Generation Missing Step-by-Step Progress & HUD Status Blindspot

- **Target Files**:
  - `frontend/src/components/views/MusicVideosView.tsx` (`handleGenerateKeyframes`, lines 707–728)
  - `frontend/src/components/video/VideoTopBar.tsx` (progress HUD rendering)
  - `frontend/src/components/video/VideoCanvasPlayer.tsx` (`activeTask` overlay)
  - `backend/app/services/video/video_orchestrator.py` (`generate_scene_keyframes`, lines 342–418)
- **Severity**: Medium (User experience confusion; user perceives generation as frozen)
- **Observable Symptoms**:
  1. The user clicks **Pre-Render Stills** (or **Regenerate Stills**).
  2. The button switches to `Keyframes…` with a spinning loader, but the central player canvas and top bar do not display any progress percentage or status message (e.g. `Diffusing Scene Still 2/8 (🎤 Vocal)`).
  3. Server logs reveal the task is actively progressing:
     ```text
     Task kf_e03e5defff7f: Diffusing Scene Still 1/8 (🎤 Vocal) - progress: 10%
     Task kf_e03e5defff7f: Diffusing Scene Still 2/8 (🎥 Cinematic) - progress: 20%
     ```
  4. The UI remains completely static for 20–40 minutes until all stills finish.
- **Deep Technical Root Cause**:
  - In `MusicVideosView.tsx`, the polling loop in `handleGenerateKeyframes` polls `videoApi.getVideoTaskStatus(taskId)`:
    ```typescript
    if (taskId) {
        while (true) {
            await new Promise(resolve => setTimeout(resolve, 2000));
            try {
                const taskStatus: any = await videoApi.getVideoTaskStatus(taskId);
                if (taskStatus.status === 'completed') { ... }
                // BUG: taskStatus is never assigned to React state!
            }
        }
    }
    ```
  - `activeTask` is the React state consumed by `VideoCanvasPlayer` and `VideoTopBar` to render the progress bar and status text. Because `setActiveTask(taskStatus)` was never called inside this polling loop, `activeTask` remained `null`.
- **Handoff Remediation Plan for Incoming AI / Engineer**:
  - In `MusicVideosView.tsx` inside the `while (true)` polling block of `handleGenerateKeyframes`, add:
    ```typescript
    const taskStatus: any = await videoApi.getVideoTaskStatus(taskId);
    if (taskStatus) {
        setActiveTask(taskStatus);
    }
    ```
  - In the `finally` block, reset `setActiveTask(null)`.

---

### `[ISSUE-VID-03]` Apple Silicon Metal Buffer Limit (558.11 GB) & Quadratic Attention Scaling

- **Target Files**:
  - `backend/app/services/video/generators/diffusers_wan.py` (lines 115–135, 150–165)
  - `backend/app/services/video/generators/diffusers_ltx.py`
  - `backend/app/core/hardware_lock.py`
- **Severity**: Critical (Previously caused fatal process crash; currently mitigated)
- **Observable Symptoms**:
  - Running Wan 2.1 video diffusion on Apple Silicon MPS previously crashed the backend with:
    ```text
    RuntimeError: MPS backend out of memory (Total allocated: ... Invalid buffer size: 558.11 GB)
    ```
- **Deep Technical Root Cause**:
  - Spatial-temporal patchification in Wan 2.1 DiT divides dimensions by spatial patch size (16) and temporal stride (4):
    $$S = \left(\frac{W}{16}\right) \times \left(\frac{H}{16}\right) \times \left(\frac{F - 1}{4} + 1\right)$$
  - For a standard 720p 4-second clip at 16 fps ($1280 \times 720$, 65 frames):
    $$S = 80 \times 45 \times 17 = 61,200 \text{ tokens}$$
  - Un-fused self-attention computes the attention score matrix $Q K^T$ of shape $(B, H, S, S)$. With 40 attention heads in `float32`:
    $$\text{Memory} = \frac{1 \times 40 \times 61,200 \times 61,200 \times 4}{1024^3} = \mathbf{558.11\text{ GB}}$$
  - Apple Silicon Metal Unified Memory limits a single buffer allocation to a fraction of physical RAM, terminating execution immediately.
- **Implemented Mitigation & Operational Bounds**:
  1. **Attention Slicing**: `pipe.enable_attention_slicing(slice_size="auto")` decomposes the head computation along discrete slices, keeping peak attention memory $<1\text{ GB}$.
  2. **VAE Tiling & Slicing**: `pipe.vae.enable_tiling()` processes latent encoding/decoding in small overlapping spatial tiles.
  3. **MPS Dimension Clamping**: On MPS, resolution is clamped to $\le 832 \times 480$ (16:9) or $\le 480 \times 832$ (9:16) and frames are bounded to $\le 33$ frames ($S \le 14,040$ tokens), guaranteeing that un-sliced intermediate peaks remain $<8\text{ GB}$.

---

### `[ISSUE-VID-04]` MiniMax Hailuo H3 33B Local MLX Execution Bottleneck & Missing Auxiliary Weights

- **Target Files**:
  - `backend/app/services/video/generators/minimax_h3.py`
  - `models/video/pipenetwork__MiniMax-H3-MLX-8bit/`
- **Severity**: Medium (Architectural Limitation)
- **Observable Symptoms**:
  - When selecting `hailuo_h3` for local rendering, the system generates high-definition 24 fps animatic clips with the exact frame lattice ($49 + 48k$) rather than executing full 33B video diffusion.
- **Deep Technical Root Cause**:
  - The local repository `pipenetwork__MiniMax-H3-MLX-8bit` (35.3 GB) provides 8-bit quantized weights for the main DiT transformer block.
  - However, full end-to-end local diffusion requires:
    1. **Text-Encoder**: Auxiliary Qwen3-VL / T5 multimodal text encoder.
    2. **VAE Latent Autoencoder**: Spatial-temporal VAE decoder to reconstruct pixels from latent video representations.
    3. **Compute Scale**: Even in 8-bit quantization on an M3 Max, un-pruned 33B dense attention requires ~1.2 hours per 5-second clip ($49$ frames).
- **Handoff Guidance**:
  - For real-time local video iteration, users and agents should prioritize **`wan_1.3b`** or **`ltx_video`** (sub-2B parameters).
  - MiniMax H3 should be routed through cloud API backends (`Fal.ai` / `Replicate`) when full 33B diffusion is required, reserving the local pipeline for rapid 24 fps director animatics.

---

### `[ISSUE-VID-05]` Keyframe Generation Diffusion Step-Level Progress & Blocking Engine Execution

- **Target Files**:
  - `backend/app/services/image_service.py` (`_render_diffusion_image`, `generate_scene_background`)
  - `backend/app/services/video/video_orchestrator.py` (`generate_scene_keyframes`)
- **Severity**: Low (Enhancement)
- **Observable Symptoms**:
  - While generating keyframes for 8 scenes, each scene takes ~4–5 minutes using `FLUX.2-klein-base-9B` (24 ODE steps).
  - During that 5-minute interval, no sub-step progress (e.g. `Step 12/24`) is emitted to the server task queue or frontend SSE stream.
- **Deep Technical Root Cause**:
  - `image_service.py` executes `mflux` or `diffusers` as a synchronous blocking function call without an injected step callback.

---

### `[ISSUE-VID-06]` LTX-Video 0.9B Resolution & Frame Lattice Divisibility Constraints

- **Target Files**:
  - `backend/app/services/video/model_specs.py` (`LTX_VIDEO_SPEC`)
  - `backend/app/services/video/generators/diffusers_ltx.py`
- **Severity**: Low (Compatibility)
- **Technical Constraints**:
  - **Spatial Divisibility**: LTX-Video requires spatial dimensions divisible by **32** (e.g. $768 \times 512$, $1280 \times 704$). Passing $1280 \times 720$ (divisible by 16 but not 32) triggers a tensor shape assertion in downsampling convolutional blocks.
  - **Temporal Divisibility**: Frame count must satisfy $F \equiv 1 \pmod 8$ (e.g. 9, 17, 25, 33, 41, 49, 97, 121, 161, 241).
  - **Native FPS**: 24 fps (unlike Wan 2.1's 16 fps).
- **Handoff Guidance**:
  - Always resolve dimensions through `model_specs.py` (`validate_dimensions_for_model`), which automatically snaps dimensions to 32-pixel increments.

---

### `[ISSUE-VID-07]` Unified Memory Eviction & Inter-Modal VRAM Collisions (LLM $\rightarrow$ Image $\rightarrow$ Video)

- **Target Files**:
  - `backend/app/services/video/video_orchestrator.py`
  - `backend/app/services/llm_service.py`
  - `backend/app/core/hardware_lock.py` (`GlobalHardwareCoordinator`)
- **Severity**: High (Mitigated)
- **Technical Context**:
  - Scene planning uses local LLMs (e.g. `Qwen3.6-35B-A3B-UD-MLX-4bit` on oMLX), occupying ~22.7 GB of unified memory.
  - Scene stills diffusion loads FLUX.2 Klein (9B), requiring ~12 GB.
  - If the LLM is not explicitly unloaded before diffusion weights load, unified memory exceeds 34 GB, triggering OS memory compression and disk thrashing.
- **Implemented Mitigation**:
  - `video_orchestrator.py` calls `llm_service.unload_local_model()` both immediately before and immediately after director planning.
  - `GlobalHardwareCoordinator.flush_memory()` purges PyTorch MPS and MLX caches between pipeline phases.

---

### `[ISSUE-VID-08]` Stem Audio Reactivity Pipeline Disconnect (Inert Envelopes)

- **Target Files**:
  - `backend/app/services/video/video_orchestrator.py` (lines 771–781)
  - `backend/app/services/video/stem_audio_reactive.py`
- **Severity**: Medium (Unfulfilled feature promise)
- **Observable Symptoms**:
  - Videos exported with "Audio Reactive" settings show static motion with no rhythmic pulse, bounce, or camera zoom synced to beats.
- **Deep Technical Root Cause**:
  - `video_orchestrator.py` calls `extract_stem_reactive_modulation()` and logs the result, but the returned envelopes (`reactivity_data`) are discarded. They are never passed to the video generator parameters nor to the final FFmpeg post-processing filters.
- **Remediation Plan**:
  - Connect `reactivity_data` to FFmpeg `zoompan` or `eq` filters during assembly to modulate camera zoom and exposure with bass/drum energy curves.

---

### `[ISSUE-VID-09]` Scene Transition Configuration Ignored (Always Hard Cuts)

- **Target Files**:
  - `backend/app/services/video/video_orchestrator.py` (`_stitch_video_segments`, lines 783–840)
- **Severity**: Medium (Aesthetic limitation)
- **Observable Symptoms**:
  - Choosing "Dissolve" or "Whip Pan" transition styles in render settings still produces hard cuts between scenes.
- **Deep Technical Root Cause**:
  - `_stitch_video_segments()` reads `transition_style` from config, but relies exclusively on the FFmpeg `concat` demuxer (`-f concat -safe 0 -i clips.txt -c copy`), which only supports hard concatenation.
- **Remediation Plan**:
  - Implement an FFmpeg complex filter graph (`xfade`) when `transition_style != "cut"` to create true beat-synced dissolves, wipes, and crossfades.

---


## 3. Cross-Repository Architectural Transfer: Insights from Milimo Video

An architectural investigation of the sister project [**Milimo Video**](https://github.com/mainza-ai/milimovideo) (`/Users/mck/Desktop/milimovideo`) revealed several battle-tested patterns that directly resolve the current challenges in Milimo Music:

### 3.1 The Slot-Based Mutual Exclusion Model (`MemoryManager`)
- **Reference**: `milimovideo/backend/memory_manager.py`
- **Pattern**: Instead of relying on ad-hoc model unloads scattered across orchestrator methods, Milimo Video defines an explicit slot mutual exclusion contract:
  ```python
  class MemoryManager:
      CONFLICTS = {
          "video": {"image"},   # LTX/Wan and Flux are mutually exclusive
          "image": {"video"},   # Flux and LTX/Wan are mutually exclusive
      }

      def prepare_for(self, slot: str) -> None:
          conflicts = self.CONFLICTS.get(slot, set())
          to_unload = conflicts & self._active_slots
          for conflicting_slot in to_unload:
              self._unload_slot(conflicting_slot)
          self._flush_memory()  # gc.collect() + torch.mps.empty_cache()
  ```
- **Application to Milimo Music**: Milimo Music spans **four** heavy modalities: Audio (HeartMuLa / YuE2 / Demucs), LLM (Qwen / DeepSeek), Image (FLUX.2), and Video (Wan / LTX). Extending `GlobalHardwareCoordinator` with explicit modality slot conflicts guarantees zero memory overlap without manual boilerplate.

### 3.2 Apple Silicon MPS VAE Decode Offload & Precision Fix (The "Black Screen" Cure)
- **Reference**: `milimovideo/backend/models/flux_wrapper.py` (lines 77–100) & `docs/ltx2-bible.md`
- **The Problem**: On Apple Silicon MPS, running VAE decoders in `bfloat16` or `float16` frequently produces completely black frames or triggers `NaN` floating point exceptions during high-resolution latent un-patching.
- **The Proven Fix**:
  1. Force `pipeline.vae.to(dtype=torch.float32)` upon pipeline initialization.
  2. Dynamically offload VAE decode to CPU when executing on MPS:
     ```python
     use_cpu_offload = (self.device == "mps" or str(self.device) == "mps")
     with torch.no_grad():
         if use_cpu_offload:
             self.ae = self.ae.to(device="cpu", dtype=torch.float32)
             z = z.to(device="cpu", dtype=torch.float32)
         dec = self.ae.decode(z).sample
         if use_cpu_offload:
             dec = dec.to(device=self.device, dtype=torch.float32)
             self.ae = self.ae.to(device=self.device, dtype=self.dtype)
     ```
  This guarantees 100% stable image/video decoding without black screens and eliminates MPS kernel allocation spikes.

### 3.3 Thread-Safe Denoising Step Callbacks with Immediate Cancellation
- **Reference**: `milimovideo/backend/tasks/image.py` (lines 136–157)
- **The Pattern**: Bridging the worker thread and the asyncio event loop for real-time progress broadcasting and instant user aborts:
  ```python
  loop = asyncio.get_running_loop()

  def flux_callback(step: int, total: int):
      progress_pct = int((step / total) * 100)
      # 1. Immediate cancellation abort within the current diffusion step
      if active_jobs[job_id].get("cancelled", False):
          raise RuntimeError(f"Job {job_id} cancelled by user.")

      # 2. Thread-safe SSE progress broadcast to frontend HUD
      asyncio.run_coroutine_threadsafe(
          broadcast_progress(job_id, progress_pct, "processing", f"Generating Image ({step}/{total})"),
          loop
      )
  ```
- **Application to Milimo Music**: Integrating this into `image_service.py` completely solves `[ISSUE-VID-05]` and guarantees real-time keyframe feedback.

### 3.4 Visual Conditioning vs. Frame-0 Static Pinning (Preventing Frozen Videos)
- **Reference**: `milimovideo/backend/tasks/video.py` (lines 150–153 & lines 387–391)
- **The Insight**: When an image is supplied to a video pipeline, passing it as frame-0 start conditioning (`input_images = [(path, 0, 1.0)]`) forces the model to anchor the beginning of the video to that exact frame. Unless intense motion prompts are used, the resulting video often appears "frozen" as a static image.
- **The Solution**:
  1. Distinguish between **Frame-0 Conditioning** (strict I2V where the shot must start from the image) and **Reference / IP-Adapter Conditioning** (semantic style/identity guidance).
  2. In prompt enhancement, automatically detect if an input image is a reference sheet or concept portrait. If detected, emit `is_reference_only: true`, extract visual features into the text prompt, and drop frame-0 image pinning so the model performs dynamic Text-to-Video.

### 3.5 Autoregressive Chained Video with "Quantum Alignment"
- **Reference**: `milimovideo/backend/tasks/chained.py` & `docs/ltx2-bible.md`
- **The Insight**: For songs spanning multiple scenes or long continuous shots exceeding the model's context window (e.g. 505 frames for LTX, 81 frames for Wan):
  - Overlap between consecutive chunks must be aligned to the VAE temporal compression lattice ($8\times$ stride for LTX):
    $$\text{latent\_slice\_count} = \left\lceil \frac{\text{overlap\_frames} - 1}{8} \right\rceil + 1$$
    $$\text{frames\_to\_trim} = (\text{latent\_slice\_count} - 1) \times 8 + 1$$
  - The tail latent slice from chunk $N$ is injected as conditioning for chunk $N+1$, and FFmpeg trims the exact decoded overlap. This prevents visual "stutter" or frozen seams between scenes.

### 3.6 Single-Frame Generation Shortcut (`num_frames == 1`)
- **Reference**: `milimovideo/backend/tasks/video.py` (lines 185–187)
- **The Pattern**: If a video generation task is dispatched with `num_frames == 1` (such as a keyframe still or album poster), the orchestrator immediately bypasses heavy video diffusion pipelines and delegates directly to `flux_inpainter.generate_image()`.

### 3.7 Standalone Image Generation Studio & Universal Visual Asset Gallery (Google Flow Paradigm)
- **Reference**: [Standalone Image Generation Studio & Visual Asset Gallery](../entities/image-studio-gallery.md) & `milimovideo/web-app/src/components/Images/ImagesView.tsx`
- **The Architectural Addition**: To permanently solve visual asset fragmentation and eliminate the blindspot of ad-hoc image generation:
  1. **Independent Studio Workspace**: An interactive visual generation canvas modeled after Google Flow where users can generate visual concepts, moodboards, character turnarounds, and album covers independently of audio synthesis.
  2. **Persistent Visual Asset Vault**: An indexed `VisualAsset` database table and filesystem archive preserving every generated image, cover, and video scene keyframe with prompt, seed, style, and aspect ratio metadata.
  3. **Universal Cross-Studio Picker**: Allows users to open the Gallery directly from any song card, Track Detail View, or Album Release page and assign any previously generated visual asset as the official album cover (`Job.cover_image_path`) with 1 click, or route it to the Video Studio as the active singing avatar portrait.

---

## 4. Architecture Pointers & File Reference Table

For any engineer or AI assistant continuing development on the AI Music Video Studio, use this reference table to navigate directly to the relevant code:

| Component | Milimo Music Path | Milimo Video Reference Path | Key Responsibilities |
|---|---|---|---|
| **Video Orchestrator** | `backend/app/services/video/video_orchestrator.py` | `milimovideo/backend/tasks/video.py` | Pipeline lifecycle, task cancellation registry (`_active_render_tasks`), keyframe generation, progress tracking. |
| **Generator Registry** | `backend/app/services/video/generator_registry.py` | `milimovideo/backend/model_engine.py` | Dynamic model dispatch (`wan_14b`, `wan_1.3b`, `ltx_video`, `hailuo_h3`, cloud backbones). |
| **Memory Manager** | `backend/app/core/hardware_lock.py` | `milimovideo/backend/memory_manager.py` | Centralized slot mutual exclusion, VRAM flush, Apple Silicon memory monitoring. |
| **Image Studio & Gallery** | `frontend/src/components/views/ImageStudioView.tsx` | `milimovideo/web-app/src/components/Images/` | Standalone Google Flow-style visual canvas, gallery grid, universal cover picker modal. |
| **Wan Generator** | `backend/app/services/video/generators/diffusers_wan.py` | — | Wan 2.1 T2V/I2V, Metal attention slicing, VAE tiling, cancellation step callbacks. |
| **LTX Generator** | `backend/app/services/video/generators/diffusers_ltx.py` | `milimovideo/LTX-2/packages/ltx-pipelines/` | Real-time DiT generation, 24 fps, modulo 32 spatial snapping, 2-stage upsampling. |
| **MiniMax H3 Generator** | `backend/app/services/video/generators/minimax_h3.py` | — | 33B DiT animatics, 49+48k lattice, cloud fallback routing. |
| **Image Diffusion** | `backend/app/services/image_service.py` | `milimovideo/backend/models/flux_wrapper.py` | FLUX.2 Klein scene background rendering, aspect ratio handling, VAE decode CPU offload. |
| **Frontend Studio View** | `frontend/src/components/views/MusicVideosView.tsx` | `milimovideo/web-app/src/components/` | Main studio workspace, keyframe polling, task cancellation triggers, multitrack timeline. |
| **Top Control Bar** | `frontend/src/components/video/VideoTopBar.tsx` | `milimovideo/web-app/src/components/` | Model selector, aspect ratio picker, render / cancel buttons, status HUD. |
| **Canvas Player** | `frontend/src/components/video/VideoCanvasPlayer.tsx` | `milimovideo/web-app/src/components/` | Video playback canvas, progress overlays, keyframe display. |

---

## 5. Priority Action Items for Incoming Engineer / AI

1. **Fix Keyframe Polling State (`[ISSUE-VID-02]`)**:
   - In `frontend/src/components/views/MusicVideosView.tsx`, update `handleGenerateKeyframes` to call `setActiveTask(taskStatus)` inside the polling loop so that clip progress and current scene status are visible in real-time.
2. **Resolve Wan 1.3B Diffusers Directory (`[ISSUE-VID-01]`)**:
   - Convert or download the `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` repository format into `models/video/Wan-AI__Wan2.1-T2V-1.3B-Diffusers` so that `_resolve_model_path()` succeeds and native 1.3B video diffusion executes locally.
3. **Adopt Milimo Video Step Callback (`[ISSUE-VID-05]`)**:
   - Apply the thread-safe `flux_callback` pattern from `milimovideo/backend/tasks/image.py` inside `image_service.py`, using `asyncio.run_coroutine_threadsafe` to broadcast step progress and check cancellation on every diffusion step.
4. **Implement MPS VAE CPU-Offloading**:
   - In `image_service.py` and `diffusers_wan.py`, add CPU offloading for VAE decoding on MPS to prevent black frames and eliminate intermediate memory spikes.
5. **Formalize Modality Slot Exclusion in Hardware Coordinator**:
   - Extend `GlobalHardwareCoordinator` with the slot conflict rules (`"audio"`, `"llm"`, `"image"`, `"video"`) modeled after `milimovideo/backend/memory_manager.py`.
6. **Deploy Standalone Image Studio & Gallery (`VisualAsset` Architecture)**:
   - Implement `VisualAsset` schema, add gallery endpoints in `image_service.py`, build `ImageStudioView.tsx`, and add the "Choose from Gallery" picker to `TrackDetailView` and `CoverStudioModal`.

---

## Related Documentation

- [AI Music Video Studio Entity](../entities/video-studio.md)
- [Director Mode v2 Concept](director-mode-v2.md)
- [Global Hardware Coordinator Entity](../entities/hardware-coordinator.md)
- [Cross-Modal Model Lifecycle Architecture](cross-modal-model-lifecycle.md)
- [Stem Audio-Reactive Video Concept](stem-audio-reactive-video.md)
