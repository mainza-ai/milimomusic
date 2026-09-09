---
title: Artwork and Static Media Architecture
type: concept
tags: [media, covers, static-files, ranged-static, pipeline, storage, title-overlay, lyrics-driven, scene-backgrounds]
created: 2026-09-09
updated: 2026-09-09
sources: [backend-api, pipeline, image-service]
aliases: [Media Serving Architecture, Cover Artwork Pipeline, Dual Static Mount]
---

# Artwork and Static Media Architecture

## 1. Overview & Context

Milimo Music generates rich multimodal artifacts for every composition:
- Full 48kHz stereo master audio (`.wav`)
- Multi-channel neural separated stems (`/audio/stems/*.wav`)
- Note-level MIDI sequences (`.mid`) and MusicXML score sheets (`.musicxml`)
- High-definition visual album artwork (`/covers/*.png`)

Previously, two architectural splits led to media playback and artwork failures:
1. **Dual Path Mount Split**: When Uvicorn anchored its working directory to `backend/` via `os.chdir(backend/)`, relative paths (`generated_audio/`, `data/covers/`) resolved into `backend/generated_audio/` and `backend/data/covers/`. However, `paths.py` anchored canonical storage to the repository root (`<REPO_ROOT>/generated_audio`, `<REPO_ROOT>/data/covers`). Standard Starlette `StaticFiles` mounts only checked a single directory, returning 404 (Not Found) for files created under the alternative tree.
2. **Spurious Vocal Stem Remix Trigger**: The pipeline automatically triggered a vocal stem re-mix whenever `final_vocal_path` was set to an existing stem, even when no voice conversion was requested (`voice_profile_id` is null). This replaced the original pristine master track with an uncalibrated remix in `backend/generated_audio`.
3. **Missing Automated Cover Generation & On-Demand Action**: Tracks lacked an automated fallback to generate cover art during generation, and users had no mechanism to generate or regenerate artwork post-creation.

---

## 2. Multi-Directory Fallback Static File Serving

To eliminate path divergence across working directories, container mounts, or CLI invocations, Starlette's `StaticFiles` was extended via `RangedStaticFiles`:

```python
class RangedStaticFiles(StaticFiles):
    def __init__(self, *args, directories: Optional[List[os.PathLike]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if directories:
            self.all_directories = [Path(d).resolve() for d in directories]
```

### Static Mount Configurations
FastAPI mounts check canonical paths first, followed by `backend/` fallback directories:

| Route Mount | Primary Directory | Fallback Directory |
|---|---|---|
| `/audio` | `<REPO_ROOT>/generated_audio` | `<REPO_ROOT>/backend/generated_audio` |
| `/generated_audio` | `<REPO_ROOT>/generated_audio` | `<REPO_ROOT>/backend/generated_audio` |
| `/stems` | `<REPO_ROOT>/generated_audio/stems` | `<REPO_ROOT>/backend/generated_audio/stems` |
| `/covers` | `<REPO_ROOT>/data/covers` | `<REPO_ROOT>/backend/data/covers` |

### Backward-Compatible Disk Mirroring
During file creation in `ImageService`, `VoiceService`, `MiniMaxProvider`, and `RealSeparator`, artifacts are saved to canonical storage and mirrored to `backend/` storage:
```python
if canonical_path.exists() and not backend_path.exists():
    shutil.copy2(canonical_path, backend_path)
```
On application startup (`create_db_and_tables`), a bidirectional reconciliation sync guarantees that any pre-existing files in either directory are immediately mirrored.

---

## 3. Album Cover Artwork Lifecycle

### 3.1 Generation Options
In the Composer Sidebar, users are provided an Apple-styled toggle:
- **Auto-generate cover artwork**: Enabled by default (`auto_generate_cover: true` in `GenerationRequest`).
- If enabled and no manual artwork was uploaded, the orchestration pipeline automatically synthesizes a matching studio cover based on track prompt, title, and musical style tags.
- If disabled, the track completes without auto-generating artwork, conserving GPU/ML compute.

### 3.2 On-Demand Manual Generation & Regeneration
Users can generate or regenerate artwork at any time after song creation:
- **API Endpoint**: `POST /jobs/{job_id}/generate-cover`
- **Request Body**: `{"style": "cinematic", "prompt": "...", "model_id": "...", "aspect_ratio": "1:1"}`
- **Universal Lookup**: Supports both 32-hex and 36-hyphenated UUID strings (`get_job_by_id`).
- **Real-Time Notification**: Emits an SSE `job_update` event with updated `cover_image_path`.
- **Frontend Actions**:
  - Hover action button on Artwork container in Track Detail view.
  - Dedicated "Generate Artwork" / "Regenerate Artwork" button chip in track action toolbar.
  - Responsive thumbnail rendering in both Table and Grid modes of the Song Library.

---

## 4. Native MLX Neural Diffusion Engine (`mflux` / `Flux2Klein`)

To avoid generating geometric procedural props when users have installed authentic diffusion weights on Apple Silicon, `ImageService` executes native MLX diffusion pipelines via `mflux`.

### 4.1 Dual Engine Strategy (MLX vs. Diffusers)
- **Apple Silicon Quantized Weights (`mflux`)**:
  When a model is detected as an MLX variant (e.g. `FLUX2-klein-9B-mlx-4bit` or containing `mlx` / `flux2` / `klein`):
  ```python
  from mflux.models.common.config import ModelConfig
  from mflux.models.flux2.variants import Flux2Klein

  is_9b = ("9b" in model_id.lower() or "9b" in local_path.lower())
  cfg = ModelConfig.flux2_klein_9b() if is_9b else ModelConfig.flux2_klein_4b()
  model = Flux2Klein(model_path=local_path, model_config=cfg)
  image = model.generate_image(
      seed=seed,
      prompt=full_prompt,
      num_inference_steps=4,
      width=1024,
      height=1024,
      guidance=1.0,
  )
  ```
- **Standard Diffusers (`torch` + `diffusers`)**:
  When model weights are unquantized PyTorch safetensors (`SDXL Turbo`, official `FLUX.1 [schnell]`), `AutoPipelineForText2Image` executes on MPS or CUDA.
- **Procedural Raster Fallback**:
  Strictly reserved for scenarios where zero model weights exist on disk and no network access is available.

### 4.2 Critical Bug Remediation: Substring Collision in Model Detection
A subtle regression previously triggered an MLX tensor reshape failure:
`"4b" in chosen_model_id.lower()` matched `"4bit"` in `custom_aitrader_flux2_klein_9b_mlx_4bit`. This forced `flux2_klein_4b()` configuration onto a 9B model, causing an attention projection mismatch `[reshape] Cannot reshape array of size 16777216 into shape (1,4096,24,128)`.
The fix explicitly checks for `9b` first (`is_9b = "9b" in ...`) and keeps dedicated pipeline caches (`_loaded_mlx_pipeline` vs `_loaded_diffusers_pipeline`).

---

## 5. Modality-Aware Active Model Persistence

Active model selections made in the **Models & HW** modal are durably stored in `data/models/active_models.json`:

```json
{
  "audio": "minimax_music3_bf16",
  "image": "custom_aitrader_flux2_klein_9b_mlx_4bit",
  "video": "custom_pipenetwork_minimax_h3_mlx_8bit"
}
```

### 5.1 Propagation Contract
1. **`model_manager.set_active_model(model_id)`**: Writes category-keyed model IDs to disk immediately.
2. **`model_manager.get_model_tree()`**: Hydrates `is_active` for each model variant based on `active_models.json`.
3. **`ModelsManagerModal.tsx`**: Renders a prominent `★ Active Engine` badge and dynamically sorts active & installed models to the top of each modality tab.
4. **Composer Sidebar & API**:
   - `selectedImageModel` dropdown defaults to the active image model (`custom_aitrader_flux2_klein_9b_mlx_4bit ✓ Ready`).
   - `api.generateJob()` forwards `cover_image_model_id`.
   - `pipeline.py` executes `image_service.generate_cover(..., model_id=req.cover_image_model_id)`.

---

## 6. Performance & Reliability Standards
1. **HTTP Audio Range Requests**:
   - Master and stem streams support single-byte HTTP 206 Partial Content responses (`Range: bytes=0-1023`).
   - `Accept-Ranges: bytes` header present on all audio responses.
   - Zero media buffering stalls or 404 deadlocks during playback.
2. **Artwork Resolution & Latency**:
   - Fast procedural visual synthesis fallback completes in `<150ms`.
   - Native MLX 4-step FLUX.2 Klein 9B diffusion completes in `~38–42s` on Apple Silicon (M-series).
   - In-memory model caching cuts repeated generation latency by eliminating checkpoint re-parsing.
   - High-definition 1024x1024 lossless PNG outputs with embedded generation metadata.
3. **URL Normalization**:
   - `coverApi.getCoverUrl(path)` seamlessly handles full URLs (`http://...`, `https://...`), absolute root-relative paths (`/covers/...`), and relative filenames (`covers/...`), preventing double-prefixing regressions.

---

## 7. Cover vs. Scene-Background Split (`ImageService`, 2026-09-09)

The video studio's B-roll path used to call `generate_cover()` for per-scene
backgrounds — one function serving two surfaces with conflicting needs:

| Concern | Cover path | Scene path (before → after) |
|---|---|---|
| Prompt suffix | `professional album cover artwork…` | was album-cover language → now `cinematic film still… no text, no watermark` (`SCENE_PROMPT_SUFFIX`) |
| Storage | `COVERS_DIR/ai_cover_*.png` + `/covers/` URL + `data/covers` mirror | was polluting the public cover library per clip → now `video_cache/scene_stills/scene_bg_*.png`, filesystem path only, kept for render debugging |
| Text | Pillow title overlay | structurally overlay-free (no `title` parameter exists on the scene path) |
| Raster fallback | Pillow album framing (border + vinyl grooves) | none — diffusion miss returns `ok: false` and `render_broll_clip` falls back to its procedural ffmpeg branch |
| Resolution | aspect-ratio enum (1:1 default) | exact video dimensions passed through (`1280×720` / `1920×1080`) |
| Style semantics | art descriptor | `SCENE_STYLE_DESCRIPTORS` maps palette keys (`neon-cyberpunk`, …) to cinematic descriptors — the palette key never reaches the image prompt |

Shared core: `_resolve_image_model()` (model resolution + active-model fallback)
and `_render_diffusion_image()` (MLX → Diffusers, no envelope) back both
`generate_cover()` and `generate_scene_background()` (`backend/app/services/image_service.py`,
`render_advanced_music_video` in `backend/app/services/video_service.py`).

---

## 8. Title-Only Text Guarantee (2026-09-09)

Exactly one renderer can put text on cover art — `_overlay_title()` (Pillow
composite, never diffusion) — and it renders at most two lines: the **song
title** plus an optional smaller **artist byline** beneath it (bottom-center,
shared legibility band, drop shadow, Inter Bold bundled under OFL).

Enforcement layers (all must hold; each is defense-in-depth for the others):

1. **LLM visual prompt** (`LLMService.generate_cover_prompt`): system prompt bans
   text/typography, title repetition, and lyric quotation; lyrics truncated to
   ~500 chars; deterministic text-free fallback when the provider is down.
2. **Diffusion prompt hygiene**: `COVER_PROMPT_SUFFIX` carries `no text, no words,
   no letters, no watermark, no typography`; `strip_text_instructions()` removes
   quoted substrings and typography/`by …` clauses from any incoming prompt.
   Raw titles are never fed verbatim to diffusion (pipeline auto-cover and
   job-regenerate route through the LLM prompt; the ArtistsView inline f-strings
   that quoted names into FLUX prompts are deleted).
3. **Overlay contract**: `title=None` → byte-identical clean art (backwards
   compatible); untitled tracks get no overlay rather than prompt-excerpt text
   (`title = job.title or None`, never `job.prompt`); titles sanitized
   (control chars stripped, 120-char cap); overlay failure is non-fatal.
4. **Artist resolution**: explicit `artist` param wins; else server-side default
   from `Job.artist_profile_id → ArtistProfile.name`, falling back through
   `Job.release_id → Release.profile_id` (`_resolve_job_artist_name` in
   `backend/app/main.py`). Standalone tracks resolve to `None` → title-only.
   Convention: track covers = title only; release covers = title + artist byline;
   artist-profile covers = name only, no byline.

Callers: ComposerSidebar (lyrics unless instrumental + title), ProjectsView
(project name), ArtistsView (profile name / release title + profile name),
pipeline auto-cover (LLM prompt + `req.title`), `POST /jobs/{id}/generate-cover`
(LLM prompt + title + resolved artist). Custom uploads (`/upload/image`) bypass
overlay by design.

