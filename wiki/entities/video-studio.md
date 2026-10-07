---
title: AI Music Video Studio
type: entity
created: 2026-09-07
updated: 2026-10-07
tags: [video, studio, wan, liveportrait, ltx-video, minimax-h3, diffusers, lipsync, karaoke, ass, director, timeline, cancellation, attention-slicing, lyric-video, typography, videotoolbox]
aliases: [VideoStudio, MusicVideosView, VideoService, VideoOrchestrator, VideoGeneratorRegistry]
---

# AI Music Video Studio

The **AI Music Video Studio** (`backend/app/services/video/` and frontend `MusicVideosView.tsx`) is Milimo Music's production-grade AI music video generation, neural singing avatar animation, and visual performance engine. It turns generated music tracks into broadcast-grade music videos with beat-matched scene cuts, neural facial lip-syncing driven by isolated vocal stems, pre-rendered scene keyframes, synchronized karaoke subtitles, cinematic video diffusion, and a non-destructive multi-track timeline editor.

## 1. Video Generator Registry & Engine Decoupling

Generation requests are resolved dynamically through the authoritative **Video Generator Registry** (`generator_registry.py`), eliminating silent fallthroughs and ensuring user-selected models run with exact architectural parameters:

| Model ID | Model Family | Max Duration | FPS | Lattice Requirement | Pipeline / Provider | Description |
|---|---|---|---|---|---|---|
| `wan_14b` | Alibaba Wan 2.1 14B | **5.0s** | 16 | $(F-1) \% 4 == 0$ | `WanPipeline` / `WanImageToVideoPipeline` | SOTA 14B DiT with 3D spatio-temporal attention, keyframe I2V and T2V |
| `wan_1.3b` | Alibaba Wan 2.1 1.3B | **5.0s** | 16 | $(F-1) \% 4 == 0$ | `WanPipeline` (T2V) | Lightweight 1.3B DiT strictly isolated from heavy 14B I2V pipelines |
| `ltx_video` | Lightricks LTX-Video 0.9B | **10.0s** | 24 | Modulo 32 size | `LTXPipeline` | Real-time high-efficiency DiT capable of up to 10s continuous generation |
| `hailuo_h3` | MiniMax Hailuo H3 33B | **15.0s** | 24 | $49 + 48k$ frames | Local MLX 8-bit / Cloud MiniMax API | SOTA 33B Omni-modal DiT with Context-IR prompt formatting |
| `cloud_fal` | Fal.ai Cloud GPU | **5.0s - 15.0s** | Adaptive | Model-specific | Serverless REST API | Offloaded Wan 2.1 / LivePortrait generation without local GPU pressure |
| `cloud_replicate`| Replicate Cloud GPU | **5.0s - 15.0s** | Adaptive | Model-specific | Replicate API Runner | Offloaded Wan 2.1 / LivePortrait execution via Replicate API token |

- **Director Mode v2 Musical Pacing**: Integrated with [Director Mode v2](../concepts/director-mode-v2.md) featuring hierarchical scored accent snapping (beats $+0.5$, downbeats $+1.8$, lyric boundaries $+2.5$), cut speed bias ($-2$ to $+2$), and model-native discrete frame increments ($F_{\text{min}} + k \cdot F_{\text{step}}$) with sub-second output trimming.
- **Keyframe Pre-Rendering**: Users can pre-render visual keyframe stills (`POST /videos/keyframes/{job_id}`) across all planned scenes to inspect and approve directorial composition before triggering full video diffusion. Cinematic plates are diffused people-free and cached under a prompt+style fingerprint, so the approved still is the exact frame Wan seeds from later (see §6).

## 2. Apple Silicon Metal Memory Protection & Attention Slicing

Dense un-fused self-attention in diffusion transformers scales quadratically with token length:
$$S = \left(\frac{W}{16}\right) \times \left(\frac{H}{16}\right) \times \left(\frac{F - 1}{4} + 1\right)$$

At $1280 \times 720$ with 65 frames, token length reaches $S = 61,200$, requiring a $(1, 40, 61200, 61200)$ tensor in `float32` that consumes **558.11 GB** of VRAM, triggering fatal Metal allocation crashes (`Invalid buffer size: 558.11 GB`).

To guarantee zero-crash execution on Apple Silicon unified memory:
1. **Attention Slicing**: Configured on pipeline initialization (`pipe.enable_attention_slicing(slice_size="auto")`), breaking the attention tensor into discrete head slices.
2. **VAE Tiling & Slicing**: Enables spatial tiled encoding/decoding (`pipe.vae.enable_tiling()`, `pipe.vae.enable_slicing()`) to avoid multi-gigabyte latent activation peaks.
3. **Adaptive Dimension & Frame Clamping (MPS)**: Clamps MPS render dimensions to $\le 832 \times 480$ (or $480 \times 832$ in 9:16) and frame counts to $\le 33$ frames ($S \le 14,040$ tokens), keeping peak attention memory strictly under 1 GB per slice while preserving the Wan lattice rule $(F-1) \% 4 == 0$.
4. **Upstream Diffusers Protection**: Polyfilled missing `ftfy` references to prevent tokenizer assertion failures during text cleanup.

## 3. Instant Task Cancellation Architecture (< 200ms)

Stopping generation previously took upwards of 2 minutes because background asyncio tasks were unreferenced and PyTorch diffusers loops executed blocking C++ operations between scene checks.

The video pipeline now guarantees instant cancellation across all execution layers:
1. **Active Task Registry**: `VideoOrchestrator` maintains `_active_render_tasks: Dict[str, asyncio.Task]` with explicit `register_render_task` and `unregister_render_task` hooks.
2. **Immediate Task Halting**: Calling `POST /videos/tasks/{task_id}/cancel` invokes `task.cancel()`, sets `cancel_event.set()`, instantly updates state to `cancelled`, and flushes VRAM/Metal cache.
3. **Diffusion Step-End Callbacks**: Injects `callback_on_step_end` into `WanPipeline`, `WanImageToVideoPipeline`, and `LTXPipeline`. If cancellation is flagged, the callback immediately raises `asyncio.CancelledError`, halting active diffusion within a single step ($<200$ms).
4. **Subprocess Termination**: Wraps all procedural and lip-sync FFmpeg subprocesses with `proc.kill()` on `CancelledError`, preventing orphaned background processes.
5. **Clean Exception Propagation**: `CancelledError` is caught without triggering procedural fallback cascades or registering false task errors.

## 4. Neural Singing Avatar & Lip-Syncing (LivePortrait)

To eliminate unnatural mouth twitching and deliver broadcast-quality vocal performances:
1. **Stem Isolation**: Routes only the isolated vocal track (`vocals.wav` / `vocals.mp3` from Demucs) to the lip-sync engine. Heavy kicks and 808 bass cannot distort lip movements.
2. **Performer Role Ownership**: Conforms to [Director Mode v2](../concepts/director-mode-v2.md) performer rules: during instrumental solos or drum cutaways, the performer's mouth is strictly kept closed (`mouth_movement: closed`), reserving lip-sync solely for the assigned active singer.
3. **LivePortrait Neural Avatar**:
   - Uses implicit keypoint representations and landmark deformation driven by audio pitch and amplitude.
   - Synthesizes organic eye blinks, micro-expressions, head nods, and realistic phonetic viseme transitions.
   - Executed under [Global Hardware Coordinator](hardware-coordinator.md) device locks to prevent VRAM exhaustion with audio pipelines.
   - Supports local Apple Silicon PyTorch MPS execution as well as cloud GPU offloading.
4. **Smooth Viseme Mesh Fallback**:
   - For low-resource environments without neural weights, a bilinear jaw mesh warp engine smoothly translates the mouth cavity and lips based on vocal power envelopes, avoiding static OpenCV overlays.
5. **Stem Audio-Reactive Modulation**:
   - Powered by [Stem Audio-Reactive Video](../concepts/stem-audio-reactive-video.md) (`stem_audio_reactive.py`), extracting clean vocal envelopes for lip-sync and percussive downbeat transients from drums/bass to drive camera zooms, pulses, and shakes.

## 5. Local-First MiniMax H3 33B Execution

The platform supports local MiniMax Hailuo H3 weights (`pipenetwork__MiniMax-H3-MLX-8bit`, 35.3 GB):
- **Local Animatic Previews**: Runs at 24 fps with exact 49+48k frame lattice alignment.
- **Hardware Realism**: Because un-quantized 33B dense attention on Apple Silicon unified memory requires ~1.2 hours per 5s clip without dedicated auxiliary pipelines, the studio clearly notifies the user of hardware demands and recommends lightweight models (`wan_1.3b` or `ltx_video`) for real-time local iterations.

## 6. Keyframe Stills Diffusion & Memory Lifecycle

Pre-rendering visual scene keyframes (`POST /videos/keyframes/{job_id}`) provides a production-grade inspection stage before video diffusion:
- **Adaptive FLUX.2 Diffusion**: Automatically recognizes whether the active image generator is a flow-matching Base model (`black-forest-labs/FLUX.2-klein-base-9B`) or a distilled Turbo/Schnell variant.
  - **Base Models**: Evaluates the full flow ODE trajectory with **24 inference steps** and **guidance scale = 3.5**, rendering crisp, photorealistic cinematic lighting, texture, and character details without blur or waxy artifacts.
  - **Distilled / Turbo Models**: Fast 4-step sampling with guidance scale = 1.0.
- **Eager Local LLM Eviction**: The visual director treatment (`video_director.py`) uses local LLMs (e.g. `Qwen3.6-35B-A3B-UD-MLX-4bit` on oMLX). To prevent OOM errors when switching from LLM prompt compiling to heavy 9B image diffusion, the backend executes an immediate HTTP unload (`POST /v1/models/{id}/unload` or `keep_alive: 0` for Ollama), dropping unified memory residency from 22.7 GB to 0 bytes before diffusion weights load.
- **Batch Diffusion Model Reuse**: During scene stills batch generation across multiple clips, models remain loaded in memory (`auto_unload=False`) to avoid repeated 15-second initialization latency per clip. Once the entire batch is rendered, a single `finally:` block unloads the weights and flushes MLX/Metal memory cache.
- **Batched Scene Plate Pre-Build** (`VideoOrchestrator._prebuild_scene_plates` → `ImageService.pregenerate_scene_backgrounds`): before the clip loop, `render_advanced_music_video` renders every missing cinematic plate inside one `GlobalHardwareCoordinator.scoped_device(..., modality="image_gen")` block — the image model loads once, renders N stills, unloads once, and only then does Wan acquire the accelerator. Cloud providers skip the pass. Progress surfaces as `Pre-building scene backgrounds... i/N (diffusing | reusing cached plate | plate ready)` mapped to 12→20 %.
  - **Prompt + Style Fingerprint Cache**: each plate is cached at `video_cache/scene_stills/scene_pregen_{job}_{scene}.png` with a `.prompt` sidecar holding `sha256(style + "\n" + prompt)[:16]`. Reuse requires a matching sidecar *and* pixels that fit the requested frame, so a rewritten director prompt, a changed visual style, or a square/wrong-size leftover triggers a re-diffusion instead of silently reusing a stale plate.
  - **Approved Retakes Are Final**: `POST /videos/retake-clip/{job_id}/{clip_index}` stamps its sidecar with an `approved:` prefix, so a frame the user hand-picked is never re-diffused by the pre-build even when the storyboard prompt differs from the retake prompt; the aspect/size guards still apply (`ImageService.stamp_scene_plate(..., approved=True)`).
  - **People-Free Plates**: pre-build stills append `SCENE_PREGEN_PROMPT_SUFFIX` ("no people, no human face…") because Wan morphs faces baked into an i2v seed into the worst artifact in the video; album covers keep `COVER_PROMPT_SUFFIX`, and storyboard **vocal** stills keep the performer visible since lip-sync — not Wan — drives those shots. A staged keyframe without a matching sidecar is never allowed to seed Wan.
  - **Storyboard Reuse**: `generate_scene_keyframes` routes cinematic scenes through the same API (`auto_unload=False`, `index_start=clip_index`) and stages plate + sidecar to `keyframes/keyframe_{job}_{scene}.png`, so the still the user approves is the exact frame Wan seeds from and render time pays no second diffusion. Covered by `backend/tests/test_scene_plates.py`.
- **Progressive Frontend Polling & Auto-Hydration**: In `MusicVideosView.tsx`, the timeline tracks are eagerly populated when still generation begins, and a 3-second progressive polling loop queries `GET /videos/keyframes/{job_id}`, progressively displaying each scene still on the timeline as soon as it is written to disk.

## 7. Fast-Path Lyric Music Video Studio (< 45s Hardware Encoded)

For artists desiring instant, broadcast-ready lyric music videos without lengthy multi-scene diffusion passes, the platform incorporates a dedicated **Fast-Path Lyric Video Engine** (`POST /videos/render-lyric/{job_id}`):
- **Hardware Acceleration Pipeline**: Employs macOS Apple Silicon `h264_videotoolbox` (or Linux `libx264` preset `fast`) to transcode full 3-minute tracks in under 45 seconds at high bitrate ($4.5$–$6.0$ Mbps) with AAC 320 kbps stereo audio.
- **Visual Background Modes & Dynamic Drift**:
  - `cover_art`: Smooth cinematic Ken Burns camera drift (`zoompan=z='min(zoom+0.0003,1.15)':d=...`) applied to high-resolution album cover artwork or custom stills.
  - `spectrum`: Procedural audio-reactive frequency spectrum visualizer overlaid onto dark gradient canvases.
  - `procedural`: Multi-scene mood platter animatics with soft color oscillations.
- **Word-Level Centisecond Karaoke ASS Scripting (`subtitle_styles.py`)**:
  - Automatically compiles word phoneme timestamps from TorchAudio MMS forced alignment into SSA/ASS scripts with centisecond `{\kf<cs>}` karaoke tags.
  - 6 Production Typography Presets:
    - `neon`: Cyberpunk Neon (Cyan fill `&H00FFFF&`, Magenta outline `&HFF00FF&`, heavy glow).
    - `spotify`: Spotify Canvas (Crisp white fill `&HFFFFFF&`, modern bold grotesque, translucent dark pill backing).
    - `kinetic_pop`: Kinetic Pop (Vibrant golden amber fill `&H00D7FF&`, dark shadow, bouncy rhythm).
    - `cinematic`: Cinematic Wide (Elegant serif, gold rim, wide letterspacing).
    - `social_vertical`: Social Vertical (9:16 mobile portrait stacked layout with centered dynamic anchor).
    - `retro_vhs`: Retro VHS (Analog CRT phosphor green fill `&H33FF33&`, scanline glow, monospace typewriter styling).
  - Aspect Ratio Safe Zones: Dynamic margin clamping ($V_{\text{margin}}$ and $H_{\text{margin}}$) guaranteeing subtitles never collide with UI controls in 16:9, 9:16 vertical (Reels/TikTok), 1:1, or 21:9.
  - Local Font Bundling: Bundles local font fallback directory with ASS `fontsdir` integration, preventing generic fallback font failures.
- **WYSIWYG 60fps Canvas Player (`LyricCanvasOverlay.tsx`)**:
  - In-browser interactive preview running at 60fps via `requestAnimationFrame` and synced directly to audio playback time.
  - Hardware-accelerated CSS `background: linear-gradient(to right, ...)` fill sweeps for live karaoke word highlights.
  - Interactive scrubbing: clicking any lyric line in the player jumps the playhead to that exact lyric start time.
- **Multitrack NLE Timeline Lyrics Track (`VideoTimelineTrack.tsx`)**:
  - Visual interactive track displaying all timed lyric phrases with proportional duration widths.
  - Micro timestamp nudging buttons (`-0.1s` / `+0.1s`) on each line pill to instantly adjust timing offsets directly to the database.
  - 1-click "Acoustically Sync Lyrics ⚡" button invoking wav2vec2 forced alignment on isolated vocal stems.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Stem Separator](stem-separator.md) | [Karaoke & Lyric Sync](karaoke-lyricsync.md)
- [Global Hardware Coordinator](hardware-coordinator.md) | [Director Mode v2](../concepts/director-mode-v2.md) | [Multitrack Timeline Editor](multitrack-editor.md)
- [Non-Destructive Multitrack Timeline](../concepts/non-destructive-multitrack-timeline.md) | [Hardware Auto-Tune](../concepts/hardware-autotune-memory-profiles.md)
- [LLM Service & Providers](llm-service.md) | [Video Troubleshooting Handoff](../concepts/video-generation-troubleshooting-handoff.md)

