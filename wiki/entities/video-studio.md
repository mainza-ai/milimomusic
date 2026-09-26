---
title: AI Music Video Studio
type: entity
created: 2026-09-07
updated: 2026-09-16
tags: [video, studio, wan, liveportrait, ltx-video, diffusers, lipsync, karaoke, ass, director, timeline]
aliases: [VideoStudio, MusicVideosView, VideoService, VideoOrchestrator]
---

# AI Music Video Studio

The **AI Music Video Studio** (`backend/app/services/video/` and frontend `MusicVideosView.tsx`) is Milimo Music's production-grade AI music video generation, neural singing avatar animation, and visual performance engine. It turns generated music tracks into broadcast-grade music videos with beat-matched scene cuts, neural facial lip-syncing driven by isolated vocal stems, pre-rendered scene keyframes, synchronized karaoke subtitles, cinematic video diffusion, and a non-destructive multi-track timeline editor.

## 1. Generative Video Models & Diffusion Engines

Frontier text-to-video and image-to-video diffusion models operate with strict spatio-temporal attention windows:

| Model ID | Model Family | Max Duration | Engine / Pipeline | Description |
|---|---|---|---|---|
| `wan_14b` | Alibaba Wan 2.1 14B | **5.0s** | `WanPipeline` / `WanImageToVideoPipeline` | SOTA 14B DiT with 3D spatio-temporal attention, keyframe I2V and T2V |
| `wan_1.3b` | Alibaba Wan 2.1 1.3B | **5.0s** | `WanPipeline` / `WanImageToVideoPipeline` | Lightweight 1.3B DiT suitable for rapid local rendering |
| `ltx_video` | Lightricks LTX-Video 0.9B | **10.0s** | `LTXPipeline` / `LTXImageToVideoPipeline` | Real-time high-efficiency DiT capable of up to 10s continuous generation |
| `hailuo_h3` | MiniMax Hailuo H3 33B | **15.0s** | Native H3 DiT / MLX | SOTA 33B Omni-modal DiT with Context-IR prompt formatting |
| `cloud_fal` | Fal.ai Cloud GPU | **5.0s - 15.0s** | Fast Serverless REST | Offloaded Wan 2.1 / LivePortrait generation without local GPU pressure |
| `cloud_replicate`| Replicate Cloud GPU | **5.0s - 15.0s** | Managed Model Runner | Offloaded Wan 2.1 / LivePortrait execution via Replicate API token |

- **Director Mode v2 Musical Pacing**: Upgraded from naive fixed bars to [Director Mode v2](../concepts/director-mode-v2.md) featuring hierarchical scored accent snapping (beats $+0.5$, downbeats $+1.8$, lyric boundaries $+2.5$), cut speed bias ($-2$ to $+2$), and model-native discrete frame increments ($F_{\text{min}} + k \cdot F_{\text{step}}$) with sub-second output trimming.
- **Keyframe Pre-Rendering**: Users can pre-render visual keyframe stills (`POST /videos/keyframes/{job_id}`) across all planned scenes to inspect and approve directorial composition before triggering full video diffusion.

## 2. Neural Singing Avatar & Lip-Syncing (LivePortrait)

To eliminate unnatural mouth twitching and deliver broadcast-quality vocal performances:
1. **Stem Isolation**: The director routes only the isolated vocal track (`vocals.wav` / `vocals.mp3` from Demucs) to the lip-sync engine. Heavy kicks and 808 bass cannot distort lip movements.
2. **Performer Role Ownership**: Conforms to [Director Mode v2](../concepts/director-mode-v2.md) performer rules: during instrumental solos or drum cutaways, the performer's mouth is strictly kept closed (`mouth_movement: closed`), reserving lip-sync solely for the assigned active singer.
3. **LivePortrait Neural Avatar**:
   - Uses implicit keypoint representations and landmark deformation driven by audio pitch and amplitude.
   - Synthesizes organic eye blinks, micro-expressions, head nods, and realistic phonetic viseme transitions.
   - Executed under [Global Hardware Coordinator](hardware-coordinator.md) device locks to prevent VRAM exhaustion with audio pipelines.
   - Supports local Apple Silicon PyTorch MPS execution as well as cloud GPU offloading.
4. **Smooth Viseme Mesh Fallback**:
   - For low-resource environments without neural weights, a bilinear jaw mesh warp engine smoothly translates the mouth cavity and lips based on vocal power envelopes, avoiding static OpenCV ellipse overlays.
5. **Stem Audio-Reactive Modulation**:
   - Powered by [Stem Audio-Reactive Video](../concepts/stem-audio-reactive-video.md) (`stem_audio_reactive.py`), extracting clean vocal envelopes for lip-sync and percussive downbeat transients from drums/bass to drive Wan 2.1 camera zooms, pulses, and shakes.

## 3. Autonomous Video Director v2

- **Musical Beat & Lyric Alignment**: Analyzes energy peaks and lyrical timestamps to segment tracks into Vocal Performance scenes (when lyrics are active) and Cinematic B-Roll scenes (instrumental breaks, drops, intros).
- **Cinematic Camera & Lighting Direction**: Directs specialized camera motions (dolly zoom, slow track, crane sweep, orbiting steadycam) and volumetric lighting designs (cyan rim, warm amber spotlights, atmospheric haze) matched to the selected aesthetic preset.
- **Model-Specific Prompt Compilation**: Translates high-level shot plans into model-specific dialects (e.g. Context-IR for MiniMax H3, LTX-2.5 embedded prompt rules, or Wan2.1 action tags).

## 4. Multitrack Timeline & Master Rendering

- **Non-Destructive Multitrack Editor**: Integrated with [Multitrack Timeline Editor](multitrack-editor.md) and [Non-Destructive Multitrack Timeline](../concepts/non-destructive-multitrack-timeline.md), enabling creators to arrange video tracks, stem audio tracks, and subtitle layers with in/out trims, opacity, crossfades, and canvas transforms (16:9, 9:16, 21:9).
- **AI Round-Trip Take**: Select any scene clip on the timeline to generate an AI retake or variation and drop it directly back into the exact timeline slot without re-editing neighboring scenes.
- **Hardware-Accelerated Single-Pass Compilation**: Compiles the composition into a single FFmpeg `-filter_complex` command using NVENC or Apple Silicon VideoToolbox, remuxed with 256k AAC audio and zero generational loss.

## 5. Keyframe Stills Diffusion & Memory Lifecycle

Pre-rendering visual scene keyframes (`POST /videos/keyframes/{job_id}`) provides a production-grade inspection stage before video diffusion:
- **Adaptive FLUX.2 Diffusion**: Automatically recognizes whether the active image generator is a flow-matching Base model (`black-forest-labs/FLUX.2-klein-base-9B`) or a distilled Turbo/Schnell variant.
  - **Base Models**: Evaluates the full flow ODE trajectory with **24 inference steps** and **guidance scale = 3.5**, rendering crisp, photorealistic cinematic lighting, texture, and character details without blur or waxy artifacts.
  - **Distilled / Turbo Models**: Fast 4-step sampling with guidance scale = 1.0.
- **Eager Local LLM Eviction**: The visual director treatment (`video_director.py`) uses local LLMs (e.g. `Qwen3.6-35B-A3B-UD-MLX-4bit` on oMLX). To prevent OOM errors when switching from LLM prompt compiling to heavy 9B image diffusion, the backend executes an immediate HTTP unload (`POST /v1/models/{id}/unload` or `keep_alive: 0` for Ollama), dropping unified memory residency from 22.7 GB to 0 bytes before diffusion weights load.
- **Batch Diffusion Model Reuse**: During scene stills batch generation across multiple clips, models remain loaded in memory (`auto_unload=False`) to avoid repeated 15-second initialization latency per clip. Once the entire batch is rendered, a single `finally:` block unloads the weights and flushes MLX/Metal memory cache.
- **Progressive Frontend Polling & Auto-Hydration**: In `MusicVideosView.tsx`, the timeline tracks are eagerly populated when still generation begins, and a 3-second progressive polling loop queries `GET /videos/keyframes/{job_id}`, progressively displaying each scene still on the timeline as soon as it is written to disk.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Stem Separator](stem-separator.md) | [Karaoke & Lyric Sync](karaoke-lyricsync.md)
- [Global Hardware Coordinator](hardware-coordinator.md) | [Director Mode v2](../concepts/director-mode-v2.md) | [Multitrack Timeline Editor](multitrack-editor.md)
- [Non-Destructive Multitrack Timeline](../concepts/non-destructive-multitrack-timeline.md) | [Hardware Auto-Tune](../concepts/hardware-autotune-memory-profiles.md)
- [LLM Service & Providers](llm-service.md)

