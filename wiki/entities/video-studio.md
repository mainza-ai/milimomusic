---
title: AI Music Video Studio
type: entity
created: 2026-09-07
updated: 2026-09-15
tags: [video, studio, wan, liveportrait, ltx-video, diffusers, lipsync, karaoke, ass]
aliases: [VideoStudio, MusicVideosView, VideoService, VideoOrchestrator]
---

# AI Music Video Studio

The **AI Music Video Studio** (`backend/app/services/video/` and frontend `MusicVideosView.tsx`) is Milimo Music's production-grade AI music video generation, neural singing avatar animation, and visual performance engine. It turns generated music tracks into broadcast-grade music videos with beat-matched scene cuts, neural facial lip-syncing driven by isolated vocal stems, pre-rendered scene keyframes, synchronized karaoke subtitles, and cinematic video diffusion.

## 1. Generative Video Models & Diffusion Engines

Frontier text-to-video and image-to-video diffusion models operate with strict spatio-temporal attention windows:

| Model ID | Model Family | Max Duration | Engine / Pipeline | Description |
|---|---|---|---|---|
| `wan_14b` | Alibaba Wan 2.1 14B | **5.0s** | `WanPipeline` / `WanImageToVideoPipeline` | SOTA 14B DiT with 3D spatio-temporal attention, keyframe I2V and T2V |
| `wan_1.3b` | Alibaba Wan 2.1 1.3B | **5.0s** | `WanPipeline` / `WanImageToVideoPipeline` | Lightweight 1.3B DiT suitable for rapid local rendering |
| `ltx_video` | Lightricks LTX-Video 0.9B | **10.0s** | `LTXPipeline` / `LTXImageToVideoPipeline` | Real-time high-efficiency DiT capable of up to 10s continuous generation |
| `cloud_fal` | Fal.ai Cloud GPU | **5.0s - 15.0s** | Fast Serverless REST | Offloaded Wan 2.1 / LivePortrait generation without local GPU pressure |
| `cloud_replicate`| Replicate Cloud GPU | **5.0s - 15.0s** | Managed Model Runner | Offloaded Wan 2.1 / LivePortrait execution via Replicate API token |

- **Musical Bar Snapping**: `VideoDirector.segment_song()` uses detected BPM to snap clip durations to exact integer musical bars (`(60.0 / BPM) * 4.0`), guaranteeing that scene cuts land precisely on musical beats.
- **Keyframe Pre-Rendering**: Users can pre-render visual keyframe stills (`POST /videos/keyframes/{job_id}`) across all planned scenes to inspect and approve directorial composition before triggering full video diffusion.

## 2. Neural Singing Avatar & Lip-Syncing (LivePortrait)

To eliminate unnatural mouth twitching and deliver broadcast-quality vocal performances:
1. **Stem Isolation**: The director routes only the isolated vocal track (`vocals.wav` / `vocals.mp3` from Demucs) to the lip-sync engine. Heavy kicks and 808 bass cannot distort lip movements.
2. **LivePortrait Neural Avatar**:
   - Uses implicit keypoint representations and landmark deformation driven by audio pitch and amplitude.
   - Synthesizes organic eye blinks, micro-expressions, head nods, and realistic phonetic viseme transitions.
   - Executed under [Global Hardware Coordinator](hardware-coordinator.md) device locks to prevent VRAM exhaustion with audio pipelines.
   - Supports local Apple Silicon PyTorch MPS execution as well as cloud GPU offloading.
3. **Smooth Viseme Mesh Fallback**:
   - For low-resource environments without neural weights, a bilinear jaw mesh warp engine smoothly translates the mouth cavity and lips based on vocal power envelopes, avoiding static OpenCV ellipse overlays.
4. **Stem Audio-Reactive Modulation**:
   - Powered by [Stem Audio-Reactive Video](../concepts/stem-audio-reactive-video.md) (`stem_audio_reactive.py`), extracting clean vocal envelopes for lip-sync and percussive downbeat transients from drums/bass to drive Wan 2.1 camera zooms, pulses, and shakes.

## 3. Autonomous Video Director

- **Musical Beat & Lyric Alignment**: Analyzes energy peaks and lyrical timestamps to segment tracks into Vocal Performance scenes (when lyrics are active) and Cinematic B-Roll scenes (instrumental breaks, drops, intros).
- **Cinematic Camera & Lighting Direction**: Directs specialized camera motions (dolly zoom, slow track, crane sweep, orbiting steadycam) and volumetric lighting designs (cyan rim, warm amber spotlights, atmospheric haze) matched to the selected aesthetic preset.

## 4. Synchronized Karaoke & Master Audio Muxing

- **ASS Subtitle Burning**: Compiles Advanced SubStation Alpha (`.ass`) karaoke scripts with real-time word/syllable timing and style-matched color highlights, burned directly into video streams using FFmpeg `-filter_complex "[0:v]subtitles='...'"`.
- **Sample-Accurate Remuxing**: Stitches individual MP4 video clips via FFmpeg concat demuxer and remuxes with 256k AAC master audio, ensuring zero audio-video drift across the entire track duration.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Stem Separator](stem-separator.md) | [Karaoke & Lyric Sync](karaoke-lyricsync.md)
- [Global Hardware Coordinator](hardware-coordinator.md) | [Sidecar Engine Manager](sidecar-engine-manager.md) | [Stem Audio-Reactive Video](../concepts/stem-audio-reactive-video.md)
