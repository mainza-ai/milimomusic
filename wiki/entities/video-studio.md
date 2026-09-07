---
title: AI Music Video Studio
type: entity
created: 2026-09-07
updated: 2026-09-07
tags: [video, studio, wan, hailuo, cogvideox, lipsync, karaoke, ass]
aliases: [VideoStudio, MusicVideosView, VideoService]
---

# AI Music Video Studio

The **AI Music Video Studio** (`backend/app/services/video_service.py` and frontend `MusicVideosView.tsx`) is Milimo Music's multi-scene video generation, lyric synchronization, and visual performance engine. It turns generated music tracks into broadcast-grade music videos with beat-matched scene cuts, artist facial lip-syncing, synchronized karaoke subtitles, and cinematic B-roll.

## 1. Generative Video Duration Constraints ("Locomotives")

Frontier text-to-video diffusion models have strict duration limits due to 3D spatio-temporal attention complexity. The video engine manages and auto-clamps these model limits:

| Model ID | Model Family | Max Duration | Description |
|---|---|---|---|
| `hailuo_h3` | MiniMax Hailuo H3 | **15.0s** | 33B Omni-Modal DiT flagship with high visual fidelity |
| `hunyuan` | Tencent HunyuanVideo | **15.0s** | Open-source 13B visual DiT sequence renderer |
| `cogvideox` | THUDM CogVideoX 1.5 | **10.0s** | 5B 3D causal VAE model (161 frames at 16 fps) |
| `wan2.1` | Wan-AI Wan 2.1 | **5.0s** | 1.3B/14B lightweight text-to-video (81 frames at 16 fps) |
| `audioreactive` | Audio-Reactive Synth | **120.0s** | Real-time procedural waveform & spectrum animation |

- **Defaulting**: When `max_clip_duration` is omitted, the engine automatically resolves to the model's exact maximum limit.
- **Clamping**: If a user specifies a duration exceeding the model's physical limit, it is automatically clamped to prevent generative failure.
- **Bar-Aligned Segmentation**: Uses song BPM (default 120) to align clip boundaries to musical bars (`(60.0 / BPM) * 4.0`), snapping cuts to natural musical phrases.

## 2. Isolated Vocal Stem Lip-Syncing

To prevent drum kicks, basslines, or synthesizers from causing unnatural mouth twitching, the lip-sync engine isolates the vocal track:
1. **Source Isolation**: Automatically pulls `vocals.wav` or `vocals.mp3` produced by Demucs neural source separation.
2. **Acoustic RMS Energy Envelope**: Calculates frame-level RMS energy ($E_t = \sqrt{\frac{1}{N}\sum x_n^2}$) and applies asymmetric attack/release ballistic smoothing ($\alpha_{attack}=0.35, \alpha_{release}=0.70$) to model human lip inertia.
3. **Facial Landmark Tracking**: Uses OpenCV Haar cascade classifiers (`haarcascade_frontalface_default.xml`) to identify character face geometry and oral cavity anchors.
4. **Viseme Mouth Deformation**: Modulates mouth aperture dynamically frame-by-frame:
   - Deep oral cavity rendering with dark crimson shading `(22, 14, 42)`.
   - Subtle upper dental highlight `(210, 218, 222)`.
   - Feathered alpha-blending of translated lower lip to maintain organic jaw contours.
   - Micro-motion head nod and studio rim glow synchronized to vocal power.

## 3. Dynamic B-Roll & Procedural Visual Synthesizer

When a clip window contains no vocals (instrumental solos, intros, drops):
- **Artwork Pan/Zoom**: Executes multi-axis Ken Burns motion with orbital sweep, subtle camera float $(x(t), y(t), z(t))$, and style-matched color grading LUTs (neon-cyberpunk, anime-cinematic, retro-vhs).
- **Procedural Canvas**: When artwork is absent, synthesizes flowing chromatic plasma with trigonometric gradient fields matching the selected visual style.

## 4. Synchronized Karaoke & Master Audio Muxing

- **ASS Subtitle Burning**: Generates Advanced SubStation Alpha (`.ass`) karaoke subtitles with highlight tags (`\k<duration>`) formatted with primary and secondary palette illumination.
- **Sample-Accurate Remuxing**: Assembles individual MP4 clips via FFmpeg concat demuxer and remuxes with uncompressed master stereo audio with zero PTS audio-video drift.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Stem Separator](stem-separator.md) | [Karaoke & Lyric Sync](karaoke-lyricsync.md)
