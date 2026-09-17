---
title: Multitrack Timeline Editor
type: entity
tags: [editor, timeline, daw, multitrack, video, audio, rendering]
created: 2026-09-16
updated: 2026-09-16
sources: [sources/maestro-creative-studio.md]
aliases: [MultitrackEditor, EditorMode, TimelineStudio]
---

# Multitrack Timeline Editor

The **Multitrack Timeline Editor** (`backend/app/services/timeline/` and frontend `EditorMode.tsx`) is Milimo Music's non-destructive composition workspace, bridging AI generation and audio-visual production.

Rather than locking creators into inflexible, monolithic AI renders, the editor gives artists complete timeline control: stacking video b-roll, isolated audio stems (`vocals`, `drums`, `bass`, `other`), sound effects, and animated karaoke typography on a unified multi-track timeline.

---

## 1. System Architecture

```
+-----------------------------------------------------------------------------------+
|                           Frontend Multitrack Editor UI                           |
|   [Tracks: Video L1, Video L2, Vocal Stem, Drum Stem, Bass, Master, Subtitles]    |
|   [Controls: Playhead, In/Out Trims, Transitions, Volume Automation, Transforms]  |
+-----------------------------------------+-----------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
   [Non-Destructive Project JSON]                        [AI Round-Trip Take API]
    - Tracks, Items, In/Out Trims                         - Select Clip -> Send to AI
    - Transforms, Transitions, Mix                        - Re-render -> Return to Slot
                |
                v
   [Single-Pass FFmpeg Compiler]
    - Hardware Encoders: NVENC / VideoToolbox / VAAPI
    - Zero Generational Loss / 256k AAC Audio
    - Export Resolutions: 16:9, 9:16 (Vertical), 21:9
```

---

## 2. Core Editing Capabilities

### 2.1 Multi-Track Layering
- **Video Tracks**: Layer multiple video clips, visual overlays, title cards, and animated stems.
- **Audio Stems**: Discrete lanes for isolated stems (Vocals, Drums, Bass, Instruments), allowing independent fader volume curves, muting, and soloing.
- **Text & Lyric Tracks**: Dynamic drawtext subtitle lanes with style palettes, font size, positioning, and box backgrounds.

### 2.2 Non-Destructive Transformations
- Canvas transformations: Scale, position offsets, cropping, opacity fading, and crossfade transitions (`fade`, `wipeleft`, `dissolve`).
- Aspect ratio selection: 16:9 (YouTube/Broadcast), 9:16 (TikTok/Shorts/Reels), 21:9 (Cinematic Ultra-Wide).

### 2.3 Hardware-Accelerated Single-Pass Export
- Translates the multi-track timeline directly into a single FFmpeg `-filter_complex` pipeline.
- Automatically selects available hardware encoders:
  - NVIDIA NVENC (`h264_nvenc`, `hevc_nvenc`, `av1_nvenc`)
  - Apple Silicon VideoToolbox (`h264_videotoolbox`, `hevc_videotoolbox`)
  - Linux VAAPI / AMF / Intel QSV
  - CPU Fallback (`libx264`, `libsvtav1`)

### 2.4 The AI Round-Trip Take
- Any timeline clip can be selected and sent to the generation provider for an AI variation or retake.
- Upon completion, the new video or audio take replaces the asset reference on the timeline while preserving all cut points, adjacent edits, volume automation, and soundtrack alignment.

---

## 3. Related Pages
- [Non-Destructive Multitrack Timeline](../concepts/non-destructive-multitrack-timeline.md)
- [AI Music Video Studio](video-studio.md)
- [Director Mode v2](../concepts/director-mode-v2.md)
- [Maestro Creative Studio Ingest](../sources/maestro-creative-studio.md)
- [Session Workspace](session-workspace.md)
