---
title: Stem-Driven Audio-Reactive Video Modulation
type: concept
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [video, audio-reactive, stems, liveportrait, wan2, modulation]
aliases: [Stem-Driven Audio Reactivity, Audio Reactive Video]
---

# Stem-Driven Audio-Reactive Video Modulation

**Stem-Driven Audio-Reactive Video Modulation** (`backend/app/services/video/stem_audio_reactive.py`) synchronizes visual movement, camera pulses, and facial lip movements with isolated acoustic stems rather than raw stereo mixes.

## The Mixed-Audio Bleed Problem

In conventional music video generators, lip sync and visual effects listen to the entire stereo audio track. This causes two major failures:
1. **Loud Drum Bleed in Lip Sync**: Heavy kicks and snare hits trigger unintended mouth opening and twitching in vocal avatars.
2. **Dull B-Roll Camera Motion**: Subtle vocal melodies or synth pads dilute percussive impacts, resulting in sluggish or off-beat camera motion.

## Stem Isolation & Dual Modulation Curves

Milimo Music leverages its neural stem separation pipeline ([BS-Roformer / HTDemucs](../entities/stem-separator.md)) to decompose audio into clean modulation channels:

```
                  ┌───────────────────────────────┐
                  │ FULL MASTER STEREO AUDIO      │
                  └──────────────┬────────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │ Neural Stem Separation    │
                   │ (BS-Roformer / HTDemucs)  │
                   └──────┬─────────────┬──────┘
                          │             │
        ┌─────────────────▼───┐     ┌───▼─────────────────┐
        │ Isolated Vocal Stem │     │ Drums & Bass Stems  │
        └─────────┬───────────┘     └───┬─────────────────┘
                  │                     │
        ┌─────────▼───────────┐     ┌───▼─────────────────┐
        │ RMS Energy Envelope │     │ Spectral Transient  │
        │ Clean Lip Sync      │     │ Onset Detection     │
        └─────────┬───────────┘     └───┬─────────────────┘
                  │                     │
        ┌─────────▼───────────┐     ┌───▼─────────────────┐
        │ LivePortrait        │     │ Wan 2.1 Video DiT   │
        │ Singing Avatar      │     │ Camera Zoom & Pulse │
        └─────────────────────┘     └─────────────────────┘
```

### 1. Isolated Vocal RMS for Lip-Sync
- Computes frame-by-frame RMS energy from `vocals.wav`.
- Scales mouth openness and facial expression purely based on singing phonemes, completely immune to kick drums, bass drops, or cymbal crashes.

### 2. Rhythm Transients for Camera Movement
- Extracts onset strength curves from `drums.wav` and `bass.wav`.
- Maps percussive downbeats to decaying zoom pulses ($1.08 \to 1.05 \to 1.02 \to 1.01$) and dynamic camera shakes in Wan 2.1 and Lightricks LTX-Video.

## Integration in Video Orchestrator

In `VideoOrchestrator.generate_advanced_video()`, the computed modulation curves drive both scene-level visual parameters and ffmpeg post-processing, producing beat-synchronized music videos.

## Related Pages

- [Video Studio](../entities/video-studio.md)
- [Stem Separator](../entities/stem-separator.md)
- [Hardware Coordinator](../entities/hardware-coordinator.md)
