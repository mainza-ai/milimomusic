---
title: Repair Segment (Inpainting Service)
type: entity
created: 2026-08-19
updated: 2026-09-16
sources: [sources/inpainting-debug.md, sources/readme.md]
tags: [inpainting, repair, repair-segment, audio-domain, crossfade, minimax, production]
aliases: [Repair Segment, InpaintingService, Inpainting]
---

# Repair Segment (Inpainting Service)

**Repair Segment** lets the user replace, repair, or regenerate a specific audio time segment of a generated track without regenerating the entire song. It preserves the surrounding acoustic and musical context (tempo, key, instrumentation, and vocals) by splicing freshly synthesized infill audio using musical beat-grid alignment and equal-power sine/cosine crossfading.

---

## 1. Modern Architecture (Milimo v2)

In Milimo v2, the `InpaintingService` was audited and completely decoupled from legacy HeartMuLa token files (`generated_tokens/*.pt`). The modern architecture operates directly in the audio domain and ties into the active generation provider:

1. **Parent Audio & Musical Profile Resolution**:
   - Audio is located on disk via `_resolve_audio_file()`.
   - `extract_audio_musical_attributes()` analyzes the parent track's BPM, beat grid, and harmonic structure.
2. **Beat-Grid Downbeat Snapping**:
   - `start_sec` and `end_sec` snap to nearest musical beats/downbeats when a valid beat grid is present, ensuring transitions occur at natural musical boundaries.
3. **Locked Acoustic Infill Generation**:
   - The infill segment (covering the gap plus crossfade margins) is synthesized via the active generation provider (MiniMax Music 3) using `build_locked_continuation_caption()`, guaranteeing matching key, rhythm, and style.
4. **Dual Equal-Power Sine/Cosine Crossfading**:
   - Smoothly splices Part 1 (head up to `start_sec + crossfade_sec`), Infill, and Part 3 (tail from `end_sec - crossfade_sec` to end):
     $$w_{\\text{out}}(t) = \\cos\\left(\\frac{\\pi}{2} t\\right), \\quad w_{\\text{in}}(t) = \\sin\\left(\\frac{\\pi}{2} t\\right), \\quad w_{\\text{out}}^2 + w_{\\text{in}}^2 = 1.0$$
   - Completely eliminates seam clicks, phase anomalies, and volume dips.
5. **Full Production Cascade**:
   - Spliced audio is passed through BS-Roformer 6-stem neural source separation, [MuScriptor](muscriptor.md) neural transcription (MIDI + MusicXML), and WhisperX timed lyric synchronization.
   - Creates a child `Job` record with `is_repair=True`, `parent_job_id`, and publishes real-time SSE progress events.

---

## 2. API & Frontend Integration

- **Endpoint**: `POST /jobs/{job_id}/inpaint`
  - Body: `TrackInpaintRequest` (`start_time: float`, `end_time: float`, `prompt: Optional[str]`, `crossfade_sec: float = 1.0`).
  - Response: `{"status": "queued", "job_id": str, "parent_job_id": str, "message": "In-painting started"}`.
- **Track Detail UI (`TrackDetailView.tsx`)**:
  - Direct `Repair Segment` action button in the track toolstrip opening `InpaintModal`.
- **Inpaint Modal (`InpaintModal.tsx`)**:
  - Interactive dual-range sliders for start/end time, visual segment timeline bar, and automatic job tracking.

---

## Related pages
- [LM-guided inpainting](../concepts/lm-guided-inpainting.md)
- [Track Extension](../concepts/track-extension.md)
- [MiniMax Music 3](minimax-music3.md)
- [MuScriptor](muscriptor.md)
- [Backend & API](backend-api.md)
