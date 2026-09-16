---
title: Track Extension
type: concept
created: 2026-08-19
updated: 2026-09-16
sources: [sources/readme.md, production-readiness-plan.md]
tags: [extension, track, generation, continuation, minimax, production]
aliases: [Extend, Track continuation, Seamless Extension]
---

# Track Extension

**Track Extension** lets Milimo continue generating from where a previous track left off,
allowing musicians and producers to build full-length songs (e.g. 60s -> 120s+)
**segment by segment** while preserving acoustic timbre, tempo, key, and musical continuity.

---

## 1. How It Works

1. **Lossless Cut Point & Slicing**:
   - The user selects where the continuation begins (`extend_from_sec`), defaulting to the parent track's end or an earlier section break.
   - The engine slices parent audio up to `extend_from_sec` losslessly at 48 kHz.
2. **Acoustic Timbre & Seed Locking**:
   - The child job inherits the parent's exact `seed`, style tags, and prompt description.
   - For MiniMax Music 3, locking seed and style tags while appending new section tags (`[Verse 2]`, `[Chorus]`, `[Outro]`) forces the autoregressive audio model to continue in the exact harmonic and vocal space.
3. **Equal-Power Crossfade Concatenation**:
   - Overlap window: $1.5\text{s}$ (user-adjustable between $0.5\text{s}$ and $3.0\text{s}$).
   - Equal-power curve: $w_{\text{out}}(t) = \cos(\frac{\pi}{2} t)$, $w_{\text{in}}(t) = \sin(\frac{\pi}{2} t)$ such that $w_{\text{out}}^2 + w_{\text{in}}^2 = 1$.
   - Prevents seam clicks, phase cancellations, or energy dips at the transition point.
4. **Full Pipeline Re-Finalization**:
   - The combined master audio is run through the entire production pipeline: stem separation (HTDemucs/BS-Roformer), [MuScriptor](../entities/muscriptor.md) neural transcription, and karaoke lyric sync.

---

## 2. API & Database Architecture

- **Endpoint**: `POST /tracks/{job_id}/extend`
  - Body: `TrackExtendRequest` (`target_duration_sec`, `extend_from_sec`, `additional_lyrics`, `prompt`, `crossfade_sec`).
  - Response: `{ "job_id": UUID, "parent_job_id": UUID, "target_duration_sec": float, "extend_from_sec": float, "status": "queued" }`.
- **Database Schema (`Job` model)**:
  - `parent_job_id: Optional[UUID]`
  - `is_extension: bool`
  - `extend_from_sec: Optional[float]`
- **Orchestration**: `pipeline.generate_audio_step` detects `job.is_extension` and calls `provider.extend(...)` passing parent audio, cut point, delta duration, and crossfade configuration.

---

## 3. User Interface Integration

- **DAW Arrange Timeline (`ArrangeTimeline.tsx`)**:
  - Direct `Extend Song` button in timeline controls header.
- **Track Studio (`TrackDetailView.tsx`)**:
  - `Extend Track` action button in the audio toolstrip.
- **Song Library (`TrackRowPlayer.tsx`)**:
  - `Extend Track` quick-action button on hover.
- **Extend Track Modal (`ExtendTrackModal.tsx`)**:
  - Visual time scrubber and duration slider (+30s, +60s, +90s, +120s, +180s presets).
  - Seam point adjustment slider.
  - Section tag inserters (`[Verse 2]`, `[Chorus]`, `[Bridge]`, `[Guitar Solo]`, `[Outro]`).
  - Equal-power crossfade window configuration.
  - Lineage indicators displaying parent seed and style tags.

---

## Related Pages

- [MiniMax Music 3](../entities/minimax-music3.md)
- [MuLaCover](../entities/mulacover.md)
- [Lyrics conditioning](lyrics-conditioning.md)
- [Session Workspace](../entities/session-workspace.md)
- [Generation Provider](../entities/generation-provider.md)
