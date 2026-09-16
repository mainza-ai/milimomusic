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

1. **Model-Native KV-Cache Roll-Forward (Option 1 & 1A)**:
   - For MiniMax Music 3 (`mlx-community/MiniMax-Music3-bf16`), the engine utilizes `generate_frame_hiddens_extended_hooked` to deterministically replay the parent track's autoregressive Qwen3 hidden states at ~11 fps up to `parent_frames` (e.g. 1500 frames for 60s).
   - Rather than generating an independent segment from scratch, the language model simply rolls forward its established KV-cache into the future, maintaining identical rhythm, harmonic voicing, timbre, and acoustic space.
2. **Deterministic Token & Caption Parity**:
   - In MiniMax Music 3, prompt text and lyrics tokens precede `<|audio_start|>`. Appending text or altering structured captions shifts Rotary Position Embeddings (RoPE) and initial embeddings, destroying deterministic replay.
   - The extension engine preserves `parent_prompt`, `parent_structured_caption`, and parent lyrics for the autoregressive conditioning tokens, guaranteeing bit-exact mathematical replay ($0.0000000$ error) across the parent frame window.
3. **Early Audio Termination Suppression**:
   - The autoregressive token loop suppresses `audio_end_token_id` (token `151670`) via `suppress_end_token = (frame_index < target_frames)` to prevent the model from exiting early when initial lyrics finish.
4. **Beat-Grid Downbeat Snapping & Equal-Power Crossfading**:
   - `extend_from_sec` snaps to the nearest musical downbeat (measure start) based on detected tempo and meter (`beat_grid`).
   - Splicing joins the parent disk audio ($0 \to \text{cut}$) with the roll-forward continuation ($\text{cut} \to \text{target}$) using equal-power sine/cosine crossfading over $1.5\text{s}$:
     $$w_{\text{out}}(t) = \cos(\frac{\pi}{2} t), \quad w_{\text{in}}(t) = \sin(\frac{\pi}{2} t), \quad w_{\text{out}}^2 + w_{\text{in}}^2 = 1$$
   - A subtle 0.25s tail fade enforces exact `target_duration_sec`.
5. **Strict Lyrics Control & On-Demand AI Drafting**:
   - **Lyrics are NEVER automatically added by default**: To prevent unintended vocal hallucination or phantom verses during extension, `auto_generate_lyrics` defaults to `False`. When empty, the child track retains the parent track's lyrics exactly as-is.
   - **On-Demand AI Drafting**: In `ExtendTrackModal.tsx`, creators can trigger the lyricist engine on demand via *"✨ Draft with AI"* (`api.generateLyrics`), allowing inspection, editing, or clearing prior to queuing.
   - **Opt-In Auto-Generation**: Users may alternatively toggle *"Auto-generate continuation lyrics with AI"* if they want the orchestrator to synthesize continuation verses automatically.

6. **Full Pipeline Re-Finalization**:
   - The extended audio is automatically run through the entire production pipeline: BS-Roformer 4-stem separation, [MuScriptor](../entities/muscriptor.md) neural transcription (MIDI + MusicXML), and forced-alignment lyric sync.

---

## 2. Forensic Validation Metrics

Live verification on parent track `27490839` (136.0 BPM gospel piano ballad) extended to 120.0s (`a7abdea8`):

- **Target Duration**: Exactly $120.00\text{s}$ (no premature cutoff).
- **Tempo Continuity**: Part 1 (0–60s) = $136.0\text{ BPM}$, Part 2 (60–120s) = $136.0\text{ BPM}$ ($\mathbf{\Delta = 0.0\text{ BPM}}$).
- **Acoustic Environment**: Spectral Centroid matched within 7% (2702 Hz vs 2516 Hz).

---

## 3. API, Database & Download Architecture

- **Extension Endpoint**: `POST /tracks/{job_id}/extend`
  - Body: `TrackExtendRequest` (`target_duration_sec`, `extend_from_sec`, `additional_lyrics`, `auto_generate_lyrics`, `prompt`, `crossfade_sec`).
  - Response: `{ "job_id": UUID, "parent_job_id": UUID, "target_duration_sec": float, "extend_from_sec": float, "status": "queued" }`.
- **Dedicated Download Endpoint**: `GET /download_track/{job_id}`
  - Resolves disk audio path via `_resolve_audio_file()`.
  - Returns `FileResponse` with explicit `Content-Disposition: attachment; filename="<title>.wav"` and `Access-Control-Expose-Headers: Content-Disposition`.
  - Frontend utilizes `api.downloadAudioTrack()` / `api.downloadUrlAsFile()` to fetch cross-origin audio files into in-memory `blob:` URLs before triggering anchor download. This bypasses HTML5 cross-origin download restrictions and prevents browsers from playing the .wav file in a new tab.
- **Database Schema (`Job` model)**:
  - `parent_job_id: Optional[UUID]`
  - `is_extension: bool`
  - `extend_from_sec: Optional[float]`
  - `parent_structured_caption: Optional[Dict[str, str]]`

---

## 4. User Interface Integration

- **DAW Arrange Timeline (`ArrangeTimeline.tsx`)**: Direct `Extend Song` button in timeline controls header.
- **Track Studio (`TrackDetailView.tsx`)**: `Extend Track` action button in the audio toolstrip; downloadable master, stems, and MIDI files via `downloadUrlAsFile`.
- **Song Library (`TrackRowPlayer.tsx`) & Global Dock Player (`GlobalAudioPlayer.tsx`)**: Direct download action buttons triggering `api.downloadAudioTrack()`.
- **Extend Track Modal (`ExtendTrackModal.tsx`)**: Visual time scrubber, downbeat snapping indicator, and duration presets (+30s, +60s, +90s, +120s).

---

## Related Pages

- [MiniMax Music 3](../entities/minimax-music3.md)
- [MuLaCover](../entities/mulacover.md)
- [Lyrics conditioning](lyrics-conditioning.md)
- [Session Workspace](../entities/session-workspace.md)
- [Generation Provider](../entities/generation-provider.md)
- [MuScriptor](../entities/muscriptor.md)
