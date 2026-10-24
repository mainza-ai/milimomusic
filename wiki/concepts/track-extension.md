---
title: Track Extension
type: concept
created: 2026-08-19
updated: 2026-10-24
sources: [sources/readme.md, production-readiness-plan.md]
tags: [extension, track, generation, continuation, minimax, production]
aliases: [Extend, Track continuation, Seamless Extension]
---

# Track Extension

**Track Extension** lets Milimo continue generating from where a previous track left off,
allowing musicians and producers to build full-length songs (e.g. 60s -> 120s+)
**segment by segment** while preserving acoustic timbre, tempo, key, and musical continuity.

> [!NOTE]
> The replay contract is strict: deterministic continuation requires
> **byte-identical conditioning inputs** to the parent run. See
> [Extension Troubleshooting](extension-troubleshooting.md) for what breaks it
> and how to diagnose failures.

---

## 1. How It Works

1. **Model-Native KV-Cache Roll-Forward (Option 1 & 1A)**:
   - For MiniMax Music 3 (`mlx-community/MiniMax-Music3-bf16`), the engine utilizes `generate_frame_hiddens_extended_hooked` to deterministically replay the parent track's autoregressive Qwen3 hidden states up to `parent_frames`, then sample only new frames.
   - Rather than generating an independent segment from scratch, the language model simply rolls forward its established KV-cache into the future, maintaining identical rhythm, harmonic voicing, timbre, and acoustic space — *iff* conditioning is byte-identical (see §2).
   - **Exact restore fast path**: if the parent run saved per-frame AR codes (`.mm3state.npz`), the KV cache is rebuilt by prefilling saved codes (no resampling, no RNG consumed). Bundles are validated (seed + conditioning fingerprint + snapshot) before use.
2. **Deterministic Token & Caption Parity**:
   - In MiniMax Music 3, prompt text and lyrics tokens precede `<|audio_start|>`. Appending text or altering structured captions shifts Rotary Position Embeddings (RoPE) and initial embeddings, destroying deterministic replay.
   - The extension conditions on the parent's **exact model-input strings**: since 2026-10-24 every generation persists `model_prompt`/`model_lyrics` verbatim (never lossy-cleaned), and extend prefers them over any reconstruction. Display fields (`lyrics`, `prompt`) may be cleaned for UI/karaoke — the replay never uses those.
   - The timed remainder slice is transcription-normalized text and must NEVER steer the replay; it travels as `tail_lyrics` for the fresh-segment fallback path only. Genuinely new (user/auto) lyrics steer explicitly and are flagged `conditioning_steered`.
3. **Early Audio Termination Suppression**:
   - The autoregressive token loop suppresses `audio_end_token_id` (token `151670`) **only for new frames past `parent_frames`** (`suppress_end = frame_index >= parent_frames`). The parent was generated with the end token allowed, so the replay range must sample identically; suppressing everywhere diverges from frame 0.
4. **Beat-Grid Downbeat Snapping & Equal-Power Crossfading**:
   - `extend_from_sec` snaps to the nearest musical downbeat (measure start) based on detected tempo and meter (`beat_grid`). Fallback grids (`120 BPM / downbeat 0.0` from failed beat detection) are repaired from the audio before snapping.
   - Splicing joins the parent disk audio ($0 \to \text{cut}$) with the roll-forward continuation ($\text{cut} \to \text{target}$) using equal-power sine/cosine crossfading. The overlap is quantized to whole beats of the real grid (never shorter than 1 beat, never longer than 1 bar); the tail starts at `cut − crossfade` by design.
   - A subtle 0.25s tail fade enforces exact `target_duration_sec` (shortfalls pad with silence).
5. **Preflight Replay Check & Honest Fallback Routing**:
   - When replaying from *reconstructed* (not stored-verbatim) conditioning, a ~30 s probe render is compared against parent disk audio *before* the full render. On mismatch the run routes to the labeled fresh-segment fallback instead of burning a full render on a doomed trajectory. Verbatim-stored and user-steered runs skip the probe (exact by construction, or divergent by intent).
   - Post-hoc, the full render's head is compared the same way (`replay_verified` tri-state on the job + SSE + UI badge). Failed raw renders are kept (`{job}_raw_failed.wav`) for forensics instead of deleted.
6. **Extension Modes** (`extension_mode`):
   - `kv_rollforward` — deterministic replay + continuation (the seamless path).
   - `state_restored` — KV cache restored from the parent's saved bundle (fastest, exact).
   - `fresh_segment_fallback` — tail-only generation spliced on (MLX/seed unavailable, or preflight failed). Expect a seam; the tail sings the remainder slice.
   - `no_parent_audio` — parent audio missing from disk; raw audio published, no splice anchor.
   - `stem_outro` — DAW-built outro looping the parent's own stems/bars (groove locked by construction; no new material sung).
7. **Strict Lyrics Control & On-Demand AI Drafting**:
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

- [Extension Troubleshooting](extension-troubleshooting.md) — symptom map, forensic recipes, root-cause catalog, CROSS case study
- [MiniMax Music 3](../entities/minimax-music3.md)
- [MuLaCover](../entities/mulacover.md)
- [Lyrics conditioning](lyrics-conditioning.md)
- [Session Workspace](../entities/session-workspace.md)
- [Generation Provider](../entities/generation-provider.md)
- [MuScriptor](../entities/muscriptor.md)
