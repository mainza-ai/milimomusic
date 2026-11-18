---
title: Music Video Pipeline Audit 2026-11
type: overview
tags: [video, audit, music-video, pipeline, production, delivery, qc, lip-sync, karaoke, reports]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [MusicVideoAudit, VideoPipelineAudit]
---

# Music Video Pipeline Audit — 2026-11

Audit of Milimo Music's **music-video workflow and pipeline** against what a
private, production deployment needs: not "does a demo clip render", but "can a
musician ship a finished video from it, twice, on any machine, and document what
it did".

Scope: `backend/app/services/video/` (~7.2 k LOC), `services/video_service.py`
(facade), `services/timeline/` (editor + render), the `/videos/*` surface in
`main.py`, and `frontend/src/components/{video,views}/`. Builds on
[Full Application Audit 2026-11](application-audit-2026-11.md) — this page
verifies its video claims and goes deeper; it does not repeat them.

Companions: [Music Video Delivery & QC Specs](concepts/music-video-delivery-specs.md)
(external standards + 2026 engine landscape) and
[Session Reporting & Reproducibility](concepts/session-reporting-and-reproducibility.md)
(the documentation/report/academic surface). **Implementation tasks live in
[Phase 7 of the remediation plan](application-remediation-plan-2026-11.md#phase-7--music-video-production-2026-11-18-added).**

> [!NOTE]
> **Remediated on 2026-11-18.** Every finding below was implemented and verified
> the same day as **Phase 7** of the [remediation plan](application-remediation-plan-2026-11.md#phase-7--music-video-production-2026-11-18-added)
> (24/25 tasks; 7.21 accessibility partial). The four audit deltas **X1–X4 are
> closed**, each with a test that fails against the pre-fix code. Findings are kept
> in past tense-unfixed form as the record of what was wrong and why it mattered.

---

## 1. What the pipeline actually is

Two independent renderers share one facade (`VideoService` → `VideoOrchestrator`),
plus a separate NLE:

| Path | Entry | Engine | Output |
|---|---|---|---|
| **Advanced** (AI scenes) | `POST /videos/render-advanced/{job_id}` → `render_advanced_music_video` (`video_orchestrator.py:748`) | Per-scene diffusion/cloud + lip sync + stem-reactive pulse | `{job}_master_mv.mp4` |
| **Lyric** (fast) | `POST /videos/render-lyric/{job_id}` → `:1605` | One FFmpeg graph: Ken Burns cover + `showwaves` + ASS karaoke | `{job}_lyric.mp4` |
| **Audio-reactive** | `POST /videos/render/{job_id}` → `video_service.py:951` | `showcqt` / cover + waveform | `{job}_reactive.mp4` |
| **NLE export** | `POST /api/editor/projects/{id}/compile` | `compile_editor_render_command` | `{project}_{task}.mp4` |

Advanced order: resolve master + vocal stem → stem-reactive curves →
`video_director.segment_song` (LLM treatment → scenes → lattice snap) →
pre-build all scene plates in one image-model load (`:654`) → per-clip render →
stem-reactive pulse → normalise every clip to plan duration and project geometry
(`:1450`) → stitch → burn karaoke ASS → remux master audio → 480p proxy +
poster.

**Verified strengths** (keep them):

- **Duration truth.** Lattice loss is compensated (`clip_normalise.py`),
  crossfade overlap is reserved per clip, and the final mux targets the song
  length instead of `-shortest` (`:1115`). `verify_assembly` reports drift.
- **Honest engine refusal.** Unimplemented engines are refused, not substituted
  with a Ken Burns animatic (`generator_registry.py:70-79`).
- **One fps through the chain**, per-task temp cleanup (`:1406`), cancel that
  actually kills FFmpeg (`_active_procs`), ASS escaping with bundled fonts
  (`subtitle_styles.py:38`), and a real 3-tier lyric aligner
  (`transcription/karaoke.py:553`).
- **Scene plates are fingerprinted and reusable**, and an approved retake plate
  is never overwritten (`stamp_scene_plate(..., approved=True)`).

---

## 2. Creative workflow (W)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| **W1** | **Per-clip renders are deleted after assembly.** `rendered_clips` are removed in `finally`; `_cleanup_task_temp` deletes every `*_{task_id}*` file. | `video_orchestrator.py:1219-1222, 1406-1432` | A retake of scene 7 re-diffuses **every** scene. No take history, no incremental re-assembly, and the editor's `ai_take_parent_id`/`take_version` can never apply to AI scenes. |
| **W2** | **"Retake" regenerates a still, not a clip.** `POST /videos/retake-clip/...` writes `keyframe_{job}_{idx}.png`; only a full re-render produces video. | `main.py:6095-6161` | The expensive step is unchanged by the button that appears to fix one shot. |
| **W3** | **Retake silently degrades to a raster placeholder** and still returns `status: ok`. | `main.py:6141-6153` | Violates the "no artifact without saying what produced it" rule the audit set for audio. |
| **W4** | **No render versioning.** Output name derives from `job.id` only; `Job.video_path` is one string. | `video_orchestrator.py:1073, 1734`; `models.py:101-102` | A second render overwrites the first silently; A/B comparison of two treatments is impossible. |
| **W5** | **No per-job render mutex.** Two `render-advanced` calls both run and write the same path. | `main.py:5885-5923` | Last writer wins; `video_path` can point at the other render's partial file. Model downloads already 409 for exactly this (`main.py:1670`). |
| **W6** | **A music-video render never produces an editor project.** Projects are created only by client `POST /api/editor/projects`. | `main.py:6540-6596` | The NLE and the AI pipeline are parallel universes: you cannot open a finished video as a timeline and re-cut it. |
| **W7** | **No approval gate.** Approved plates are respected, but nothing blocks a render on unapproved plates. | `_scene_plate_is_current:613` | The review step is advisory; a render can consume plates nobody looked at. |

## 3. Musical analysis and sync (M)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| **M1** | **Downbeats are every 4th beat**, first beat assumed to be a bar start. | `audio_analysis.py:126` | In 3/4, 6/8, half-time, or any pickup, cuts land on beats 2–4 — and downbeats carry the highest non-section cut weight (1.8). This is the single largest source of "the edits feel off". |
| **M2** | **Sections are 16 s RMS windows, not structure analysis.** | `audio_analysis.py:150-186` | A quiet chorus is labelled `Bridge`. Section boundaries carry weight 3.0 (strongest anchor) *and* the label is fed to the LLM director, so a wrong label corrupts the treatment too. |
| **M3** | **The generation plan's own structure is discarded.** Milimo generates the song from tagged lyrics (`[Verse]`/`[Chorus]`) and stores `timed_lyrics_json`, yet the video path re-derives structure from energy. | `audio_analysis.py:131-188` vs `models.py:110` | Exact section truth already exists in-repo and is unused. Highest-leverage single fix in this audit. |
| **M4** | **`camera_zoom` and `vocal_envelope` are computed song-wide and never read.** Only `beat_hits` is consumed. | `stem_audio_reactive.py:52-104`; no consumers outside the module | Two full-song librosa passes per render for nothing; the module docstring's claim that the vocal envelope drives lip sync is false. Zero tests reference the module. |
| **M5** | **The audio-reactive pass re-encodes each clip at `-preset ultrafast`.** | `stem_audio_reactive.py:167-175` | Another lossy generation, at the worst preset, on every clip with a beat hit. |
| **M6** | **Two alignment sources are never reconciled.** The lyric path prefers `timed_lyrics_json`; the advanced path re-aligns from raw lyrics when absent. | `video_orchestrator.py:1078-1097, 1670-1682` | A timing fix made in the DAW may not reach the burned lyrics. |

## 4. Engines (E)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| **E1** | **Local Wan clamps to ≤832×480 and ≤33 frames on MPS** while the UI offers 720p/1080p. | `diffusers_wan.py:165-178` | The delivered 1080p file is upscaled 480p, and nothing reports it (plan 4.2 asked for exactly this — see **X3**). |
| **E2** | **Cloud clips are hard-capped at 5 s** with one fixed model id. | `cloud_video.py:139, 170` | Planned 15 s scenes become 5 s clips that the stitcher stretches. |
| **E3** | **No cost estimate or budget guard** for cloud renders. | no `cost`/`price`/estimate in `services/video/` | fal bills per generated second; a 40-scene render is unbounded spend the user cannot preview. Plan 4.5's acceptance required it. |
| **E4** | **"LivePortrait neural lip sync" is in practice always the jaw-warp fallback — and reports success.** The provider delegates to `SmoothVisemeFallbackProvider` when weights are missing *or* `is_available` is False (which only checks torch+MPS/CUDA), returning its `True`. | `lip_sync/live_portrait.py:41-48, 76-91, 128-157` | `fallback_used` stays `False`, so no banner. Even with weights, the code shells `inference.py --driving_audio` — a CLI contract real LivePortrait does not offer (it is video-driven). |
| **E5** | **No face extraction.** Lip sync needs an explicit portrait; keyframes are never face-detected. | `resolve_face_image:235-250` | Users with only album art silently get b-roll instead of a performer, with no prompt to add one. |
| **E6** | **No subject consistency mechanism** beyond "I2V from the same portrait": no reference conditioning, IP-Adapter, LoRA, or last-frame chaining. | `diffusers_wan.py:87-107`; deferred item 4.11 | Eight scenes look like eight different people. This is the quality ceiling versus hosted tools. |

## 5. Delivery and QC (D)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| **D1** | **The master is encoded 2–3 times, lossy each time**: normalise → stitch → optional subtitle re-encode, all `libx264 veryfast` at default CRF. | `clip_normalise.py:78-86`; `video_orchestrator.py:1566-1574, 1126-1136` | Generation loss compounds on the one artifact the user publishes. **Measured:** a delivered master is 0.80 Mbps video. |
| **D2** | **The master lacks `+faststart`** (only the proxy has it). **Measured on a real render:** `moov@14552439, mdat@44` — the index is at the end. | `video_orchestrator.py:1126-1147` | Publishing over HTTP requires downloading the whole file before playback/seek. The lyric path *does* set it (`:1794`) — so the two paths disagree. |
| **D3** | **Audio is re-encoded to AAC 256 kbps with no loudness normalisation or true-peak limiting**, while the platform owns a real `-14 LUFS` mastering stage. | `video_orchestrator.py:1133, 1150, 1162`; `transcription/mastering.py` | Videos disregard the platform's own loudness policy; WAV masters get a lossy transcode with no A/B. |
| **D4** | **No QC pass.** Nothing checks black/freeze frames, A/V drift, dropped frames, or measures delivered loudness. `verify_assembly` only compares durations and only logs. | `clip_normalise.py:106-119`; `:1108-1113` | Broken renders are found by watching them. |
| **D5** | **One hardcoded delivery profile.** 8-bit 4:2:0 H.264 + AAC, one resolution and aspect per render, fixed `-b:v 8000k` in the NLE for *any* resolution, no caption sidecar on the advanced path. | `video_orchestrator.py:1126-1191`; `editor_projects.py:296-305` | No YouTube/Vevo/broadcast/social presets, no 4K ladder, no SRT/VTT deliverables. See [delivery specs](concepts/music-video-delivery-specs.md). |
| **D6** | **`/videos/render/{job_id}` reports success without checking FFmpeg.** `render_audio_reactive_video` returns a URL unconditionally; the endpoint then writes `job.video_path`. | `video_service.py:1001-1010`; `main.py:5516-5531` | A failed encode yields "completed" and a dead URL. |
| **D7** | **Retention GC can orphan video references.** `prune_generated_media` deletes `.mp4` under `generated_audio` and never nulls `video_path`. | `core/storage_gc.py:126-158` | With retention on, jobs point at deleted videos with no signal. |
| **D8** | **The plate cache has no policy.** GC prunes `video_cache/keyframes`; plates actually live in `generated_audio/videos/keyframes`. | `storage_gc.py:31-38`; `video_service.py:50` | Re-render cost is either unbounded growth or 24 h amnesia. |

## 6. Workflow surface / UI (U)

The UI is genuinely deep — 15 aesthetics, 6 typography presets, pacing bias, cast
scoping, per-scene retake, treatment re-conception, a section ruler and a
nudgeable lyric track. It is also where honesty and durability concentrate.

| # | Finding | Evidence | Impact |
|---|---|---|---|
| **U1** | **"Split A/B Comparison" is decorative.** The component has exactly one `<video>`; the divider and Take A/B labels overlay it and clip nothing. | `VideoCanvasPlayer.tsx:324-355` (`1` `<video>` at `:189`) | The only affordance for judging a retake compares a video with itself. |
| **U2** | **Retake/re-imagine edits are not persisted** — in-memory `planResult` only; `saveStoryboard` is called from reorder alone. | `MusicVideosView.tsx:646, 1312-1344` | Direction work is lost on reload; the master is unchanged. |
| **U3** | **Retake camera/lighting choices are dropped server-side** (`RetakeClipRequest` has no such fields) and the toast claims success. | `ClipRetakeModal.tsx:147-176`; `models.py:906-911` | Silent no-op presented as success. |
| **U4** | **The provider picker is not wired.** `getVideoProviders()` has no callers; "Free"/"Fast" are static; `cloud_replicate` is a reachable state string that D4 disabled. | `api.ts:1714`; `VideoInspectorDock.tsx:514, 530` | Availability and capability are asserted, not queried. |
| **U5** | **Regenerate silently downgrades settings** — the restore whitelist covers 4 of 15 styles and 3 of 6 subtitle presets. | `MusicVideosView.tsx:1170, 1179` | A re-render can differ from the approved video. |
| **U6** | **Renders are not resumable across reloads** (task id in a memory ref; only planning uses `sessionStorage`). | `MusicVideosView.tsx:400, 583, 974` | Reload during a 10-minute render loses the HUD while the GPU keeps working. |
| **U7** | **Poll failures are swallowed forever** (empty catches, no attempt cap/timeout); the keyframe loop is `while (true)`. | `MusicVideosView.tsx:480-482, 847-878, 1013, 1083` | A dead backend is indistinguishable from a slow render. |
| **U8** | **The completion state is destroyed by the next effect**, and `completed` + `video_url: null` ends silently. | `MusicVideosView.tsx:554, 996, 1062` | The `fallback_used` banner is usually never seen. |
| **U9** | **Cancel is optimistic** — toast and local status fire even with no task id. | `MusicVideosView.tsx:911-942` | "Cancelled" can be false. |
| **U10** | **The proxy and poster the backend produces are unused**; the player streams the master. | backend `:1179-1187`; no `_480p`/`_poster` in `frontend/src` | Scrubbing pulls the full file. |
| **U11** | **Fabricated data becomes render input**: missing timings invented as `idx*15`/`15 s`, `bpm: 120` hardcoded in the plan request, restored storyboards cast without aggregate fields. | `MusicVideosView.tsx:347-350, 565, 668-670` | The plan the renderer receives can disagree with what the user saw. |
| **U12** | **The timeline cannot be edited**: no drag/trim/split, only ±1 swaps; timeline length is the last clip's end, not the song; reorder knowingly desyncs. | `VideoTimelineTrack.tsx:181-187, 197-207`; `MusicVideosView.tsx:626-651` | Real timing fixes require a re-render. |
| **U13** | **Accessibility is demo-grade on the primary path**: unlabelled track `<select>`, modals without dialog semantics/focus trap/Escape, click-only timeline clips, mouse-only compare, unlabelled sliders, 8–9 px type. | `VideoTopBar.tsx:81-92`; `ClipRetakeModal.tsx:64`; `KeyframeZoomModal.tsx:25`; `VideoTimelineTrack.tsx:335, 507` | Keyboard/screen-reader users cannot select a track or seek. |
| **U14** | **Only the final mp4 and keyframe PNGs leave the app** — no caption sidecar, no storyboard project export, no share link, no poster. | `MusicVideosView.tsx:1368-1372` | Delivery and hand-off are manual; the *decisions* are not portable. |

## 7. Reporting, documentation, research

Detailed inventory and standards live in
[Session Reporting & Reproducibility](concepts/session-reporting-and-reproducibility.md).
The video-specific conclusions:

- **Storage is ahead of emission.** Provenance, quality, seeds and engines are
  persisted per job (`models.py:69,113-125`) but never emitted as a document.
- **Audio has export parity that video does not.** `/transcribe/export` covers
  MIDI/MusicXML/Ableton/LRC/SRT/stems; the video side has no cut list, storyboard,
  EDL/OTIO, captions sidecar, or credits.
- **The pack ships the unmastered file.** Studio/Project packs key off
  `job.audio_path`, not `mastered_path` (`main.py:4656`), and mastering
  measurements are never persisted (`models.py` has no such columns).
- **MusicXML misreports musical facts**: `<fifths>0</fifths>`, 4/4, `divisions=4`,
  and a literal `<encoding-date>2026-08-21</encoding-date>`
  (`muscriptor_provider.py:465-516`) — disqualifying as a research artifact.

## 8. Audit deltas — claimed done, not actually done

> **Closed 2026-11-18** by 7.1 (X1), 7.10 (X2), 7.3 (X3) and 7.7 (X4).

| # | Plan item | Claimed | Reality |
|---|---|---|---|
| **X1** | 1.14 "Video lip sync: set `fallback_used`/`fallback_reason`; require a real weight manifest for `is_available`" | UI banners the jaw-warp fallback | `is_available` still only checks torch+MPS/CUDA (`live_portrait.py:41-48`); the *internal* fallback returns `True`, so the orchestrator never sets `fallback_used` (`:929-936`). Reclassify as open — it is **E4**. |
| **X2** | 4.8 "feed `camera_zoom` into the Ken Burns branch" | done | `camera_zoom` is computed and never read anywhere (grep). Re-open as **M4**. |
| **X3** | 4.2 "report the delivered resolution" | done | No delivered-resolution or upscale notice exists in task status, API or UI. Re-open (folded into **E1**). |
| **X4** | 4.5 (deferred, unverified) "cloud UI gated on key **and** a cost estimate" | implemented, unverified | No cost estimate exists anywhere in `services/video/`. Re-open as **E3**; the fal path itself remains unverified without a key. |

Two of these (X1, X2) are exactly the class of defect Phase 1 existed to close:
a capability that reports success while doing something else, and work that is
computed then discarded. Phase 1's status line should read 17/18, not 18/18.

## 9. Priorities (all delivered — see Phase 7)

Each priority below became a Phase 7 task and is now implemented; kept as the
rationale for ordering.

**P0 — truth (a user must be able to believe the output):** E4/X1 lip-sync
honesty · D6 render success verification · W3/U3 retake honesty · U1 fake
compare · X4/E3 cloud cost · U5 config fidelity.

**P1 — the pipeline a musician can actually iterate with:** W1/W2 per-clip
persistence + incremental re-render · W4 takes/versioning · W5 mutex · M1/M2/M3
real harmony and structure (start with M3, it is nearly free) · D1/D2/D3 single
quality-targeted encode + faststart + loudness · D4 QC gate.

**P2 — delivery and documentation:** D5 export presets and caption sidecars ·
UI resilience (U6/U7/U8/U9) · accessibility (U13) · reporting/reproducibility
(Phase 7.5).

Not worth doing: chasing a bigger local diffusion model before E6 (consistency)
and D1–D3 (delivery) are fixed — the bottleneck is not the model.

## Related

[Music Video Delivery & QC Specs](concepts/music-video-delivery-specs.md) ·
[Session Reporting & Reproducibility](concepts/session-reporting-and-reproducibility.md) ·
[AI Music Video Studio](entities/video-studio.md) ·
[Application Remediation Plan](application-remediation-plan-2026-11.md) ·
[Full Application Audit 2026-11](application-audit-2026-11.md) ·
[Stem Audio-Reactive Video](concepts/stem-audio-reactive-video.md) ·
[Index](index.md)
