---
title: Application Remediation Plan 2026-11
type: overview
tags: [plan, remediation, security, honesty, lifecycle, migrations, daw, video, ci]
created: 2026-11-16
updated: 2026-11-18
sources: [application-audit-2026-11.md, decisions-2026-11.md, music-video-pipeline-audit-2026-11.md]
aliases: [remediation plan nov 2026, audit plan, fix plan v3]
---

# Application Remediation Plan — 2026-11

Implementation plan for the findings in
[Full Application Audit 2026-11](application-audit-2026-11.md), with scope fixed by
[Decision Record 2026-11](decisions-2026-11.md), plus **Phase 7** from the
[Music Video Pipeline Audit 2026-11](music-video-pipeline-audit-2026-11.md) (added
2026-11-18; 110 tasks total, 81 of the original 85 verified or deliberately deferred).
Ordered by **value ÷ risk**, with a hard dependency spine so each phase leaves the app
shippable.

Every task names the files to touch and an observable acceptance criterion. Effort is in
focused engineer-days.

## Implementation status (2026-11-18)

**76 of 85 original plan tasks are implemented; 4 of those were reopened after
re-verification (§Corrections) and 9 are deferred with reasons below.** Phase 7 adds
25 tasks from the music-video audit. Every phase is shippable at its current state.

| Phase | Status | Evidence |
|---|---|---|
| **0 — Stop the bleeding** | ✅ **16/16** | exploit chain closed; YuE2 + Training Studio deleted; suite runs in 50 s |
| **1 — Provenance contract** | ✅ **18/18** | `quality`/`provenance` on every job; 1.14 **closed by 7.1** — lip sync now reports the engine that ran |
| **2 — Spine** | ✅ **14/15** | durable queue for generation, bounded retries, one recovery policy, stall sweeper, **Alembic 0001–0007 + backups**, one UUID encoding, real FKs, soft delete, retention, lock ownership |
| **3 — DAW core** | ✅ **16/17** | clip editing with trim/split/fades, workspace undo, tempo map + metronome, sample-accurate loop, clipboard, quantize strength, velocity lane, MIDI import, automation lanes, server-side session recall |
| **4 — Media truth** | ✅ **13/16** | clip duration/geometry normalisation, real timeline export, storyboard persistence, fal-only cloud, scoped cancel, subtitle escaping, temp hygiene, proxies, single fps, performer scoping, real audio analysis; 4.8 **closed by 7.10**, 4.2's reporting clause **closed by 7.3** (4.5 remains unverified without a live fal key) |
| **5 — Platform & API** | 🟡 **9/12** | one error envelope, 202/201, typed bodies everywhere, idempotency, SSE replay + topics, rate limits by template, share links, accessible confirms |
| **6 — Hardening** | ✅ **8/8** | CI lint/audit/migration-smoke/unit/E2E jobs, universal lockfile, `npm ci`, observability, log hygiene, capability matrix |
| **10 — Memory** | 🟡 **11/12** | added 2026-11-18 from the [Memory Management Audit](memory-management-audit-2026-11.md); footprint telemetry, MLX/MPS caps, bounded pipeline cache, budgeted agent history, WAL limit and one release path shipped |
| **9 — Performance** | 🟡 **8/12** | added 2026-11-18 from the [Performance Audit](performance-audit-2026-11.md); ETag/304 + catalogue memo + boot warm, thread cap, compression decision, lazy LLM SDKs, measured encoder choice, benchmark harness and parallel normalisation shipped |
| **8 — Capability upgrades** | 🟡 **11/19** + 1 partial | added 2026-11-18 from the [Video Capability Landscape](video-capability-landscape-2026-11.md); verifier/reshoot, character library, external avatar tiers, licence gate, capability matrix, viseme timing, stall sweeper shipped — the weight-dependent adoptions are gated on smoke tests |
| **7 — Music video production** | ✅ **29/29** | added 2026-11-18 from the [Music Video Pipeline Audit](music-video-pipeline-audit-2026-11.md); implemented and verified the same day, plus follow-ups 7.26–7.29 (consistency, lip-sync registry, render reconciliation, chain scope) |

**Verification (2026-11-18, after the execution pass):** backend `800 passed, 2 skipped,
3 deselected` (0 failures); frontend `38 vitest tests`;
`tsc -b` + `vite build` green (entry 588 kB / 163 kB gz); `ruff` bug-class gate
clean; API parity 178 routes / 182 client calls; Alembic at `0008`.

### Deferred — with reasons, not silently dropped

| Task | Why deferred |
|---|---|
| 2.13 one storage root | Deleting the boot-time mirror moves ~5.5 GB; it needs an operator-run migration with a verify step, not an automated boot action |
| 3.10 lazy stem decode + LRU | A buffer-lifecycle change in the transport; the loading path is correct today, just eager |
| 4.11 retake seed/character continuity | Needs a persisted per-job reference plate + img2img path through the image service |
| 4.16 collapse the two video service layers | ~1.3 k LOC of duplication; a pure refactor with no behaviour change, best done as its own review |
| 5.1 split `main.py` into routers | Deliberately last: it would have made every other phase a merge conflict |
| 5.9 frontend data layer (TanStack Query) | Cross-cutting rewiring of ~180 call sites; the highest-value remaining frontend work |
| 5.11 full WCAG 2.2 AA programme | Dialog semantics, assertive errors, reduced motion and the `xs` breakpoint shipped; the full audit (hotkey scoping, i18n) remains |
| 4.5 fal end-to-end | Implemented (upload-then-URL); **unverified** without a key — see the decision record |

### Corrections from the music-video audit (2026-11-18)

Four tasks were marked done that the code does not support. They are reopened with
new IDs rather than silently re-opened in place, because the fix is larger than the
original task text. Full evidence:
[Music Video Pipeline Audit §8](music-video-pipeline-audit-2026-11.md#8-audit-deltas--claimed-done-not-actually-done).

| Task | Original acceptance | What is actually true | Now tracked as |
|---|---|---|---|
| 1.14 | UI banners the jaw-warp fallback | `is_available` still checks only torch+MPS/CUDA; the *internal* fallback returns `True`, so `fallback_used` is never set | **7.1** |
| 4.8 | `camera_zoom` feeds the Ken Burns branch | `camera_zoom` and `vocal_envelope` are computed song-wide and read by nothing | **7.10** |
| 4.2 | Delivered resolution reported | never reported anywhere; the 1080p master is upscaled 480p on MPS | **7.3** |
| 4.5 | Cloud UI gated on key **and** cost estimate | no cost estimate exists in `services/video/` | **7.7** |

**All four were closed by Phase 7 on 2026-11-18** (7.1, 7.10, 7.3, 7.7), with
regression tests that fail against the pre-Phase-7 code. Phase 1 is back to
**18/18** and Phase 4 to **13/16**. Nothing else in the earlier phases was
invalidated.


## Sequencing principle

Three tracks, not one:

- **Track A — Trust:** make the product stop lying (provenance, capability, cancel).
- **Track B — Spine:** make state, schema, and jobs real (queue, Alembic, FKs, recovery).
- **Track C — Surface:** make the DAW and studios actually editable.

**Track C starts only after Track B lands CANCELLED + the claim helper**, because every
editor feature eventually needs a reliable save and a reliable undo of a job state.
**Track A runs first and continuously** — it is the cheapest path to user trust and it
prevents users from shipping fake content.

## Phase 0 — Stop the bleeding (Week 1, ~8 days)

Fix the things that are dangerous or actively destroying user work. Nothing here is a
refactor.

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 0.1 | Kill the arbitrary-file-read class: introduce `resolve_public_media_url()`; reject absolute paths and `..` in every request-sourced path; route `commit-vocal`, `lead-sheet`, `thumbnail`, `gallery/route` through it | `core/paths.py`, `main.py:4371-4392,3538,5769,5802` | `POST commit-vocal {"master_path":"/etc/passwd"}` → 400; regression test per endpoint |
| 0.2 | Contain the SPA catch-all with `Path.resolve()` + `is_relative_to(dist)` | `main.py:5980-5991` | `GET /../../.env` → 404 |
| 0.3 | Make auth fail-closed: `MILIMO_ALLOW_ANONYMOUS` must be opt-in; wrap static mounts in the auth middleware; drop `?auth=` except for a short-lived SSE ticket | `core/security.py`, `main.py:545-549` | Booting with no token and no explicit allow flag refuses to serve non-loopback |
| 0.4 | Honour `HOST`/`MILIMO_HOST` in the launcher; refuse `0.0.0.0` without a token | `milimo:158,165` | `./milimo start` binds 127.0.0.1 by default |
| 0.5 | Piano-roll unmount must flush, not cancel, the debounced save; add dirty guard + `beforeunload` | `PianoRoll.tsx:335-339,643-646` | Edit → switch mode within 700 ms → reload → edit present |
| 0.6 | Fix `provider_registry.get` → `get_provider` (in-paint dead) and take the device lock in the repair path | `inpainting_service.py:258`, `:123-270` | Repair returns regenerated audio; two concurrent repairs serialise |
| 0.7 | `Assert`-style tests: a test walks every `task_queue.*` and `video_service.*` call site against `dir()` | `tests/test_task_queue_contract.py` | Adding a bad call fails CI |
| 0.8 | Boot reconcile must repair **every** `ACTIVE_RUN_STATUSES` value, not just `running`; age-bound the blocking query | `main.py:339-344,1386-1400,1914` | The 5 stuck `queued` album runs clear on next boot; artist delete works |
| 0.9 | `JobStatus.CANCELLED` + `INTERRUPTED`; stop writing `FAILED` on user cancel; atomic claim (`WHERE status='queued'`) so cancel cannot be resurrected | `models.py:9-13`, `pipeline.py:91-97`, `main.py:4560-4576` | Cancel a queued job → it never runs and never flips back |
| 0.10 | Disable the fake timeline "compile" endpoint (or make it 501) until Phase 4.4 | `main.py:5918-5936` | Endpoint no longer returns a command as if it rendered |
| 0.11 | Delete `title="Action"` everywhere; move `{...rest}` after `title` in the `Button` primitive | `primitives.tsx:60-61` + 40 call sites | `grep -r 'title="Action"'` = 0 |
| 0.12 | Make the build green: update `test_scene_plates.fake_render` to accept `cancel_check`, and stop `image_service.py:943` swallowing a `TypeError` as "pre-build failed" | `tests/test_scene_plates.py:27-33`, `image_service.py:938-944` | 7 scene-plate tests pass; a broken renderer signature fails loudly |
| 0.13 | Stop the suite hanging: `pytest-timeout`, `asyncio_mode = auto`, `HF_HUB_OFFLINE=1` in CI, mock or `@pytest.mark.heavy` the three live-model tests | `pytest.ini`, `ci.yml`, `tests/test_production_v2.py:225,244,521` | Full suite completes unattended in CI |
| 0.14 | **D1 — delete YuE2 + Training Studio**: remove the provider, the trainer, the `/api/training/jobs*` routes, the `/data/training_auditions` mount, the `yue2_studio` UI tab and `trainingApi`; purge any `yue2_*` files already written to `models/loras` | `providers/yue2_provider.py`, `services/training/`, `main.py:5939-5968`, `VoiceStudioModal.tsx`, `api.ts:1783-1810` | `grep -ri yue2 backend/app frontend/src` = 0; no code path can write a `.safetensors` it did not train |
| 0.15 | **D3 — de-stub the keepers**: remove `extend()`/`repair_segment()` from HeartMuLa and `repair_segment()` from MuLaCover; drop the capability flags they imply; mark HeartMuLa `legacy: true` and hide it from the default picker | `heartmula_provider.py:152-187`, `mulacover_provider.py:227-241`, `registry.py:34-37` | No provider advertises a method that returns its own input |

**Exit criteria:** no unauthenticated file read; no silent MIDI loss; cancel means cancel;
in-paint works; the suite has a contract test that catches dead method calls; **no shipping
feature can fabricate a model artifact.**

## Phase 1 — The provenance contract (Weeks 2–3, ~9 days)

One rule, enforced mechanically: **no artifact may reach `COMPLETED` without saying what
produced it and how degraded it is.**

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 1.1 | Define `Provenance` once (`source`, `degraded`, `reason`, `method`) and add it to `GeneratedAudioResult`, `TranscriptionResult`, `SeparationResult`, `ImageResult`, `VoiceResult`, `VideoClip` | `providers/base.py`, `transcription/*`, `services/image_service.py`, `video/types.py` | Type-level: every result has it |
| 1.2 | Persist a single `job.provenance_json` + a derived `job.quality` (`neural` \| `degraded` \| `placeholder`); surface it on Job and SSE | `models.py`, `pipeline.py:197-201` | `/jobs/{id}` exposes `quality` |
| 1.3 | MuScriptor: delete the hardcoded I–vi–IV–V fallback; on failure persist `transcription_status="unavailable"` + reason and write **no** MIDI/MusicXML | `muscriptor_provider.py:376-473` | A failed transcription never populates the piano roll |
| 1.4 | ~~YuE2 honesty~~ — **resolved by deletion (D1, task 0.14)**; nothing to flag because nothing ships | — | — |
| 1.5 | Voice conversion: never `copyfile`, never hard-fail the job; record `method` ∈ `{svc, spectral_morph, skipped}`; rename `NeuralSVCService` → `SpectralMorphSVC` | `voice_service.py:409-475,585`, `voice/neural_svc.py:19-22` | Failing SVC degrades honestly and does not fail the pipeline |
| 1.6 | Derive capability manifests from tests: `supports_x == True` only if the method exists and is not a stub; remove `supports_segment_repair`/`supports_lora`/`supports_voice_conversion` where untrue | `providers/*.py`, `providers/base.py` | New test `test_capability_manifests_are_earned.py` |
| 1.7 | `ProviderNotFound` instead of silent MiniMax substitution; record `provider_substituted_from` when a fallback is explicit | `providers/registry.py:138-140` | Selecting a missing model → 400 `model_not_installed` |
| 1.8 | Drop `extend()` from providers that regenerate a song; pipeline falls back to the concatenate+crossfade path | `musicgen/stable_audio/heartmula/mulacover` providers | Extension never produces an unrelated track |
| 1.9 | Report real duration: `requested_duration_sec` + `truncated_reason`; stop overwriting `job.duration_ms` silently | `stable_audio_provider.py:149`, `musicgen_provider.py:150`, `pipeline.py:193` | 3-minute request on a 47 s model surfaces the truncation |
| 1.10 | Fix the separation lie: `source_id` from the model actually loaded; UI text from `result.stems.keys()`; remove `"htdemucs_6s" if False` | `real_separator.py:34,187`, `pipeline.py:261,278` | UI never offers a guitar stem that does not exist |
| 1.11 | Remove invented instruments from per-instrument stems, or mark them `derived_from` + `synthetic_derivation=True` | `instrument_stems.py:264-312` | No instrument appears that transcription did not find |
| 1.12 | Transcribe the **delivered** master (after remix), not the raw generation | `pipeline.py:340-399` | MIDI matches the exported audio |
| 1.13 | Image: persist `engine` on `VisualAsset`, badge procedural plates, add "regenerate with model" | `image_service.py:637-648`, `ImageStudioView.tsx:123` | Gallery distinguishes FLUX from gradient |
| 1.14 | Video lip sync: set `fallback_used`/`fallback_reason`; require a real weight manifest for `is_available` | `lip_sync/live_portrait.py:39-148`, `video_orchestrator.py:903` | UI banners the jaw-warp fallback |
| 1.15 | Mark `cogvideox`/`hunyuan`/H3-local as `implemented: false`; disable in picker; refuse the render rather than substituting procedural | `generator_registry.py:67-78`, `minimax_h3.py:155-178` | No engine labelled "diffusion" renders a Ken Burns |
| 1.16 | Register the `*-Diffusers` Wan repos the loader can actually run; make a missing `model_index.json` a surfaced error | `model_manager.py:549,561`, `diffusers_wan.py:51-85` | Download → render works end to end, or fails loudly |
| 1.17 | `StaticWaveform`: never cache a fabricated envelope; show "waveform unavailable" + retry | `StaticWaveform.tsx:11-46` | Cache holds only real peaks |
| 1.18 | Export modal: either implement server-side transcode or label the pills "filename only" and disable them | `MasteringExportModal.tsx:185-233` | Choosing "MP3" never yields a `.mp3`-named WAV |
| 1.19 | Honest progress everywhere: indeterminate stripe when `progress == null`; real `step/total_steps`; per-frame diffusers callback, FFmpeg `-progress`, HF byte callbacks | `App.tsx:938-946,1722`, `video_orchestrator.py:432,882`, `main.py:1621` | No fabricated percentages remain |

**Exit criteria:** `grep` for `used_fallback`/`degraded` covers every result type; a
"placeholder content" audit query (`quality != 'neural'`) returns a UI-visible list.

## Phase 2 — Spine: jobs and schema (Weeks 3–5, ~12 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 2.1 | Move music generation onto `DurableTaskQueue`; one worker, GPU lane concurrency 1, bounded depth; 429 `queue_full` with position/ETA | `core/task_queue.py`, `main.py:3471`, new `core/worker.py` | Restart mid-generation resumes or re-dispatches queued jobs |
| 2.2 | `attempts`, `max_attempts`, `next_retry_at` on Job/Task; bounded retry with jitter classified by error type | `models.py`, `pipeline.py:549-558` | Transient OOM retries once, then fails with the typed reason |
| 2.3 | Fix `DurableTaskQueue`: close connections (`contextlib.closing`), single-statement `UPDATE`, `INSERT OR IGNORE`, add `fail_task` | `core/task_queue.py:75-81,126-131,185-218` | fd count stable over 1 k updates; cancel path works |
| 2.4 | `last_heartbeat_at` + a stuck-job sweeper; unified recovery policy table replacing the three divergent reconcilers | `core/recovery.py`, `main.py:336-393` | A wedged job becomes `stalled` instead of `processing` forever |
| 2.5 | Cancel honesty: thread `cancel_event` into separation + transcription + cloud video poll loops; register every FFmpeg child in `_active_procs`; only flip to `cancelled` after acknowledgement | `real_separator.py`, `muscriptor_provider.py`, `cloud_video.py`, `video_orchestrator.py:1120-1140` | Cancel during transcription stops within 2 s |
| 2.6 | Alembic baseline: stamp the current live schema as `0001`, batch mode for SQLite, run `alembic upgrade head` as an explicit step; **delete the boot-time DDL and the PK rewrite** | new `backend/alembic/`, `main.py:165-310`, delete `migrate_db.py` | Boot does zero DDL; a schema mismatch fails fast with a version message |
| 2.7 | Backup before migrate (`sqlite3 .backup`, N rotations) + documented restore | `scripts/`, `alembic/env.py` | Every upgrade leaves a timestamped backup |
| 2.8 | One UUID encoding: `UUID` type + FK everywhere, normalise at the Pydantic boundary; one-time repair script for existing mixed rows | `models.py:96-101`, `main.py:794-851` | 36-char and 32-hex inputs resolve identically; `get_job_by_id` drops to one lookup |
| 2.9 | Real foreign keys with `ON DELETE SET NULL`, `PRAGMA foreign_key_check` on boot; delete the 40-line manual cascade | `models.py`, `main.py:4530-4552` | Deleting a session leaves no dangling `job.session_id` |
| 2.10 | Add the missing indexes (`job(status, created_at DESC)`, `job.project_id`, `job.session_id`, `agentrun.parent_run_id`) and unique constraints (`playlisttrack(playlist_id, job_id)`, `agentassignment(profile_id, role)`) via migration | `models.py` | `EXPLAIN QUERY PLAN` uses the index on `/history` |
| 2.11 | `job_artifact(job_id, kind, path)` child table; delete by lookup instead of glob-glue; add `deleted_at` soft delete + explicit `?purge=true` | `models.py`, `main.py:4437-4503` | Delete removes the remixed master and `ai_cover_*.png` too |
| 2.12 | Retention + disk guard: `MILIMO_MEDIA_RETENTION_DAYS`, disk gauge on `/health`, refuse new jobs below N GB free, add `scene_stills` to GC | `core/storage_gc.py:28-33`, `main.py:552` | Disk growth is bounded and observable |
| 2.13 | Single storage root; delete the boot-time `copy2` mirror and the 5-root resolution shotgun | `main.py:323-331`, `core/paths.py:109-406` | One `generated_audio`, no resurrection of deleted files |
| 2.14 | Fix the GPU-lock contract: owner tokens on `acquire`/`release`, delete the duplicate `music_service.gpu_lock`, pass a real `modality` from the pipeline | `core/hardware_lock.py:174-213`, `pipeline.py:73-78` | Releasing without holding raises; eviction hooks fire on release |
| 2.15 | Move model eviction / cache flush / `detect_tempo` / lyric alignment / diffusion off the event loop (`to_thread` / executor) | `hardware_lock.py:188,232-276`, `main.py:3550`, `pipeline.py:359-377`, `diffusers_wan.py:219` | SSE heartbeats never stall during a render |

**Exit criteria:** kill -9 mid-generation → restart → no lost work, no stuck state, no
orphan files, no schema drift. This phase is what makes everything after it safe.

## Phase 3 — DAW core (Weeks 5–8, ~22 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 3.1 | Introduce a `Clip` model (`start`, `offset`, `duration`, `gain`, `fades`) and make Arrange a real editor: drag, trim, split (S), fade handles, snap grid | `ArrangeTimeline.tsx`, `AudioEngineContext.tsx` | A clip can be trimmed and split, and playback honours it |
| 3.2 | Workspace-level command stack: every mutation goes through `dispatch({apply, revert})`; ⌘Z/⇧⌘Z scoped to the workspace | `SessionWorkspace.tsx:1203-1217`, new `stores/history.ts` | Mixer move, loop change, lyric edit, stem switch all undo |
| 3.3 | Tempo map `[{t, bpm, sig}]` + metronome scheduled on the AudioContext clock + count-in | `SessionWorkspace.tsx:1804-1974`, `NotationViewer.tsx:63` | Click tracks 4/4 and 6/8; count-in before `playAll()` |
| 3.4 | Sample-accurate loop: `source.loop/loopStart/loopEnd` or double-buffered next pass; 5 ms gain ramp on stop/seek | `SessionWorkspace.tsx:776-785,897-907` | No gap or click at the loop seam or on pause |
| 3.5 | Piano roll toolset: velocity lane, ⌘C/⌘V/⌘X/⌘A, split/legato, quantize options (strength/iterative/end), scale snap on edit, nudge | `PianoRoll.tsx:419-595,1646-1660` | A MIDI editor tests the standard operations |
| 3.6 | Keyboard scope fix: shortcuts work from the panel, not only when the grid has focus | `PianoRoll.tsx:613-619` | ⌘Z after clicking Quantize still undoes |
| 3.7 | MIDI import (`.mid` → `NoteEvent[]` → commit) | `PianoRoll.tsx`, `api.ts` | Import a MIDI file and see it on the grid |
| 3.8 | Persist the whole session server-side (`/workspace/{id}/session`, debounced, versioned), localStorage as offline mirror | `SessionWorkspace.tsx:229-275` | Arrangement + mix survive cache clear and another browser |
| 3.9 | Fix A/B audition mid-playback silence; reset duration on track change; guard localStorage volume parse | `SessionWorkspace.tsx:661-663`, `AudioEngineContext.tsx:63-66,158-224` | A→B switch while playing keeps audio flowing |
| 3.10 | Lazy stem decode + LRU buffer cache; compute peaks once per buffer off-thread | `SessionWorkspace.tsx:369-379,710-755,959-967` | Workspace opens without decoding 300 MB |
| 3.11 | Kill the re-render storm: `useSyncExternalStore` selectors, split contexts enforced by lint, memoize the 5 parses in Track Studio | `AudioEngineContext.tsx:660-663`, `TrackDetailView.tsx:341-359` | Playback does not re-render library rows |
| 3.12 | Code splitting: `lazy()` per view + `manualChunks`, icon deep-imports | `App.tsx:14-45`, `vite.config.ts` | Entry chunk < 400 KB gz |
| 3.13 | Guard render-time JSON with `safeJsonParse` everywhere; per-view `ErrorBoundary` | `GlobalAudioPlayer.tsx:81-85`, `App.tsx:1753-2242` | A corrupt row degrades one panel, not the app |
| 3.14 | "Play All" passes the project queue; add a waveform-unavailable state | `ProjectsView.tsx:196-202`, `StaticWaveform.tsx` | Play All plays the project |
| 3.15 | Remove `select-none` from the shell; add `screens.xs`; `prefers-reduced-motion` block | `App.tsx:1242`, `tailwind.config.js`, `index.css` | Text is copyable; tab labels show on narrow screens |
| 3.16 | **D5 — automation lanes** (in-scope): `automation: {param: [{t, v, curve}]}` per track on the session payload; a lane under the arrange timeline; schedule with `setValueCurveAtTime`/`linearRampToValueAtTime` on play and seek; 4 params (volume, pan, send, master). Read-only display first (~4 d), then drawing + undo (~4 d) | `SessionWorkspace.tsx:631-641`, `ArrangeTimeline.tsx`, `stores/*` | A volume curve drawn in the lane is audible on playback and survives reload |
| 3.17 | **D5 — tempo-locked generation** (replaces time-stretch): pass the project BPM/key into `GenerationRequest` and condition the model on it, so a track is *born* at the right tempo instead of being stretched | `providers/minimax_provider.py` (conditioning), `models.py` `GenerationRequest`, `ComposerSidebar.tsx` | Generating at 96 BPM yields a 96 BPM track within ±1 BPM of the beat grid |

**Out of scope (D5):** any time-stretch / warp / `atempo` / Rubber Band in the live
transport. If conformance of *imported* audio is ever needed, it ships as an export-time
bounce only — see [D5](decisions-2026-11.md#d5--automation-lanes-and-time-stretch-automation-yes-time-stretch-no).

**Exit criteria:** the Arrange tab is editable, undo covers the workspace, tempo/meter are
real, and the app opens fast on a 4-minute 4-stem project.

## Phase 4 — Media studio truth (Weeks 8–10, ~10 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 4.1 | Frame-lattice correctness: render `model_render_duration`, trim/pad each clip to `target_duration`, assert `\|Σ clips − song\| < 0.05 s`, re-time otherwise | `director_music_timing.py:194-216`, `video_orchestrator.py:1071,1391-1400` | A 40-clip video ends exactly with the song |
| 4.2 | Normalise every clip before assembly (`scale`/`pad`/`fps`/`setsar`); report the delivered resolution | `video_orchestrator.py:1357-1400` | Mixed Wan(16 fps)/LTX(25 fps) concat is valid |
| 4.3 | Persist the storyboard (`PATCH /videos/{job}/storyboard`); make reorder a real timeline move that recomputes windows and warns on desync | `main.py:5025`, `VideoTimelineTrack.tsx:196-206` | Reload keeps scene edits |
| 4.4 | Real compile: enqueue an editor render, run FFmpeg in a worker, stream progress, return a URL; apply `clip.transform` + `transition_in/out` + `setpts` | `main.py:5918`, `editor_projects.py:222-232` | Export produces a playable MP4 with the configured transforms |
| 4.5 | **D4 — cloud is fal only**: upload-then-URL instead of data URIs, resolve endpoint ids from a table, provider-aware clip lengths; add one mocked-transport contract test; delete the Replicate branches (`cloud_video.py:161-180`, `cloud_lipsync.py:73-78`) and the H3-cloud path | `cloud_video.py:99-180`, `cloud_lipsync.py:60-90`, `minimax_h3.py:96` | A fal key produces a real clip; `grep -r REPLICATE` = 0; cloud UI gated on key **and** a cost estimate |
| 4.6 | Performer scoping actually applied: add the field to `SceneClip` and inject into `prompt` | `video_director.py:756-799,1146-1165`, `types.py:41-64` | Instrumental shots get `mouth_movement: closed` |
| 4.7 | Resolve audio before analysis; synthetic analysis sets `plan.fallback_used` with a reason | `video_director.py:305-313` | A relative `audio_path` no longer silently yields a 120 BPM grid |
| 4.8 | One fps through the reactive chain; feed `camera_zoom` into the Ken Burns branch; skip the re-encode when there are no hits | `video_orchestrator.py:806,966,136-147` | Beat pulses land on the beat |
| 4.9 | Temp hygiene: per-task temp dir + `finally: rmtree`; cascade keyframes/plates/thumbs on job and asset delete | `video_orchestrator.py:1120-1123`, `main.py:2919-2935,4923-4980` | Delete reclaims the bytes |
| 4.10 | ASS escaping + real fonts dir; log the resolved font fallback | `subtitle_styles.py:15-24,377` | A `{` in a lyric renders literally |
| 4.11 | Retake continuity: persist a per-job seed + character reference plate; apply the same grade; accept `reference_image_path` | `main.py:5445-5480` | Scene 7 retake matches scenes 1–6 |
| 4.12 | Per-job diffusion cancel (job→task map) instead of the global singleton | `video_orchestrator.py:209-216`, `image_service.py:85-96` | Cancelling one job does not abort another |
| 4.13 | Serve model specs from one place (`/videos/models`); delete the four duplicate tables | `model_specs.py:56-169`, `video_service.py:61-70`, `video_director.py:167-176`, `MusicVideosView.tsx:38-46` | UI sliders match backend clamps |
| 4.14 | Availability = weights ∧ arch ∧ VRAM; show "weights missing — 28 GB" instead of a live option; real hardware labels | `video_orchestrator.py:256-265`, `model_manager.py:199-231`, `VideoInspectorDock.tsx:513-516` | No "⚡ Ready" without weights |
| 4.15 | Video preview proxy (480p) + first-frame thumbnail + regenerate action | `VideoCanvasPlayer.tsx:173-195`, render completion | Timeline scrubbing does not stream a 1080p master |
| 4.16 | Simplify the two service layers: one façade that delegates; delete the legacy OpenCV lip-sync kept alive by tests | `video_service.py:457-808`, `video_orchestrator.py:231-246` | ~1.3 k LOC removed, one truth about face images |

**Exit criteria:** every rendered clip is what its label says, audio and video end together,
and a failed render is visibly failed.

## Phase 5 — Platform & API (Weeks 10–12, ~7 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 5.1 | Split `main.py` into `app/api/v1/{jobs,generation,transcription,agents,media,projects,videos,system}.py`; `app/db.py` owns `engine`; delete `os.chdir` and the service→`main` back-imports | `main.py` (6 k LOC), `agents/orchestrator/*`, `services/video/video_orchestrator.py:1150`, `services/image_service.py:1028` | No service imports the HTTP layer; tests import modules individually |
| 5.2 | Typed request bodies for all 27 untyped routes; `Query(le=…)` bounds on every list; one `Page` dependency with a real `total` | `main.py` (many) | OpenAPI has zero `additionalProperties: true` bodies |
| 5.3 | One error envelope `{error:{code,message,details}}`; one `AppError`; override the `HTTPException` handler | `core/errors.py`, `main.py:469-512` | The frontend has one error path |
| 5.4 | Status codes: `202`+`Location` for queued, `201` for creates, `204` for deletes | all mutating routes | `POST /generate/music` → 202 |
| 5.5 | Idempotency-Key on GPU-spending POSTs (generate, download, produce, chat) | `core/idempotency.py` | Double-submit creates one job |
| 5.6 | **D2 — single-user, multi-tenant-ready**: mandatory fail-closed auth + one hashed API key with scopes; `owner_id: str = "local"` columns + a single `scoped()` query helper (so a hosted mode later is a migration, not a rewrite); **signed read-only share links** for sharing; document Team/LAN mode. No registration, roles, or per-user libraries | `models.py`, `core/security.py`, new `core/scoping.py` | Share link works without a token; a read-only key cannot delete; `owner_id` exists on every user-owned row |
| 5.7 | SSE: monotonic ids, `Last-Event-ID` replay from a bounded ring, topic filtering (`?job_id=`), signed ticket instead of `?auth=` | `main.py:5734-5763`, `music_service.py:58-77`, `api.ts:615-623` | A reconnect after sleep loses no events; tabs do not cross-talk |
| 5.8 | Rate limiting by route **template** with costs, covering separation/voice/video/training | `core/ratelimit.py:20-49` | Varying the job id does not reset the bucket |
| 5.9 | Frontend data layer: typed client with timeout + `AbortSignal` + retry, TanStack Query keys, one `jobsById` normalised store; delete module-scope caches | `api.ts` (1900 LOC), `App.tsx`, `SongsView.tsx:24`, `ProjectsView.tsx:39-40` | No duplicate fetch, no stale cache, no hung request |
| 5.10 | `useConfirm()` + toast-based errors; eliminate `alert`/`confirm` and the 12 silent `console.error` catches | `App.tsx:729,1115,1165,1224`, `ProjectsView.tsx:130-272` | Every failure is visible and dismissible |
| 5.11 | Accessibility pass: remove global `select-none` (done in 3.15), focus traps on all modals, `role="alert"` for errors, scope single-key transport hotkeys, reduced-motion | `primitives.tsx:251`, `AudioEngineContext.tsx:552-579`, `utils/toast.ts:16` | WCAG 2.2 AA on the shell; axe-clean modals |
| 5.12 | Config: atomic writes via `win_safe_files`, batch provider updates, `override=False`, `chmod 600` on `llm_config.json` | `config_manager.py:21-23,213-214` | A crash mid-save cannot truncate keys |

**Exit criteria:** versioned, typed, idempotent API; auth with scopes; SSE that survives a
laptop sleep; one error path in the UI.

## Phase 6 — Continuous hardening (from Week 1, ~1 day/week)

| ID | Task | Acceptance |
|---|---|---|
| 6.1 | CI spine: `ruff` + `mypy` on `app/`, `pytest --cov-fail-under=60`, `pip-audit`, Playwright E2E, `docker build`, `alembic upgrade head` smoke | CI green on a clean clone; a lint error fails the build |
| 6.2 | Test hermeticity: `pytest-timeout=120`, `asyncio_mode=auto` in `pytest.ini`, `HF_HUB_OFFLINE=1`, no test may touch the network or a real model (mark `@pytest.mark.heavy` and exclude by default) | Full suite < 5 min, deterministic, cannot hang |
| 6.3 | Pin everything: `uv.lock` + `uv sync --frozen`, `package-lock.json` + `npm ci` on **both** sides, Dependabot | Identical env on dev/CI/prod; the logfire/opentelemetry clash cannot recur |
| 6.4 | Align Python versions (CI 3.12 vs image 3.11); **remove the CUDA device reservation from the CPU image (D6)** — do not add a CUDA base image this cycle | The "GPU" compose profile no longer implies CUDA it does not have |
| 6.5 | Observability: request id, access log, structured JSON logs, `/health` with DB + disk + queue depth, per-job token/cost accounting (LLM usage is currently discarded on `generate_text`) | One dashboard answers "what is running and what did it cost" |
| 6.6 | Log hygiene: kill `print()`, rotate `milimo.log`/`ai_debug.log`, stop writing full lyrics to a CWD file by default | No unbounded log files |
| 6.7 | `./milimo stop` kills only recorded PIDs | No more SIGTERM to a stranger's dev server |
| 6.8 | **D6 — capability matrix enforcement**: `hardware_autotune` emits the platform matrix; the engine picker disables unavailable engines with a stated reason; README headline claims match the matrix | On a CUDA box the UI says "MiniMax requires Apple Silicon MLX" and offers fal, instead of failing at runtime |

## Phase 7 — Music video production (2026-11-18, added)

Source: [Music Video Pipeline Audit 2026-11](music-video-pipeline-audit-2026-11.md).
Effort estimates are focused engineer-days for one person who knows the codebase.
Every task names an observable acceptance criterion — several are literally a test
that failed before implementation.

> **Status: 29/29 implemented and verified on 2026-11-18.** Regression suites added:
> `test_video_truth.py`, `test_video_takes.py`, `test_video_qc.py`,
> `test_video_delivery.py`, `test_music_structure.py`,
> `test_video_incremental_retake.py`, `test_symbolic_and_manifest.py`,
> `test_video_endpoints_smoke.py`, `test_video_consistency.py`,
> `test_lipsync_registry.py`, plus `frontend/tests/video-a11y.test.ts`
> (+100 tests). Three real bugs were found by the new tests while implementing:
> `-af loudnorm` combined with `-c:a copy`, a take re-save copying a file onto
> itself, and two icon-only dialog buttons with no accessible name.

**Sequencing:** 7A must precede 7B because a musician cannot iterate on a pipeline
that lies about what it produced. 7B precedes 7C because export presets and UI
resilience amplify whatever the pipeline emits. 7D can run in parallel with 7C.

### 7A — Truth (P0, ~9 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 7.1 | **Lip-sync honesty.** Return fallback metadata from the provider instead of a bare `True`; set `fallback_used`/`fallback_reason` from it; require a weight manifest *and* an audio-driven engine for `is_available`; stop shelling `--driving_audio` at LivePortrait (it is video-driven) — either wire Wan 2.2 S2V (Apache-2.0) / EchoMimicV3 (Apache-2.0) or declare the jaw-warp fallback the product and label it. | `lip_sync/live_portrait.py`, `lip_sync/fallback.py`, `video_orchestrator.py:914-946`, `VideoInspectorDock.tsx` | No live-sync weights → task status carries `fallback_reason="procedural jaw-warp"`; test asserts the banner state, not just the boolean |
| 7.2 | **Render success verification.** `/videos/render/{job_id}` must check FFmpeg exit code and file existence before returning, emit the error envelope, and never write `video_path` on failure. | `video_service.py:951-1010`, `main.py:5498-5534` | A stubbed failing encoder produces HTTP 500 with the FFmpeg tail and leaves `video_path` unchanged |
| 7.3 | **Retake honesty + delivered-resolution reporting.** Persist retake prompt/camera/lighting server-side; mark raster fallbacks `degraded` with a reason; report the resolution actually rendered (and the upscale factor) in task status and on the card. | `main.py:6095-6161`, `models.py:906-911`, `video_orchestrator.py` (`get_video_providers`/task status) | Reload keeps retake edits; a procedural plate is flagged in the status; the HUD says `832×480 upscaled to 1920×1080` when that is the truth |
| 7.4 | **Takes and versioning.** New `VideoRender` row per render (config, status, artifacts, qc, seed, created_at) with `job.video_path` pointing at the selected take; cascade delete; UI take switcher. | `models.py`, `main.py` (`/videos/*`), `MusicVideosView.tsx` | Two renders coexist; switching takes changes the player; deleting a take removes its files |
| 7.5 | **Replace the fake A/B compare** with a real two-source player (works on 7.4's takes) and make it keyboard-operable. | `VideoCanvasPlayer.tsx:324-355` | Two `<video>` sources, divider clips both, arrow-key control |
| 7.6 | **Per-job render mutex.** 409 `render_in_progress` for a second render of the same job. | `main.py:5885-5923` | Second concurrent call returns 409; test mirrors the model-download guard |
| 7.27 | **Lip-sync engine registry.** The request's `lip_sync_engine` was ignored and LivePortrait was always instantiated. A registry now reports every engine with real availability, licence and the reason it cannot run; an unavailable choice falls back to the jaw-warp animatic *with a recorded reason*, and `GET /videos/lipsync-engines` feeds the inspector. | new `lip_sync/registry.py`, `video_orchestrator.py`, `main.py`, `VideoInspectorDock.tsx` | Choosing an engine changes what runs; the picker shows "unavailable" with the reason; tests cover the fallback contract |
| 7.7 | **Cloud cost estimate and gate.** Price table → per-render estimate in the `/videos/render-advanced` response and UI; require confirmation above a threshold. | `cloud_video.py`, `main.py`, `VideoInspectorDock.tsx` | Estimate shown before spend; no more hardcoded "Free"/"Fast" badges |

### 7B — Pipeline correctness (P1, ~14 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 7.8 | **Real downbeats and metre.** Integrate `all-in-one` (MIT) or an equivalent beat/downbeat model; fall back to beat grid + lyric anchors, and record which was used. | `audio_analysis.py:106-128` | A 3/4 fixture yields bar-start downbeats; source recorded on the plan |
| 7.9 | **Structure from the generation plan first.** Parse the tagged lyrics (`[Verse]`/`[Chorus]`/…)+ `timed_lyrics_json` into sections; librosa novelty only as fallback. | `audio_analysis.py:131-188`, `video_director.py:1021-1232` | Section labels match the generation tags; a test asserts the tag path is used when present |
| 7.10 | **Consume or delete the dead curves.** Feed `camera_zoom` into the Ken Burns/procedural branch (and the beat pulse), and either wire `vocal_envelope` into the lip-sync amplitude or delete both passes. | `stem_audio_reactive.py`, `video_orchestrator.py:829-837,992-1003` | Grep test: every curve returned by the module is read; new unit tests for the module (currently zero) |
| 7.11 | **Persist per-clip renders + incremental re-render.** Keep `clips/{task}/{idx}.mp4` and the plan JSON; retake re-renders one scene and re-stitches without touching the rest. | `video_orchestrator.py:903-1030,1219-1222`, `main.py:6095` | A one-scene retake renders ~1/N of the work; test asserts other clip mtimes are unchanged |
| 7.12 | **One quality-targeted encode.** Lossless (or CRF ≤12) intermediates, then exactly one delivery encode; drop the `ultrafast` pulse pass into the normalise filter. | `clip_normalise.py:78-86`, `video_orchestrator.py:1566-1574`, `stem_audio_reactive.py:167-175` | `ffprobe` shows the preset's target bitrate; no more than two encode generations end-to-end |
| 7.13 | **Master delivery encoding.** `-movflags +faststart`, audio passthrough when the source is AAC else ≥320 kbps, loudness to −14 LUFS / TP −1 dB, measured value persisted. | `video_orchestrator.py:1126-1191` | moov-before-mdat assertion test; `ebur128` within ±1 LU of target; loudness stored on the render row |
| 7.14 | **QC gate.** New `services/video/qc.py`: `blackdetect d=0.1`, `freezedetect d=0.05`, `silencedetect d=0.1`, A/V offset by cross-correlation (≤1 frame), `ebur128`, frame-count check; store `qc_json`; surface pass/warn/fail. | new module + `video_orchestrator.py` + UI | Planted-fault fixtures (1 s black, 5-frame black, 1.5 s freeze, audio dropout) are all caught — FFmpeg defaults catch 1 of 6 |

| 7.28 | **Render reconciliation at boot.** A killed server left `videorender` rows in `rendering` forever, so the UI showed a take that never completed. Boot now marks them failed with a stated reason. | `services/video/takes.py`, `main.py` (lifespan) | A crash mid-render yields a visible terminal state, not a ghost |
| 7.29 | **Chain scope.** `chain_scope: section \| all` keeps continuity inside a verse/chorus and resets at a section change, so `chain` mode no longer makes the whole video feel like one shot. | `services/video/consistency.py`, `video_orchestrator.py`, `models.py` | Same-section cuts chain; a chorus does not inherit the verse frame |
| 7.26 | **Subject consistency (audit E6).** `consistency_mode: off \| character \| chain`: a persisted character bible injected once into every scene prompt, and (in `chain`) the previous clip's final frame as the next scene's i2v seed. Approved plates always win. | new `services/video/consistency.py`, `video_orchestrator.py`, `VideoInspectorDock.tsx`, `models.py` | Two consecutive scenes share the stated identity; re-stamping is idempotent; the mode is visible in the inspector |

### 7C — Delivery and workflow (P2, ~12 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 7.15 | **Export presets.** Table-driven: YouTube 1080p/2160p, Vevo HD (20 Mbps, no slate), Canvas (3–8 s, 9:16, silent), broadcast ProRes. | new `services/video/export_presets.py`, `video_orchestrator.py`, `VideoTopBar.tsx` | Each preset passes ffprobe assertions (codec, bitrate, resolution, duration, audio rate) |
| 7.16 | **Caption sidecars.** Always write SRT/VTT (+ ASS for the advanced path) beside the render; burn-in becomes independent of sidecar emission. | `video_orchestrator.py:1076-1097`, `karaoke.py:620` | Sidecars exist for every render, parse, and match burned timings |
| 7.17 | **Multi-aspect variants.** One plan → 16:9 + 9:16 + 1:1 masters. | `video_orchestrator.py:801-808`, `main.py` | Three masters, durations equal within 1 frame |
| 7.18 | **Frontend resilience.** Server-side task lookup by job (`GET /videos/tasks?job_id=`), resume after reload, SSE instead of 1 s polling, attempt caps/backoff/timeouts, real error text. | `main.py`, `MusicVideosView.tsx:400-522,847-878`, `api.ts` | Reload mid-render restores the HUD; a dead backend surfaces "connection lost" instead of spinning |
| 7.19 | **Config fidelity.** Persist and restore the *full* render config (drop the style/preset whitelists), schema-validate on regenerate. | `MusicVideosView.tsx:1166-1198`, `api.ts` | All 15 styles and 6 presets round-trip through regenerate |
| 7.20 | **Wire the provider picker.** Call `/videos/providers`, show availability, key state and cost; remove the phantom `kling` claim. | `VideoInspectorDock.tsx:498-535`, `video_orchestrator.py:272-307` | No hardcoded capability badges; unavailable providers are disabled with a reason |
| 7.21 | **Accessibility for the video view.** Labels, dialog semantics/focus trap/Escape, keyboard seek, reduced-motion, ≥10 px load-bearing type. | `VideoTopBar.tsx`, `ClipRetakeModal.tsx`, `KeyframeZoomModal.tsx`, `VideoTimelineTrack.tsx`, `VideoCanvasPlayer.tsx`, `utils/useReducedMotion.ts` | Keyboard-only path from song pick → plan → render → download; **verified by `frontend/tests/video-a11y.test.ts`** guarding labels, dialog semantics, Escape, keyboard-seeking blocks, the 10 px floor and reduced-motion coverage (a browser axe run remains a manual step) |

### 7D — Reporting and reproducibility (P2, ~8 days)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 7.22 | **Artifact manifest.** One builder (`app/core/artifact_manifest.py`) → `GET /jobs/{id}/manifest.json`: seed, params, model id + version + git hash, verbatim prompt/lyrics, stage provenance, artifact SHA-256s, measured loudness. | new module, `main.py`, `models.py` | Manifest validates against its own schema; every field is populated for a neural and a degraded job |
| 7.23 | **Provenance in packs.** Add `provenance_json`, `quality`, model version to `metadata.json`; ship `mastered_path` when present; add a stem manifest. | `main.py:4645-4727, 5305-5396, 1069-1124` | A pack opened by someone else states what produced each stem; mastered audio is the one inside the pack |
| 7.24 | **MusicXML correctness.** Emit measured key/`fifths`, time signature, computed `divisions`, real encoding date. | `muscriptor_provider.py:445-620`, `minimax_provider.py:308-389` | Exported score round-trips key and metre through a validator |
| 7.25 | **Video report + cut list, and audio-only bounce.** Storyboard JSON + per-scene table (prompt, duration, provider, seed, qc) exportable with the MP4; `POST /api/editor/projects/{id}/bounce` for a WAV/FLAC mixdown. | `main.py`, `services/timeline/render.py:144-193`, `editor_projects.py:160-310` | A video ships with a reproducible cut list; the NLE can produce a mixdown |

### Phase 7 exit criteria

- Nothing in the video path reports success while doing something else (7.1–7.6).
- A single scene can be re-rendered and re-assembled without re-diffusing the film (7.11).
- The delivered master is faststart, loudness-compliant, encoded at a targeted bitrate, and passes an automated QC gate that catches sub-second faults (7.12–7.14).
- At least one DSP-shaped preset and a caption sidecar exist (7.15–7.17).
- A finished session can be handed to someone else as a manifest + report + cut list (7.22–7.25).

## Phase 8 — Capability upgrades (2026-11-18, added)

Source: [Video Capability Landscape 2026-11](video-capability-landscape-2026-11.md).
Phase 7 made the pipeline honest; this phase makes it capable. Every adoption task
is gated on a **smoke test on target hardware** — vendor VRAM/timing claims are
not accepted as evidence, and a failure is recorded in the capability matrix rather
than silently degrading.

> **Status: 8/19 implemented, 1 partial.** Shipped: plate verifier + reshoot (8.1,
> now with optional CLIP semantics), character library (8.3), external avatar tiers
> (8.4 — EchoMimicV3, Wan 2.2 S2V and InfiniteTalk contracts), licence gate in CI
> (8.9), capability matrix (8.10), render stall sweeper (8.12), viseme-driven Tier 0
> (8.13), and motion-compensated delivery interpolation (8.6).
> The [caveat investigation](caveat-investigation-2026-11.md) then added 8.15–8.19,
> which are **code** gaps (not weight gaps) discovered while verifying the rest.

### 8A — Identity and consistency (G3, G6, G7)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| ✅ 8.1 | **Verifier + reshoot loop.** Generate 2–3 candidate plates per scene, score identity/instruction adherence against the character bank, reshoot the loser. AutoMV's ablation credits this with the largest measured consistency gain — with no identity model. | new `services/video/verifier.py`, `image_service.pregenerate_scene_backgrounds`, `video_orchestrator._prebuild_scene_plates` | A deliberately off-character candidate is rejected and re-rolled; the accepted plate records its score |
| 8.2 | **Reference → keyframe → I2V.** `generate_scene_background(reference_images=[...])` using **FLUX.2 klein 4B** (Apache-2.0, up to 4 refs); the character plate conditions every scene still. | `services/image_service.py`, `video_orchestrator.py` | Same character id ⇒ plates that pass 8.1 without prompt changes |
| ✅ 8.3 | **Character library.** `Character` row (plate, description, optional LoRA), CRUD, `character_id` on render config, plate reused across songs. | `models.py` + Alembic, `main.py`, video UI | Two different songs using one character share the plate hash |
| 🟡 8.4 | **Avatar provider interface + tier ladder** (3 of 5 engines wired). One `generate(image, audio, prompt, duration, fps)` contract with per-provider metadata (`licence`, `territory_blocked[]`, `min_vram_gb`, `fps`), so routing is a check rather than a doc note: `procedural` (tier 0), **`echomimic_flash`** (tier 1, Apache-2.0, 12 GB, 25 fps, `infer_flash.py`), `infinitetalk` / `longcat_mlx` (tier 2), `wan_s2v` / `liveavatar` (tier 3, 80 GB). | new `lip_sync/providers/`, `lip_sync/registry.py`, `lip_sync/base.py` | An EU user on an 8 GB card can never be routed to Hunyuan-Avatar; each provider declares licence + VRAM + fps |
| ✅ 8.13 | **Tier-0 viseme timing.** Drive the jaw-warp from the MMS forced-alignment word timings we already compute for subtitles, instead of an RMS envelope. | `lip_sync/fallback.py`, `transcription/karaoke.py` | The always-available path opens the mouth on syllables, not on loudness |
| 8.14 | **Wan VACE reference conditioning** (14B tier). `WanVACEGenerator` passing `reference_images`; registry/model-spec entry. | new `generators/diffusers_vace.py`, `model_specs.py`, `model_manager.py` | Conditioned clip keeps identity where the text-only clip drifts; refuses honestly without weights |

### 8B — Resolution and timing (G2, G5)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 8.5 | **FastMetal Wan 2.2 TI2V-5B** (Apache-2.0, 1280×704 @24 fps). New generator variant + catalogue entry; remove the 832×480 MPS clamp for this path. | `generators/diffusers_wan.py`, `model_specs.py`, `model_manager.py` | Smoke test measures wall-clock and peak memory on the target Mac; a failure is recorded, not hidden |
| ✅ 8.6 | **Interpolation quality** (MIT) replacing `minterpolate`. **8.6a shipped:** `minterpolate` blend keeps 16 fps Wan 2.1 out of the delivery path; it is a stopgap and is labelled as such. | `services/video/clip_normalise.py`, `video_orchestrator._finalise_master` | A 16→24 conversion shows no duplicated-frame runs; the master records which backend ran |
| 8.7 | **1080p uplift stage**: FlashVSR+ (CUDA) / SeedVR2-3B via MLX (Mac) as an optional post-process with honest unavailability. | new `services/video/upscale.py`, delivery presets | A 720p master reaches 1080p with temporal consistency; unavailable hardware says so |

### 8C — Motion (G4)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| 8.8 | **Motion transfer** via `WanAnimatePipeline` from a user-supplied driving clip; registry entry labelled "motion transfer (video-driven)", never "singing". | new generator, `lip_sync/registry.py` naming, UI | A dance clip drives the character; the UI never conflates it with lip-sync |

### 8D — Truth and operations (G8, G9, G10)

| ID | Task | Files | Acceptance |
|---|---|---|---|
| ✅ 8.9 | **Licence gate (shipped, CI-enforced).** `core/licensing.py` + `GET /system/licences` classify every model and report blockers (currently `flux_2_klein_9b`, `flux-non-commercial-license`). **8.9b:** CI fails when a reachable default is non-commercial/unknown unless explicitly acknowledged; **8.9c:** territory blocks (`EU/UK/US/South Korea` exclusions) become route-level checks, not prose. | `core/licensing.py`, CI, `.env.example`, avatar registry | A non-commercial or territory-blocked default cannot ship unnoticed |
| 8.15 | **Attach a reference plate to a character (G14).** Creation from the UI sends only a description, so `has_plate` is false for every UI-created character and the verifier has nothing to compare against. | `VideoInspectorDock.tsx`, `MusicVideosView.tsx`, `characters.py` | A UI-created character carries a plate, and the verifier reports a real score |
| 8.16 | **Reference-conditioned scene stills (G11).** `generate_scene_background(reference_images=[...])` plumbed to the image backend that supports it; honest "backend cannot condition" otherwise. | `services/image_service.py`, `video_orchestrator.py` | The character plate visibly influences the plate, not just the score |
| ✅ 8.17 | **Plan/render config agreement (G12).** `/videos/plan` accepts `character_id`, `consistency_mode`, `chain_scope` and applies them, so the approved storyboard matches the render. | `VideoPlanParams`, `video_director.py`, `main.py` | Approving a storyboard then rendering produces the same scene descriptions |
| ✅ 8.18 | **Real export cancellation (G13).** Run delivery transcodes as killable subprocesses; a cancelled export stops FFmpeg and leaves no take. | `services/video/deliver.py`, `main.py` | Cancelling an export stops CPU within 2 s and creates no take row |
| ✅ 8.19 | **Dependency licence audit (G16).** Classify installed Python packages too, not just catalogue models. | `core/licensing.py`, `scripts/check_licences.py` | An InsightFace-class dependency appears in the gate output |
| ✅ 8.10 | **Video capability matrix** endpoint + UI: per engine, required VRAM/resolution/fps and measured status, so the ceiling is visible *before* a render. | `model_specs.py`, `/videos/models`, `VideoInspectorDock.tsx` | The picker states "720p needs CUDA" instead of failing after 40 minutes |
| 8.11 | **Ground-truth the cloud path.** One real fal render with a key; refresh the price table from the provider's docs. | `pricing.py`, script | Estimate within 20% of the invoice; the path is no longer theoretical |
| 8.12 | **Render stall sweeper.** ✅ shipped — `sweep_stalled_renders()` fails a take that has made no progress within `MILIMO_RENDER_STALL_SECONDS` (default 1 h), in the existing sweeper loop. | `services/video/takes.py`, `main.py` | A wedged render reaches a terminal state instead of showing progress forever |

### Phase 8 exit criteria

- A 720p/24 fps master is produced **natively** on the reference Mac, with measured time and memory.
- Identity holds across a chorus without prompt-level tricks, and the mechanism is reusable across songs.
- The singing avatar is a real model on at least one supported path, with `jaw_warp` as the stated floor.
- No reachable default is non-commercial or unclassified without a written acknowledgement.
- The capability matrix answers "can this machine do that?" before the render starts.

## Phase 9 — Performance (2026-11-18, added)

Source: [Performance Audit 2026-11](performance-audit-2026-11.md). One user, one
machine: **bandwidth is free, CPU/GPU seconds are not.** Every task below replaces a
belief with a measurement, and the four popular-but-wrong moves (more workers,
`aiosqlite`, `orjson`, blanket compression) are recorded as *rejected* with evidence.

> **Status: 7/12.** Shipped: conditional GET + catalogue memo + boot warm (9.1),
> FFmpeg thread cap (9.2), compression decision (9.3), lazy LLM SDK imports (9.4),
> measured encoder selection (9.5), the benchmark harness (9.9) and parallel clip
> normalisation (9.10).
> Measured: boot **4.64 s → 3.83 s**; model tree 1 103 ms → 4 ms; 304s 1.0–1.4 ms;
> encoder benchmark picks x264 (0.21 s) over VideoToolbox (0.31 s) for 1 s of 1080p.
> **9.8 shipped** as `scripts/bench_sqlite.py` (measured; knobs left unset because the
> gain did not appear). Still open: SSE progress (9.6), list virtualisation (9.7) and
> the React Compiler / parse budget (9.11) — all product-surface work.

| ID | Task | Files | Acceptance |
|---|---|---|---|
| ✅ 9.1 | **Conditional GET + catalogue memo.** ETag/304 on the five expensive read endpoints; model tree memoised for 15 s with a generation counter bumped on install/delete/activate; boot warms the catalogue. | `core/perf.py`, `services/model_manager.py`, `main.py` | A repeat read returns 304 in <5 ms; a new install appears immediately |
| ✅ 9.2 | **FFmpeg thread budget.** `-threads 4` by default (`MILIMO_FFMPEG_THREADS`, 0 = unlimited) on delivery and normalise. | `export_presets.py`, `clip_normalise.py`, `deliver.py` | Commands carry the cap; the app stays responsive during a render |
| ✅ 9.3 | **Compression decision.** Documented as *not* a win on loopback; middleware kept opt-in for LAN clients with streaming paths excluded. | `core/perf.py`, `main.py` | Default responses have no `content-encoding`; `/events` never compressed |
| ✅ 9.4 | **Lazy LLM SDK imports.** `llm_service`, `google.genai`, `openai`, `anthropic` are imported eagerly and cost ~1.3 s of a 4.6 s boot. | `services/llm_service.py`, `main.py` | Cold import drops by ≥1 s; first agent call pays it instead |
| ✅ 9.5 | **Measured encoder policy.** Benchmark x264 vs VideoToolbox once per machine, cache the winner, and default to the one that finishes sooner when the user is waiting. | new `services/video/encoder_bench.py`, `subtitle_styles.py` | The recorder reports which encoder ran and why; no "hardware" claim without a measurement |
| 9.6 | **SSE progress instead of 1 s polling.** The video view polls; `/events` already has replay. | `MusicVideosView.tsx`, `video_orchestrator.py` | A render reports progress with zero repeat requests while idle |
| 9.7 | **Virtualise long lists + keep the playhead out of React state.** | `SongsView.tsx`, `VideoTimelineTrack.tsx` | 10 k-row lists mount in <100 ms; scrubbing does not re-render the tree |
| 9.8 | **SQLite knob measurement.** Benchmark `cache_size`/`mmap_size`/`temp_store` before adopting any of them. | `db.py`, benchmark script | Either a measured win is kept, or the knobs stay unset with a note |
| ✅ 9.9 | **Performance harness in CI.** `hyperfine` on two hot endpoints, a bundle-size budget, and a `py-spy` recipe documented for hang diagnosis. | `.github/workflows/ci.yml`, `scripts/bench.sh` | A >30 % latency or >10 % bundle regression fails the build |
| ✅ 9.10 | **Parallel clip normalisation + concat join.** Clips are independent; normalise them concurrently and join with the demuxer instead of one giant graph. | `video_orchestrator.py`, `clip_normalise.py` | A 20-clip assembly spends materially less wall time at the same quality |
| 9.11 | **Frontend parse budget.** React Compiler evaluation and a `rollup-plugin-visualizer` report before any further manual chunking. | `vite.config.ts`, docs | A measured parse-time number, not a chunk-size guess |
| 9.12 | **Startup budget.** Defer `torch` until the first tensor is needed; measure boot to first served request. | `main.py`, `core/hardware_lock.py` | Boot budget documented and enforced by the harness |

### Phase 9 exit criteria

- Every performance claim in the docs cites a measured number on target hardware.
- A regression in endpoint latency or bundle size fails CI.
- Renders do not starve the editor (thread cap + a stated encoder policy).
- Idle CPU is ~0 when nothing is rendering (no polling).

## Phase 10 — Memory (2026-11-18, added)

Source: [Memory Management Audit 2026-11](memory-management-audit-2026-11.md). The
sibling of Phase 9: on a unified-memory machine, "how fast" and "what is resident" are
the same question. Everything here was measured before it was changed; the largest
finding (MLX's cache default exceeding physical RAM) came from reading the installed
library rather than a blog.

> **Status: 11/12.** Shipped: footprint telemetry + `/system/memory` (10.1), MLX/MPS
> caps (10.2), bounded Wan pipeline cache (10.3), budgeted agent history (10.4), WAL
> size limit (10.5), one release path (10.6), WAL checkpoint job (10.7), memory trend +
> leak verdict (10.8), process hygiene in `/health` (10.10), peak reset per load (10.11)
> and the model budget guard (10.12). Remaining: per-agent memory namespaces (10.9) —
> a product feature, not plumbing.

| ID | Task | Files | Acceptance |
|---|---|---|---|
| ✅ 10.1 | **Footprint telemetry.** Report `phys_footprint` (Metal-aware) beside RSS, plus MLX active/cache/live/peak and MPS allocated/driver. | `core/memory.py`, `main.py` | `/system/memory` shows a footprint that moves when accelerator memory is released (RSS does not) |
| ✅ 10.2 | **Accelerator caps at boot.** MLX cache capped (`min(12 GiB, 10 % RAM)`), MPS fraction capped (0.6) before the first tensor. | `core/memory.py`, `main.py`, `video_orchestrator.py` | A model unload returns memory instead of parking it in a 161 GiB cache |
| ✅ 10.3 | **Bounded pipeline cache.** Wan pipelines evicted oldest-first (default 1), hooks removed, moved off device. | `generators/diffusers_wan.py` | Rendering 14B then 1.3B does not keep both resident |
| ✅ 10.4 | **Budgeted agent history.** Character + message budget, system turn preserved, truncation reported to the caller. | new `core/agent_memory.py`, `main.py` | A 200-turn session cannot exceed the prompt budget; the response says what was dropped |
| ✅ 10.5 | **WAL size limit.** `journal_size_limit=64 MiB` on the engine and the task queue. | `main.py`, `core/task_queue.py` | A long reader cannot grow the `-wal` file without bound |
| ✅ 10.6 | **One release path.** `release()` clears MLX + MPS pools, reports footprint before/after, and collects cycles only when asked. | `core/memory.py`, `core/hardware_lock.py` | `gc.collect()` is gone from hot paths; a deprecated `mx.metal` call stays gone (test) |
| ✅ 10.7 | **WAL checkpoint job + session hygiene.** Periodic `wal_checkpoint(TRUNCATE)`; audit for `Session` objects held across LLM/FFmpeg calls. | `core/task_queue.py`, `main.py` | The `-wal` file returns to ~4 MiB after a burst |
| ✅ 10.8 | **Memory ring buffer + slope alarm.** Sample footprint/MLX/MPS per minute; flag a positive slope *at idle*. | new `core/memory_guard.py` | `/system/memory/history` shows a trend; a synthetic leak trips it |
| 10.9 | **Per-agent memory namespaces.** Separate the critic's context from the songwriter's; distill durable facts instead of replaying turns. | `agents/runtime/context.py`, `agents/*` | One agent's 200 turns do not enter another's prompt |
| ✅ 10.10 | **Process hygiene in `/health`.** fds, threads, zombie children; `-nostdin` + `wait()` audit for every FFmpeg spawn. | `main.py`, `services/video/*`, `transcription/*` | A zombie or fd leak is visible in the health payload |
| ✅ 10.11 | **Peak reset per model load.** `reset_peak_memory()` after a load so peak describes one job, not the lifetime. | `core/memory.py`, generators | Peak readings stay interpretable across model swaps |
| ✅ 10.12 | **Model budget guard.** Compare requested size against `max_recommended_working_set_size` and refuse with a reason instead of swapping. | `services/model_manager.py`, registry | Loading an 80 GB model on a 32 GB box fails fast with a stated reason |

### Phase 10 exit criteria

- The app can state its own memory (footprint, accelerator split, swap) at any moment.
- Nothing grows without a bound: caches, histories, WAL, task registries.
- A leak is detectable as a slope, not as a crash.
- An unload actually returns memory to the OS on both MLX and MPS paths.

## Decisions — closed (D1–D6)

All six scope questions are resolved in [Decision Record 2026-11](decisions-2026-11.md).
Summary and where each lands in this plan:

| # | Decision | Where it lands |
|---|---|---|
| D1 | Delete YuE2 + Training Studio | 0.14 (deletion), 1.4 (closed) |
| D2 | Single-user, multi-tenant-ready | 5.6 (shrunk to ~1 week) |
| D3 | Keep HeartMuLa (demoted) + MuLaCover, de-stub both | 0.15, 1.6, 1.8 |
| D4 | Cloud = fal only | 4.5 |
| D5 | Automation yes, time-stretch no | 3.16, 3.17 |
| D6 | No CUDA generation; publish + enforce matrix | 6.4, 6.8 |

Two claims behind D3/D4 are structural rather than tested and should be smoke-tested before
budget is committed: an end-to-end **fal** clip (no key has ever been configured) and one
end-to-end **MuLaCover** cover generation.

## Dependency graph

```
Phase 0 (bleeding + D1/D3 deletions)
   ├── Phase 1 (provenance) ──────────────┐
   └── Phase 2 (spine: cancel/claim/Alembic/FKs)
             ├── Phase 3 (DAW core + D5 automation) ─┤
             ├── Phase 4 (media truth + D4 fal) ─────┤
             └── Phase 5 (platform/API + D2 auth) ───┴── Phase 6 (continuous + D6 matrix)
                                                            │
                                     Phase 7A (truth: 7.1–7.7) ── Phase 7B (7.8–7.14)
                                                            └── Phase 7C (7.15–7.21)
                                                                Phase 7D (7.22–7.25)  ← independent
```

Phase 8 depends on Phase 7 (the delivery path and the truth layer it audits) and on
hardware availability for its smoke tests. Inside Phase 8, 8.1 precedes 8.2 (the
verifier scores what the reference conditioning produces), and 8.5 precedes 8.7
(upscaling is only worth wiring once 720p is native).

Phase 7A depends on Phase 4 (it reopens 4.8 and extends the polyphony of the media
pipeline); 7B assumes 7A's honesty so quality work is measurable; 7C amplifies
whatever 7B emits; 7D only needs Phase 1's provenance records, so it can start
immediately.

Phase 5.1 (router split) is deliberately last-ish: doing it earlier would make every other
phase a merge conflict. The exception is `app/db.py`, which is a 1-hour change and should
happen in Phase 0.

## Definition of done for the whole programme

- `quality != 'neural'` is a first-class, queryable, UI-visible property of every artifact.
- A capability manifest can only claim what a test proves.
- No code path can write a model artifact it did not train (D1).
- Kill -9 at any point leaves no lost work, no stuck row, and no orphan file.
- `alembic upgrade head` is the only way the schema changes.
- The Arrange tab edits audio; undo covers the workspace; tempo and meter are real;
  automation curves are audible (D5).
- Video duration equals song duration within 50 ms **and** Σ clip frames equals the song's
  frame count at delivery fps.
- No video artifact reports success unless its file exists and its QC gate passed.
- A single scene can be re-rendered and re-assembled without re-rendering the film.
- Every delivered master is faststart, loudness-measured, caption-complete, and has a
  reproducible manifest/cut list.
- The published capability matrix matches what the app does on macOS, CUDA and CPU (D6).
- CI blocks all of the above from regressing.

## Related

[Decision Record 2026-11](decisions-2026-11.md) ·
[Full Application Audit 2026-11](application-audit-2026-11.md) ·
[Music Video Pipeline Audit 2026-11](music-video-pipeline-audit-2026-11.md) ·
[Production Readiness Plan](production-readiness-plan.md) ·
[UI/UX Plan](ui-ux-plan.md) ·
[Roadmap](roadmap.md) · [Architecture](architecture.md) · [Index](index.md)
