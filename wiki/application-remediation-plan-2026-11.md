---
title: Application Remediation Plan 2026-11
type: overview
tags: [plan, remediation, security, honesty, lifecycle, migrations, daw, video, ci]
created: 2026-11-16
updated: 2026-11-16
sources: [application-audit-2026-11.md, decisions-2026-11.md]
aliases: [remediation plan nov 2026, audit plan, fix plan v3]
---

# Application Remediation Plan — 2026-11

Implementation plan for the findings in
[Full Application Audit 2026-11](application-audit-2026-11.md), with scope fixed by
[Decision Record 2026-11](decisions-2026-11.md). Ordered by **value ÷ risk**, with a hard
dependency spine so each phase leaves the app shippable.

Every task names the files to touch and an observable acceptance criterion. Effort is in
focused engineer-days.

## Implementation status (2026-11-16)

| Phase | Status | Evidence |
|---|---|---|
| **0 — Stop the bleeding** | ✅ **Complete** (16/16) | `415 → 449` tests green; exploit chain closed; YuE2 + Training Studio gone |
| **1 — Provenance contract** | ✅ **Complete** (18/18) | `provenance.py`, `quality` on Job + SSE, honest transcription/voice/separation/image/video |
| **2 — Spine** | 🟡 **Part 1 done** (9/15) | queue integrity, one recovery policy, stall sweeper, **Alembic + backups**, indexes/unique constraints, disk guard, lock ownership |
| **3 — DAW core** | 🟡 **Bug-fix subset done** (5/17) | code splitting (**1.25 MB → 568 kB entry**), per-view ErrorBoundary, guarded+memoized parses, A/B audition fix, honest progress |
| **4 — Media truth** | ⬜ **Not started** in code (1.15/1.16 landed in Phase 1) | engine honesty + refusal is in; lattice trim, storyboard persistence, real compile remain |
| **5 — Platform & API** | ⬜ **Not started** | decision-dependent auth landed in 0.3; typed bodies/envelope/SSE replay remain |
| **6 — Hardening** | 🟡 **Part done** | CI lint/audit/migration-smoke/E2E jobs, `ruff` bug-class gate clean, pytest-timeout + offline env |

**Verification at time of writing:** backend `449 passed, 2 skipped, 3 deselected`
(0 failures, 104 s — the suite previously hung indefinitely); frontend `tsc -b` + `vite
build` green; live server on `:8010` migrated to Alembic `0003` and serving
`/health` with db/disk/queue.

### Deferred — with reasons, not silently dropped

| Task | Why deferred |
|---|---|
| 2.1 music jobs onto the durable queue | Needs a worker loop + claim semantics; the recovery policy and CANCELLED state it depends on are now in place, so this is the natural next step |
| 2.2 bounded retries | Columns (`attempts`, `max_attempts`) ship; the retry classifier needs per-provider error taxonomy work |
| 2.8 one UUID encoding | Requires a data-repair pass over existing rows; must follow a verified backup |
| 2.9 real foreign keys | SQLite needs table rebuilds + orphan repair; do it behind Alembic batch mode with the backup path now available |
| 2.11 `job_artifact` table + soft delete | Touches every delete path; better after 2.9 |
| 3.1–3.5 clip editing, workspace undo, tempo map, loop accuracy, piano-roll tools | ~14 days of DAW work; the highest-value remaining item, untouched because it needs a sustained build |
| 4.1–4.16 media truth | Lattice trim + assembly normalisation are the two that matter most; the engine-honesty half already shipped |
| 5.1–5.12 API/platform | Router split deliberately last (merge-conflict cost); one error envelope and typed bodies are the next-best slice |
| 6.3 exact dependency pinning | `uv.lock` generation needs a full resolution run; `npm ci` is already used in CI |

> [!IMPORTANT] **Scope is closed (D1–D6).** YuE2 and the Training Studio are deleted, not
> fixed; the product is **single-user with multi-tenant-ready columns**; HeartMuLa is kept
> but demoted and MuLaCover is kept and de-stubbed; cloud video is **fal only**; automation
> lanes are in, time-stretch is out (replaced by tempo-locked generation); CUDA is **not** a
> generation target, and a published capability matrix replaces the implied promises.
> Net effect vs. the first draft: **+≈2 days, −≈6–8 weeks**. See
> [D1–D6](decisions-2026-11.md) for the evidence behind each.

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
| 4.1 | Frame-lattice correctness: render `model_render_duration`, trim/pad each clip to `target_duration`, assert `|Σ clips − song| < 0.05 s`, re-time otherwise | `director_music_timing.py:194-216`, `video_orchestrator.py:1071,1391-1400` | A 40-clip video ends exactly with the song |
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
```

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
- Video duration equals song duration within 50 ms.
- The published capability matrix matches what the app does on macOS, CUDA and CPU (D6).
- CI blocks all of the above from regressing.

## Related

[Decision Record 2026-11](decisions-2026-11.md) ·
[Full Application Audit 2026-11](application-audit-2026-11.md) ·
[Production Readiness Plan](production-readiness-plan.md) ·
[UI/UX Plan](ui-ux-plan.md) ·
[Roadmap](roadmap.md) · [Architecture](architecture.md) · [Index](index.md)
