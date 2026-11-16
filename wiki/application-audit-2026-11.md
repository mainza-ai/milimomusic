---
title: Full Application Audit (2026-11-16)
type: overview
tags: [audit, gaps, security, reliability, daw, honesty, video, image, ops, ci]
created: 2026-11-16
updated: 2026-11-16
sources: []
aliases: [full audit, app audit nov 2026, gap report v3]
---

# Full Application Audit — 2026-11-16

Whole-application audit of Milimo Music: every backend route, the generation and
transcription pipeline, the DAW workspace, the video/image studios, persistence, and
ops. Six parallel deep-dives plus direct verification of every P0 claim.

Companion plan: [Application Remediation Plan 2026-11](application-remediation-plan-2026-11.md).
Supersedes the status snapshots in [Production Readiness Audit](production-readiness-audit.md)
and [UI/UX Audit](ui-ux-audit.md) — many of their findings are now fixed, and several
of those fixes introduced or exposed new defects recorded here.

## Verdict

**Breadth is genuinely exceptional; trust and lifecycle are still the weak axis.**
The product ships an enormous amount of real capability — sample-locked multitrack
transport, neural transcription, stem separation, forced-aligner karaoke sync, album
agents, a video director, a gallery, a project model. Roughly a third of it is
*provably not doing what its label says*, and the failures cluster in the same three
places every time: **fallbacks that lie about provenance**, **lifecycle states that no
one owns**, and **contracts (schema, API, capability) that are declared but not
enforced**.

Compared with the 2026-08-21 audit, the *number* of dishonest fallbacks has gone **up**,
not down: the original leak (synth audio) was fixed with a `used_fallback_synth` flag,
but the same pattern was never applied to transcription, voice conversion, image
generation, lip sync, or the YuE2 provider — and one provider actively sets the flag
to `False` while synthesizing.

## Health baseline (measured)

| Metric | Value |
|---|---|
| Backend Python (app only) | 36,824 LOC / 166 files |
| Largest backend file | `backend/app/main.py` — 6,004 LOC |
| HTTP routes | **162**, all on `app`, zero `APIRouter` |
| Routes with `response_model` | 13 / 162 |
| Frontend TS/TSX | 35,606 LOC / 66 files |
| Largest frontend file | `frontend/src/App.tsx` — 2,406 LOC |
| Frontend unit tests | **0** |
| E2E specs | 2 files / 255 LOC — **not run in CI** |
| JS shipped | **1.26 MB single chunk** (`dist/assets/index-*.js`), zero `React.lazy` |
| Backend tests | 53 files |
| Migrations | **none** (no Alembic; ad-hoc `ALTER TABLE` + primary-key rewrite on every boot) |
| Auth | single optional bearer token, **off by default** |
| Generated media on disk | `generated_audio` 8.5 GB + `backend/generated_audio` 5.5 GB + `data` 2 GB |

## Test suite status (measured)

Run with the only working plugin shape
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -p pytest_asyncio.plugin`, `HF_HUB_OFFLINE=1`):

| Result | Count |
|---|---|
| Passed | **348** (313 in the main run + 35 for `test_video_service`/`test_video_pipeline`) |
| Skipped | 2 |
| Failing (deterministic) | **7** — all in `test_scene_plates.py` |
| Hanging | **3** — in `test_production_v2.py` |
| Cannot collect | `test_mulacover.py` (`pretty_midi` not installed) |

Specific defects:

- **The suite cannot collect at all** in this environment. An installed `logfire` plugin
  raises `ImportError: cannot import name 'ReadableLogRecord' from opentelemetry.sdk._logs`.
  Unpinned dependencies make the dev env mutually inconsistent. `-p pytest_asyncio` alone
  does not register `asyncio_mode`, so 98 async tests fail with "async def functions are
  not natively supported"; `pytest.ini` declares no `asyncio_mode`.
- **Three tests in `test_production_v2.py` do real model / network work and hang
  indefinitely** with 0 % CPU:
  `test_cover_image_generation_with_flux` (drives `AutoPipelineForText2Image.from_pretrained`
  through `/generate/cover-image`, `image_service.py:437`),
  `test_huggingface_search_and_custom_model` (live `GET /models/search?q=flux`, no mock),
  and `test_image_service_raster_png_generation` (`generate_cover(...)` → diffusion).
  There is **no `pytest-timeout`**, so CI can hang until the job limit; `ci.yml` sets no
  offline flags.
- **Seven tests in `test_scene_plates.py` fail deterministically on `main`**:
  `_pregen` now passes `cancel_check=` but the test's `fake_render` signature was never
  updated, so all 7 abort with `TypeError` that `image_service.py:943` swallows as a
  warning. The regression guards for the scene-plate cache are therefore dead, and
  **CI cannot be green as committed**.
- CI runs backend pytest + a frontend build. No lint, no types, no coverage gate, no
  security scan, no E2E, no Docker build, no `alembic upgrade` smoke. Python 3.12 in CI vs
  3.11 in the Dockerfile.
- CI runs backend pytest + a frontend build. No lint, no types, no coverage gate, no
  security scan, no E2E, no Docker build. Python 3.12 in CI vs 3.11 in the Dockerfile.

## A. P0 — Trust failures (the product lies about its own output)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| A1 | **YuE2 "48 kHz 3B foundation model" is a 4-chord sine synth** and returns `used_fallback_synth=False` while `initialize()` only sets a flag | `providers/yue2_provider.py:110-140, 217-240` | Advertised flagship open engine generates a sine arpeggio and *falsifies the provenance flag built to prevent exactly this* |
| A2 | **MuScriptor failure writes a hardcoded I–vi–IV–V MIDI + MusicXML at 124 BPM / C major** and stores it as a real transcription | `transcription/muscriptor_provider.py:376-473`; consumed at `orchestration/pipeline.py:450-453` | Every transcription failure silently becomes 8 s of fake notation in the piano roll and notation view; `TranscriptionResult` has no `used_fallback` field |
| A3 | **Voice conversion "succeeds" as a fixed EQ shelf, then as `shutil.copyfile`** — no conversion flag anywhere | `services/voice_service.py:409-475, 585` | User selects a voice profile, gets the original singer with +3.5 dB at 350 Hz, labelled "converted"; pipeline then remixes the master with it |
| A4 | **Segment repair (in-paint) is 100 % dead** — `provider_registry.get(...)` does not exist (class exposes `get_provider`) | `services/inpainting_service.py:258`; `providers/registry.py:87,190` | The whole "repair a bad bar" feature raises `AttributeError` on every request and reports as a generic engine failure |
| A5 | **`repair_segment()` is a no-op in 4 of 5 providers** (returns the input file untouched) while every capability manifest advertises `supports_segment_repair=True` | `minimax_provider.py:2084-2098`, `hf_audio_provider.py:281-295`, `heartmula_provider.py:173-187`, `mulacover_provider.py:227-241` | The provider contract promises in-painting the interface does not deliver |
| A6 | **Voice conversion hard-fails an otherwise successful job** when separation was skipped | `orchestration/pipeline.py:314-328`; `voice_service.py:378-380` | Optional feature turns a completed generation into a failed job |
| A7 | **Piano-roll edits are silently discarded on unmount** — the comment says "flush pending saves", the code only clears the timers | `frontend/.../PianoRoll.tsx:643-646` vs 700 ms debounce at `:335-339` | Switching Listen↔Piano Roll or closing the workspace inside 700 ms loses every edit, including a 200-note paste |
| A8 | **Timeline "compile/export" renders nothing** — builds an FFmpeg command string and returns it | `main.py:5918-5936` | The NLE export is cosmetic; `output_path` never exists |
| A9 | **Audio/video desync by design** — lattice trim math is computed then thrown away | `video/director_music_timing.py:194-216` (no consumers); `generators/diffusers_wan.py:178,231`; `video_orchestrator.py:1071` (`-shortest`) | 0.1–0.25 s lost per clip; a 40-clip video ends seconds before the song and every later cut drifts off the beat |

## B. P0 — Security

| # | Finding | Evidence |
|---|---|---|
| B1 | **Unauthenticated arbitrary local file read → API-key exfiltration.** `POST /tracks/{id}/commit-vocal` accepts `master_path` unvalidated and stores it as `job.audio_path`; `GET /download_track/{id}` serves whatever that path points at. `resolve_audio_file` trusts absolute paths. Two requests read `backend/llm_config.json` (plaintext provider keys) or `.env`. Same primitive via `/transcribe/lead-sheet`, `/api/v1/thumbnail/{filename}`, `/api/gallery/route`. | `main.py:4371-4392, 4411-4413`; `core/paths.py:129-130` |
| B2 | **SPA catch-all does path traversal**: `os.path.join(dist, full_path)` with no `realpath` containment, on a `:path` converter that passes `..` through | `main.py:5980-5991` |
| B3 | **Auth is opt-in and mounts bypass it**: `require_auth` returns early when no token is set (the default); `/stems`, `/generated_audio`, `/data/training_auditions` are `app.mount`ed and therefore outside the dependency; the token is also accepted via `?auth=` | `core/security.py:20,27-46`; `main.py:545-549,5968` |
| B4 | **The launcher binds `0.0.0.0`** and ignores `HOST`/`MILIMO_HOST`, so the documented "LAN exposure is explicit opt-in" is false for anyone using `./milimo start` | `milimo:158,165`; `main.py:6002` |
| B5 | **GPU compose profile runs on a CPU image** — `python:3.11-slim` + an `nvidia` device reservation, no CUDA base, no torch CUDA index | `Dockerfile:11`; `docker-compose.yml:26-32` |

## C. P1 — Lifecycle: states nobody owns

- **Generation jobs are not on the durable queue.** They are Starlette `BackgroundTasks`
  (`main.py:3471`); only the video orchestrator uses `DurableTaskQueue`.
- **Boot recovery destroys queued work.** `reconcile_orphan_jobs` marks *queued and
  processing* jobs `FAILED` (`main.py:353-367`); the durable queue meanwhile parks video
  tasks at `paused` (`task_queue.py:245-254`) that nothing ever resumes — 8 rows are
  stuck there in the live DB.
- **Cancelling a queued job resurrects it.** Cancel writes `FAILED`, but the pending
  background task later writes `PROCESSING` unconditionally (`pipeline.py:91-97`) before
  the first terminal check.
- **Cancel cannot reach the expensive step.** The only mid-pipeline checks are after
  generation and at finalize; separation and transcription have none. `DELETE /jobs/{id}`
  deletes the row and files while the pipeline keeps writing (`main.py:4504-4550`).
- **`JobStatus` has no `CANCELLED`.** User cancel is stored as `FAILED` with a free-text
  message (`models.py:9-13`), and `_abort_if_terminal` tests for `cancelled`/`interrupted`
  values the enum can never produce (`pipeline.py:43`).
- **Five album runs stuck in `queued` since August 2026 permanently block artist and
  release deletion** (409 `active_run`). Boot reconcile only repairs `running`
  (`main.py:339-344`); `cancel_agent_run` likewise (`main.py:1393`). Verified in the live
  DB: `agentrun` has `queued|5`, all `album_orchestrator`, dated 2026-08-25.
- **`DurableTaskQueue` leaks a SQLite connection per call** (`sqlite3` connection
  context manager commits but never closes; `task_queue.py:75-81`), and
  `video_orchestrator.py:205` calls `task_queue.fail_task(...)`, which does not exist.
- **`video_service.update_task` does not exist** either (`main.py:5395`), so keyframe
  failures raise `AttributeError` and the durable task never reaches a terminal state.
- **No retries, no attempts, no backoff, no heartbeat, no stuck-job watchdog.** Transient
  OOM/503/full-disk permanently fails a job.
- **GPU lock discipline is inconsistent**: `/tracks/{id}/separate` (`main.py:2576-2581`)
  and the entire inpainting service (`inpainting_service.py:123-270`) take no device
  lock; `release_device` releases unconditionally with no owner token
  (`hardware_lock.py:208-212`); the pipeline acquires with `consumer="Audio Pipeline"`
  and no modality, so the eviction hook lookup on release matches nothing.
- **Diffusion blocks the event loop**: synchronous `pipe(...)` inside `async def`
  (`generators/diffusers_wan.py:219,262`; `diffusers_ltx.py:141`), and the lyrics agents
  call blocking `requests.post` inside `async def` (`services/lyrics_graph.py:373,400`).

## D. P1 — Persistence: no schema authority

- **No migration tooling.** 38 `ALTER TABLE` statements plus a **primary-key rewrite**
  (`UPDATE job SET id = …`) run inside the request-serving process on every boot
  (`main.py:165-293, 295-310`); failures are `print()`ed and ignored. `migrate_db.py` is
  a dead, CWD-relative, divergent copy.
- **Zero foreign keys** except `SessionMessage.session_id`. Verified orphans in the live
  DB: 2 `playlisttrack` rows and 2 `job.release_id` values point at rows that no longer exist.
- **Two UUID encodings for the same relationship**: `session.id` is 32-hex, `job.session_id`
  holds 36-char hyphenated values. Which form the client sends decides whether tracks are
  visible in their own session; `get_job_by_id` tries five lookup strategies to compensate.
- **Declared indexes were never materialized** on existing tables (`job.project_id`,
  `job.session_id`, `agentrun.parent_run_id`) — `create_all` does not add indexes to
  existing tables.
- **Thirteen JSON-in-TEXT columns** with no version, no validation, no queryability; the
  stems blob is read by four different code paths with four different key conventions.
- **`updated_at` never bumps in the pipeline** → stuck-job detection is impossible.
- **No retention, no quota, no disk guard**: `storage_gc.py:28-33` prunes only
  `video_cache`/`uploads`/`temp`; `generated_audio` (8.5 GB) is never scanned. Cascade
  delete globs by naming convention and **misses** `{id}_remixed_master.wav` and
  `ai_cover_*.png`, which is how 466 directories accumulated.
- **Dual storage roots** with a boot-time `shutil.copy2` that resurrects deleted files
  (`main.py:323-331`).

## E. P1 — Capability & provider contract

- Unknown provider id silently becomes MiniMax (`providers/registry.py:138-140`) with no
  substitution record on the job.
- `is_ready()` returns `True` for any HF repo id containing `/`, before any weights exist
  (`musicgen_provider.py:60`, `stable_audio_provider.py:55`).
- `supports_lora=True` on MiniMax although no LoRA code exists in that provider
  (`minimax_provider.py:1197`; `music_service.load_lora_checkpoint` returns `False`).
- `extend()` on MusicGen / StableAudio / YuE2 / HeartMuLa / MuLaCover generates a **fresh
  unrelated song**; the pipeline calls whichever `extend` exists.
- Requested duration silently truncated (StableAudio 47 s, MusicGen 120 s) and the
  pipeline overwrites `job.duration_ms` from the result, erasing the difference.
- **Stem separation claims 6-stem BS-Roformer while running 4-stem HTDemucs**: dead
  ternary `"htdemucs_6s" if False else "htdemucs"`, while `source_id` defaults to
  `bs_roformer_6stem` and the UI is told guitar and piano exist.
- Per-instrument stems **invent** instruments by copying piano notes into Clarinet and
  Guitar parts without removing the originals (`transcription/instrument_stems.py:264-312`).
- Transcription runs on the **pre-voice-conversion** audio while the delivered master is
  the remixed one (`pipeline.py:340-346` vs `:391-399`).
- Producer LLM runs **inside** `generate()` at `temperature=0.7`, so identical seed+prompt
  is not reproducible (`minimax_provider.py:1374-1387`).
- Hardcoded placeholder lyrics are injected as if written, with no provenance marker
  (`llm_service.py:1418-1439, 1638-1639`); `generate_lyrics_async` can return the literal
  string `"Generation failed."` as lyrics (`llm_service.py:1145-1148`).
- MusicXML export is hardcoded `C major` / 4/4, drops every note after the 8th per
  measure, and engraves all durations as quarter notes (`muscriptor_provider.py:511-603`).

## F. P1 — Video & image studios

- **LivePortrait is effectively never executed** (requires `inference.py` inside the weights
  dir, which HF repos do not ship) → a Haar-cascade jaw warp is rendered and
  `fallback_used` is **not** set (`video/lip_sync/live_portrait.py:56-148`,
  `fallback.py:190`, `video_orchestrator.py:903`).
- **Registry sells engines that do not exist**: `cogvideox`, `hunyuan`, unknown keys →
  `ProceduralVideoGenerator()` (`generator_registry.py:67-78`); MiniMax H3 logs a
  paragraph then generates an animatic (`minimax_h3.py:155-178`).
- **Downloaded weights do not match what the loader wants** — registry registers
  `Wan2.1-T2V-1.3B/14B` while the generator only accepts `*-Diffusers` repos, so a 28 GB
  download ends in a silent Ken Burns render with a ✅ Ready badge.
- **Cloud providers are wired so they cannot work** — base64 data URIs as `image_url`,
  `{"version": "wan-video/wan-2.1-14b"}` where Replicate needs a hash, and hard 5 s clips
  (`cloud_video.py:137-166`, `cloud_lipsync.py:73-78`).
- **Performer scoping / `mouth_movement: closed` is computed then discarded** — `SceneClip`
  has no such field and the prompt never receives it (`video_director.py:756-799, 1146-1165`).
- **Mixed resolutions/rates concatenated without normalization**; Wan on MPS silently
  clamps to ≤832×480 while the UI claims 1080p (`diffusers_wan.py:163-170`).
- **Scene edits and clip reordering are client-only** — no storyboard persistence, and a
  reorder detaches every later scene from its lyric window (`VideoTimelineTrack.tsx:196-206`).
- **Timeline render ignores transforms and transitions** (`editor_projects.py:222-232`).
- **Procedural placeholder artwork is presented as AI output** in Image Studio
  (`image_service.py:637-648`; `ImageStudioView.tsx:123`), engine recorded as
  `studio_procedural_raster` and never badged.
- **Cancel is global**: `cancel_keyframe_generation` calls the singleton
  `image_service.cancel_active_diffusion()`, aborting *any* job's cover generation
  (`video_orchestrator.py:209-216`, `image_service.py:85-96`).
- **Progress is fabricated at four layers** (`video_orchestrator.py:432, 882, 1631, 1711`;
  download progress only moves per completed file).
- **Temp-file leak per render**, orphan keyframes/plates/thumbnails on delete, `scene_stills`
  absent from the GC list (`video_orchestrator.py:1120-1123`; `storage_gc.py:28-33`).
- **Subtitle ASS has no escaping** — a `{` or `}` in a lyric corrupts the caption line; and
  `fontsdir` points at `backend/app/assets/fonts`, which does not exist
  (`subtitle_styles.py:15-24,377`).

## G. P1 — Frontend (DAW & shell)

- **No clip editing exists.** Arrange renders static gradient divs; there is no region
  model, no drag/trim/split/fade/crossfade/snap (`ArrangeTimeline.tsx:276-348`).
- **Undo/redo covers only piano-roll note arrays** — mixer, loop region, lyrics, mastering,
  stem-source switches are unrecoverable (`PianoRoll.tsx:344-379` only).
- **No metronome, count-in, record arm, tempo map or meter change** (verified: 0 hits for
  `metronome`/`automation`/`timeStretch`/`warp` in `frontend/src`); `NotationViewer.tsx:63`
  hardcodes 4/4.
- **No MIDI import** — export only.
- **`GlobalAudioPlayer` parses `timed_lyrics_json` during render with no guard**
  (`:81-85`) while `safeJsonParse` exists and is unused there — a corrupt row replaces the
  whole app and takes unsaved piano-roll edits with it.
- **Switching A/B master audition mid-playback silences the session** (`SessionWorkspace.tsx:661-663`).
- **All stems AND the master are decoded eagerly on mount** (~315 MB float32 for a 3-minute
  4-stem track) before the user presses play (`SessionWorkspace.tsx:710-755, 959-967`).
- **30 Hz `currentTime` reaches every consumer of the merged audio context**, re-rendering
  every library row and the 2,386-line Track Studio during playback
  (`AudioEngineContext.tsx:660-663`); `TrackDetailView` re-parses five JSON blobs per render,
  30×/s (`TrackDetailView.tsx:341-359`).
- **Export format/sample-rate selectors only rename the file** — MP3/96k yields a WAV named
  `.mp3` (`MasteringExportModal.tsx:185-233`).
- **Fabricated waveforms are cached as real**: `StaticWaveform` falls back to a hash-driven
  envelope and memoizes it for the session (`StaticWaveform.tsx:11-26,41-46`).
- **SSE token in the query string** (`api.ts:616-617`); **no `onerror`, no reconnect, no
  replay**; a page reload mid-generation loses all progress.
- **Fabricated progress**: `progress: data.progress || 25` (`App.tsx:938-946`), and
  "Step 1/4" hardcoded for a 10-step pipeline (`App.tsx:1722` vs `:1131`).
- **`title="Action"` on ~40 controls** — including the shared `Button` primitive, where
  `{...rest}` is spread *before* `title="Action"` so caller tooltips are discarded
  (`primitives.tsx:60-61`).
- **Command palette advertises shortcuts that do not exist** (`1-8`, `D`, `V`, `R`, `⌘N`)
  and one that is bound to something else (`M` = mute).
- **`select-none` on the root container** disables text selection app-wide (`App.tsx:1242`).
- **Zero code splitting** — all 11 views + 6 modals in one 1.26 MB chunk.

## H. P1 — Ops, CI, packaging

- No dependency pinning anywhere: `requirements.txt` is 100 % `>=`; no `uv.lock`; `npm install`
  instead of `npm ci` in Docker and CI. This is the direct cause of the broken local test env.
- **CI cannot be green as committed**: 7 deterministic failures in `test_scene_plates.py`
  (stale `fake_render` signature vs the new `cancel_check` kwarg), and 3 tests in
  `test_production_v2.py` hang on real model/network calls with no `pytest-timeout`.
  `pytest.ini` has no `asyncio_mode`.
- CI: no `ruff`/`mypy`, no coverage gate, no `pip-audit`, no E2E, no Docker build, no
  migration smoke; Python 3.12 in CI vs 3.11 in the image.
- `./milimo stop` kills whatever owns port 8000 with no ownership check (`milimo:138-140`);
  logs never rotate.
- Provider API keys sit in a world-readable `backend/llm_config.json` (mode 644), written
  non-atomically on every update (`config_manager.py:213-214` — `win_safe_files.atomic_write_json`
  exists and is not used).
- Health check is a liveness stub: no DB ping, no disk, no queue depth, no model state.

## I. Missing functionality vs. a professional studio

| Expected | Status |
|---|---|
| Clip/region editing, fades, crossfades, snap | **Absent** |
| Arrangement undo/redo | **Absent** |
| Tempo map, time signature change, metronome, count-in | **Absent** |
| Automation lanes (volume/pan/send) | **Absent** |
| Time-stretch / warp / groove quantize | **Absent** |
| MIDI import | **Absent** (export only) |
| Multi-user accounts, per-user libraries, sharing | **Absent** (single optional token, singleton profile) |
| Project-level video settings, export presets, aspect variants | **Absent** |
| Render queue with priorities / preview proxies / cost estimate | **Absent** |
| Caption text+timing editor, thumbnail regeneration | **Absent** |
| Model download resume, per-model disk accounting | **Absent** (in-memory ledger) |
| Background task library view (queue depth, retry) | **Absent** |
| Backup / restore / retention UI | **Absent** |

## J. What is genuinely excellent — preserve

1. **Extension replay verification** — preflight probe, post-hoc correlation + tempo check,
   beat-quantized crossfade, forensic retention of a diverged render
   (`minimax_provider.py:589-727, 1971-1977`). This is the model for how *every* "did it
   actually work?" question should be answered.
2. **The fallback-provenance pattern itself** (`used_fallback_synth` / `fallback_reason` →
   Job → SSE). Correct shape; the work is applying it everywhere.
3. **Sample-locked multitrack transport** with buffer pre-decode, split audio context, and a
   deliberately throttled playhead (`SessionWorkspace.tsx:789-879`, `AudioEngineContext.tsx:415-441`).
4. **Hotkey scope arbitration stack** (last-pushed-wins with a consumed contract and throw
   isolation, `utils/hotkeyScope.ts:20-38`).
5. **`_abort_if_terminal` orphan-work guards** and the honest comments explaining why a
   blocking MLX call cannot be preempted.
6. **GlobalHardwareCoordinator** — modality eviction hooks, eager/TTL policies, MLX+CUDA+MPS
   cache flushing, 600 s acquisition timeout with an actionable error.
7. **Upload hardening** — extension whitelist → streamed cap → magic-byte sniff → randomized
   name, with SVG deliberately excluded and the reason documented (`core/uploads.py:16,56-116`).
8. **Honest hardware-encoder probing** — actually encodes a test frame before trusting
   VideoToolbox/NVENC (`subtitle_styles.py:429-471`). The template for all availability checks.
9. **Agent/album concurrency guards + ledger** — 409 `run_active`, typed error map, per-run
   attempt records.
10. **Scene-plate caching with approval stamps** so a user-approved retake is never
    re-diffused over (`image_service.py:768-826`).

## K. Systemic patterns (fix the pattern, not the instance)

1. **Provenance is per-feature, not a contract.** One flag exists for audio; transcription,
   voice, image, lip sync, and one provider have none. → *Every `Result` type carries
   `source`/`degraded`/`reason`, and no `COMPLETED` row may lack it.*
2. **Capability manifests are written by hand and never tested.** → *Derive them: a capability
   is `True` only if a test proves the method is not a stub.*
3. **Lifecycle states are written by many owners, transitioned by none.** → *One claim/CAS
   transition helper; one recovery policy table; `CANCELLED` exists as a state.*
4. **The schema has no authority.** No migrations, no FKs, two UUID encodings, indexes
   declared but absent. → *Alembic baseline + `PRAGMA foreign_key_check` on boot.*
5. **Contracts are declared in prose, enforced nowhere.** 162 routes, 13 response models,
   27 untyped bodies, one optional token, `title="Action"`, palette shortcuts that do not
   exist. → *Typed bodies, one error envelope, generated OpenAPI used by the frontend client.*

## Severity tally (verified + corroborated)

| Severity | Count | Concentration |
|---|---|---|
| P0 | 14 | Honesty (5), security (5), data loss / no-op features (4) |
| P1 | 46 | Lifecycle, persistence, capability contract, video, DAW |
| P2 | 38 | API design, performance, cleanup, a11y |
| P3 | 21 | Dead code, polish, consistency |

## Related

[Decision Record 2026-11](decisions-2026-11.md) ·
[Application Remediation Plan](application-remediation-plan-2026-11.md) ·
[Production Readiness Audit](production-readiness-audit.md) ·
[UI/UX Audit](ui-ux-audit.md) ·
[Architecture](architecture.md) ·
[Roadmap](roadmap.md) · [Index](index.md)
