---
title: Album Orchestrator — Implementation Plan (R1–R4)
type: concept
tags: [plan, agents, orchestrator, album, run-lifecycle]
created: 2026-08-22
updated: 2026-08-25
sources: [agent-foundation.md, artist-profiles-vision.md]
aliases: [r2 plan, album pipeline plan]
---

# Album Orchestrator — Implementation Plan (R1–R4)

Deep-investigation-derived plan for the unique product layer. Decisions locked:
gated-by-default album execution w/ autopilot toggle · track duration = 120s +
energy×120s (energy-scaled, 120–240s) · missing crew = warn-and-proceed.

## Verified integration seams (build on these)

- `_build_override(profile)` injection → `_get_provider` works (proven live).
- `await music_service.generate_task(job_id, req, engine)` converts HTTP fire-and-forget into a blockable sequential step; GPU lock serializes per-track children automatically.
- EventManager accepts arbitrary payloads; `/events` auth survives via `?auth=`.
- Full artifact inventory per track documented (audio/midi/xml/stems_json/notes/beat_grid/timed_lyrics/caption/provenance flags).

## Seed→Song mapping gaps (must-handle in bridge)

| Seed field | Destination | Note |
|---|---|---|
| story_seed | `run_lyrics_graph(topic=…)` | clean fit |
| suggested_style_tags | tags comma-string | validate vs StyleRegistry; genre-first ordering (MiniMax positional destructure trap) |
| working_title / mood / energy | NO param | bridge synthesizes steering prose (`user_message`) |
| placement_hint | release ordering only | keep out of lyric prompt |
| concept_statement/recurring_motifs | ride inside topic/user_message prose | cohesion |

Friction traps: GenerationRequest.duration_ms defaults 30s (bridge must set explicitly);
Job.voice_profile_id silently dropped (model field missing); weak inputs trigger hidden
producer rewrite (pre-enrich so thresholds `<30 chars` are false); dual error conventions
(graph raises MaxRetriesExceededError vs chat_with_lyrics_async returns error dict).

## Run/event infrastructure gaps

EventManager queues unbounded (dead QueueFull handler); pure broadcast no scoping;
pipeline checks cancel only inside providers; AgentRun lacks parent_run_id/state_json/
progress/budget_json + boot reconciliation; frontend connectToEvents registers only
job_update/job_progress (additive extraEventTypes fix); polling fallback required (no replay).

## Phases

### R1 — Auth loop closure (½d)
axios Authorization interceptor (localStorage milimo_auth_token) + EventSource ?auth= ·
XFF-aware limiter key · .env.example MILIMO_AGENT_TIMEOUT · remove stray root generated_tokens/.

### R2a — Schema completions (2d)
Job.release_id + Job.voice_profile_id model fields · EventManager maxsize=512 ·
AgentRun boot reconciliation (running→interrupted) + columns parent_run_id/state_json/
progress/budget_json.

### R2b — Run lifecycle engine (4-5d)
active_runs threading.Event registry (+shutdown_runs in lifespan) · POST
/agents/runs/{id}/cancel (clone jobs cancel incl DB fallback) · between-step cancel
checks · RunEventBus wrapper stamping run_id/parent_run_id on run_update/run_progress
(existing payload vocabulary → FloatingStatusWidget unmodified) · state_json cursor
written transactionally per step · BudgetTracker fed from child usage · resume endpoint
skipping completed steps.

### R2c — Frontend run UX (2d)
connectToEvents extraEventTypes param · run_id filter · run-progress drawer + cancel.

### R3 — Artist domain completion (3-4d)
Shared role enum validated backend-side · assignment/profile overrides injected as
policy chain heads · project-scoped profiles in ArtistsView · auto-persist Vision→Release
on success · Toasts replace alert()/silent catches · profile deep-links + dirty guard.

### R4 — Songwriter bridge + first album (1.5-2wks)
`create_track_from_seed(profile, release, seed, …)` implementing verified chain:
tags validate/order → chat_with_lyrics_async(current="", user_message=steering w/
working_title+mood+energy, topic=concept+story_seed, tags) → sanitize → rewrite_caption
→ GenerationRequest(explicit duration_ms=120s+energy*120s, seed, structured_caption,
project_id) → Job(release_id, artist_profile_id) → await generate_task → attach.
AlbumOrchestrator: vision→per-seed children (sequential), budget tracker, gated pauses,
release status transitions, minimal progress UI.


## Implementation Status (as of 2026-08-25)

| Phase | State | Notes |
|---|---|---|
| R1 auth loop | DONE | axios interceptor · EventSource ?auth= · extraEventTypes |
| R2a schema | DONE | Job.release_id/voice_profile_id fields + migration · bounded EventManager queues (512) · AgentRun orchestration cols (model+DB) · boot reconciliation |
| R2b run lifecycle | DONE | RunRegistry cancels · /agents/runs/{id}/cancel+resume · BudgetState · strong-ref task spawning |
| R2c frontend drawer | PARTIAL | ArtistsView subscribes run_progress (live stage text); full drawer pending |
| R3 artist domain | PARTIAL | overrides verified live; role enum validation, Toasts, deep-links pending |
| R4 songwriter+album | LIVE-BACKEND | SongwriterAgent (explicit JSON contract) · bridge create_track_from_seed · AlbumOrchestrator gated/autopilot · vision free-reuse · routes produce/resume — 'Ignition Hymn' written & queued live; full-audio completion pending stable server |
| Lifecycle solidification | DONE | _abort_if_terminal ×3 pipeline checkpoints · provider orphan-audio discard · instance lock (PID lockfile) · shutdown reporting — see log 2026-08-25 |

**Remaining:** full-album audio E2E on owner-run server · frontend Produce button + run drawer · R3 polish items · mlx_audio step-callback preemption (upstream).

## Related
[Artist Profiles & Album Agents](artist-profiles-vision.md) · [Agent Foundation](agent-foundation.md)
