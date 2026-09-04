---
title: Artist Section — Production Gap Report & Implementation Plan
type: concept
tags: [audit, artists, production, plan, backend, frontend]
created: 2026-08-29
updated: 2026-08-29
sources: [artist-section-audit.md, artist-phases-execution-spec.md, album-orchestrator-plan.md]
aliases: [artist gap report, artist production plan]
---

# Artist Section — Production Gap Report & Implementation Plan (2026-08-29)

> [!NOTE] **Status: Phases E–H shipped (2026-08-29).** All 29 gaps closed except:
> full World-Builder lore *generation* (lore surface + grounding shipped; generation is a
> vision item), in-app playback from tracklist (ships via "Studio" handoff to the global
> player), and two pre-existing `workspace.spec.ts` E2E failures unrelated to artists
> (fail identically on clean HEAD — separate fix needed). Evidence: 152 backend tests,
> 5 artist E2E specs, tsc+build green, commit `80f6eda` (E+F) + G/H commit.

Evidence-based audit of the artist domain against the production bar ("not a demo").
Every claim verified against code today. Supersedes the mechanical A–D phase list in
[artist-phases-execution-spec](../sources/artist-phases-execution-spec.md) where they conflict.

## Shipped & verified working

| Piece | Evidence |
|---|---|
| Profile CRUD + cover endpoint | `backend/app/main.py:1018-1103` |
| Crew role enum validation (C4) | 422 + allowed list, server-side |
| Release create + tracklist endpoint (C1) | `main.py:1147`, `main.py:1160` |
| Tracklist UI w/ artifact chips | `ArtistsView.tsx:704-734` |
| Run ledger + `GET /agents/runs?profile_id=` (C5 backend) | `main.py:866` |
| Identity dirty-guard beforeunload (C3 partial) | `ArtistsView.tsx:163-170` |
| Album orchestrator: gated produce, resume cursor, budget, WAL recovery | `album.py`, `test_album_recovery.py` |
| Jobs land in Explore (bridge sets project_id/release_id/artist_profile_id) | `bridge.py:144-159` |
| Experiencer vision auto-persist to `release.vision_json` on produce flow | `experiencer_bridge.py:64` |

## P0 — Correctness & data integrity (must fix before any UX work)

1. **Tracklist retry dupes (spec C2).** `release_tracks` returns *every* `Job` row for the
   release (`main.py:1167-1169`). Orchestrator retries append new Jobs for the same seed;
   `state_json.job_ids` (`album.py:191`) is the only winner record. Fix: resolve the latest
   album run cursor for the release, prefer cursor jobs in cursor order, dedupe remaining
   rows by seed (newest wins), keep manual (non-orchestrated) jobs chronologically.
2. **`Release.status` is written by nothing.** Model has `planned|in_progress|completed`
   (`models.py:319`) but the orchestrator never touches it; the UI recomputes rollups per
   request. Fix: orchestrator sets `in_progress` on start, `completed` (or `partial`) on
   terminal state; add a validated transition helper (planned→in_progress→completed,
   failure → back to `in_progress` with error surface).
3. **No concurrent-produce guard.** `POST /releases/{id}/produce` (`main.py:1187`) spawns a
   run without checking for an active run on the same release → double runs, double spend.
   Fix: 409 if an `AgentRun` for this release is `queued|running|awaiting_approval`; add
   unique-active-run check inside the orchestrator too (race between request and spawn).
4. **Orphan integrity on profile delete.** `DELETE /profiles/{id}` removes assignments only
   (`main.py:1087-1103`); releases + their Jobs are orphaned and unreachable (profile-scoped
   list 404s). Production policy: block delete while a run is active; cascade-delete
   releases whose jobs have no project home OR offer explicit "keep discography, detach
   from artist" — must be one deliberate, tested behavior, not silence.
5. **Silent catches still live in ArtistsView** (violates the P1 honesty principle):
   list-load `catch(console.error)` (`ArtistsView.tsx:94-96`), `addCrewMember`/`removeCrewMember`
   (`:209,221`), `handleDeleteProfile` (`:191`), `saveVisionAsRelease` (`:319`). All become
   honest error toasts.
6. **SSE cross-talk.** Experiencer stage-text subscribes to *any* `run_progress` message
   (`ArtistsView.tsx:229-234`) — concurrent album/agent runs corrupt the stage readout.
   Fix: filter by `run_id` (the envelope response provides it) like the album banner does.
7. **Broken saveState hack.** `setSaveState('error' as never)` immediately overwritten by
   `setSaveState('idle')` (`ArtistsView.tsx:184-185`) — save failures are invisible. Make
   `error` a real state with a visible inline message.

## P1 — Missing core functionality

8. **Release lifecycle API absent.** Only POST /releases + scoped list exist. Add:
   `GET /releases/{id}`, `PATCH /releases/{id}` (title/description/status via transition
   helper), `DELETE /releases/{id}` (policy: jobs detach to plain history, artifacts kept),
   ownership validation on tracks/produce. Plus frontend: rename, edit description, delete.
9. **Crew override chain is dead data.** `AgentAssignment.model_provider/model/config_json`
   and `ArtistProfile.default_provider/default_model` are stored but **never read anywhere**
   (grep-verified). Wire resolution order `assignment override → profile default → global
   default` in `run_agent` (`main.py:805-828`) and the orchestrator child `RunContext`
   (`album.py:169`), and surface the resolved model in the run ledger. Config: validate
   `config_json` against a small allowlist, never freeform injection.
10. **`lore_json` is unwired end-to-end.** Column exists (`models.py:290`), zero read/write
    paths, absent from `ArtistProfileUpdate`, World Builder output isn't persisted per artist.
    Implement: PATCH surface + World-Builder grounding reads/writes profile lore + detail UI
    viewer (read-only first, editable later).
11. **No run recovery after reload.** Active album banner dies on refresh (runId lives only
    in state). On mount: `GET /agents/runs?profile_id=&status=running` (+ queued/awaiting)
    and reattach banner; same for experiencer runs.
12. **Autopilot has no UI.** `produce(autopilot)` param exists; UI hardcodes `false`
    (`ArtistsView.tsx:57`). Add explicit toggle with a cost warning.
13. **Per-seed retry absent.** A failed track can only be fixed by re-running the whole
    album. Add `POST /releases/{id}/tracks/{seed}/retry` (orchestrator single-seed path
    reusing `create_track_from_seed`) + UI button on failed rows.
14. **`image_prompt` is a dead field** on create/update schemas — either wire it to the
    image pipeline for cover generation or remove it from the API surface.
15. **Pagination missing** on `GET /profiles`, `/profiles/{id}/releases`, `GET /agents/runs`
    (limit-only). Offset+total pattern to match `/history`.

## P2 — UX / product completion (A/B/C remainder)

16. **A1 guided create stepper** — 4-step modal: name+project → bio (examples, counter) →
    tags (chips from StyleRegistry) → optional cover (uploadCoverImage→setCover then create).
    Current create is 3 bare inputs with duplicated inline validation.
17. **A2 + B1 list stats** — `GET /profiles?with_stats=1` (crew_count, release_count,
    last_activity via max(updated_at, max release.created_at), single grouped queries); card
    rendering. **Includes fixing the duplicate-thumbnail render bug** (`ArtistsView.tsx:371-385`
    — the cover block renders twice, malformed nesting).
18. **B2 sort** (name | activity), **B3 shimmer skeletons** (replace spinner),
    **B4 empty-state CTA button**, **B5 `role="list"`/`listitem`** semantics on the grid.
19. **C5 run-history panel** — backend ready; add `agentsApi.listRuns(profileId)` + panel in
    detail (agent, status, latency, tokens, error, created).
20. **C6 deep-links** — `?view=artists&id=` read on mount + popstate in App, passed as
    `initialProfileId`; URL synced (replaceState) on open/close of detail.
21. **C7 finish** — `ErrorBoundary` gains optional `sectionName` prop; App already passes it
    (uncommitted diff).
22. **C3 finish** — dirty-guard on back-to-list navigation (beforeunload already done).
23. **Listening experience** — artifact chips open raw files in new tabs. Add in-app
    playback from tracklist (reuse global player) + explicit "Open in Studio" (deep-link to
    track-detail) since Jobs already appear in Explore.
24. **Crew UX** — human role labels; model-override editors on assignment rows (blocked on #9).

## P3 — Systemic quality (Phase D)

25. **D3 `useValidatedForm`** hook adopted by create stepper + identity editor.
26. **D2 a11y sweep** — focus-trapped Modal (primitives), htmlFor labels, keyboard list nav.
27. **D1 artist E2E** — `e2e/artists.spec.ts` (create→detail→cover→produce-gated) and wire a
    Playwright job into `ci.yml` (verified: CI has backend + build jobs only, **no e2e job**).
28. **Project scoping enforcement** — ArtistsView currently lists all profiles (no project
    filter passed); profile-scoped endpoints don't validate project membership. Enforce
    server-side + pass project context from the UI.
29. **Perf** — stats aggregate in one round-trip (#17); tracklist for 30-track releases is
    O(n) rows + per-row artifact dict — fine at this scale, but index `Job.release_id`
    (verify exists) and keep rollup server-side.

## Implementation plan (production order)

| Phase | Contents | Exit criteria |
|---|---|---|
| **E — Integrity** (#1-#7) | C2 dedupe, release status transitions, produce guard, delete policy, honest errors, SSE filter, saveState fix | All silent catches gone; retry run leaves one row per seed; double-produce → 409; delete policy tested; tsc+build+114 tests green |
| **F — Lifecycle & crew engine** (#8-#15) | Release GET/PATCH/DELETE + UI, override chain live, lore_json path, run recovery, autopilot UI, per-seed retry, pagination | Overrides observable in run ledger; lore round-trips; reload reattaches active run; new endpoint tests |
| **G — UX completion** (#16-#24) | A1/A2/B1-B5/C5-C7/C3, playback + open-in-studio | Full keyboard+screen-reader list nav; deep-link lands on profile; stats render; E2E-ready DOM |
| **H — Systemic** (#25-#29) | form hook, focus trap, artist E2E + CI job, project scoping | `npx playwright test` green incl. artists flow in CI; scoping tests |

Verification gates per phase: `tsc + vite build` · backend suite (114+ new) · manual smoke on
live servers · CI green. No phase marked done without evidence.

Related: [artist-section-audit](../sources/artist-section-audit.md) ·
[artist-phases-execution-spec](../sources/artist-phases-execution-spec.md) ·
[album-orchestrator-plan](album-orchestrator-plan.md) · [artist-profiles-vision](artist-profiles-vision.md)
