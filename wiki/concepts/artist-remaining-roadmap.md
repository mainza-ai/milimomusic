---
title: Artist Domain — Remaining Roadmap (post Phases E–H)
type: concept
tags: [plan, artists, roadmap, voice, agents, production]
created: 2026-08-29
updated: 2026-08-29
sources: [artist-production-gap-report.md, artist-profiles-vision.md]
aliases: [remaining roadmap, artist next phases]
---

# Artist Domain — Remaining Roadmap

> [!NOTE] **Status: Waves 1+2 shipped (2026-08-29, commits `9a6af51`, `27cb5b3`).**
> Done: C2 indexed run lookups, A3 lore→songwriter, A1 voice identity, A2 World-Builder
> agent + lore generation, D1 workspace.spec fix, B1 playback, B2 track ordering,
> B3 budget caps, B4 live release chips, A4 cover generation, C3 ledger retention.
> **Remaining: Wave 3 / conditional only** — stylist+critic agents, shared Modal
> primitive, pagination UI + server-side search, run observability, ops docs.
> **A6 (LoRA checkpoint links) is DEFERRED by owner decision** — HeartMuLa is
> legacy-only and not part of the current stack; revisit only if HeartMuLa
> sampling returns.

Everything still open after Phases E–H shipped (commits `80f6eda`, `c27420c`, `a9f0136`).
Verified against code 2026-08-29. Ordered P1 → P3; each phase has an exit gate.

## P1 — Flagship differentiators (the artist product's "why")

### A1. Artist voice identity (2–3d) — **the missing flagship**
Every track today uses the default voice. The offline cloning stack already exists
(`voice_service`: profiles + `convert_vocals`), and the pipeline already consumes
`GenerationRequest.voice_profile_id` (`pipeline.py:229`) — nothing links an artist to one.
- Model: `ArtistProfile.voice_profile_id` (+ column migration via the existing PRAGMA path).
- Resolution: assignment-level `voice_profile_id` (crew `config_json`) → profile default → none.
- Bridge: `create_track_from_seed` passes `voice_profile_id=resolved` into `GenerationRequest`
  (currently hardcodes `None`, `bridge.py:159`); pipeline applies it as today for SVC.
- API: list available voice profiles on the profile detail (`GET /voice/profiles` exists).
- UI: "Singing voice" selector on the artist page (None = provider default); per-release override.
- Tests: resolution order unit test; bridge passes the id; honest fallback when a voice profile is deleted (dangling id → warn, don't fail the album).
- Exit gate: an album produced with a linked voice renders with that voice; delete the voice mid-album → tracks still complete with a ledger warning.

### A2. World-Builder agent (2d)
Registry has exactly one agent (`experiencer`); songwriter runs direct. Lore has a full
surface but nothing *generates* it.
- New `app/agents/world_builder/`: persona + `WorldLore` schema (hometown, era, influences,
  appearance, lore facts, contradictions-to-avoid); registered in `AGENTS` with `profile_id`
  carrying the artist.
- Route: `POST /profiles/{id}/lore/generate` → runs the agent grounded on name/bio/tags,
  persists JSON to `lore_json`, ledger row via the standard `run_agent` path (overrides +
  lore grounding already flow for any profile-scoped run).
- UI: "Generate world lore" on the lore editor; regenerate keeps a manual-edit lock (don't
  clobber unsaved edits without confirm).
- Tests: agent output validates; persistence; ledger row carries profile_id.
- Exit gate: artist lore round-trips generated → edited → grounding blocks in experiencer prompts.

### A3. Lore steers the songwriter (0.5d)
Experiencer gets `ctx.artist_lore`; songwriter children don't. `album_context` already
carries title/concept/artist to the songwriter — add `artist_lore` there (one line in
`album.py` + `bridge.py`, prompt block in `songwriter/agent.py`).
- Exit gate: songwriter messages include the lore block (test alongside existing tag-cohesion tests).

### C2. Indexed run↔release lookups (1d)
Guards and cursor resolution parse `input_json` of **every** album run per request
(`release_tracks`, produce guard, retry guard, delete guard). Fine at 50 runs, table-scan
at 50k. Add a real `AgentRun.release_id` column (indexed, backfilled in the startup
migration), write it in `produce`/`retry`, and query it directly.
- Exit gate: no `album_run_release_id` JSON parsing on hot paths; backfill covers legacy rows; tests updated.

### D1. Fix the 2 pre-existing `workspace.spec` failures (0.5d)
Landing tagline + Prompt/Lyrics selectors no longer match the UI (fail on clean HEAD;
reproduced during the G+H gate). Update the spec to current UI truth — unblocks fully
green CI e2e.

## P2 — Listening, ownership & ops polish

### B1. Tracklist playback via the global player (1d)
Inline play button per completed track: fetch the Job (tracklist already has the id) and
hand its audio URL to `GlobalAudioPlayer` instead of only the raw-file link / Studio jump.

### B2. Release track ordering (1.5d)
Tracklist order is the vision's seed order; users can't curate. `Release.track_order_json`
column (migration), `PATCH /releases/{id}/track-order` with the job-id array (validate all
jobs belong to the release), up/down controls in the tracklist header.

### B3. Budget caps UI (0.5d)
`produce` already accepts `{"budget": {"deadline_s"}}` — expose a selector
(off / 15m / 30m / 60m) beside the autopilot toggle; surface budget-exceeded state in the banner.

### B4. Per-release live chip (0.5d)
`GET /releases/{id}` already returns `active_run` — render a "producing…" pulse on release
rows (poll while any release is active) instead of showing state only inside the tracklist.

### A4. Artist + release cover generation (1.5d)
`/covers/generate` + `/covers/generate-prompt` exist (data/covers static mount). Wire:
- Profile: "Generate identity image" → prompt from lore/tags → upload flow already exists.
- Release: `Release.cover_image_path` (column migration) + same generation path; show as
  tracklist header art and in Explore when tracks land there.

### C3. Run-ledger retention (0.5d)
`agent_runs` grows unbounded (full I/O JSON per run). Pruning: env-configured retention
(default 30d) swept at startup + `DELETE /agents/runs?older_than=`; never prune rows
referenced by an active run's cursor.

## Wave 3 — implementation plan (investigated 2026-08-29)

### 3A. Stylist + Critic agents (2.5–3d) — the crew's last two real jobs

**Insertion points (verified):** `create_track_from_seed` runs songwriter → sanitize →
caption → generation. The Stylist slots between sanitize and caption (refines the draft's
tags *before* the rich prompt bakes them in); the Critic reviews the draft *before*
generation — an honest text-level review (lyrics vs seed story + lore consistency).
Audio review would require audio analysis and is out of scope.

**Design:**
- `app/agents/stylist/`: persona + schemas (`StylistBrief{seed, draft, artist_name, artist_lore?}`
  → `StylingChoice{style_tags: List[str] 2–6, rationale}`). Registered like the World-Builder.
  `order_tags_genre_first` still runs AFTER the stylist as the deterministic guard.
- `app/agents/critic/`: persona + schemas (`CriticBrief{seed, draft}` →
  `Critique{verdict: 'pass'|'revise'|'concern', score: float 0–1, notes, contradictions}`).
- **Revise path (bounded):** verdict `revise` → ONE songwriter revision round fed by the
  critic's notes → ONE re-review; final verdict recorded either way (never a loop).
- **Graceful degradation (non-negotiable):** stylist/critic LLM failures must never kill the
  track — catch, record in the ledger attempts, proceed with the unrefined draft (same
  pattern as the caption step).
- **Cost control:** opt-in per run — `POST /releases/{id}/produce` gains
  `{"crew": {"stylist": true, "critic": true}}`, default OFF (+1–2 LLM calls per track).
  Retry runs inherit the flags via their input_json. Two checkboxes beside autopilot.
- **Persistence/surface:** critic output lands in the album run cursor
  (`state.reviews[slot]`); `release_tracks` joins it by slot → each tracklist row can show
  a verdict chip + notes on demand. Stylist tags need no extra persistence — they're baked
  into the Job's rich prompt.
- **Exit gate:** seeded album run with crew on → stylist tags land in the Job prompt,
  critic verdict shows on the tracklist row, `revise` path produces exactly one revision,
  forced-LLM-failure runs still complete.

### 3B. Shared Modal primitive (1d)

**Investigation:** all four modals (InpaintModal, LLMSettingsModal, StyleManagerModal,
PathsSettingsModal) repeat `fixed inset-0 … backdrop-blur` overlays + X button with no
focus trap (InpaintModal has Escape only); ArtistsView's create modal has the only full
trap — inline. No portal needed (fixed positioning suffices).

**Design:** `<Modal open onClose title? widthClass? children>` in `ui/primitives.tsx`:
Tab trap + Escape + overlay click, initial focus to first focusable, restore focus to the
opener on close, body scroll lock. Migrate ArtistsView first (delete its inline trap),
then the four modals one commit each.

**Exit gate:** no modal overlay outside `primitives.tsx`; keyboard pass — Tab cycles, Esc
closes, focus returns to opener; all e2e still green.

### 3C. Artists-list pagination + server-side search (1d)

**Gap:** search is client-side over the loaded page — with paging, page-2 artists are
invisible to search. Backend list already has limit/offset/total but no text filter.

**Design:** add `q` to `GET /profiles` (case-insensitive LIKE across name/bio/tags);
frontend search becomes a debounced (250 ms) server query; pager = Prev/Next +
"N–M of T" footer (limit 24); skeletons and empty/miss states already exist; keep
`with_stats=1` on every page. Tests: q matches name/bio/tag, misses return empty;
pagination totals; e2e with mocked API (search + pager navigation).

**Exit gate:** search spans ALL profiles while paged; zero client-side filtering remains.

### 3D. Run observability (1d)

**Design:** `GET /agents/runs/stats?profile_id=&window=` aggregating the same bounded
query as the ledger list — counts by status, success rate, tokens in/out sums, latency
p50/p95 (computed in Python over parsed values; SQLite has no percentile), per-agent
breakdown. UI: a footer line inside the run-history panel ("142 runs · 91% success ·
p50 4.1s · p95 38s · 1.2M out-tokens") via `agentsApi.runStats(profileId)`.

**Exit gate:** seeded profile renders real aggregates; endpoint shape tested.

### 3E. Ops documentation (0.5d, docs only)

**Investigation:** multi-instance is already *prevented*, not just discouraged —
`acquire_instance_lock` hard-fails a second boot (stale-lock grace 30 s) with
`MILIMO_ALLOW_MULTI_INSTANCE=1` as the explicit escape hatch. C1 is therefore docs.

**Design:** README "Operations" section documenting: the instance lock + escape hatch +
`MILIMO_LOCK_FILE`; `MILIMO_RUN_RETENTION_DAYS` (ledger retention, 30d default, 0=off);
`MILIMO_AGENT_TIMEOUT`; `MILIMO_AUTH_TOKEN` / `MILIMO_CORS_ORIGINS`; `MILIMO_MAX_DURATION_S`;
SQLite WAL backup guidance (use `VACUUM INTO` or stop the server — don't copy a live WAL
trio). Also mark `artist-phases-execution-spec.md` as superseded (pending micro-fix).

**Exit gate:** every `MILIMO_*` env var documented; roadmap page closes C1.

### Suggested order & budget
3A (2.5–3d) → 3B (1d) → 3C (1d) → 3D (1d) → 3E (0.5d) ≈ **6–7.5d**. 3B/3C/3D are
independent and can interleave; 3A stands alone. Gates per item: backend suite (160+),
tsc/build, artist e2e extended, CI green.

## Micro-fixes (fold into the next touching commit)
- `songwriter/agent.py:36` — `'(unnamed}'` → `'(unnamed)'` (typo in fallback text).
- `artist-phases-execution-spec.md` status header still says "A-partial shipped" — mark
  superseded by the gap report.

## Suggested sequence
**Wave 1 (P1):** C2 → A3 → A1 → A2 → D1 (≈6d) — A1/A2 are the product differentiators;
C2 unblocks clean guards first; D1 makes CI fully green.
**Wave 2 (P2):** B1 → B3 → B4 → B2 → A4 → C3 (≈5.5d).
**Wave 3 (P3):** as demand appears (A5 stylist/critic first when quality signals exist).

## Verification gates (every wave)
tsc + vite build · full backend suite (153+) with new tests per feature · artist E2E
extended for the shipped flow · live smoke (servers up, one gated album run) · CI green.
