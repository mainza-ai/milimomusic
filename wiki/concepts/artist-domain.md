---
title: Artist Domain — Current State
type: concept
tags: [artists, crew, albums, releases, voice, production, reference]
created: 2026-08-29
updated: 2026-09-03
sources: [artist-phases-execution-spec.md, artist-section-audit.md]
aliases: [artist domain, artists current state]
---

# Artist Domain — Current State

What the artist section **does** as of 2026-09-03 (Phases E–H + roadmap Waves 1–3
shipped, OpenCode DeepSeek v4 Flash baseline active, and genuine MiniMax Music 3
neural generation verified end-to-end). For history and reasoning see the [gap report](artist-production-gap-report.md)
and [roadmap](artist-remaining-roadmap.md); for the original dream, the
[vision page](artist-profiles-vision.md).

## Data model (`app/models.py`)

- **ArtistProfile** — identity (name, bio, tags), `lore_json` (World-Builder canon),
  `cover_image_path`, `voice_profile_id` (singing voice), `default_provider/model`,
  optional `project_id`.
- **AgentAssignment** — which agent serves an artist at which role, with
  `model_provider/model` overrides (the crew override chain head) and `config_json`.
- **Release** — album container: `status` lifecycle (`planned → in_progress → completed`),
  `vision_json` (persisted ExperiencerVision), `track_order_json` (curation),
  `cover_image_path`.
- **AgentRun** — full ledger for every agent/orchestrator/retry run: I/O JSON, attempts,
  tokens, latency, budget, progress, `parent_run_id`, indexed `profile_id` +
  `release_id` (album cursors; never pruned by retention).
- **Job** — generated tracks carry `release_id` + `artist_profile_id` provenance and
  `voice_profile_id` per track.

## Album pipeline (one seed → one track)

`create_track_from_seed` (orchestrator bridge), all LLM work behind
[ResiliencePolicy](agent-foundation.md):

1. **Songwriter** writes the draft (lore + revision blocks included in the prompt)
2. sanitize + validate
3. *[crew flag]* **Stylist** refines tags → `order_tags_genre_first` guard
4. *[crew flag]* **Critic** reviews → `revise` = ONE songwriter revision + one re-review
5. caption rewrite → rich prompt (tags baked in, producer can't rewrite them)
6. `GenerationRequest` (explicit duration/seed + artist `voice_profile_id`) → Job
7. generation + transcription/stems pipeline (voice conversion degrades gracefully
   if the linked voice profile vanished)

The [Album Orchestrator](album-orchestrator-plan.md) wraps this: gated produce
(pause/approve per track) or autopilot, budget deadline, resume cursor in
`state_json` (`slot_jobs` + `completed_seeds` + `failed_jobs` + `reviews`), WAL
crash recovery, 409 guards against concurrent runs, single-seed retry.

> [!NOTE] **Production verification & OpenCode DeepSeek v4 Flash baseline (2026-09-03).**
> - **OpenCode baseline:** OpenCode Zen (`deepseek-v4-flash`) is the default baseline
>   across backend and frontend. Experiencer, World-Builder, Songwriter, Stylist, and
>   Critic agents all execute against `deepseek-v4-flash` via Cloudflare-protected endpoints.
> - **Database decoupling:** SQLite/SQLAlchemy sessions in `AlbumOrchestrator.execute` and
>   `retry_single_seed` are decoupled into short atomic transactions, committing and closing
>   before long async LLM and GPU generation awaits.
> - **Live 2-track album production:** Album "Aurora Borealis" for artist "Nova Eclipse"
>   produced 2 tracks with authentic MiniMax Music 3 neural inference (`used_real_inference: true`),
>   BS-Roformer 4-stem separation on Apple Silicon MPS, and MuScriptor transcription to
>   note-level MIDI and MusicXML.
> - **Critic verdicts:** Joined by slot to tracklist rows (`pass 0.80`, `pass 0.85`).
> - **UI provenance badge:** Displays emerald `● MiniMax Music 3 (Neural)` badge for
>   verified neural inference tracks.

**Vision handoff:** the Experiencer Studio's "Save as Release" persists the full
vision onto the release, so `produce` uses its exact seeds and track count —
a 2-track EP is producible (previously produce re-imagined with the default 5).

## Key endpoints

| Area | Endpoints |
|---|---|
| Profiles | `GET/POST /profiles` (`with_stats`, `q`, `limit/offset`) · `GET/PATCH/DELETE /profiles/{id}` (delete = block active runs, cascade releases, detach jobs) · `PATCH …/cover` · `POST …/lore/generate` |
| Crew | `PUT /profiles/{id}/assignments` (role enum + provider validation) |
| Releases | `GET/POST /releases` · `GET/PATCH/DELETE /releases/{id}` · `PATCH …/track-order` · `GET …/tracks` (slot-cursor dedupe + review join + lifecycle status) · `POST …/produce` (autopilot, budget, crew) · `POST …/tracks/{job}/retry` |
| Runs | `GET /agents/runs` (profile-scoped, paginated) · `GET …/stats` (success rate, p50/p95, tokens, per-agent) · `GET/POST …/cancel · /resume` · `DELETE /agents/runs` (retention prune) |
| Agents | `GET /agents` · `POST /agents/{name}/run` — see [Artist Crew Agents](../entities/artist-crew-agents.md) |

## Frontend surface (`ArtistsView.tsx`)

- **List**: server-side search (`q`, debounced), sort (activity/name), 24/page with
  "N–M of T" pager, shimmer skeletons, empty-state CTA, stat cards (crew/releases/
  last activity), full keyboard/screen-reader semantics.
- **Guided create**: 4-step stepper (name+project → bio w/ counter+example → tags with
  style chips → optional cover), shared-`Modal` a11y (trap, Escape, focus restore).
- **Detail**: identity editor (validated form, dirty-guard + URL sync), singing-voice
  selector, world-lore editor + World-Builder generation, crew management with model
  overrides, experiencer studio (SSE-filtered stage text), releases (rename/describe/
  delete/art generation/lifecycle chips), tracklist (artifact chips, in-app play,
  Studio handoff, retry, reorder, review chips), run-history panel with aggregates.
- **Integrity**: zero silent catches (honest toasts everywhere), run recovery on
  reload, deep-links `?view=artists&id=`, per-section ErrorBoundary.

## Related
[Artist Crew Agents](../entities/artist-crew-agents.md) · [Album Orchestrator Plan](album-orchestrator-plan.md) ·
[Artist Profiles & Album Agents](artist-profiles-vision.md) · [Gap Report](artist-production-gap-report.md) ·
[Remaining Roadmap](artist-remaining-roadmap.md) · [Voice Studio (SVC)](../entities/voice-service.md)
