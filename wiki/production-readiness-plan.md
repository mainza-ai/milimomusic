---
title: Production Readiness — Implementation Plan
type: overview
tags: [plan, production, security, reliability, packaging, artists, minimax]
created: 2026-08-21
updated: 2026-09-03
sources: [production-readiness-audit.md, artist-production-gap-report.md, artist-remaining-roadmap.md]
aliases: [prod plan, remediation plan, production-roadmap-v2]
---

# Production Readiness — Implementation Plan (Updated 2026-09-03)

Comprehensive, production-grade roadmap remediating gaps across generation, creative agents, transcription, persistence, and the artist domain. Designed for **production deployment, NOT a demo**.

## Locked Decisions & Baseline
- **Generation Flagship**: [MiniMax Music 3](entities/minimax-music3.md) is the primary default engine.
- **HeartMuLa Legacy Isolation**: HeartMuLa (the v1 3B model in `heartlib/`) is legacy. All boot hooks that trigger duplicate MiniMax initialization are severed. Inpainting and track extensions are re-architected natively for MiniMax/audio-domain conditioning rather than HeartMuLa tokens. A6 LoRA training remains deferred.
- **LLM Runtime Baseline**: OpenCode (`opencode`) running `deepseek-ai/deepseek-v4-flash-0731` is the active default for development, testing, and creative agent crews (World-Builder, Experiencer, Songwriter, Stylist, Critic).
- **Persistence & Architecture**: SQLite with WAL mode stays as the primary self-hosted database; ad-hoc migrations replaced with Alembic; browser `localStorage` replaced with backend REST models for Playlists and Profiles.
- **Live Verification**: Both backend (`localhost:8000`) and frontend (`localhost:5173`) servers run continuously during implementation for end-to-end verification.

## Phased Production Roadmap

### Phase 1: Purge Legacy Zombies & Fix Runtime/Artist Crashes (Immediate)
- **Audio Upload Crash**: Fix `backend/app/main.py:518` where `/transcribe/upload` references undefined `filename` instead of `safe_name` (guaranteed HTTP 500 on audio import).
- **The `(unnamed)` Artist Bug**: In `backend/app/agents/orchestrator/album.py:147` & `line 387`, replace `getattr(release, "artist_name", "")` with `session.get(ArtistProfile, release.profile_id).name`. Release has no `artist_name` column, which previously caused all album tracks to generate for an anonymous `(unnamed)` artist.
- **Dropped `voice_profile_id`**: In `backend/app/agents/orchestrator/bridge.py:290`, persist `voice_profile_id=voice_profile_id` on the `Job` row instead of hardcoding `None`.
- **HeartMuLa Boot Hooks**: Sever lines 84–90 in `music_service.py` where querying `get_provider("heartmula")` falls back to MiniMax and initializes the 29GB model twice.
- **Tracklist Stem URLs**: Update `ArtistsView.tsx:1463` to parse `stems_json` and render individual stem links or a modal instead of linking directly to raw JSON (which caused 404s).
- **Deceptive Audio Fallbacks**:
  - Remove the 4x stereo master copy in `real_separator.py:185`.
  - Remove the hardcoded 8s C–Am–F–G chord progression in `muscriptor_provider.py:320`.
  - Fail honestly with typed diagnostic error envelopes.

### Phase 2: Artist Domain Hardening & Crew Resolution
- **Stylist & Critic Overrides**: In `album.py`, resolve individual `chain_head` overrides for `"stylist"` and `"critic"` via `resolve_chain_head(session, profile_id, agent_name)` so user crew configurations are respected.
- **Release Track Detach/Attach API**:
  - Add `DELETE /releases/{release_id}/tracks/{job_id}` to detach unwanted tracks from a release without deleting user history.
  - Add `POST /releases/{release_id}/tracks` to attach existing studio jobs to a release.
- **Continuous Album Playback**: In `ArtistsView.tsx`, pass the album's tracklist (`tracks.tracks`) to `playTrack(job, tracks.tracks)` so the player queue is populated with all tracks in the release.

### Phase 3: Linux & CI Cross-Platform Hardening
- **Lazy MLX Imports**: Make `import mlx.core as mx` in `minimax_local_hooks.py` conditional and lazy, allowing Linux servers and Ubuntu GitHub Actions CI runners to boot without `ModuleNotFoundError`.
- **Honest Platform Matrix**: Eliminate procedural synthesizer fallback; return typed `HardwareIncompatibilityError` when hardware requirements are not met.
- **Dependency Pinning**: Declare `demucs>=4.0.0` in `requirements.txt`; pin `pydantic-ai` and `pydantic-graph`.

### Phase 4: Durable Task Queue & Async Operations
> [!NOTE] **Design locked 2026-09-03** — see [Task Queue](entities/task-queue.md) for the full design.
- **Durable Task Queue**: Introduce `backend/app/core/queue.py` with SQLite backing. Separate GPU lane (`concurrency=1`) from DSP/IO lane (`concurrency=2`).
- **Locked semantics**: on restart, re-enqueue ALL tasks — `queued` stay queued, `running` re-enqueue with `attempts+1` (replaces fail-honest `reconcile_orphan_jobs` for queued work).
- **Async Operations**: Convert `/transcribe/upload`, `/mastering/match`, `/jobs/{id}/voice-convert` from blocking HTTP endpoints into queued async tasks returning `202 Accepted` with SSE progress tracking; `/jobs/{id}/inpaint` and `/models/download` also move onto the queue (fixing the fire-and-forget `asyncio.create_task` and the in-memory download ledger).
- **Streamed Model Downloads**: Replace in-memory ledger with streamed Hugging Face downloads providing chunk-level progress and resumability, persisted in `TaskRecord`.

### Phase 5: Real Audio Services (RVC SVC & MiniMax Extension)
> [!NOTE] **Design locked 2026-09-03** — see [Singing Voice Conversion](concepts/singing-voice-conversion.md) and [Track Extension](concepts/track-extension.md).
- **Real Singing Voice Conversion**: Replace `shutil.copyfile` in `voice_service.py` with real RVC v2 inference (RMVPE pitch extraction + HuBERT acoustic representations). **Locked approach: vendor the MIT-licensed RVC-WebUI inference modules** (`SynthesizerTrnMs768NSFsid` v2, RMVPE, ContentVec) into `backend/app/vendor/rvc/`; inference only — users import standard `.pth` checkpoints, no voice training. Includes fixes for the `voice-convert` endpoint arg bug (`main.py:2961`) and the empty-file write (`voice_service.py:146`).
- **Native MiniMax Track Extension**: The open MLX hook cannot consume reference audio, so **locked approach: analysis-conditioned extension** — parent BPM/key/caption/lyrics-derived conditioning, LLM lyric continuation, equal-power crossfade mixdown, honest `conditioning` metadata (no fabricated audio-embedding claims).

### Phase 6: Database Domain Completion & Alembic
> [!NOTE] **Design locked 2026-09-03** — see [Playlists & Studio Profile](concepts/playlists-profiles.md).
- **Alembic Baseline**: Versioned database migrations under `backend/alembic/`; batch mode for SQLite; `0001_baseline` stamped onto legacy DBs after one final legacy column-patch pass; ad-hoc `PRAGMA/ALTER` blocks (`main.py:132-232`) retired after a transition release.
- **Database Models**: Add formal `Playlist`, `PlaylistTrack`, and `StudioUserProfile` SQLModel tables (`PlaylistTrack` replaces the localStorage `songIds` array with ordered rows + unique constraint).
- **Frontend Storage Migration**: Wire `PlaylistsView.tsx` and `ProfileView.tsx` to backend REST endpoints, deprecating `localStorage` via a one-time import; device-local prefs (theme/volume/composer) intentionally stay local.

### Phase 7: Live Server Boot, Frontend Hardening & Verification
- **Video Studio Storyboard**: Replace mock `setTimeout` in `MusicVideosView.tsx` with backend `/generate/storyboard` endpoint.
- **Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` supporting CPU and NVIDIA CUDA GPU profiles.
- **Live Verification**: Run FastAPI (`:8000`) and Vite (`:5173`) servers; execute automated pytest and Playwright suites.

## Related Pages
- [Production Readiness Audit](production-readiness-audit.md)
- [Artist Domain](concepts/artist-domain.md)
- [MiniMax Music 3](entities/minimax-music3.md)
- [HeartMuLa](entities/heartmula.md)
- [Wiki Index](index.md)
