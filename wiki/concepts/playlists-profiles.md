---
title: Playlists & Studio Profile (Backend Persistence)
type: concept
created: 2026-09-03
updated: 2026-09-03
tags: [database, playlists, profiles, alembic, migrations, phase-6]
aliases: [Playlist model, StudioUserProfile, localStorage migration]
sources: [production-readiness-plan.md]
---

# Playlists & Studio Profile (Backend Persistence)

> [!NOTE] **Status: design locked 2026-09-03, not yet shipped.** Phase 6 of the
> [production plan](../production-readiness-plan.md).

Browser `localStorage` for playlists and the studio profile moves to formal SQLModel
tables + REST endpoints, so library state survives browsers/devices and can cascade
properly when tracks are deleted.

## New tables (`backend/app/models.py`)

| Table | Columns | Notes |
|---|---|---|
| `Playlist` | `id`, `name`, `description`, `cover_color`, `created_at`, `updated_at` | replaces `milimo_playlists`; `cover_color` keeps the Tailwind gradient string |
| `PlaylistTrack` | `id`, `playlist_id` (FK, indexed), `job_id` (indexed), `position` | replaces the `songIds` array; `unique(playlist_id, job_id)`; `position` gives stable ordering |
| `StudioUserProfile` | `id`, `artist_name`, `bio`, `avatar_image_path`, timestamps | singleton row (GET auto-creates the default) |

Plus `PlaylistCreate/Update`, `PlaylistAddTrack`, `StudioUserProfileUpdate` schemas.

## Alembic baseline (replaces ad-hoc migrations)

Schema drift is currently patched by `PRAGMA table_info` / `ALTER TABLE` blocks in
`main.py:132-232`. Phase 6 introduces `backend/alembic/`:

- `env.py` wired to SQLModel metadata + the existing engine; **batch mode** for SQLite
  ALTERs.
- `0001_baseline` = the current schema. Boot logic: run the legacy column patcher once →
  `alembic stamp 0001` if the DB is unversioned → `upgrade head`. Fresh DBs go straight
  through migrations.
- `0002_playlists_profiles` = the three new tables. The ad-hoc `ALTER` blocks retire
  after one transition release.

## API routes

- `GET /playlists`, `POST /playlists`
- `GET /playlists/{id}`, `PATCH /playlists/{id}`, `DELETE /playlists/{id}`
- `POST /playlists/{id}/tracks` (`{track_id, position?}`)
- `DELETE /playlists/{id}/tracks/{job_id}`
- `PUT /playlists/{id}/tracks` (full reorder: array of job_ids)
- `GET /profile/studio` (auto-create default), `PUT /profile/studio`

`/profiles` is already taken by the [artist domain](artist-domain.md) — hence
`/profile/studio`.

## Frontend migration

| localStorage key | Fate |
|---|---|
| `milimo_playlists` (`PlaylistsView.tsx:30,69`, cleanup in `App.tsx:506`) | **migrated** — one-time import to backend, then cleared |
| `milimo_artist_name`, `milimo_artist_bio` (`ProfileView.tsx:15-25`) | **migrated** — same one-time import |
| `milimo_volume`, `milimo_theme`, composer prefs (`milimo_duration`, `milimo_seed`, …) | **kept** — device-local by design, explicitly out of scope |

One-time import flow: if the backend returns an empty collection AND localStorage holds
data → POST the local state → clear the key. `App.tsx` job-delete cleanup moves
server-side (DB cascade removes `PlaylistTrack` rows when a Job is deleted).

## Tests (planned)

`backend/tests/test_playlists_api.py` (CRUD, reorder, unique constraint, job-delete
cascade), `test_studio_profile.py` (singleton upsert), and a migration smoke test
(fresh DB `upgrade head`).

## Related pages

- [Production plan — Phase 6](../production-readiness-plan.md) · [Backend & API](../entities/backend-api.md)
- [Artist Domain](artist-domain.md) (owns `/profiles`) · [Frontend](../entities/frontend.md)
