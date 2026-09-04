---
title: Implementation Backlog — Full Gap Ledger
type: concept
tags: [backlog, gaps, roadmap]
created: 2026-08-25
updated: 2026-08-25
---

# Full Gap Ledger (leave-no-stone-unturned audit)

## A. Album pipeline (in flight)
- [ ] Confirm 5/5 seeds complete + release.status flip → 'completed'
- [ ] Album-level mastering pass (consistent loudness across tracks)
- [ ] Cover art generation per release; track titles from songwriter into UI lists
- [ ] Resume-after-restart for mid-generation tracks (cursor has job_ids; job requeue untested)

## B. R3 artist domain polish
- [ ] Role enum shared backend+frontend w/ validation (currently free strings)
- [ ] Toasts replace alert()/silent catches in ArtistsView
- [ ] Profile deep-links (?view=artists&id=) + unsaved-changes guard
- [ ] Project-scoped profiles enforced in ArtistsView queries
- [ ] Assignment model overrides injected as policy chain heads (plumbing verified, not wired)

## C. Performance
- [ ] C1 benchmark: 8bit/4bit vs bf16 (speed + listening test)
- [ ] Flow steps knob validation (12 vs 30 quality delta)
- [ ] Dynamic RTF learning (store measured per-machine; replace static estimate)

## D. Platform gaps (never started)
- [ ] Voice cloning (RVC) real weights + consent flow — Phase 2 B-item still open
- [ ] HeartMuLa keep/drop decision execution (21GB)
- [ ] Training Studio E2E verification
- [ ] Multi-user/auth productization (single-token auth today)
- [ ] Alembic migrations replacing ad-hoc ALTERs
- [ ] CI (GitHub Actions: pytest+tsc+build on push)
- [ ] Packaging/distribution story (README quickstart claims untested)
- [ ] Frontend test coverage (zero tests today)
- [ ] SSE replay/persistence for missed events (polling fallback only)
- [ ] Budget token accounting wired into child runs (deadline-only today)

Priority order proposed: A → B → D(CI/Alembic first) → C.
