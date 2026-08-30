---
title: Artist Section Audit (2026-08-26)
type: source
tags: [audit, artists, frontend, backend]
created: 2026-08-26
---

# Why it failed for the user
3 defects: 5 silent catches (failures invisible) · synchronous agent runs block
HTTP 30-240s (UI looks hung) · stale dev-server bundle risk.

# Backend gaps
Releases: no GET list (405), no PATCH/DELETE, no release→tracklist endpoint,
status written by nothing. Crew: assignment model overrides stored-but-never-injected;
profile defaults unused; no GET /agents/runs list. Identity: lore_json/cover_image
unwired; voice_profile_id unconsumed. Structural: cascade delete unverified, no
pagination, project scoping unenforced.

# Frontend gaps
Zero tests/browser verification · no skeletons/empty/error states · no tracklist
view · no release edit/delete · roles freeform · no deep-links · root-only
ErrorBoundary · no project scoping.

# Plan
P1 runnable+truthful (async runs, kill silent catches, tracklist endpoint+UI) →
P2 release lifecycle (PATCH/DELETE/cascade/status transitions/ownership) →
P3 crew+identity real (chain-head injection, role enum, lore/cover, run history) →
P4 polish (states, E2E, a11y, SSE replay).
