---
title: Artist Phases — Execution Spec (approved, pending build-out)
type: source
tags: [plan, artists, ui]
created: 2026-08-26
---

# Status: SUPERSEDED (2026-08-29) — all A–D phases shipped and verified; see
# [Artist Production Gap Report](../concepts/artist-production-gap-report.md) and
# [Artist Remaining Roadmap](../concepts/artist-remaining-roadmap.md) for the live plans.

## Phase A rest
A1 Guided create stepper: replace 3-input block (ArtistsView ~line 348) with 4-step
modal: (1) name+project selector, (2) bio w/ examples+char counter, (3) tags w/
suggestion chips from StyleRegistry, (4) optional cover via coverApi.uploadCoverImage
→ profilesApi.setCover. Progress dots, disabled-next on invalid, toast on success,
auto-open detail of created profile.
A2 List rows: cover thumbnail (cover_image_path || gradient placeholder), crew size +
release count (needs detail fetch or aggregate endpoint GET /profiles?with_stats=1 —
add backend aggregate), last activity (max(updated_at)).

## Phase B
B1 Backend GET /profiles?with_stats=1 returning counts per profile.
B2 Search input (client-side filter), sort dropdown (name/activity).
B3 Skeleton rows while loading (existing Spinner primitive → shimmer variant).
B4 Empty state w/ "Create your first artist" CTA. B5 aria-labels/role="list".

## Phase C
C1 Tracklist section in detail: fetch GET /releases/{id}/tracks (shipped) — list w/
artifact chips (audio/midi/xml/stems links when non-null), status dot, real-inference
badge. C2 Dedupe retry rows via orchestrator job_ids cursor (state_json).
C3 Dirty-guard: compare editName/editBio/editTags vs detail; beforeunload + nav warn.
C4 Role enum: shared const in api.ts + backend validation (ReleaseCreate pattern) —
allowed: experiencer, songwriter, producer, stylist, critic.
C5 Run history panel: needs backend GET /agents/runs?profile_id= (add endpoint; ledger
has profile_id column? verify — else filter client-side from runs GET).
C6 Deep-link: App passes ?view=artists&id= → ArtistsView initialProfileId prop.
C7 Section ErrorBoundary wrapper.

## Phase D
D1 Playwright E2E: create→detail→cover→produce-gated flow; add to CI (needs frontend
job restored — unblock lockfile platform pin first: remove darwin-arm64 rollup from
package-lock or add overrides).
D2 a11y sweep: focus trap Modal (primitives.tsx), labels htmlFor, keyboard list nav.
D3 Form pattern: extract useValidatedForm hook; adopt in create+identity editors.

## Verification gates per phase
tsc+build green · 114+ backend tests · manual smoke via running servers · CI green
(requires D1 lockfile fix). Do not mark done without evidence.
