---
title: UI/UX & Design Audit (2026-08-21)
type: overview
tags: [audit, ui, ux, design-system, daw, accessibility]
created: 2026-08-21
updated: 2026-08-21
sources: []
aliases: [ui audit, design audit]
---

# UI/UX & Design Audit (2026-08-21)

Four parallel deep-dives (UX flows/IA, visual design system, DAW workspace
interactions vs Logic/Ableton/GarageBand standards, accessibility). Companion plan:
[ui-ux-plan](ui-ux-plan.md).

**Verdict:** the Web Audio transport engine is production-grade; the layers on top
(interaction honesty, design-system discipline, editor capability) were demo-grade.
Headline findings: the app's entire CSS animation layer was inert (~43 dead classes);
design tokens existed with **zero** consumers (161 hardcoded hexes, 20 dark-surface
variants); several visible controls were fake; one editor bug corrupted scores.

## F1 Honesty failures (all fixed 2026-08-21 unless noted)
Mixer peak meters were `Math.random()` → real per-channel AnalyserNode taps.
TrackDetail stem Solo/Mute decorative → real multi-stem preview mixer.
Arrange waveforms simulated sine bars → real decoded-amplitude peaks + true clip widths.
Fabricated metadata ("-14.0 LUFS", "48kHz FLAC", "Quantization Grid 1/16") removed/bound to real data.
Progress theater (timer-driven 25→55→85%, pinned fake 70% bars) → honest indeterminate states.
Mastering failure reported success → now reports failure. Hardcoded "Engine Ready" pill → real `/health` polling.

## F2 Logic bugs (fixed)
Piano-roll filtered-index delete corruption · notation rhythm glyphs bucketed in raw
seconds (only correct ≈70BPM; fed exported MusicXML) + hardcoded 4/4 · workspace hotkeys
drove the global player over the session mix (dual audio) · invalid Tailwind classes
(`w-18`, `py-0.2`, undefined `shadow-apple-2xl`) silently breaking layout/elevation.

## F3 Missing functionality (was)
No undo/redo anywhere; single-note selection; no move/resize/snap/quantize/velocity;
no zoom; scrubbing glitched (per-step source rebuild); loop start-to-end only; % faders
without dB; stem-source switch wiped mixes; nothing persisted per track; only 20 tracks
reachable (pagination props never wired); refresh destroyed all state; hidden generation
params (seed/temp/cfg/topk had no setters); generation completion had no hand-off;
Sessions browser unreachable; export scattered across 4 surfaces.

## F4 Design system state (at audit)
Tokens defined in `index.css` but 0 consumers; no primitives (≥4 CTA recipes; 15 ad-hoc
modals none closing on Escape); ≥6 spinner idioms; Combobox 100% light-only; Toast forced
white slab + off-palette indigo; 31 emoji icons beside lucide; glassmorphism diverging from
the reference guide's flat surfaces; micro-type epidemic (256 instances <12px); dead code
(App.css landmine, unused apple.* palette, GradientButton/AudioVisualizer).

## F5 Accessibility snapshot (DEFERRED by owner — recorded, not lost)
≈35–40% of WCAG 2.2 AA; fails Level A on 2.1.4 (unmodified single-key hotkeys) and
4.1.3 (zero status messages for a minutes-long async core loop). Functional fixes that
double as UX fixes WERE applied (hotkey scoping/modal Escape via primitives); the full
AA program remains deferred until before public/EU-facing release.
i18n readiness ~10% (English-only locked for now).

## What was already excellent (preserve)
Sample-accurate multi-stem transport; dual stem-source switching with solo-in-place;
buffer pre-decode; karaoke word highlighting; notation engraving fundamentals;
MediaSession integration; ProjectsView empty states/breadcrumbs; fallback-synthesis
honesty chip; "In Dev" badge pattern.

## Related
[UI/UX Plan](ui-ux-plan.md) · [Production Readiness Audit](production-readiness-audit.md) · [Session Workspace](entities/session-workspace.md)
