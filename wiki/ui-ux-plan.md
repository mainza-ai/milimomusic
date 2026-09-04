---
title: UI/UX Implementation Plan — Progress Log
type: overview
tags: [plan, ui, ux, design-system, editor]
created: 2026-08-21
updated: 2026-08-21
sources: [ui-ux-audit.md]
aliases: [ui plan]
---

# UI/UX Implementation Plan — Progress Log

Remediates [ui-ux-audit](ui-ux-audit.md). Owner decisions (2026-08-21): keep
glassmorphism but **discipline it**; accessibility program **deferred** (functional
fixes stay); piano roll = **full editor**; English-only.

## Delivered 2026-08-21 (this wave)

**U0 Truth pass — complete**
Real AnalyserNode meters · real arrange waveforms + true clip widths + synced clickable
ruler + claim-based lane mapping + vertical scroll · real Solo/Mute via multi-stem preview
mixer · fabricated chips removed/bound · real SSE-progress honesty (indeterminate widget,
no timer theater) + stale-closure fix · mastering failure honesty · real `/health` pill ·
piano-roll delete corruption fixed · notation BPM-derived rhythm glyphs + `BEAT_UNIT` ·
workspace-scoped hotkeys (hotkeyScope stack) · shadow/keyframe/class fixes.

**U1 Disciplined-glass foundations — core landed**
`surface-*` token tiers in Tailwind theme · all animation keyframes defined (43 dead
classes live) · primitives library `ui/primitives.tsx` (Button/Modal portal+Esc+trap/
Spinner/Toggle/Badge/`cn`) · Combobox dark-mode+ARIA · Toast de-indigoed/dark-aware ·
piano rail light mode · lucide-only in PianoRoll · 10px type floor (8/9px raised) ·
dead code deleted (`App.css`, GradientButton, AudioVisualizer).

**U2 Full editor & workspace reality — delivered**
PianoRoll v2: undo/redo (100-step) · multi-select/marquee · drag-move snapped ·
edge-resize · transpose (Shift=octave) · ⌘D duplicate · snap selector+quantize ·
velocity editing · h/v zoom · follow-playhead · batched persistence with honest save
chip (incl. Retry) · editor hotkey scope. Transport: throttled seek (no scrub glitches),
A-B loop region. Mixer: dB faders + double-click unity reset + per-source mix memory.
Per-track session persistence (mixer/mode/source/loop survive refresh).

**U3 Journeys — key items delivered**
Refresh-safe deep links (mount reads URL; workspace writes it) · completion hand-off
toast w/ Play/Open Studio (SSE+polling) · Load More wired + empty states + debounced
server search (20-track ceiling gone) · advanced generation params exposed
(temp/cfg/topk/seed/seed-lock) · accessible Toggles replace div-switches.

**Perf pass (same day)** — synth node lifecycle teardown (audio-graph leak), sorted-window
scheduler (binary search), memoized JSON parses + render layers (SessionWorkspace/
ArrangeTimeline/PianoRoll), notation coarse-time, global engine tick →30fps,
**HistoryFeed rebuilt**: server-computed cached peaks endpoint + StaticWaveform +
TrackRowPlayer (zero per-card engines, wavesurfer.js dependency removed entirely),
engine `playTrack(startAt)` for row seek.

## Deferred (recorded)
Full WCAG AA program · i18n foundation · metronome/count-in · lyric timing nudge ·
Modal-primitive rollout to all dialogs · ⌘K palette · bulk actions · unified Export
Center · Sessions-browser surfacing/naming repair · first-run wizard · full icon sweep ·
Button adoption across remaining views · canvas piano roll · TanStack Query migration.

## Related
[Audit](ui-ux-audit.md) · [Production Readiness Plan](production-readiness-plan.md) · [Agent Foundation](concepts/agent-foundation.md)
