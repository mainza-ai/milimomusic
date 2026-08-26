---
title: Product Vision Audit — Differentiator & Enterprise-Readiness Analysis
type: concept
tags: [audit, ux, vision, competitive]
created: 2026-08-25
---

# What exists in the market (due diligence)
Suno/Udio: instant songs, zero control, cloud-only, no identity/memory.
Stable Audio: clips. AIVA: MIDI composition. RipX/Moises: stems only.
ComfyUI: powerful local gen, hostile UX, no music-native workflow.
**Nobody** combines: persistent artist identity + narrative-driven agent crews +
full neural transcription/stems + open local engine + real DAW editing.
That combination is the company. Everything below serves it.

# Gap vs "enterprise-ready" (honest)
Current ArtistsView = functional plumbing, not a product: plain rows, raw JSON-ish
states, no artist visual identity, no listening experience, no sense of "journey",
weak empty/loading/error states, no onboarding, alerts instead of toasts.

# Options analyzed
## A. Album Journey View ★ differentiator
Render the Experiencer's emotional_arc as a navigable timeline; each seed = station
with mood/energy/story; generated tracks dock onto their stations; unmade stations
show what WILL exist. The narrative IS the interface. No competitor has this.
Cost: ~3-4d. Risk: low (data already in vision_json).

## B. Design-system substrate ★ table stakes
Tokens (spacing/type/color), consistent primitives everywhere, skeleton loaders,
empty states with next-action, toast system, focus management. Without B nothing
feels enterprise regardless of features. Cost: ~2d.

## C. Listening Room ★ wow-factor
Album playback with real-time stem toggling (vocal/drum/bass mute mid-song) using
existing separated stems; journey timeline doubles as scrubber map. Cost: ~3d
after B.

## D. Crew Activity Feed ★ trust/transparency
Agents visible as workers: songwriter drafts, failures, retries, token spend —
like a CI pipeline for creativity. Pairs with honesty principle. Cost: ~2d.

# Recommendation (order matters)
B → A → D → C. Substrate first (nothing great sits on mud), then the narrative
moat (A), then transparent agents (D), then listening magic (C).
Each phase ships demoable value; total ≈ 10 working days.

# Explicitly rejected
Node-graph composer (ComfyUI territory, months of work, wrong audience);
cloud-only SaaS pivot (abandons open-source ethos); more generation providers
(doesn't differentiate).
