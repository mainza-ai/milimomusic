---
title: Source — Milimo Music v2 Refactor Plan
type: source
created: 2026-08-19
updated: 2026-08-19
tags: [source, roadmap, v2]
aliases: [v2 refactor plan, milimo-music-v2-refactor-plan]
---

# Source — Milimo Music v2 Refactor Plan

**"Milimo Music v2 — Refactor & Upgrade Plan: From HeartMuLa music generator → full
open-source AI production DAW."** **Raw location:** `devs/milimo-music-v2-refactor-plan.md`
(immutable source).

## Key contents (summary)
- **Verified source material**: repo inventory of mainza-ai/milimomusic, MiniMax-AI/MiniMax-Music3,
  muscriptor/muscriptor.
- **Licensing**: non-commercial open source; MuScriptor MIT code / CC BY-NC 4.0 weights;
  MiniMax license to verify; produce `LICENSES.md`.
- **Product thesis**: generate → auto-transcribe (MuScriptor) → editable multitrack.
- **Architecture** (§3): Generation Provider abstraction; MiniMax default, HeartMuLa option;
  MuScriptor producer-edit; model & adapter management; dependency-currency policy;
  voice training (RVC/SVC); new backend capabilities (BS-Roformer, Matchering, WhisperX).
- **Frontend/UI** (§4): Suno-class IA kept; session grows a workspace
  (Listen/Arrange/Piano Roll/Notation/Mix).
- **Phased plan** (§5) and beyond (Milimo Video).

## Entities it feeds
- [Roadmap (v2)](../roadmap.md), [MiniMax Music 3](../entities/minimax-music3.md),
  [MuScriptor](../entities/muscriptor.md), [v2 references](../entities/v2-references.md),
  [Frontend](../entities/frontend.md), [AI Co-Writer](../entities/ai-cowriter.md).

## Related pages
- [Roadmap (v2)](../roadmap.md) | [Index](../index.md)
