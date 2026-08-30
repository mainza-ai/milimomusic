---
title: Artist Profiles & Album Agents — The Ultimate Vision
type: concept
tags: [vision, agents, artist-profiles, album, orchestration, roadmap]
created: 2026-08-21
updated: 2026-08-21
sources: []
aliases: [artist profiles, album mode, creative flywheel]
---

# Artist Profiles & Album Agents — The Ultimate Vision

> [!NOTE] **Update (2026-08-29): most of this vision is now built.** Artist
> profiles, agent crews (Experiencer/World-Builder/Stylist/Critic), lore canon,
> voice identity, gated album production with resume/budget/retry, release
> lifecycle and a full inspectable UI all ship — see the current state in
> [Artist Domain](artist-domain.md) and the crew page
> ([Artist Crew Agents](../entities/artist-crew-agents.md)). Still vision-only:
> per-agent memory scoping, artist-file export, world_state tools.

Owner-stated north star (2026-08-21): within a **Project**, the user creates
**Artist Profiles** (unlimited). Each profile has **specific agents assigned**.
The user then says *"create an entire album"* — and the assigned agents
(world builders, experiencers, songwriters…) autonomously produce a full 10+
track album *particular to that artist*. Never done before; the differentiator
vs Suno-class black boxes: everything is inspectable, editable, and grounded in a
persistent artist identity.

## The hierarchy

```
Project  (workspace: name, palette, bpm/key defaults — exists today)
  └── ArtistProfile  (NEW — the unit of identity & agent assignment)
        ├── Identity: name, bio, lore/world doc, genre DNA, influences,
        │   visual identity, voice/timbre preference, default style tags,
        │   LoRA checkpoints (Training Studio tie-in)
        ├── AgentAssignments: [{role, agent_def, model_profile override}]  ← data, not code
        ├── World State: structured lore the world-builder reads/writes (tools)
        ├── Discography: releases → tracks (Jobs with provenance)
        └── Memory: conversations, decisions, style evolution
```

## The album production flow

```
User: "Create a 12-track album for this artist"
        ↓
Album Orchestrator (top meta-agent, long-running resumable run)
 1. WORLD BUILDER      expands/reads artist lore → album concept + arc (WorldLore schema)
 2. ARTISTIC DIRECTOR  tracklist plan: themes per track, sequence, energy curve,
                       which songs are singles; AlbumPlan schema (typed)
 3. PER TRACK (fan-out over the existing Job pipeline):
       Songwriter   lyrics (Co-Writer graph, reused) + title
       Style Curator tags/caption (Structured Captions, StyleRegistry-grounded)
       EXPERIENCERS simulated listener personas critique draft → revision loop
       [generation] existing generate→stems→transcribe pipeline per track Job
 4. ALBUM CRITIC coherence pass across tracks (motifs, key/BPM arc, pacing)
 5. Human review gates between phases (approve plan / approve tracks / approve master)
Artifacts: album_plan.json · world bible update · N track Jobs linked to the release
```

Key properties: every stage is a typed artifact the user can edit and re-run from;
generation cost is bounded by budget hooks; runs are resumable (kill-safe) because
they reuse the durable job-engine patterns.

## Data-model gap analysis (verified against `models.py`)

Exists today: `Job` is rich (project_id, session_id, full transcription/provenance);
`Session` is project-scoped chat; `SessionMessage` stores role/content/preset JSON.
Missing entirely:

| New entity | Purpose | Notes |
|---|---|---|
| **ArtistProfile** table | identity, bio/lore JSON, default tags/model prefs, visual refs, LoRA checkpoint links | FK: project_id. Jobs gain `artist_profile_id` |
| **AgentAssignment** (JSON on profile or table) | which agents serve this artist + per-artist model/provider overrides | rides ConfigManager |
| **Release / Album** table | title, artist_profile_id, track ordering, status | Jobs gain `release_id`; enables "album" as first-class |
| **agent_runs** table | run_id, agent, status, input/output refs, usage_json, error | foundation page §runtime |
| **world_state** store | versioned lore documents the world-builder mutates via tools | SQLite now, Alembic later |

## What this demands from the runtime

See [AI Agent Foundation](agent-foundation.md) for the full investigation. The
vision specifically requires: memory (G4) — an artist's lore/history must persist
across runs; tools (G3) — world_state read/write IS the world builder; budgets
(G5) — a 12-track generation run is real money; resumability — album runs span
hours/days; streaming progress (G9) — users watch the album being made.

## Creative flywheel (why this wins)

Artist profile accumulates: lore → discography → trained LoRA voice/style →
experiencer feedback history → sharper agent output. Each album makes the next
one better *for that artist specifically*. No black-box competitor offers
persistent, editable artist identity with inspectable multi-agent craft.

## Open questions (for owner)

1. "Experiencers" = simulated listener-persona critics? (assumed yes; naming open)
2. Do albums auto-generate audio per track inside the run, or produce plans/drafts
   first with audio as an approved second phase? (cost implications)
3. Artist profile sharing/export format (portable "artist file"?)
4. Can one agent serve multiple artists concurrently with per-artist memory scoping?

## Related
- [AI Agent Foundation](agent-foundation.md) — the runtime this stands on
- [AI Co-Writer](../entities/ai-cowriter.md) · [Orchestration Pipeline](generation-pipeline.md) · [Roadmap](../roadmap.md)
