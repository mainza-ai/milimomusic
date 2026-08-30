---
title: Artist Crew Agents
type: entity
tags: [agents, crew, artists, world-builder, stylist, critic, experiencer]
created: 2026-08-29
updated: 2026-08-29
sources: []
aliases: [crew agents, the crew, artist agents]
---

# Artist Crew Agents

The artist section's registered LLM agents — every one runs through
[ResiliencePolicy](agent-foundation.md) (provider failover, typed errors, usage
capture), so per-artist model overrides (assignment → profile default → global)
apply uniformly. See [Artist Domain](../concepts/artist-domain.md) for how they
fit together.

## The four registered agents (`app.agents.registry.AGENTS`)

| Agent | Role in the crew | Invocation |
|---|---|---|
| **Experiencer** | Imagines the lived journey inside an album concept → `ExperiencerVision` (arc + per-song seeds). Also auto-persists vision onto `release.vision_json` during album production. | `POST /agents/experiencer/run` (profile-scoped, in-request) |
| **World Builder** | Keeps the artist's canonical world document (`WorldLore`: origin, era, appearance, musical DNA, binding lore facts, contradictions-to-avoid). Persists to `ArtistProfile.lore_json`. | `POST /profiles/{id}/lore/generate` |
| **Stylist** | Curates the final 2–6 style tags per song draft pre-generation; `order_tags_genre_first` still runs after as the deterministic guard. | In-pipeline only (crew flag) |
| **Critic** | Pre-generation review: `pass / revise / concern` + score + notes + lore contradictions. `revise` triggers exactly ONE songwriter revision, then one re-review — never a loop. | In-pipeline only (crew flag) |

World Builder, Stylist and Critic each live in `app/agents/<name>/` with the same
shape as the Experiencer: `persona.py` (system prompt) + `schemas.py` (pydantic
contracts) + `agent.py` (build_messages + run) — persona and contract only, no
provider code.

## Where Stylist + Critic hook into the album pipeline

Inside [create_track_from_seed](../concepts/artist-domain.md#album-pipeline)
(`bridge.py`), after sanitize and before caption/generation:

```
songwriter draft → sanitize → [stylist: refine tags] → [critic: review]
     → (revise? ONE songwriter revision → re-review once) → caption
     → GenerationRequest (tags + voice baked in) → Job → generation
```

- **Opt-in per run**: `POST /releases/{id}/produce {"crew": {"stylist": bool, "critic": bool}}`
  — default OFF (+1–2 LLM calls per track). Retry runs inherit the flags.
- **Graceful degradation (locked principle):** a crew agent failing NEVER kills the
  track — the failure is recorded and the track proceeds unrefined/unreviewed.
- **Bounded revision:** max one revision round + one re-review, whatever the verdict.
- **Persistence:** critic verdicts land in the album run cursor
  (`state_json.reviews[slot]`) and join onto [tracklist rows](../concepts/artist-domain.md)
  by `seed_slot`; stylist tags need no extra persistence (baked into the Job's rich prompt).

## Related
[Artist Domain](../concepts/artist-domain.md) · [Agent Foundation](../concepts/agent-foundation.md) ·
[Artist Profiles & Album Agents](../concepts/artist-profiles-vision.md) · [Voice Studio (SVC)](voice-service.md)
