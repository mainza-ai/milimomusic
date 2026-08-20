---
title: Track Extension
type: concept
created: 2026-08-19
updated: 2026-08-19
sources: [sources/readme.md]
tags: [extension, track, generation, continuation]
aliases: [Extend, Track continuation]
---

# Track Extension

**Track Extension** lets Milimo continue generating from where a previous track left off,
allowing the creation of longer compositions **segment by segment**.

## How it works
- Generation continues from a prior track's context (the tail audio/history becomes the new
  prompt's reference/context), rather than starting fresh.
- The backend links jobs via `parent_job_id` on the `Job` model, so an extension is recorded
  as a child of the original generation (see [Backend & API](../entities/backend-api.md)).
- The output is a longer, continuous composition built incrementally.

## Related pages
- [HeartMuLaGenPipeline](../entities/heartmulagenpipeline.md) | [Backend & API](../entities/backend-api.md)
- [Lyrics conditioning](lyrics-conditioning.md)
