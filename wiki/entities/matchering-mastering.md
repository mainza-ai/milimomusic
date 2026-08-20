---
title: Matchering Reference Mastering
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [mastering, matchering, lufs, dsp]
aliases: [MasteringEngine, Matchering, reference mastering]
---

# Matchering Reference Mastering

The **MasteringEngine** (`transcription/mastering.py`) provides **reference mastering** —
matching a target's loudness (LUFS), frequency spectrum, and RMS to a reference track.
It backs the **Mix** tab's "Matchering DSP Reference Master" button in the
[Session workspace](session-workspace.md).

## API
- `match_master(target_audio_path, reference_audio_path, job_id, target_lufs=-14.0) → MasteringResult`.
- Exposed as `POST /mastering/match/{job_id}` with `MasteringRequest{target_lufs, reference_job_id}`
  (default broadcast target **-14.0 LUFS**).
- Returns `mastered_audio_path`, `target_lufs`, `spectral_match_score`.

> [!WARNING] **Fidelity note.** The current implementation is largely a **stub/placeholder**:
> it reports progress steps but effectively copies/short-circuits the file; it does not yet
> run real Matchering DSP (the v2 plan lists Matchering as the intended engine). The frontend
> also masks mastering failures with a success message. Treat reference mastering as
> scaffolded, not production-grade.

## Related pages
- [Session workspace](session-workspace.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Backend & API](backend-api.md) | [Stem separator](stem-separator.md)
