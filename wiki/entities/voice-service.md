---
title: Voice Studio (SVC)
type: entity
created: 2026-08-20
updated: 2026-08-30
tags: [voice, svc, rvc, cloning, consent, singing]
aliases: [VoiceService, Voice Training Studio, SVC]
---

# Voice Studio (SVC)

The **Voice Service** (`services/voice_service.py` + frontend `VoiceStudioModal`) is
Milimo's offline **Singing Voice Conversion (SVC)** and vocal-identity feature. Users train
a voice profile and select a "Sing as…" voice in the Composer.

## Voice profiles
- `VoiceProfile` dataclass: id, name, description, sample_audio_path, status
  (`ready`/`training`/`failed`), `consent_confirmed`, `f0_method` (`rmvpe` default,
  also `crepe`/`harvest`/`pm`), sample_rate (40k default).
- Stored as JSON under `backend/data/voice_profiles/` (default starter profiles "Aria",
  "Marcus").
- Endpoints: `GET/POST /voice/profiles`, `DELETE /voice/profiles/{id}`.
- **Consent gate**: `create_profile(name, description, consent_confirmed, ...)` **raises
  `ValueError` if `consent_confirmed` is false** — the UI requires a mandatory checkbox
  before training.

## Conversion
- `convert_vocals(vocal_stem_path, profile_id, job_id, pitch_shift=0)` runs SVC on an
  isolated vocal stem. Wired into the [orchestration pipeline](../concepts/generation-pipeline.md)
  as **optional Step 3** when `GenerationRequest.voice_profile_id` is set.
- The Composer exposes a **"Sing as…" voice selector** (default "Default AI Voice" +
  trained profiles + "+ Train New Voice…").

> [!WARNING] **Fidelity note.** As implemented, `convert_vocals` currently copies/emits the
> vocal stem rather than running a real RVC/SVC model. The v2 plan intends **RVC v2**
> (via Applio forks) or So-VITS-SVC as the actual engines; those weights aren't wired in
> this revision. Treat SVC as scaffolded with a consent + profile-management layer in place.

## Related pages
- [Session workspace](session-workspace.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Backend & API](backend-api.md) | [Stem separator](stem-separator.md) | [Roadmap (v2)](../roadmap.md)


---

**Artist linkage (A1, 2026-08-29):** an [artist profile](../concepts/artist-domain.md) can link a voice profile (`ArtistProfile.voice_profile_id`); album tracks resolved from that profile run SVC conversion automatically, degrading gracefully if the profile is deleted.
