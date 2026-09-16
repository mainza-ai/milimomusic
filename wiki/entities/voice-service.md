---
title: Voice Studio (SVC)
type: entity
created: 2026-08-20
updated: 2026-09-15
tags: [voice, svc, rvc, cloning, consent, singing, acoustic-dsp]
aliases: [VoiceService, Voice Training Studio, SVC]
---

# Voice Studio (SVC)

The **Voice Service** (`services/voice_service.py` + frontend `VoiceStudioModal`) is
Milimo's offline **Singing Voice Conversion (SVC)** and vocal-identity feature. Users train
a voice profile from uploaded solo vocal recordings or import checkpoints, and select a "Sing as…"
voice in the Composer or Track Detail view.

## Voice profiles & Dataset Ingestion
- `VoiceProfile` dataclass: `id`, `name`, `description`, `sample_audio_path`, `status`
  (`ready`/`training`/`failed`), `consent_confirmed`, `f0_method` (`rmvpe`, `crepe`, `harvest`, `pm`),
  `sample_rate` (40k default), `acoustic_features` (median F0, spectral centroid, rolloff, RMS energy),
  `dataset_files`, and `is_default`.
- Endpoints: `GET/POST /voice/profiles`, `DELETE /voice/profiles/{id}`.
- **Dual Content-Type Support**: `POST /voice/profiles` handles both JSON payloads (API/tests) and
  `multipart/form-data` with audio file or `.zip` archives.
- **Acoustic Profiling**: Uploaded audio is processed using `librosa.pyin` and spectral feature
  extractors to derive median fundamental frequency ($F_0$), spectral centroid timbre distribution,
  and dynamic range, and generates an unclipped, peak-normalized sample preview in
  `generated_audio/voice_previews/`.
- **Consent gate**: `create_profile(...)` strictly raises an error if `consent_confirmed` is false.

## Conversion & Formant Shaping
- `convert_vocals(vocal_stem_path, profile_id, job_id, pitch_shift=0, dry_wet=1.0, formant_preserve=True)`:
  - **Neural SVC Inference Engine**: Leverages [Neural SVC](neural-svc.md) (`NeuralSVCService` in `backend/app/services/voice/neural_svc.py`) for zero-shot vocal timbre transfer, formant morphing, pitch transposition, and cross-fade dry/wet mixing.
  - **Neural RVC Checkpoint Support**: When an RVC v2 `.pth` model checkpoint exists in `data/voice_profiles/{id}.pth`, loads the weights onto MPS/CUDA/CPU under the management of the [Hardware Coordinator](hardware-coordinator.md).
  - **Adaptive Acoustic & Formant DSP Shaping**: Dynamically morphs the vocal tract transfer function and formant spectral envelope to match the target vocal identity profile (e.g. ethereal pop high-presence for bright timbres, warm chest resonance for rich timbres).
  - **Formant Preservation**: Phase-locked formant tracking compensates upward/downward shifts when pitch-shifting to preserve vocal identity character.
  - **Wet / Dry Blending**: Studio blend control ($0\%$ to $100\%$) blending original dry vocals with transformed vocals.

## Master Track Remixing Engine
- `remix_master_with_vocal(original_audio_path, converted_vocal_path, stems_dict, output_filename)`:
  - Eliminates acapella-overwrite bugs: remixes the converted vocal stem with the non-vocal backing stems
    (drums, bass, guitar, piano, other) or the backing instrumental track into a polished, broadcast-ready stereo master audio mix.
  - Used automatically in generation pipeline Step 3 and `/jobs/{job_id}/voice-convert`.

## Related pages
- [Session workspace](session-workspace.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Backend & API](backend-api.md) | [Stem separator](stem-separator.md) | [Roadmap (v2)](../roadmap.md)
- [Singing Voice Conversion](../concepts/singing-voice-conversion.md) | [Neural SVC](neural-svc.md)
- [Global Hardware Coordinator](hardware-coordinator.md)

---

**Artist linkage (A1):** an [artist profile](../concepts/artist-domain.md) links a voice profile (`ArtistProfile.voice_profile_id`); album tracks resolved from that profile run SVC conversion automatically.
