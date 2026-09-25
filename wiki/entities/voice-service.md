---
title: Voice Studio (SVC)
type: entity
created: 2026-08-20
updated: 2026-09-24
tags: [voice, svc, rvc, cloning, consent, singing, acoustic-dsp, vocal-booth]
aliases: [VoiceService, Voice Training Studio, Vocal Studio, SVC]
---

# Voice Studio (SVC)

The **Voice Service** (`services/voice_service.py` + frontend `VocalStudioView` and `VoiceStudioModal`) is
Milimo's offline **Singing Voice Conversion (SVC)**, vocal-identity cloning, and vocal production workstation.
It enables producers to train custom singer identities from audio files or live microphone input,
transpose musical pitch with phase-locked formant tracking, transform isolated vocal stems into target singer
timbres, and automatically remix polished stereo master songs.

## 1. Workstation Architecture (3-Zone Vocal Studio)
In the frontend, Vocal Studio is exposed as a first-class navigation view (`vocal-studio` route in `App.tsx`)
and as an expansive modal workstation launched directly from the DAW arrange timeline or composer:
- **Zone 1: Master Studio Bar**:
  - 3-Way Mode Switcher: `Voice Conversion` vs `Voice Identities` vs `Live Vocal Booth`.
  - Source Song Dropdown Selector with automatic vocal stem detection badge (`Vocal Stem Isolated ✓`).
  - Active Voice Identity chip with 1-click sample audio auditioning.
- **Zone 2: Dual Workspaces**:
  - **Conversion Mode**: Target voice identity selector on the left; [Vocal DSP Rack](#3-vocal-dsp-rack--conversion-engine) on the right.
  - **Identity Library Mode**: Dataset ingestion form (name, description, F0 method, consent gate) alongside saved voice cards with median $F_0$ (Hz) and acoustic timbre badges.
  - **Vocal Booth Mode**: [In-Browser Microphone Vocal Booth](#2-in-browser-vocal-booth-microphone-recording).
- **Zone 3: Full-Width Audition & A/B Transport**:
  - Tri-state player (`Converted Vocal` vs `Original Stem` vs `Full Master Remix`) with scrubber, skip controls, WAV export, and 1-click hand-off to the DAW workspace.

## 2. In-Browser Vocal Booth (Microphone Recording)
- Component: `VocalBoothRecorder.tsx`.
- Uses Web Audio API `AudioContext` and `AnalyserNode` to display a 16-band real-time VU frequency peak meter during recording.
- Features a 3-second animated countdown, live elapsed timecode, take review player, and direct conversion of browser audio blobs into `File` payloads passed to `voiceApi.createProfile`.

## 3. Vocal DSP Rack & Conversion Engine
- Component: `VocalDSPRack.tsx`.
- **Musical Pitch Transposition**: Transposes vocal pitch across $-12$ to $+12$ semitones with quick octave presets (`+12 Octave Up M→F`, `-12 Octave Down F→M`, `+3 Minor 3rd`, `+7 Perfect 5th`).
- **Phase-Locked Formant Preservation**: Compensates resonance bands opposite to pitch shifts to prevent chipmunk / slow-down artifacts.
- **Wet / Dry Mix Control**: Studio balance slider blending $0\%$ (dry original vocal) to $100\%$ (target singer timbre).
- **Pitch Extraction Algorithms**: RMVPE (recommended high-quality vocal tracking), CREPE, Harvest, and PM.
- **Backend Neural SVC Engine**: `NeuralSVCService` (`backend/app/services/voice/neural_svc.py`) executing STFT spectral morphing, harmonic excitation residual, and target envelope modulation, with automatic fallback to biquad formant shaping.

## 4. Voice Profiles & Dataset Ingestion
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

## 5. Master Track Remixing Engine
- `remix_master_with_vocal(original_audio_path, converted_vocal_path, stems_dict, output_filename)`:
  - Eliminates acapella-overwrite bugs: remixes the converted vocal stem with the non-vocal backing stems
    (drums, bass, guitar, piano, other) or the backing instrumental track into a polished, broadcast-ready stereo master audio mix.
  - Used automatically in generation pipeline Step 3 and `/jobs/{job_id}/voice-convert`.

## 6. DAW Multitrack Integration
- Clicking "Voice Convert" on any vocal track in `ArrangeTimeline.tsx` invokes `useModalStore.openVoiceConvert(job, track.audioUrl)`.
- `App.tsx` binds this state to open `VoiceStudioModal` with the track and vocal stem pre-selected, allowing instant round-trip vocal re-voicing without leaving DAW context.

## Related pages
- [Session workspace](session-workspace.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Backend & API](backend-api.md) | [Stem separator](stem-separator.md) | [Roadmap (v2)](../roadmap.md)
- [Singing Voice Conversion](../concepts/singing-voice-conversion.md) | [Neural SVC](neural-svc.md)
- [Global Hardware Coordinator](hardware-coordinator.md)

---

**Artist linkage (A1):** an [artist profile](../concepts/artist-domain.md) links a voice profile (`ArtistProfile.voice_profile_id`); album tracks resolved from that profile run SVC conversion automatically.
