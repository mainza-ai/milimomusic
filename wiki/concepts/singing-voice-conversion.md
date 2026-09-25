---
title: Singing Voice Conversion (RVC v2 & Acoustic DSP)
type: concept
created: 2026-09-03
updated: 2026-09-24
tags: [rvc, svc, rmvpe, hubert, voice, acoustic-dsp, vocal-studio, vocal-booth]
aliases: [RVC, SVC, voice conversion]
sources: [production-readiness-plan.md]
---

# Singing Voice Conversion (RVC v2 & Acoustic DSP)

Milimo Music's **Singing Voice Conversion (SVC)** pipeline allows producers to clone vocal
identities, ingest custom vocal datasets via file upload or live in-browser microphone capture,
and convert isolated vocal stems into target vocal identities with real-time A/B split auditioning
and master track remixing.

## 1. Dataset Ingestion & Acoustic Profiling

When a user trains a new voice identity in the **Vocal Studio**:
1. **Dual Ingestion Paths**:
   - **File Upload**: Upload clean solo vocal recordings (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) or `.zip` archives.
   - **Live In-Browser Vocal Booth**: Direct microphone recording via `VocalBoothRecorder.tsx` with Web Audio API `AudioContext` and `AnalyserNode` frequency VU peak meters, 3-second countdown, and instant take review playback.
2. **Mandatory Consent**: Enforces legal rights verification before processing.
3. **Acoustic Feature Extraction**:
   - **Fundamental Frequency ($F_0$)**: Extracted using probabilistic YIN (`librosa.pyin`) across C2 (~65 Hz) to C7 (~2093 Hz).
   - **Spectral Centroid**: Measures vocal brightness distribution across the frequency spectrum.
   - **Spectral Rolloff & RMS Energy**: Measures harmonic dispersion and average vocal loudness.
4. **Normalized Sample Preview**: Generates an unclipped, peak-normalized preview audio file saved to `generated_audio/voice_previews/` and instantly previewable in the UI.

## 2. Interactive Vocal DSP Rack & Musical Transposition

The studio's `VocalDSPRack.tsx` provides musical production parameters:
- **Pitch Transposition**: Full octave $-12$ to $+12$ semitone range with quick production presets:
  - `+12 Octave Up (Male → Female)`
  - `-12 Octave Down (Female → Male)`
  - `+3 Minor 3rd Harmonizer`
  - `+7 Perfect 5th Harmonizer`
- **Phase-Locked Formant Preservation**: Automatically warps spectral resonance bands inversely to pitch shifts, preventing unnatural chipmunk and slow-down artifacts.
- **Dry / Wet Balance**: Linear blend between dry original vocal stem ($0\%$) and converted singer timbre ($100\%$).
- **F0 Extraction Algorithms**: Configurable between RMVPE, CREPE, Harvest, and PM.

## 3. High-Fidelity Neural SVC & Formant DSP Shaping Engine

When running without a pre-trained `.pth` checkpoint, the pipeline executes acoustic timbre transfer through the [Neural SVC](../entities/neural-svc.md) engine (`backend/app/services/voice/neural_svc.py`):
- **Phase-Locked Pitch Shifting**: Shifts pitch semitones using phase vocoder preservation.
- **Formant & Equalization Tuning**:
  - `aria` (Ethereal Pop): Highpass filter at 120 Hz, presence boost at 3.2 kHz (+3.0 dB), air brilliance shelf at 8.5 kHz (+2.5 dB).
  - `marcus` (Warm Soul/R&B): Chest resonance boost at 350 Hz (+3.5 dB), warmth at 1.2 kHz (+1.5 dB), top-end taming at 6.5 kHz (-1.5 dB).
  - Custom profiles: Adaptive spectral envelope warping based on target profile reference recordings.
- **Formant Preservation Compensation**: Adjusts resonance bands opposite to pitch shifts (+/- 12 semitones) to preserve natural vocal tract character.
- **Wet / Dry Blend**: Seamless blending between original dry vocal and transformed vocal ($0\%$ to $100\%$).

## 4. Master Track Remixing & Tri-State Audition Transport

To avoid acapella-overwrite bugs where backing instruments are lost during voice conversion:
- The `remix_master_with_vocal` engine combines the converted vocal stem with all non-vocal stems (drums, bass, guitar, piano, other) or the backing instrumental track.
- Re-aligns sample rates, pads waveforms, balances gains, and applies peak normalization to 0.95.
- Produces a complete, cohesive stereo master song for `Job.audio_path` while preserving the converted vocal stem in `stems_json["vocals"]`.
- The frontend `VocalAuditionPlayer.tsx` provides immediate tri-state A/B switching between `Original Vocal Stem`, `Converted Vocal Stem`, and `Full Master Remix` with waveform scrub bar and 1-click routing to the multitrack DAW workspace.

## Related pages

- [Voice Studio (SVC)](../entities/voice-service.md) · [Neural SVC](../entities/neural-svc.md)
- [Global Hardware Coordinator](../entities/hardware-coordinator.md) · [Task Queue](../entities/task-queue.md)
- [Track extension](track-extension.md) · [Orchestration pipeline](generation-pipeline.md)
