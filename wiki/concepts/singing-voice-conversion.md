---
title: Singing Voice Conversion (RVC v2 & Acoustic DSP)
type: concept
created: 2026-09-03
updated: 2026-09-07
tags: [rvc, svc, rmvpe, hubert, voice, acoustic-dsp]
aliases: [RVC, SVC, voice conversion]
sources: [production-readiness-plan.md]
---

# Singing Voice Conversion (RVC v2 & Acoustic DSP)

Milimo Music's **Singing Voice Conversion (SVC)** pipeline allows users to clone vocal
identities, ingest custom vocal datasets with legal consent verification, and convert isolated vocal stems
into target vocal identities while maintaining a polished stereo master track mix.

## 1. Dataset Ingestion & Acoustic Profiling

When a user trains a new voice identity in the **Voice Training Studio**:
1. **Upload Formats**: Users upload clean solo vocal recordings (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) or `.zip` archives.
2. **Mandatory Consent**: Enforces legal rights verification before processing.
3. **Acoustic Feature Extraction**:
   - **Fundamental Frequency ($F_0$)**: Extracted using probabilistic YIN (`librosa.pyin`) across C2 (~65 Hz) to C7 (~2093 Hz).
   - **Spectral Centroid**: Measures vocal brightness distribution across the frequency spectrum.
   - **Spectral Rolloff & RMS Energy**: Measures harmonic dispersion and average vocal loudness.
4. **Normalized Sample Preview**: Generates an unclipped, peak-normalized preview audio file saved to `generated_audio/voice_previews/` and instantly previewable in the UI.

## 2. Neural RVC v2 Checkpoint Inference

If a user imports or places an RVC v2 `.pth` model checkpoint in `data/voice_profiles/{profile_id}.pth`:
- Loads model weights (`net_g`, `params`, `weight`) onto hardware (`mps`, `cuda`, or `cpu`).
- Extracts pitch curve and applies semitone pitch shifting via `torchaudio.functional.pitch_shift`.
- Generates converted vocal waveform and applies wet/dry ratio blending.

## 3. High-Fidelity Acoustic & Formant DSP Shaping Engine

When running without a pre-trained `.pth` checkpoint, the pipeline executes acoustic timbre shaping:
- **Formant & Equalization Tuning**:
  - `aria` (Ethereal Pop): Highpass filter at 120 Hz, presence boost at 3.2 kHz (+3.0 dB), air brilliance shelf at 8.5 kHz (+2.5 dB).
  - `marcus` (Warm Soul/R&B): Chest resonance boost at 350 Hz (+3.5 dB), warmth at 1.2 kHz (+1.5 dB), top-end taming at 6.5 kHz (-1.5 dB).
  - Custom profiles: Adaptive formant filtering based on extracted $F_0$ and spectral centroid.
- **Formant Preservation Compensation**: Adjusts resonance bands opposite to pitch shifts (+/- 12 semitones) to preserve natural vocal tract character.
- **Wet / Dry Blend**: Seamless blending between original dry vocal and transformed vocal ($0\%$ to $100\%$).

## 4. Master Track Remixing Engine

To avoid acapella-overwrite bugs where backing instruments are lost during voice conversion:
- The `remix_master_with_vocal` engine combines the converted vocal stem with all non-vocal stems (drums, bass, guitar, piano, other) or the backing instrumental track.
- Re-aligns sample rates, pads waveforms, balances gains, and applies peak normalization to 0.95.
- Produces a complete, cohesive stereo master song for `Job.audio_path` while preserving the converted vocal stem in `stems_json["vocals"]`.

## Related pages

- [Voice Studio (SVC)](../entities/voice-service.md) · [Task Queue](../entities/task-queue.md)
- [Track extension](track-extension.md) · [Orchestration pipeline](generation-pipeline.md)
