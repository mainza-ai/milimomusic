---
title: Neural & Spectral Drum Tracker
type: entity
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [drums, dsp, transcription, midi, librosa, symbolic, mulacover]
aliases: [Drum Tracker, drum_tracker, Drum Conditioning Engine]
---

# Neural & Spectral Drum Tracker

The **Drum Tracker** (`backend/app/transcription/drum_tracker.py`) provides specialized percussive transcription and rhythm conditioning extraction from separated drum stems or full audio tracks.

## Overview

While standard pitch-detection models (like Basic Pitch or standard MuScriptor) excel on melodic instruments (vocals, keys, brass), they fail on non-tonal transients produced by drums. The Drum Tracker combines **spectral flux onset envelope detection**, **sub-band energy gating**, and **musical quantization** to extract standard General MIDI drum events.

## Frequency Sub-Bands & Note Mapping

| Instrument | MIDI Note | Frequency Range | Detection Criteria |
|------------|-----------|-----------------|---------------------|
| **Kick Drum** | Note 36 (`C1`) | Sub-bass ($20\text{ Hz} - 120\text{ Hz}$) | High low-band transient energy with dominant sub-bass peak |
| **Snare Drum** | Note 38 (`D1`) | Mid-body ($180\text{ Hz} - 1200\text{ Hz}$) | Concentrated mid-frequency punch with rapid noise decay |
| **Hi-Hat** | Note 42 (`F#1`) | High air ($5000\text{ Hz} - 16000\text{ Hz}$) | High-frequency spectral flux with crisp transient decay |

## Algorithmic Workflow

```
┌────────────────────────────────────────────────────────┐
│  ISOLATED DRUMS STEM (from BS-Roformer / HTDemucs)     │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ Librosa Spectral Flux     │
              │ Onset Envelope Detection  │
              └─────────────┬─────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
┌────────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
│ Low-Pass Filter │ │Band-Pass (Mid)│ │High-Pass Filter │
│ (20 - 120 Hz)   │ │(180 - 1200 Hz)│ │(5000 - 16000 Hz)│
│ Kick Candidate  │ │Snare Candidate│ │Hi-Hat Candidate │
└────────┬────────┘ └───────┬───────┘ └────────┬────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │ Energy Gating & De-Duplication
              ┌─────────────▼─────────────┐
              │ 16th-Note Grid            │
              │ Tempo Alignment (BPM)     │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ (onset, pitch, duration)  │
              │ SymbolicCondition Drums   │
              └───────────────────────────┘
```

## Integration with MuLaCover

In `SymbolicHub.transcribe_milimo_neural()`, the drum events extracted by `transcribe_drums_from_stem` are serialized into the `SymbolicCondition.drums` tensor:
- Format: integer triplets `(onset_16th, pitch, duration_16th)` where pitch is 36 (kick), 38 (snare), or 42 (hi-hat).
- Injected into channel 9 of the exported `drums.mid` file for DAW multitrack review and PianoRoll editing.
- Conditioned directly onto the MuLaCover autoregressive model alongside lead melody and chords.

## Related Pages

- [MuLaCover](mulacover.md)
- [MuScriptor](muscriptor.md)
- [Stem Separator](stem-separator.md)
- [Session Workspace](session-workspace.md)
