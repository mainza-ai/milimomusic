---
title: Stem Separation (Dual-Engine)
type: entity
created: 2026-08-20
updated: 2026-08-21
tags: [stems, separation, roformer, bs-roformer, muscriptor, per-instrument, daw]
aliases: [StemSeparator, stem separation, BS-Roformer, Roformer, real_separator]
---

# Stem Separation (Dual-Engine)

Milimo produces DAW stems from **two complementary engines** and lets the user choose
which source the DAW reflects — production-grade, never a single hard-coded path:

| Engine | Mechanism | What it yields | Source |
|--------|-----------|----------------|--------|
| **BS-Roformer** (primary) | Real neural **source separation** of the master audio | Dynamic isolated *audio* stems: `vocals`, `drums`, `bass`, `guitar`, `piano`, `other` (6-stem) | `transcription/real_separator.py` |
| **MuScriptor** (secondary) | **Note-level transcription** then per-instrument rendering | One audio stem per distinct instrument + GM program numbers | `transcription/instrument_stems.py` |

Both stem sets are written onto `Job.stems_json`, and the
[Session workspace (DAW)](session-workspace.md) shows a dynamic **stem-source toggle**
(Neural Stems ↔ Per-Instrument Parts) so the user picks how they work — BS-Roformer for real
isolated master sources, MuScriptor for instrument-by-instrument editing.

## BS-Roformer — real neural separation (`real_separator.py`)
- SOTA **BS-Roformer / MelBand-Roformer** architecture (via `audio-separator`), supporting dynamic stem counts (4, 5, 6+ stems).
- Accelerates across **CUDA → Apple Silicon MPS → CPU** natively.
- Returns a structured `SeparationResult` containing dynamic `stems`, `source_id`, `sources_available`, and `stem_count`.
- `separate_sources(master_wav, out_dir, job_id)` writes stems into `{job_id}_{stem_name}.wav` and returns dynamic URLs.
- Inference runs in a worker thread off the main event loop.

## MuScriptor per-instrument parts (`instrument_stems.py`)
- One audio stem per distinct `instrument` in the transcription's `notes[]`
  (pitch + timing + velocity → dependency-free numpy/soundfile tone rendering).
- Family-timbre mapping: `drums`, `bass`, `guitar`, `keys`, `sustained`, else `plucked`.
- `render_instrument_parts(notes, job_id, duration_sec)` returns `(parts, programs)`:
  - `parts` — `{instrument_name: "/audio/stems/{job}_part_{slug}.wav"}`
  - `programs` — `{instrument_name: GM program int}` (via a GM program table that mirrors
    MuScriptor's own web app: Acoustic Piano=0, Electric Bass=33, Voice=52, Flutes=73, …).
- **File-namespacing:** per-instrument files use a `_part_` prefix so they can never collide
  with the neural `{job_id}_drums.wav` etc.

> [!NOTE] Honest distinction: BS-Roformer produces **real separated audio** from the waveform;
> MuScriptor parts are a *faithful per-instrument rendering of the transcribed notes*
> (a listening/editing aid), not a neural separation of the original. Both are real data the
> DAW plays; neither is a fake/oscillator placeholder.

## `stems_json` shape
```json
{
  "vocals": "/audio/stems/<job>_vocals.wav",
  "drums":  "/audio/stems/<job>_drums.wav",
  "bass":   "/audio/stems/<job>_bass.wav",
  "guitar": "/audio/stems/<job>_guitar.wav",
  "piano":  "/audio/stems/<job>_piano.wav",
  "other":  "/audio/stems/<job>_other.wav",
  "stems_source": "bs_roformer_6stem",
  "instrumental_parts": { "Drums": "/audio/stems/<job>_part_drums.wav", "Electric Bass": "..." },
  "instrument_programs": { "Drums": 0, "Electric Bass": 33 },
  "sources_available": ["bs_roformer_6stem", "muscriptor"],
  "default_source": "muscriptor"
}
```

## Production resilience
- Neural separation failure (heavy model load, missing weights, resource pressure) is **non-fatal**:
  the [orchestration pipeline](../concepts/generation-pipeline.md) degrades gracefully to
  the per-instrument parts rather than failing the whole job.
- The legacy mid-side DSP filter-bank (`transcription/stem_separator.py`) was removed in Fix 3 in favor of the dual neural + MuScriptor architecture.

## Related pages
- [Orchestration pipeline](../concepts/generation-pipeline.md) | [Backend & API](backend-api.md)
- [MuScriptor](muscriptor.md) | [Session workspace](session-workspace.md)
- [Matchering mastering](matchering-mastering.md)
