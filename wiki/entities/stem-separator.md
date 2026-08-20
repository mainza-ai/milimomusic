---
title: Stem Separation (Dual-Engine)
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [stems, separation, demucs, htdemucs, muscriptor, per-instrument, daw]
aliases: [StemSeparator, stem separation, HTDemucs, Demucs, real_separator]
---

# Stem Separation (Dual-Engine)

Milimo produces DAW stems from **two complementary engines** and lets the user choose
which source the DAW reflects — production-grade, never a single hard-coded path:

| Engine | Mechanism | What it yields | Source |
|--------|-----------|----------------|--------|
| **HTDemucs** (primary) | Real neural **source separation** of the master audio | 4 isolated *audio* stems: `vocals`, `drums`, `bass`, `other` | `transcription/real_separator.py` |
| **MuScriptor** (secondary) | **Note-level transcription** then per-instrument rendering | one audio stem per distinct instrument + GM program numbers | `transcription/instrument_stems.py` |

Both stem sets are written onto `Job.stems_json`, and the
[Session workspace (DAW)](session-workspace.md) shows a **stem-source toggle**
(4 Master Stems ↔ Per-Instrument) so the user picks how they work — HTDemucs for real
isolated master sources, MuScriptor for instrument-by-instrument editing.

## HTDemucs — real neural separation (`real_separator.py`)
- Uses Meta's open-source **HTDemucs** (`demucs.pretrained.get_model("htdemucs")`), cached as
  a process singleton (loaded once per server lifetime, cross-thread lock).
- Runs **CUDA if available, else CPU** — deliberately *not* MPS (HTDemucs' ungated conv1d
  op has >65536 output channels that the MPS backend can't run in this torch release).
- `separate_sources(master_wav, out_dir, job_id, shifts=1)` → real
  `{job_id}_{vocals|drums|bass|other}.wav` + `/audio/stems/...` URLs.
- Runs off the event loop via a worker thread.

## MuScriptor per-instrument parts (`instrument_stems.py`)
- One audio stem per distinct `instrument` in the transcription's `notes[]`
  (pitch + timing + velocity → dependency-free numpy/soundfile tone rendering).
- Family-timbre mapping: `drums`, `bass`, `guitar`, `keys`, `sustained`, else `plucked`.
- `render_instrument_parts(notes, job_id, duration_sec)` returns `(parts, programs)`:
  - `parts` — `{instrument_name: "/audio/stems/{job}_part_{slug}.wav"}`
  - `programs` — `{instrument_name: GM program int}` (via a GM program table that mirrors
    MuScriptor's own web app: Acoustic Piano=0, Electric Bass=33, Voice=52, Flutes=73, …).
- **File-namespacing:** per-instrument files use a `_part_` prefix so they can never collide
  with the HTDemucs `{job_id}_drums.wav` etc.

> [!NOTE] Honest distinction: HTDemucs produces **real separated audio** from the waveform;
> MuScriptor parts are a *faithful per-instrument rendering of the transcribed notes*
> (a listening/editing aid), not a neural separation of the original. Both are real data the
> DAW plays; neither is a fake/oscillator placeholder. Separating *real recording audio*
> is HTDemucs' job; instrument-level note editing is MuScriptor's.

## `stems_json` shape
```json
{
  "vocals": "/audio/stems/<job>_vocals.wav",
  "drums":  "/audio/stems/<job>_drums.wav",
  "bass":   "/audio/stems/<job>_bass.wav",
  "other":  "/audio/stems/<job>_other.wav",
  "stems_source": "htdemucs",
  "instrumental_parts": { "Drums": "/audio/stems/<job>_part_drums.wav", "Electric Bass": "..." },
  "instrument_programs": { "Drums": 0, "Electric Bass": 33 },
  "sources_available": ["htdemucs", "muscriptor"],
  "default_source": "htdemucs"
}
```

## Production resilience
- HTDemucs failure (heavy model load, missing weights, resource pressure) is **non-fatal**:
  the [orchestration pipeline](../concepts/generation-pipeline.md) degrades gracefully to
  the per-instrument parts rather than failing the whole job.
- The legacy DSP filter-bank (`stem_separator.py`) remains only as a backward-compat
  fallback for very old jobs without instrument parts — not the primary DAW source.

## Related pages
- [Orchestration pipeline](../concepts/generation-pipeline.md) | [Backend & API](backend-api.md)
- [MuScriptor](muscriptor.md) | [Session workspace](session-workspace.md)
- [Matchering mastering](matchering-mastering.md)
