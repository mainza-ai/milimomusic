---
title: MuScriptor
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [muscriptor, transcription, midi, musicxml]
aliases: [muscriptor]
---

# MuScriptor

**MuScriptor** (Kyutai × Mirelo) is a **multi-instrument Automatic Music Transcription**
model — **not a generator**. It turns any audio (including Milimo's own generated output)
into **MIDI + sheet music**. It is the keystone of Milimo's "producer-edit" story and is
now **fully integrated** as a git submodule.

## Integration status (current)
- **Git submodule**: `muscriptor/` → `https://github.com/muscriptor/muscriptor.git` (branch `main`),
  declared in `.gitmodules` — cloned with `--recurse-submodules`.
- **Backend wrapper**: [`MuScriptorProvider`](backend-api.md) (`transcription/muscriptor_provider.py`)
  loads `TranscriptionModel.load_model("small")` (Apple Silicon MPS/CPU) and exposes
  `transcribe(audio) → TranscriptionResult{midi_path, musicxml_path, notes, beat_grid, bpm, key}`.
- Runs in the [orchestration pipeline](../concepts/generation-pipeline.md) as **Step 4** and
  powers the standalone `/transcribe/upload` endpoint for user audio.

## Capabilities (as integrated)
- **Multi-instrument polyphonic transcription** — extracts Piano, Bass, Drums, Vocal melody.
- **MIDI (`.mid`)** — multi-track, Logic/Ableton/FL Studio/Pro Tools compatible.
- **MusicXML 3.1 sheet music (`.musicxml`)** — automatic **Grand Staff** engraving with
  Treble (𝄞) + Bass (𝄢) clefs, beat/tempo via `beat-this`.
- Optional **export formats** via `/transcribe/export/{job_id}/{format}`:
  `midi`, `musicxml`, `ableton`, `lrc`, `srt`.
- Note events are stored as JSON on the `Job` (see [Backend & API](backend-api.md)).

> [!NOTE] The provider has a graceful fallback transcription (C–Am–F–G chord progression)
> used if real inference errors, so the DAW always has data to render.

## Licensing
Code **MIT**; model weights **CC BY-NC 4.0 (non-commercial only)**. Terms prohibit
transcribing audio you don't hold rights to — surfaced in the app's upload flow
([LICENSES.md](../sources/readme.md)).

## Related pages
- [Backend & API](backend-api.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Session workspace](session-workspace.md) | [v2 reference projects](v2-references.md)
- [Roadmap (v2)](../roadmap.md)
