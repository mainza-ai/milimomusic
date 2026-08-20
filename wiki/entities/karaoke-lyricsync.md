---
title: Karaoke & Lyric Sync
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [karaoke, lyrics, sync, lrc, srt, timestamps]
aliases: [LyricSyncEngine, lyric sync, karaoke]
---

# Karaoke & Lyric Sync

The **LyricSyncEngine** (`transcription/karaoke.py`) produces **word/line-level timestamps**
so tracks can play synchronized karaoke lyrics and export `.lrc` / `.srt` files. It backs
the **Lyrics** mode of the [Session workspace](session-workspace.md) and the pipeline's
timed-lyric step.

## Capabilities
- `align_lyrics(lyrics, duration_sec)` → `TimedLine[]` (`{text, start, end, words[]}`),
  stored on the `Job` as `timed_lyrics_json`.
- `generate_lrc(...)` → standard `.lrc` lyrics files.
- `generate_srt(...)` → standard `.srt` subtitle files.
- Exported via `GET /transcribe/export/{job_id}/{format}` for `lrc` and `srt`.

> [!WARNING] **Fidelity note.** Current alignment **evenly distributes** words/lines across
> the total duration (uniform partitioning, no acoustic forced alignment). The v2 plan lists
> **WhisperX** forced alignment (sub-100ms word timestamps) as the upgrade path. Handle the
> generated timestamps as estimates.

## Related pages
- [Session workspace](session-workspace.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Backend & API](backend-api.md)
