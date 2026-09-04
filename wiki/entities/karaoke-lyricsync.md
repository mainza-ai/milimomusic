---
title: Karaoke & Lyric Sync
type: entity
created: 2026-08-20
updated: 2026-09-03
tags: [karaoke, lyrics, sync, lrc, srt, timestamps, mms-fa, forced-alignment]
aliases: [LyricSyncEngine, lyric sync, karaoke, MMS_FA]
---

# Karaoke & Lyric Sync

The **LyricSyncEngine** (`backend/app/transcription/karaoke.py`) produces **sub-100ms word- and line-level timestamps**
so tracks can play synchronized karaoke lyrics and export standards-compliant `.lrc` and `.srt` files. It backs
the **Lyrics** mode of the [Session workspace](session-workspace.md), the bottom persistent audio player, and the 4-step orchestration pipeline.

## 3-Tier Architecture

To achieve production-grade synchronization without failing on silent or synthetic tracks, `LyricSyncEngine` implements a resilient 3-tier strategy:

1. **Tier 1: Neural Acoustic Forced Alignment (`torchaudio.pipelines.MMS_FA`)**:
   - Evaluates the isolated vocal stem (`BS-Roformer` or `HTDemucs`), resampled to 16,000 Hz mono.
   - Cleans the lyric transcript into normalized dictionary tokens (`a-z'`).
   - Runs CTC acoustic emission through Meta's Multilingual Forced Aligner backbone.
   - Generates exact millisecond-accurate word boundaries (`TimedWord`), preserving vocal pauses and instrumental solos without lyrics bleeding across silence.
2. **Tier 2: Multi-interval Adaptive Acoustic VAD**:
   - If neural alignment is unavailable or fails, computes a dynamic 75th-percentile vocal energy envelope (`p75 * 0.25`).
   - Segments the audio into active singing intervals with 400ms hangover and 300ms minimum segment length.
   - Maps stanzas directly to discrete vocal bursts, preserving instrumental breaks.
3. **Tier 3: Syllable-Weighted Heuristic Fallback**:
   - Proportional syllable timing used when audio is absent or completely silent (offline testing).

## Section Header Deconfliction & Frontend Tracking

- **Section Headers (`is_section: true`)**: Tags such as `[Verse 1]`, `[Chorus]`, and `[Outro]` are identified, isolated, and given cue timestamps during the silent gaps preceding the sung section. They strictly never overlap with sung lyric lines.
- **Frontend Active-Line Logic**: In `SessionWorkspace.tsx` and `GlobalAudioPlayer.tsx`, `activeLineIndex` ignores section headers (`!line.is_section`), preventing header banners from stealing the active lyric highlight or disrupting word-level progress.
- **On-Demand Re-alignment**: Recomputing timestamps via `POST /tracks/{job_id}/realign_lyrics` automatically resolves the track's separated vocal stem (`stems.vocals` or `stems.part_vocals`) and true audio duration.

## Endpoints
- `POST /tracks/{job_id}/realign_lyrics` → recompute acoustic alignment on demand.
- `GET /tracks/{job_id}/lrc` & `GET /transcribe/export/{job_id}/lrc` → export `.lrc` synchronized lyrics.
- `GET /transcribe/export/{job_id}/srt` → export `.srt` subtitles.

## Related pages
- [Session workspace](session-workspace.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Backend & API](backend-api.md) | [MiniMax Music 3](minimax-music3.md)

