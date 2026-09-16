---
title: Inpainting & Segment Repair Architecture
type: concept
created: 2026-08-19
updated: 2026-09-16
sources: [sources/inpainting-debug.md]
tags: [inpainting, repair, audio-domain, crossfade, beatgrid, lm-guided]
aliases: [LM-guided repair, Inpainting pipeline, Segment Repair]
---

# Inpainting & Segment Repair Architecture

**Inpainting & Segment Repair** is the strategy behind Milimo's
[Repair Segment](../entities/inpainting.md). It allows creators to selectively rewrite flawed, clipping, or uninspired sections of a track while preserving the remainder.

---

## 1. Evolution Across Generations

| Phase | Engine / Domain | Mechanism | Limitations |
|-------|-----------------|-----------|-------------|
| **Phase 1 (Legacy)** | HeartMuLa (RVQ Tokens) | Masked language model token infill spliced in RVQ space (12.5 Hz), then HeartCodec decoded. | Coupled to local token caches (`.pt`), sensitive to RVQ codebook shifts, silent or repeating on missing history. |
| **Phase 2 (Milimo v2)** | Active Provider (MiniMax Music 3 / MLX) + Audio Domain | Beat-grid downbeat alignment, locked continuation caption, infill synthesis, and dual equal-power sine/cosine crossfading. | Robust across providers; eliminates token cache dependencies and seam artifacts. |

---

## 2. Phase 2 Audio-Domain Repair Workflow

```
1. LOCATE & ANALYZE
   parent_audio = _resolve_audio_file(job.audio_path)
   extract_audio_musical_attributes(BPM, beat grid)

2. BEAT-GRID ALIGNMENT
   snap start_sec, end_sec to downbeats/beats

3. INFILL GENERATION
   generate segment of duration (end - start + 2*crossfade)
   locked to parent BPM, key, and instrumentation

4. DUAL EQUAL-POWER SPLICING
   Part 1 (0 -> start + crossfade)
   Infill (crossfade -> end - start + crossfade)
   Part 3 (end - crossfade -> duration)
   w_out = cos(pi/2 * t), w_in = sin(pi/2 * t)

5. POST-REPAIR CASCADE
   BS-Roformer 6-stem separation -> MuScriptor transcription -> WhisperX lyric sync
```

---

## Related pages
- [Repair Segment (Inpainting Service)](../entities/inpainting.md)
- [Track Extension](track-extension.md)
- [MiniMax Music 3](../entities/minimax-music3.md)
