---
title: Music Video Delivery & QC Specs
type: concept
tags: [video, delivery, qc, loudness, eb128, vevo, youtube, licensing, models, standards]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [VideoDeliverySpecs, DeliveryQC, MusicVideoStandards]
---

# Music Video Delivery & QC Specs

What a finished music video must actually *be* to be accepted by a DSP, and what
2026's generation engines can and cannot supply. Compiled for the
[Music Video Pipeline Audit 2026-11](../music-video-pipeline-audit-2026-11.md);
external facts are cited to primary sources, with uncertainty flagged in §10.

---

## 1. One video is not a deliverable — it is four

Milimo currently produces one H.264/AAC file per render
([audit §5, D5](../music-video-pipeline-audit-2026-11.md)). A production pipeline
needs at minimum:

| Target | Container / codec | Video | Audio | Constraints |
|---|---|---|---|---|
| **YouTube upload** | MP4, moov at front, no edit lists | H.264 High, progressive, 2 consecutive B-frames, closed GOP = fps/2, 4:2:0, BT.709 | AAC-LC **48 kHz, ~384 kbps** | 1080p ≈ 8 Mbps (24–30 fps) / 12 (48–60); 2160p ≈ 35–45 / 53–68 |
| **Vevo (DSP delivery)** | .mov/.mp4 | H.264 **or ProRes**; 2K/4K **ProRes only** | **AAC 44.1 kHz 320 kbps CBR** (HD/vertical); **LPCM 48 kHz** for 2K/4K | HD 20 Mbps (min 7.5), 2K 30, 4K 60; de-interlaced; **no slates, title or end cards**; needs ISRC + `YouTubeChannelName` metadata |
| **Spotify Canvas** | MP4 or JPG, **silent** | 9:16 vertical | none | **3–8 s**, height 720–1080 px |
| **Archive / broadcast** | ProRes 422 HQ or lossless | native, 23.976–60 | LPCM | −23 LUFS / −1 dBTP (EBU R128) |

Two traps worth designing around: vertical or square clips **≤3 min** are classified
by YouTube as Shorts and are **not deliverable to Vevo**; and a video already
published on another channel cannot be delivered to Vevo. Captions are demanded
as **TTML** by some DSPs, which no current Milimo path can emit.

## 2. Loudness: a music video needs two masters

| Context | Target | True peak |
|---|---|---|
| Broadcast (EBU R 128 v5) | **−23.0 LUFS**, ±0.2 LU in QC | ≤ −1 dBTP |
| Streaming/platform (song master, untouched) | ≈ **−14 LUFS** class | ≤ −1 dBTP; Spotify asks ≤ −2 dBTP if the master is louder than −14 |
| Netflix (reference) | −27 LUFS | ≤ −2 dBTP |

The rule that matters: **do not re-normalise the song to −23 for the video** —
the music video would sit 9 dB quiet next to the record. Ship the song master as
the video's audio track and produce a separate broadcast deliverable.

Milimo already owns a real `−14 LUFS` mastering stage with a true-peak guard
(`backend/app/transcription/mastering.py`), and the video path ignores it
([audit D3](../music-video-pipeline-audit-2026-11.md)).

## 3. QC: the FFmpeg-defaults trap

A production QC gate must override FFmpeg's defaults. A documented test planted
six faults (1 s black gap, 5-frame black gap, 1.52 s freeze, single repeated
frame, 0.3 s audio dropout) and **the default detectors found one of six** —
all three filters default to a 2 s minimum. At `blackdetect d=0.1`,
`freezedetect d=0.05`, `silencedetect d=0.1` all six were caught, while a
legitimate held title card was still correctly flagged.

Checklist by layer:

| Layer | Checks |
|---|---|
| **Conformance** | container/codec/profile/level, resolution, SAR/DAR, frame count ↔ duration, channel layout, decode errors |
| **Picture** | `blackdetect`, `freezedetect`, flash frames, `cropdetect` (letterbox), `idet` (interlace), `signalstats` (range), `scdet` (cuts) |
| **Sync** | structural only: per-stream `start_time`/`duration` mismatch. **FFmpeg cannot judge lip sync** — use an A/V cross-correlation offset check (≤1 frame) plus a human spot-check |
| **Audio** | `ebur128` integrated + true peak vs target, `silencedetect`, clipping, mono compatibility |
| **Delivery** | loudness report, captions present/valid, duration vs spec (±1.5 frames), text inside title-safe |

Colour: SDR delivery is Rec.709 limited-range 4:2:0 with tags written (players
assume BT.709 when untagged and convert full→limited). HDR needs BT.2020 + PQ or
HLG, 10/12-bit, and measured metadata — not a tone-mapped SDR export.

## 4. Frame-exactness is the whole ballgame

Every model quantises clip length: H3 renders `5 + 17k` frames, LTX `frames % 8 == 1`
(default 121), Wan `(F−1) % 4 == 0`, HuMo 97 frames @25 fps, Veo 3.1 offers 4/6/8 s.
Pipelines that work **over-render, trim from frame 0, and conform every clip to one
timebase** so the sum of frames equals the song's frame count. Pipelines that
concatenate raw model output drift progressively: ~0.2 s per clip becomes ~4 s
over a 20-clip song.

Milimo already does the hard part (`clip_normalise.py`, `verify_assembly`) — but
only per clip. The missing assertion is global: **Σ clamp(frames) == song frames
at delivery fps**, checked as a test, not a log line.

## 5. Engine landscape, late 2026 (delivery-relevant subset)

Local/open, licence-first for a commercially deployable product:

| Model | Licence | Clip / res | Audio | Local footprint |
|---|---|---|---|---|
| **Wan 2.2 A14B / TI2V-5B** | **Apache-2.0** | 480P/720P; 5B = 1280×704 @24 fps ~5 s | silent | 80 GB / 24 GB (≈8 GB with offload) |
| **Wan 2.2 S2V-14B** | **Apache-2.0** | 480P/720P, minute-scale | **audio-in → talking/singing video** | 80 GB class |
| **LTX-2.5** | LTX-2 Community (free < $10M revenue; watermark/provenance must stay) | 121-frame default, upscalers to 4K | **joint A/V** | ~32 GB VRAM |
| **HunyuanVideo 1.5** | Tencent Community — **excludes EU/UK/South Korea** | 480p/720p + SR | silent | ~14 GB with offload |
| **MiniMax H3 33B** | H3 Community — **excludes EU/UK/US/South Korea** | 4–15 s, 768p local | 32 kHz stereo | ≈21 GB INT8 |
| **SkyReels V3** | Skywork Community (conditional) | 5 s @720p; A2V ≤200 s audio | audio-in avatar | 24 GB+ |
| **CogVideoX 1.5 / Mochi** | Apache-2.0 | 480p class | silent | stale (no release since 2024) |

Audio-driven / lip-sync, with the licensing trap made explicit:

| Model | Task | Licence verdict |
|---|---|---|
| **EchoMimicV3** | image+audio → body | **Apache-2.0**, weights included, singing demos |
| **LatentSync 1.5/1.6** | re-dub, latent diffusion | Apache-2.0 code; ⚠ weights reported OpenRAIL++ and its detector non-commercial |
| **MuseTalk 1.5** | re-dub | MIT code+weights; dependencies carry own terms |
| **LivePortrait** | **video**-driven portrait animation | MIT, **but bundled InsightFace models are research-only** |
| **Wav2Lip / Sonic** | re-dub / portrait | **non-commercial** |
| **HunyuanVideo-Avatar** | multi-character avatar | Tencent Community (EU/UK/KR excluded) |
| **InfiniteTalk** | long-form | ⚠ conflicting reports (Apache-2.0 vs custom) |

**Singing caveat.** Nearly all of these are trained on *speech*. Sustained vowels
and phoneme timing are decent; melisma, wide vibrato and breathy onsets are not,
and an off-axis or mic-occluded mouth defeats them. The practical 2026 pattern is
to route singing shots to an **S2V/audio-conditioned** model fed the isolated
vocal stem, and to keep general T2V for b-roll.

## 6. Consistency: bound the drift, don't chase it

Ranked by cost-effectiveness (all are used by shipping pipelines):

1. **Reference conditioning** — Kling 4.0 allows up to 15 reference assets +
   10 keyframes; H3 Ref2VA 9 images/3 videos/3 audio; SkyReels V3 1–4 images.
2. **Last-frame chaining** — carry the previous clip's final frame as the next
   clip's first frame (AutoMV does this explicitly for continuity).
3. **Native multi-shot generation** — LTX-2.5 generates connected scenes in one
   pass, holding identity/lighting across cuts.
4. **Character bible** — a persisted description (face, hair, skin, clothing,
   age) injected into every prompt; AutoMV's ablations credit it with real gains.
5. **LoRA / IC-LoRA subject locks** — per-character fine-tunes.
6. **Reference-video motion transfer** — Wan 2.2 Animate, LivePortrait.

Milimo has none of these beyond "reuse the same portrait"; that is why its scenes
read as unrelated shots ([audit E6](../music-video-pipeline-audit-2026-11.md)).

## 7. What the strongest open song→video pipelines do

The only pipeline with a published benchmark is **AutoMV** (Apache-2.0,
arXiv 2512.12196): MIR preprocessing (structure + Demucs + Whisper + captioning)
→ screenwriter agent → director agent → per-shot renderer that **routes singing
shots to an S2V model with the vocal stem** and b-roll to a general video model →
**verifier agent (VLM)** that scores realism, instruction-following and character
consistency and triggers regeneration → assembly. Benchmark: 60 songs, 12
criteria, expert score 2.42; the authors note LLM judges still trail humans.

Locally deployable equivalents (MusicVision, FeverSlop, muvid, Glitchframe,
song2mv, mondo-music-video) are engineering-grade rather than benchmarked, and
their shared lessons are the useful part:

- over-render and trim to a single frame grid; assert total duration within one frame;
- persist word timings **and manual corrections** so re-runs don't clobber them;
- export an EDL/FCPXML alongside the MP4 so the cut is editable elsewhere;
- three of four muvid modes are free, deterministic and key-free — a good pattern
  for "works with no GPU, no API key".

## 8. Standards for documenting generated media

- **Provenance packaging**: RO-Crate 1.3 (folder + `ro-crate-metadata.json`,
  `CreateAction` with `instrument`/`object`/`result`), Croissant 1.1 (JSON-LD with
  W3C PROV-O; Hugging Face auto-generates it), `CITATION.cff` 1.2 (GitHub renders
  it; Zenodo reads it for the DOI record).
- **Content credentials**: C2PA 2.4 / Content Credentials — signed *claims*, not
  AI detection; a 2026 formal-methods review disputes its security guarantees, so
  treat it as disclosure/attribution and say so.
- **Symbolic**: MusicXML 4.0 is a W3C Community Group Final Report (not a
  Standard), and `.mxl` is the compressed container; MIDI 2.0 requires UMP/SMF2 —
  emitting MIDI 1.0 SMF is not "MIDI 2.0".
- **Identifiers**: ISRC identifies a *recording or music video*, never a release;
  releases need UPC/EAN or GRid; ISWC identifies the composition.
- **DDEX ERN 4.3.2** is current and, since 4.3.1, carries explicit **generative-AI
  disclosure fields** for recordings and contributions — the machine-readable place
  a DSP will look. Vevo's own feed parser is documented against ERN 3.8.

## 9. Mapping to Milimo's gaps

| Standard / lesson | Milimo today | Task |
|---|---|---|
| YouTube bitrate ladder + faststart | one profile, no faststart on the master | 7.3, 7.4 |
| Vevo ProRes/H.264 20 Mbps, no slates | no presets, no ProRes | 7.4 |
| Canvas 3–8 s 9:16 silent | no variant export | 7.4 |
| −14 LUFS on the video track | audio re-encoded, no loudness | 7.3 |
| QC at non-default thresholds | no QC at all | 7.5 |
| Σ frames == song frames | per-clip only | 7.2 |
| Reference conditioning / last-frame chaining | none | 7.6 |
| Audio-conditioned singing shots (S2V) | jaw-warp fallback reporting success | 7.1, 7.7 |
| Apache-2.0-only local stack | Wan 2.1; LivePortrait w/ InsightFace | 7.7 |
| EDL/OTIO hand-off | none | 7.9 |
| ERN 4.3.2 AI disclosure | none | 7.12 |

## 10. Uncertainty — stated, not hidden

- **Sora 2 API retirement (2026-09-24)** is stated on OpenAI's own video docs page;
  extraordinary enough that it should be re-checked against the deprecations page
  before it is cited in a public document.
- **Leaderboard positions** (Veo 3.1 at #26, Elo 963) are one dated snapshot of the
  Artificial Analysis T2V board (2026-11-18, ±13–15 at 95%).
- **Runway Gen-4.5 "1247 Elo, #1"** is a 2025-12-01 claim and is absent from the
  current board — do not cite it as current.
- **LTX-2 clip length** conflicts inside Lightricks' own materials (10 s vs 20 s;
  card default 121 frames).
- **LatentSync weights licence** is reported as both Apache-2.0 and OpenRAIL++.
- **Spotify Canvas dimensions**: the official page's "720–1080 px tall" contradicts
  the widely used 1080×1920 export; follow the written range until tested.
- **MMS aligner accuracy** figures come from a vendor benchmark on 27 songs;
  directionally consistent with FA-Bench, not peer-reviewed.
- **Vevo TTML captions** comes from a distributor, not Vevo's own docs.

## Sources

Delivery: [Vevo specs](https://support.vevo.com/hc/en-us/articles/360033310854-Music-Video-Specs) ·
[Vevo feed delivery](https://support.vevo.com/hc/en-us/articles/8624200408731-Feed-Delivery) ·
[Spotify Canvas](https://support.spotify.com/co-en/artists/article/canvas-guidelines/) ·
[Spotify loudness](https://support.spotify.com/us/artists/article/loudness-normalization/) ·
[EBU R 128](https://tech.ebu.ch/docs/r/r128.pdf) ·
[EBU R 128 s2](https://tech.ebu.ch/docs/r/r128s2.pdf) ·
[QCTools](https://mediaarea.net/QCTools/Getting_Started).
Models: [Wan 2.2](https://github.com/Wan-Video/Wan2.2) ·
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) ·
[HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) ·
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) ·
[SkyReels V3](https://github.com/SkyworkAI/SkyReels-V3) ·
[EchoMimicV3](https://github.com/antgroup/echomimic_v3) ·
[LatentSync](https://github.com/ByteDance/LatentSync) ·
[LivePortrait LICENSE](https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE).
Pipelines: [AutoMV paper](https://arxiv.org/abs/2512.12196) ·
[MusicVision](https://github.com/tsondo/musicvision) ·
[FeverSlop](https://github.com/H256/feverslop-ts) ·
[muvid](https://github.com/reeleehq/muvid) ·
[all-in-one](https://github.com/mir-aidj/all-in-one).
Standards: [RO-Crate 1.2](https://www.researchobject.org/ro-crate/specification/1.2) ·
[Croissant 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html) ·
[CITATION.cff](https://citation-file-format.github.io/) ·
[C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html) ·
[ERN release notes](https://ern.ddex.net/electronic-release-notification-message-suite-part-1-definitions-of-messages/annex-a-(informative)-release-notes) ·
[MusicXML 4.0](https://www.w3.org/2021/06/musicxml40) ·
[MIDI 2.0 core](https://midi.org/midi-2-0-core-specification-collection).
