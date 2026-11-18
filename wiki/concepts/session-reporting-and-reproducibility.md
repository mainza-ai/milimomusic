---
title: Session Reporting & Reproducibility
type: concept
tags: [reports, documentation, export, provenance, reproducibility, academic, standards, metadata, ddex, musicxml]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [ReportingAudit, Reproducibility, ArtifactManifest]
---

# Session Reporting & Reproducibility

How Milimo Music turns a finished session into **reports, results, documentation
and research-grade artifacts** — what it emits today, what it records but never
emits, and what a private production deployment needs.

Part of the [Music Video Pipeline Audit 2026-11](../music-video-pipeline-audit-2026-11.md);
standards references are in
[Music Video Delivery & QC Specs](music-video-delivery-specs.md#8-standards-for-documenting-generated-media).
**Tasks: Phase 7.5–7.12 of the [plan](../application-remediation-plan-2026-11.md#phase-7--music-video-production-2026-11-18-added).**

---

## 1. What a user can obtain today (verified inventory)

| Capability | Backend | Frontend | Maturity |
|---|---|---|---|
| Master transcode — WAV/FLAC/MP3 × 44.1/48/96 kHz, real FFmpeg | `main.py:4846-4908` | `MasteringExportModal.tsx:248` | working |
| Reference mastering (Matchering) or −14 LUFS + peak guard | `main.py:1132-1188`, `transcription/mastering.py` | `MasteringExportModal.tsx:140-160` | working, **measurement discarded** |
| Stems ZIP | `main.py:1069-1124` | `MasteringExportModal.tsx:306` | partial — audio only, no manifest, flat names |
| **Studio Pack** (master + stems + instruments + MIDI/MusicXML/notes + LRC/txt + `metadata.json`) | `main.py:4645-4727` | `TrackDetailView.tsx:717` | working — the most complete artifact |
| Project Studio Pack (album folder) | `main.py:5305-5396` | `ProjectsView.tsx:194` | partial — per-track metadata omits seed/params/provenance |
| MIDI / MusicXML export | `main.py:1001-1016` | `PianoRoll.tsx:1913`, `NotationViewer.tsx:177` | partial — see §4 |
| Engraved PDF scores / parts / tab | `main.py:1190-1200` → `muscriptor/utils/sheets.py` | `NotationViewer.tsx:605-690` | partial — needs external MuseScore ≥4; failures swallowed |
| LRC / SRT timed lyrics | `main.py:1017-1030`, `:2798-2834` | LRC: yes; **SRT: no caller** | working backend, half-exposed |
| "Ableton export" | `main.py:1031-1067` | none | **stub** — a JSON descriptor, not `.als` |
| Video (reactive / advanced / lyric / NLE) | `main.py:5498, 5885, 5926, 6599` | `MusicVideosView.tsx:1370` | working; MP4 only |
| Signed expiring share link | `main.py:4578-4593` | `MasteringExportModal.tsx:279` | working token; **page returns JSON**, not a player |

**Absent entirely** (grep-verified): any `/report`, `/analysis`, `/summary`
endpoint; any CSV writer (`import csv` = 0); any Markdown/HTML artifact writer;
any credits/liner-notes/ISRC/ISWC/UPC fields; `CITATION.cff`; any transcription
confidence or accuracy metric.

## 2. Recorded but never emitted

This is the core finding: **the platform collects more provenance than it can
hand to anyone.**

| Fact | Recorded at | Emitted where |
|---|---|---|
| Seed (including auto-chosen) | `models.py:69`, `main.py:3747` | Studio Pack `metadata.json` only |
| Provider / model id | `models.py:73` | Studio Pack |
| Sampling params (temperature/cfg/topk) | `models.py:85-87` | Studio Pack |
| Exact caption + lyrics fed to the model | `models.py:83-84` | Studio Pack `prompt`, `lyrics.txt` |
| **Per-stage provenance + quality** (`neural`/`degraded`/`placeholder`) | `models.py:113-127`, `core/provenance.py` | **UI badge only — never in any export** |
| Beat grid | `models.py:109` | Studio Pack |
| Word-level lyric timings | `models.py:110` | LRC/SRT, video burn-in |
| Measured LUFS / true peak after mastering | computed in `mastering.py:107-165` | **HTTP response only, never persisted** |
| Detected key + `key_confidence` | `minimax_provider.py:308-389` | **transient use only** |
| Model *version* (`capabilities.version`) | `providers/base.py:24` | `/health` only |
| Video render config | `models.py:102` | DB only (drives "regenerate") |

Consequences for the video path specifically: `video_config_json` exists but is
not exportable, the storyboard is persisted server-side with no export
(`PATCH /videos/{job}/storyboard`), captions are burned to pixels with no
sidecar on the advanced path, and the MusicXML written to the pack reports
*fixed* musical facts.

## 3. Gaps — production deliverables

| Expected deliverable | Status |
|---|---|
| Stem package **with** metadata (engine, sample rate, bit depth, peak/LUFS per stem) | missing — ZIP is audio-only (`main.py:1090-1101`) |
| Release-ready master + **loudness report** (LUFS, dBTP, per-target) | measured then thrown away; no dBTP value at all |
| Packs ship the **mastered** file | missing — packs key off `job.audio_path`, not `mastered_path` (`main.py:4656`) |
| Session interchange (AAF/OMF/`.als`/`.rpp`/DAWproject) | missing; "Ableton" is a JSON descriptor |
| **Audio-only mixdown of the arrangement** | missing — the only timeline render target is `.mp4` (`render.py:169-193`) |
| Embedded metadata (ID3v2, BWF/iXML, XMP) in exports | missing — bare FFmpeg transcode |
| Credits / liner notes / identifiers (ISRC, UPC, publisher) | missing |
| Album/release package | missing — Release is a container with no export route |
| Caption sidecars for video (SRT/VTT/TTML) | missing |
| Cut list / EDL / FCPXML / OTIO for the video | missing |
| Human-readable share page | JSON only |

## 4. Gaps — research and academic use

A paper built on Milimo output today can state provider name, seed, sampling
params, caption and lyrics, plus a coarse quality verdict. It **cannot** state
the exact weights, code revision, environment, measured loudness, transcription
confidence, or any error metric — and the symbolic export is wrong:

- `muscriptor_provider.py` writes `<fifths>0</fifths>` and 4/4 unconditionally,
  `divisions=4`, and a literal `<encoding-date>2026-08-21</encoding-date>`
  (`:465-516`). A notation artifact that misstates key and metre is not usable as
  research output.
- No per-note confidence is produced (`muscriptor_provider.py:199-223`;
  `muscriptor/events.py:21-25`), so no calibration or error analysis is possible.
- No ground-truth comparison, no CER/F1/`mir_eval`, no listening-test harness.

### The reproducibility minimum (from ISMIR-style practice)

1. **Matched seeds** passed into the generation call, with the prompt sets released
   as a manifest.
2. **Dataset identity by checksum**, not name (`mirdata`-style validation).
3. **PRIMAD** framing: Platform, Research question, Implementation, Data,
   experimental setup.
4. **Listening tests** that state N, stimulus duration, anchors, scale and
   statistics; MUSHRA with a declared low anchor is the house style.
5. **Failures recorded honestly** (`BLOCKED`/`NOT_TESTED` rows), not folded into
   a pass.

### Machine-readable packaging

- **`CITATION.cff`** 1.2.0 — cheapest provenance win: GitHub renders a BibTeX
  block and Zenodo reads it for the DOI record.
- **RO-Crate 1.3** — a folder + `ro-crate-metadata.json`; provenance as
  `CreateAction` (`instrument` → software, `object` → input, `result` → output).
- **Croissant 1.1** — JSON-LD with W3C PROV-O chain-of-custody; the shape a
  dataset card should take.
- **C2PA 2.4** content credentials — signed *claims*; explicitly **not** AI
  detection, and disputed as a security control, so label it as disclosure.
- **DDEX ERN 4.3.2** — carries explicit generative-AI disclosure fields since
  4.3.1; the machine-readable place a DSP will look.
- **Identifiers** — ISRC = recording/video, ISWC = composition, UPC/GRid = release.
- **MusicXML 4.0** (`.mxl`) and **MIDI 2.0** only if UMP/SMF2 is emitted.

## 5. Recommendations (ranked by value / effort)

| # | Recommendation | Value | Effort |
|---|---|---|---|
| 1 | **One `artifact_manifest` builder** (`app/core/artifact_manifest.py`) emitted at `GET /jobs/{id}/manifest.json`: seed, params, model id + version, verbatim prompt/lyrics, stage provenance, artifact SHA-256s, app version + git hash, measured loudness. Everything else consumes it. | highest — makes a session citable and debuggable | M |
| 2 | **Persist mastering measurements** (`mastered_lufs`, `mastered_true_peak_db`, `method`, `mastered_at`) and include them in every pack. | release compliance | S |
| 3 | **Provenance in packs** — add `provenance_json` + `quality` + `capabilities.version` to `metadata.json`; stamp the model version at generation time. | academic differentiator already computed | S |
| 4 | **Stem manifest** (`stems/manifest.json`: engine, path, rate, bit depth, channels, duration, peak/LUFS) and `Track_NN_Title/Stem.wav` naming. | interchange with any DAW | S–M |
| 5 | **Fix MusicXML** to emit measured key/fifths, time signature, computed divisions, real encoding date. | correctness, not polish | M |
| 6 | **Audio-only bounce** of the NLE project (`POST /api/editor/projects/{id}/bounce`) — the DAW currently cannot render a mixdown. | largest single production gap | M |
| 7 | **Video report + cut list**: storyboard JSON, per-scene prompt/duration/provider/seed, QC results, caption sidecars. | hand-off and auditability | M |
| 8 | **Credits/liner notes + identifiers** on Release/Job, printable in the pack. | release workflows | M |
| 9 | **Note-level confidence** from the transcription event stream, persisted in `notes_json`. | research | M |
| 10 | **`CITATION.cff`** + a `docs`-level reproducibility statement (via wiki, since `docs/` is read-only). | best value per line | S |

## 6. Honesty notes

- `docs/TRAINING_STUDIO.md` documents `/training/*` endpoints that no longer
  exist. The wiki records the retirement
  ([Training Studio](../entities/training-studio.md)); the stale source doc stays
  because sources are immutable — this page is the correction of record.
- The share endpoint is described as a "public share page" in the API-parity
  allowlist (`scripts/check_api_parity.py:31`); it returns JSON. Either build the
  page or rename the claim.
- Mastered exports exist, but every pack and every video ships the unmastered
  file. Until that is fixed, "mastering" is a UI-only outcome for deliverables.

## Related

[Music Video Pipeline Audit 2026-11](../music-video-pipeline-audit-2026-11.md) ·
[Music Video Delivery & QC Specs](music-video-delivery-specs.md) ·
[AI Music Video Studio](../entities/video-studio.md) ·
[Multitrack Timeline Editor](../entities/multitrack-editor.md) ·
[Application Remediation Plan](../application-remediation-plan-2026-11.md) ·
[Index](../index.md)
