---
title: Singing Voice Conversion (RVC v2)
type: concept
created: 2026-09-03
updated: 2026-09-03
tags: [rvc, svc, rmvpe, hubert, voice, phase-5]
aliases: [RVC, SVC, voice conversion]
sources: [production-readiness-plan.md]
---

# Singing Voice Conversion (RVC v2)

> [!NOTE] **Status: design locked 2026-09-03, not yet shipped.** Phase 5A of the
> [production plan](../production-readiness-plan.md). The service layer it upgrades is
> documented in [Voice Studio (SVC)](../entities/voice-service.md).

Real **RVC v2** singing voice conversion replaces the copy/pitch-shift placeholder in
`voice_service.convert_vocals`. Voice model *training* stays out of scope — inference
only: users import standard RVC v2 `.pth` checkpoints.

## Inference stack (vendored)

Adapted from the MIT-licensed RVC-WebUI inference modules into `backend/app/vendor/rvc/`
(attribution headers required):

| Module | Role |
|---|---|
| `models.py` | `SynthesizerTrnMs768NSFsid` (v2, 768-dim features) + `.pth` checkpoint loader (`params`/`weight` dicts) |
| `f0.py` | **RMVPE** pitch extraction (primary; respects the profile's `f0_method`) |
| `hubert.py` | ContentVec (`content-vec-best`) HuBERT acoustic features |
| `pipeline.py` | Full SVC forward: RMVPE F0 → HuBERT features → synthesis → 40 kHz output |

- Synthesis params: `pitch_shift` (semitones), `filter_radius`, `protect`,
  `rms_mix_rate`; `index_rate=0` initially (no faiss dependency).
- Device: **MPS with CPU fallback**, override via `MILIMO_RVC_DEVICE` (MPS op coverage
  varies; CPU is always correct, just slower).
- Weights (`content-vec-best`, `rmvpe.pt`) download into `data/models/rvc/` through the
  [Model Manager](../entities/model-manager.md) tree + the Phase 4 IO lane download task
  (chunk-level progress, resumable) — see [Task Queue](../entities/task-queue.md).

## Honest conversion modes

`convert_vocals` reports what actually ran in its metadata:

- **`method: "rvc_v2"`** — checkpoint present at `data/voice_profiles/{profile_id}.pth`
  → real neural conversion.
- **`method: "pitch_shift"`** — no checkpoint, but the caller set an explicit
  `pitch_shift` → torchaudio DSP shift, labeled as such.
- **`VoiceModelMissingError`** (typed) — no checkpoint and no pitch shift requested.
  The current silent behaviors are removed: the empty-file write when the vocal stem is
  missing (`voice_service.py:146-147`) and the unlogged clean copy.

## Bug fixes included (audit 2026-09-03)

- `POST /jobs/{job_id}/voice-convert` (`main.py:2961`) passes an output *path* as the
  `job_id` positional argument of `convert_vocals` and ignores the returned path — the
  created Job's `audio_path` points at audio that is never written. Fixed by the
  Phase 4 queue conversion (child Job created `queued`, handler runs conversion with the
  correct signature and writes the real path).
- The checkpoint existence check (`main.py`-side `os.path.exists(model_ckpt)` in
  `voice_service.py:150-152`) only logs; it never loads the model. The vendored
  `models.py` loader makes it real.

## Wiring

- [Orchestration pipeline](generation-pipeline.md) step 3 keeps its call signature —
  `convert_vocals` becomes genuinely neural with no pipeline changes.
- `POST /voice/profiles/{id}/checkpoint` (planned) imports a standard RVC v2 `.pth`;
  Training Studio shows honest per-profile status: *inference-ready* vs *DSP-only*.

## Tests (planned)

`backend/tests/test_voice_conversion.py`: missing checkpoint → typed error (no empty
file regression); mocked tiny checkpoint through the pipeline; pitch-shift fallback
labeling; queue-backed endpoint producing a child Job with a valid `audio_path`.

## Related pages

- [Voice Studio (SVC)](../entities/voice-service.md) · [Task Queue](../entities/task-queue.md)
- [Track extension](track-extension.md) (Phase 5 sibling) · [Orchestration pipeline](generation-pipeline.md)
