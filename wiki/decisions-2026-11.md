---
title: Decision Record 2026-11 (Scope & Scope-Outs)
type: overview
tags: [decisions, scope, yue2, providers, multi-user, cloud-video, dsp, cuda, hardware-matrix]
created: 2026-11-16
updated: 2026-11-16
sources: [application-audit-2026-11.md]
aliases: [ADR 2026-11, scope decisions, decision log nov 2026]
---

# Decision Record — 2026-11

Six scope decisions that gate the remediation plan. Each was investigated against the
code before deciding; every recommendation below cites the evidence it rests on.
Records **D1–D6** are binding for the 2026-11 remediation cycle and change the shape of
[Application Remediation Plan 2026-11](application-remediation-plan-2026-11.md).

Rules for this page: a decision is **superseded** by appending a new record with a later
date, never by editing history. Each decision carries a **revisit trigger** — the
condition under which it should be reopened.

| ID | Decision | Verdict | Net effect on plan |
|---|---|---|---|
| D1 | YuE2 provider + Training Studio | **Delete** | −1 provider, −1 feature surface |
| D2 | Single-user vs multi-user | **Single-user, multi-tenant-ready** | −6 to −8 weeks |
| D3 | Legacy providers | **Keep HeartMuLa (demoted) + MuLaCover; delete Training Studio** | 2 keepers de-stubbed |
| D4 | Cloud video scope | **fal only; delete Replicate + H3-cloud** | Cloud becomes load-bearing |
| D5 | Automation lanes + time-stretch | **Automation yes; time-stretch no** | +~10 d, −2 wk DSP risk |
| D6 | CUDA as generation target | **No for generation; yes for everything else** | Publish + enforce matrix |

---

## D1 — YuE2 provider and Training Studio: **delete both**

**Decided 2026-11-16.** Status: approved.

**Evidence.**
- `providers/yue2_provider.py:110-140` is a 4-chord sine generator over a hardcoded
  `[220.0, 174.61, 261.63, 196.0]` (A-minor → F → C → G). `generate()` never reads
  `prompt`, `lyrics`, `seed`, `structured_caption` or LoRA weights.
- `initialize()` only sets `_is_ready = True` (`:56-61`); capabilities still claim a 3B
  48 kHz model, LoRA, voice conversion, extension and segment repair (`:38-54`).
- `:239` returns `used_fallback_synth=False` on synthesized audio — it *falsifies* the
  provenance flag the rest of the codebase is built on.
- `services/training/yue2_trainer.py:120-200` is `await asyncio.sleep(0.5)` ×4 followed by
  writing a **20-byte fake checkpoint** (`f.write(b"YUE2_LORA_V24_CHECKPOINT")`) and
  reporting "Training complete! LoRA saved to …". Audition WAV paths are recorded but
  never created.
- No YuE2 weights exist on disk (`models/audio` = `HeartMuLa__MuLaCover`, `voice_svc`; the
  HF cache has no YuE2 entry).
- Exposed live in the UI: `VoiceStudioModal` → `yue2_studio` tab → `trainingApi`.

**What we do.** Remove `providers/yue2_provider.py`, `services/training/`,
`POST|GET /api/training/jobs*`, the `/data/training_auditions` mount, the `yue2_studio`
tab and `trainingApi` (`api.ts:1783-1810`); update the 3 tests that reference them.
Preserve the *idea* on the roadmap only.

**What we explicitly do not do.** Do not "make it honest" by flipping the flag — a
fabricated `.safetensors` is the real liability: a user will try to load it.

**Consequence.** Milimo ships **no open-weights audio engine** until one is genuinely
integrated. That is an accurate statement of the product today, and it is cheaper to say
than to fake (see [D3](#d3--legacy-providers-keep-heartmula-demoted-and-mulacover-delete-the-training-studio)).

**Revisit trigger.** A real YuE2 (or comparable) checkpoint plus a real trainer exists as
a separate, independently tested project. Effort to reclaim: ~1 day re-registration.

---

## D2 — Product shape: **single-user, multi-tenant-ready**

**Decided 2026-11-16.** Status: approved.

**Evidence.**
- No `owner_id` on any table (`models.py`); `StudioUserProfile` is a singleton
  (`id: str = "default"`, `models.py:683`).
- `core/instance_lock.py:31-53` **refuses to boot a second backend instance** by design,
  to protect GPU + DB integrity.
- The GPU coordinator's `asyncio.Lock` is process-local (`core/hardware_lock.py:28`), so
  mutual exclusion breaks the moment two workers run.
- SQLite WAL, single writer, 10 s busy timeout; a read during the audit returned
  `database is locked`.
- The rate limiter is in-memory and per-process (`core/ratelimit.py:26-35`).

**Cost of real multi-user:** `owner_id` + row filters across ~12 tables (1 wk) · per-user
media isolation + signed URLs (1 wk) · auth, scopes, sessions, reset (1 wk) · per-user
queue fairness (3 d) · **then** Postgres + a cross-process GPU lock + a deployment
redesign (2–3 wk). ≍ 6–8 weeks, on top of Phase 2, for a product whose differentiator is
running large local models on the user's own hardware.

**What we do.**
1. Mandatory auth, fail-closed (plan 0.3/0.4).
2. **Signed read-only share links** — this is the 90 % of "multi-user" that users actually
   request, at ~2 days.
3. Add `owner_id: str = "local"` columns and one `scoped()` query helper **now** (~2 days)
   so a hosted mode later is a migration, not a rewrite.
4. Document a **Team/LAN mode**: one token, one DB, N clients, single instance.
5. Delete "multi-user" wording from README and roadmap.

**What we explicitly do not do.** No per-user libraries, no role management, no
registration flow, no cluster deployment in 2026-11.

**Consequence.** Phase 5.6 shrinks from 6–8 weeks to ~1 week. The freed budget funds
Phases 3 and 4 (the DAW and media-truth work users can actually see).

**Revisit trigger.** A hosted/SaaS product decision, or a verified need for two people to
edit the same project concurrently.

---

## D3 — Legacy providers: **keep HeartMuLa (demoted) and MuLaCover; delete the Training Studio**

**Decided 2026-11-16.** Status: approved.

These are three different situations, not one.

| Provider | Reality | Verdict |
|---|---|---|
| **HeartMuLa** | Real: 2,920 LOC `heartlib`, weights present (`heartlib/ckpt/HeartMuLa-oss-3B`, `HeartCodec-oss`), `generate()` drives `HeartMuLaGenPipeline` (`heartmula_provider.py:100-150`) | **Keep, demoted** |
| **MuLaCover** | Real: 11,958 LOC vendored, torch engine loading Qwen + codec (`services/mulacover/mulacover_engine.py`), weights present, only provider that consumes reference audio | **Keep, de-stub** |
| **Training Studio (YuE2)** | Theater — see [D1](#d1--yue2-provider-and-training-studio-delete-both) | **Delete** |

**What we do — HeartMuLa.** Mark `legacy: true`, hide from the default engine picker, add
a deprecation notice with a removal target, remove the no-op `extend()` (`:152-171`) and
`repair_segment()` (`:173-187`), and stop advertising `supports_track_extension` /
`supports_segment_repair` / `supports_voice_conversion`. Keep it because it is **the only
non-MLX generation path** — which is exactly what makes [D6](#d6--cuda-as-a-generation-target-no-for-generation-yes-for-everything-else) affordable.

**What we do — MuLaCover.** Remove `repair_segment` (`mulacover_provider.py:227-241`);
make `extend()` stop treating the parent as a cover reference; reconcile capability flags
with reality. It is woven into 13 backend + 11 frontend files and represents a genuine
product category (cover/remix), so fixing costs less than deleting.

**What we explicitly do not do.** No new features for either keeper this cycle; no LoRA
training of any kind.

**Revisit trigger.** HeartMuLa: remove entirely when a CUDA-capable generation path exists
via another route. MuLaCover: promote to first-class when reference-audio conditioning
becomes a headline feature.

---

## D4 — Cloud video: **fal only; delete Replicate and H3-cloud**

**Decided 2026-11-16.** Status: approved.

**Evidence.**
- `.env.example:60-61` documents `FAL_KEY` and `REPLICATE_API_TOKEN`; **neither is present
  in `.env`** — nothing on this path has been exercised in practice.
- fal uses the documented queue submit → poll → fetch pattern
  (`cloud_video.py:99-110, 132-155`) — structurally correct, except it posts a base64
  **data URI** as `image_url`, which fal will not accept.
- Replicate posts `{"version": "wan-video/wan-2.1-14b"}` (`:161-166`) — not a version hash,
  and `self.model` is ignored. It 400s every time. `cloud_lipsync.py:73-78` has the same
  data-URI defect.
- 141 GB of local video weights are already on disk, so local is the differentiator — but a
  laptop user needs *something*.

**What we do.** fal becomes the single cloud backbone for video and lip sync
(`fal-ai/wan-i2v`, `fal-ai/wan-t2v`, `fal-ai/live-portrait`): upload → hosted URL → submit
→ poll → download. Add one contract test with a mocked transport. Gate the cloud UI on a
real key check **plus a cost estimate** — this spends real money per click.

**What we explicitly do not do.** No Replicate support, no MiniMax-H3-cloud, no additional
cloud vendors. No cloud path may silently substitute for a local render.

**Consequence.** Cloud stops being decorative and becomes the answer for Linux/NVIDIA
users who cannot run the MLX flagship — i.e. it carries [D6](#d6--cuda-as-a-generation-target-no-for-generation-yes-for-everything-else).

**Revisit trigger.** A paying deployment needs a second vendor for redundancy, or fal's
pricing/quality regresses.

---

## D5 — Automation lanes and time-stretch: **automation yes, time-stretch no**

**Decided 2026-11-16.** Status: approved.

**Automation — build it.** The graph is already one `GainNode` + `StereoPannerNode` per
stem driven by `setTargetAtTime` (`SessionWorkspace.tsx:631-641`). Scope: `automation:
{param: [{t, v, curve}]}` on the session payload, a lane under the arrange timeline, and
`setValueCurveAtTime`/`linearRampToValueAtTime` scheduling on play and seek. Four
parameters (volume, pan, send, master).

Cost: **~4 days** read-only display, **~8 days** with drawing and undo. It depends on the
clip model (3.1) and workspace undo (3.2), so it lands *inside* Phase 3, not after it.

**Time-stretch — do not build.** The project already tried it and removed it. Log entry
2026-10-24:

> "Killed librosa time_stretch from the chain (phase-vocoder flips/smears phase even at
> 0.03 %: measured corr −1.0 on a near-unity stretch)"

No `atempo`/`rubberband`/`soundtouch` exists in `backend/app` today. Options and costs:
Rubber Band binary (~4 d + cross-platform native dependency risk) · hand-written WSOLA
(~2 wk) · SoundTouchJS in an AudioWorklet (~1 wk, preview quality) · FFmpeg `atempo`
bounce-only (~2 d, artefacty beyond ±10 %). All of them attack the DAW's strongest asset:
sample-locked, drift-free playback.

**What we do instead.** Ship **tempo-locked generation** (~2 days): request the target BPM
at generation time rather than stretching finished audio. If alignment is ever required
for export, do it there — offline, in a bounce — never in the live transport.

**Revisit trigger.** Users need to conform imported external audio (not generated) to a
tempo map; then implement export-time WSOLA or Rubber Band, still never in the transport.

---

## D6 — CUDA as a generation target: **no for generation, yes for everything else**

**Decided 2026-11-16.** Status: approved.

**Evidence.** The flagship is MLX-only: there is no torch fallback, and
`minimax_local_hooks.py:213, 362, 552` raise `mlx_audio is not available on this platform`.
Under `MILIMO_STRICT_INFERENCE=1` that is a hard error; otherwise it degrades to the synth.
The 37 `torch.cuda` references are device selection and telemetry for torch-side services
(separator, transcriber, image, video, HeartMuLa). The Docker "GPU" profile installs **CPU
wheels** on `python:3.11-slim`, so it never had CUDA.

**The matrix — publish it, and enforce it in the UI.**

| Capability | Apple Silicon (MLX) | NVIDIA CUDA | CPU |
|---|---|---|---|
| MiniMax Music 3 (flagship generation) | ✅ | ❌ | ❌ |
| HeartMuLa (legacy generation) | ✅ MPS | ✅ | ⚠ slow |
| MuLaCover | ✅ | ✅ | ⚠ |
| Separation / transcription / mastering / DAW / agents | ✅ | ✅ | ✅ |
| Image (FLUX) · Video (Wan diffusers) | ✅ MPS | ✅ | ⚠ |
| Cloud generation / lip sync (fal) | ✅ | ✅ | ✅ |

**What we do.** `hardware_autotune` emits the matrix; the UI **disables** unavailable
engines with the reason ("requires Apple Silicon MLX" / "weights missing — 28 GB") instead
of failing at runtime. Delete the CUDA device reservation from the CPU Dockerfile. Correct
the README headline claims to match the matrix.

**What we explicitly do not do.** No `Dockerfile.cuda`, no cu121 wheel pinning, no CUDA
OOM-path testing this cycle — that is ~2 weeks (pin wheels, audit all 37 sites, test OOM
recovery) for zero users who cannot already use fal.

**Revisit trigger.** A deployment or sponsor requires local CUDA generation, or a
torch/CUDA backend for the flagship appears upstream.

---

## Combined effect on the plan

```
D1 (delete YuE2 + Training)  ──► removes the fake-artifact class outright
D3 (keep/demote the real ones) ─► de-stubs HeartMuLa + MuLaCover
D6 (no CUDA generation)      ─┬─► D4 cloud (fal) becomes the Linux/NVIDIA answer
                              └─► D3 HeartMuLa becomes the only local non-MLX gen
D5 (no time-stretch)         ──► replaces a ~2-week DSP risk with ~2 days of
                                 tempo-locked generation
D2 (single-user)             ──► Phase 5.6 drops 6–8 weeks → ~1 week, funding Phases 3–4
```

Net: **adds ≈2 days** (tempo-locked generation, ownership columns, matrix plumbing),
**removes ≈6–8 weeks** (multi-user), and eliminates the two most damaging shipping
artefacts — a fake 48 kHz "foundation model" and a 20-byte fake LoRA checkpoint.

## Verification gaps carried forward

Two claims behind **D4** and **D3** are structural, not tested. Smoke-test both before
committing budget:

1. **fal end to end** — the queue pattern is correct on paper, but with no key the path has
   never run. One hour with a key: submit a 5 s `fal-ai/wan-t2v` clip, confirm the file
   lands and the poll loop exits cleanly.
2. **MuLaCover cover generation** — the engine loads real weights and a Qwen text encoder,
   but no end-to-end cover was executed during the audit. One hour: generate a 30 s cover
   from a short reference and confirm stems + transcription follow.

## Related

[Application Remediation Plan 2026-11](application-remediation-plan-2026-11.md) ·
[Full Application Audit 2026-11](application-audit-2026-11.md) ·
[Generation Provider](entities/generation-provider.md) ·
[YuE2 Music](entities/yue2-music.md) ·
[MuLaCover](entities/mulacover.md) ·
[HeartMuLa](entities/heartmula.md) ·
[Track Extension](concepts/track-extension.md) ·
[Index](index.md)
