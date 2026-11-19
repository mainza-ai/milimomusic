---
title: Caveat Investigation 2026-11
type: overview
tags: [audit, caveats, video, lip-sync, upscaling, licensing, ci, gaps]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [CaveatInvestigation, Phase8Caveats]
---

# Caveat Investigation — 2026-11

The Phase 8 delivery ended with five stated caveats. This page investigates each
one to answer a single question: **is it a weights problem, a code problem, or a
decision nobody has made?** The distinction decides the plan — only one of those
three is fixed by waiting.

Companions: [Video Capability Landscape](video-capability-landscape-2026-11.md) ·
[Music Video Pipeline Audit](music-video-pipeline-audit-2026-11.md) ·
[Plan — Phase 8](application-remediation-plan-2026-11.md#phase-8--capability-upgrades-2026-11-18).

---

## 1. Verdicts

| # | Caveat as stated | Verdict | Evidence |
|---|---|---|---|
| **C1** | "Seven tasks need weights" | **Mixed** — 3 are pure weights, 2 are code+weights, 1 is a code gap | `image_service` has **no reference-image parameter at all** (8.2), so no amount of downloading fixes it; `WanVACEPipeline` exists in the installed diffusers (8.14 is weights only); FastMetal (8.5) needs a generator variant plus weights |
| **C2** | "RIFE detected but not bundled; `minterpolate` softens" | **Code gap, now fixed** | The filter used `mi_mode=blend`, which *dissolves* neighbours. It now defaults to `mi_mode=mci:mc_mode=obmc:me_mode=bidir:vsbmc=1` (true motion-compensated interpolation); `MILIMO_INTERP_MODE=blend` remains a fast opt-out. RIFE still needs a frame pipeline — see 8.6b |
| **C3** | "8.4 wires 2 of 4 tiers" | **Code gap, partly fixed** | InfiniteTalk is now wired through a JSON-template launcher that preserves the engine's own schema; LongCat remains a `torchrun` launcher and is deliberately not half-wired |
| **C4** | "Verifier is perceptual, not semantic" | **Code gap, fixed with a gate** | `score_candidate` now carries an optional CLIP similarity (`semantic`), which is `None` — explicitly, not zero — when no embedding model is installed. `needs_reshoot` adds a semantic floor, so "same composition, different subject" is re-rolled |
| **C5** | "`MILIMO_NONCOMMERCIAL_OK` not enforced in CI" | **Process gap, fixed** | `scripts/check_licences.py` + a `licence-gate` CI job. Verified: exit 1 without acknowledgement, 0 with it, and the finding is printed either way |

## 2. What investigating the caveats uncovered

Each caveat turned out to sit on top of a larger gap. These are new:

| # | New gap | Why it matters |
|---|---|---|
| **G11** | **The character plate never reaches generation.** Consistency compares against it (verifier) and uses it as an i2v seed, but `generate_scene_background` is text-only. | The library exists, the reference exists, and the model never sees it. This is now the single largest consistency gap. |
| **G12** | **The plan endpoint ignores character/consistency config.** `/videos/plan` accepts `character_desc` but not `character_id`, `consistency_mode` or `chain_scope`, so the storyboard the user approves can differ from what the render does. | A preview that disagrees with the render is the exact class of dishonesty Phase 7 removed elsewhere. |
| **G13** | **Export/variant cancellation is cosmetic.** The task is registered and cancellable in the UI, but `transcode` runs inside `asyncio.to_thread`; killing the task does not stop FFmpeg. | A "Cancelled" export keeps burning CPU and leaves a take behind. |
| **G14** | **Character plates are API-only.** Creation from the UI sends a description, never a portrait, so `has_plate` is false for every UI-created character. | The verifier silently has nothing to compare against, which disables the feature users just enabled. |
| **G15** | **Delivery ignores the capability matrix.** The interpolation target is chosen from the source rate, not from what the device/engine can deliver, so a 16 fps chain and a 25 fps chain get the same treatment. | The matrix tells the user the truth before the render; delivery should not then contradict it. |
| **G16** | **The licence gate covers the model catalogue only.** Python dependencies (an InsightFace-class detector, for example) are invisible to it. | The audit's first licence finding was a *dependency* (InsightFace), not a model. |

## 3. What this changes about "fully functional"

The honest reading of the current build:

- **Fully functional today:** music generation → stems → transcription → lyric
  sync → storyboard → multi-scene video → karaoke burn → delivery presets → QC →
  takes → manifest → character library (text-only) → verifier (perceptual, plus
  semantic when CLIP is present) → jaw-warp avatar with real viseme timing.
- **Functional with weights:** EchoMimicV3-Flash / Wan 2.2 S2V avatars (contract
  live), Wan 2.2 720p, VACE reference conditioning.
- **Not functional yet, by code:** reference-conditioned scene stills (G11), the
  plan/render config agreement (G12), export cancellation (G13), UI plate attach
  (G14).

## 4. Recommended order

1. **G14 → G11** (plate attach, then reference-conditioned stills). Without the
   plate there is nothing to condition on, and without conditioning the verifier is
   judging plates it could have influenced.
2. **G12** (plan receives the render config) — cheap, removes a disagreement class.
3. **G13** (real cancellation) — run the transcode as a killable subprocess.
4. **G15** (delivery consults the matrix) — makes the pre-render promise binding.
5. **G16** (dependency licence audit) — the class that produced the first finding.
6. Then the weight-gated adoptions (8.5, 8.14, 8.8, 8.7, 8.11).

## 5. Uncertainty

- **CLIP as an identity proxy** is a semantic check, not a face recogniser: it says
  "same kind of subject", not "same person". A real face embedding needs a licence
  decision (InsightFace-class models are research-only), which is why G16 matters.
- **InfiniteTalk's request schema** is taken from its example file; the template
  mechanism exists precisely because that schema may change between releases. It is
  not verified against a live checkout.
- **`mci` cost**: motion-compensated interpolation is materially slower than blend.
  The default changed for quality; a 3-minute 480p→24 fps pass is now a
  minutes-scale step, not seconds.

## Related

[Video Capability Landscape](video-capability-landscape-2026-11.md) ·
[Music Video Pipeline Audit](music-video-pipeline-audit-2026-11.md) ·
[Plan — Phase 8](application-remediation-plan-2026-11.md#phase-8--capability-upgrades-2026-11-18) ·
[Index](index.md)
