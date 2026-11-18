---
title: Video Capability Landscape 2026-11
type: overview
tags: [video, research, models, lip-sync, upscaling, consistency, roadmap, licensing, apple-silicon]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [VideoLandscape, ModelResearch, CapabilityRoadmap]
---

# Video Capability Landscape — 2026-11

Phase 7 made the video pipeline **honest and deliverable**. It did not make it
**capable**: the singing avatar is still a jaw-warp animatic, MPS clamps Wan to
832×480 at 16 fps, and identity continuity is text + frame-chaining rather than
reference conditioning.

This page answers *what to adopt next, in what order, and why*, with the local
library reality checked against the November-2026 open-source landscape.

Companions: [Music Video Pipeline Audit](music-video-pipeline-audit-2026-11.md) ·
[Delivery & QC Specs](concepts/music-video-delivery-specs.md) ·
[Remediation Plan — Phase 8](application-remediation-plan-2026-11.md#phase-8--capability-upgrades-2026-11-18).

---

## 1. What the installed stack already gives us

Most "just add model X" plans miss this: the local `diffusers` build already ships
the pipelines, so the cost is **weights and wiring**, not a port.

| Fact (verified on this machine, 2026-11-18) | Value | Consequence |
|---|---|---|
| `diffusers` | **0.39.0** | Modern Wan/LTX classes present |
| Reference conditioning for Wan | **`WanVACEPipeline`**, optional `reference_images` | An IP-Adapter-equivalent exists, no new dependency |
| Wan 2.2 I2V A14B | `model_index.json` → **`WanImageToVideoPipeline`** | The class we already use |
| Wan 2.2 TI2V-5B | `model_index.json` → **`WanPipeline`** (`expand_timesteps`, `boundary_ratio`) | 720p on the existing class |
| VACE weights | `Wan-AI/Wan2.1-VACE-14B-diffusers` (tagged `WanVACEPipeline`) | Apache-2.0, diffusers-native |
| Upscaling | **`LTX2LatentUpsamplePipeline`** | First-party latent upscale path |
| Joint A/V + in-context refs | **`LTX2Pipeline`**, `LTX2InContextPipeline` | One family, licence-gated |
| Audio→video | **nothing** (`WanAnimatePipeline` is pose/face-video driven) | Lip-sync stays external |

> [!IMPORTANT]
> `WanAnimatePipeline` is **video-driven**, not audio-driven. It is the right tool
> for motion transfer from a dance clip and the wrong tool for singing. Conflating
> the two is how the "LivePortrait" path became a lie.

## 2. The gaps, stated precisely

| Gap | Present behaviour | "Fixed" looks like |
|---|---|---|
| **G1** No audio-driven singing | `jaw_warp` animatic; registry states the reason | A model warps the mouth from the vocal stem |
| **G2** Resolution ceiling | 832×480 / 16 fps on MPS; no upscale, no interpolation | 720p/24 native, or a real upscale + interpolate stage |
| **G3** Identity drift | character bible (text) + frame chain | Reference-image conditioning |
| **G4** No motion transfer | portrait + audio only; no body | `WanAnimatePipeline` from a driving clip |
| **G5** No frame interpolation | fps forced by duplication (now `minterpolate`, a stopgap) | RIFE-class interpolation at delivery |
| **G6** No reusable character | consistency is per-render config | Character library objects |
| **G7** Plates are text-only | `image_service` has no reference/img2img input | Reference-conditioned keyframes |
| **G8** No licence gate | licence strings recorded, never read | Classification + blocker report (now implemented) |
| **G9** No video capability matrix | users discover ceilings after a long render | Engine picker states VRAM/resolution reality |
| **G10** Cloud path unverified | list-price estimate only; `fal` never exercised | One real cloud render, measured |

## 3. Resolution: the finding that changes the plan

**The 832×480 clamp is a runtime limit of our current Wan 2.1 path, not a limit of
open models on Apple Silicon.**

`FastVideo/FastMetal-5B-QAD` runs **Wan 2.2 TI2V-5B natively at 1280×704, 121
frames, 24 fps** — 151 s baseline / 47 s fast mode, **9.34 GiB peak on an M4 Max**
(Apache-2.0, pre-quantised INT8 for Metal). Weights verified Apache-2.0 via the HF
API; timings are author-reported.

That single swap addresses G2 *and* most of G5: TI2V-5B is **24 fps natively**,
so the 16→24 problem largely disappears. (The A14B models remain 16 fps.)

| Path | Native | Licence | Local (Mac) | Integration |
|---|---|---|---|---|
| **Wan 2.2 TI2V-5B (FastMetal QAD)** | **1280×704 @24** | Apache-2.0 | **9.3 GiB reported** | Medium — new pipeline config, not a new class |
| Wan 2.2 I2V A14B | 1280×720 @16 | Apache-2.0 | 60–80 GB class | Low — existing class |
| Wan 2.1 VACE 14B | 480P/720P | Apache-2.0 | ~20 GB (wrapper) | Medium — reference plumbing |
| LTX-2.5 (two-stage) | 1080p/4K *via latent upscale + refine* | LTX Community (<$10M ARR) | ~38 GB int4 on Mac | High + licence gate |
| HunyuanVideo 1.5 | 720p (+SR to 1080p) | **excludes EU/UK/KR** | 24 GB | Disqualified as a default |
| MiniMax H3 | 768p | **excludes US/EU/UK/KR** | 144 GB BF16 | Disqualified |

**Nobody ships true single-pass 1080p open weights at sane VRAM in this cycle.**
Every "1080p/4K" open claim is half-res denoise → ×2 latent upscale → refine.
That is fine — it just means 1080p is a *pipeline*, not a model choice.

### Upscalers and interpolation (only after native 720p)

| Tool | Licence | 3-min 480→1080 | Mac | Verdict |
|---|---|---|---|---|
| **FlashVSR+** | Apache-2.0 | ~4 min | CUDA only | Best value if any CUDA node exists |
| **SeedVR2-3B** | Apache-2.0 | ~2.8 h | MLX ports work; **ComfyUI MPS video fails** (`conv3d`) | Mac fallback via MLX |
| Real-ESRGAN | BSD-3 | ~15–25 min | stale macOS build | No temporal consistency |
| SUPIR / GIMM-VFI / FLUX.1 Kontext | **non-commercial** | — | — | Excluded |
| **RIFE 4.26** | **MIT** | 21 ms/frame @1080p | `rife-mlx` exists | **Default interpolator** |

Upgrading the base model beats upscaling 480p by a wide margin: a 2026 QoMEX study
found diffusion-VSR metrics unreliable and upscalers largely indistinguishable,
while composition and motion are decided at the latent resolution. Our own
arithmetic agrees — 14B at 480p/33 frames costs ~82 min per 2-second clip.

## 4. Identity: what actually works

The strongest local lever is **not** an identity adapter — it is AutoMV's
*verification loop*. Their ablation (arXiv 2512.12196v2, Nov 2026) shows removing
the **character bank** drops expert character-consistency 3.07 → 1.22, and the
**verifier-driven reshoot** cuts drift errors by 62% — with no identity model at
all. They also use last-frame chaining, exactly as we do.

For genuine reference conditioning on this hardware:

| Method | Base | Licence | Mac | Note |
|---|---|---|---|---|
| **Reference → keyframe → I2V** | FLUX.2 klein **4B** + Wan 5B | **Apache-2.0** | ✅ | The pragmatic local answer: generate the plate with references, then I2V from that plate |
| **Wan VACE** `reference_images` | Wan 2.1 14B | Apache-2.0 | via wrapper | Diffusers-native; needs 14B weights |
| Wan2.2-Animate-2 | Wan 2.2 14B | Apache-2.0 | ❌ no MLX | Motion transfer, not identity |
| Stand-In (CVPR 2026) | Wan 2.1 14B | Apache-2.0 | ❌ CUDA wrapper | Best FaceSim reported (0.724) |
| LTX IC-LoRA "Ingredients" | LTX-2.5 | LTX Community | ⚠️ memory-tight | Reference-sheet identity across cuts |
| FLUX.1 Kontext / Redux / IP-Adapter | FLUX.1 | **non-commercial** | ✅ | Blocked by licence |

> [!WARNING]
> **Our own default image model is non-commercial.** The licence audit added this
> cycle (G8) resolves the effective default to `flux_2_klein_9b`, whose HF metadata
> is `license_name: flux-non-commercial-license`. The Apache-2.0 sibling
> (`FLUX.2-klein-4B`) is the shipping-safe choice. For a deliberately
> non-commercial open-source project this is consistent, but it **must be replaced
> before any commercial deployment** — and it is now reported by
> `GET /system/licences` instead of being invisible.

## 5. Singing: licences decide the shortlist, VRAM decides the tier

The uncomfortable finding: **no 2026-grade singing-head model runs on Apple
Silicon through its own code.** They are CUDA DiT stacks with `flash-attn`/fp8 in
the hot path. The only Mac paths are third-party MLX ports (LatentSync; LongCat
Avatar), so the design must be a **tier ladder**, not one "best model".

| Tier | Engine | Licence | VRAM | Length · res · fps | Singing evidence |
|---|---|---|---|---|---|
| **0 — always** | jaw-warp animatic (+ real viseme timing) | in-house | 0 | unlimited | n/a (procedural) |
| **1 — default local** | **EchoMimicV3-Flash** | **Apache-2.0**, no territory clause | **12 GB**, 8 steps, no face mask | 81-frame chunks (3.2 s @25 fps) · ≤768² · **25 fps** | speech-trained; singing out-of-distribution |
| **2 — local big** | **InfiniteTalk** / **LongCat-Avatar 1.5** | Apache-2.0 / **MIT** | ~12–24 GB / int8 | streaming · 480–720P | InfiniteTalk ships a *singing* example; LongCat has an MLX port |
| **3 — CUDA/cloud** | **Wan 2.2 S2V-14B** → **LiveAvatar** | Apache-2.0 | 80 GB (48 GB fp8 LiveAvatar) | auto length · 720P · 16 fps | **explicitly trained on singing** |
| **Polish** | LatentSync 1.6 (lips-only repair) | ⚠ **unresolved** (OpenRAIL++ vs Apache) | 18 GB | video-in/out · 512² · 25 fps | speech |

**Recommendation:** **EchoMimicV3-Flash first** — it is the only candidate that is
Apache-2.0 with no territory clause, fits a 16–24 GB card, runs at native **25 fps**
(matching the timeline), and exposes a real argparse entry (`infer_flash.py`) with
built-in 16 kHz resample and −23 LUFS normalisation. Wan 2.2 S2V is the quality
tier, not the default: 80 GB and 16 fps make it a cloud job.

**Tier 0 is worth fixing, not just keeping.** The jaw-warp already has real
viseme timing available: the karaoke engine produces MMS forced-alignment word
timings we currently use only for subtitles. Driving the mouth from those timings
instead of an RMS envelope is a cheap, always-available quality win on any laptop.

> [!WARNING]
> **LatentSync's licence is unresolved.** Code is Apache-2.0; the HF card says
> **OpenRAIL++**, the maintainer answered "openrail++" in a discussion, and the
> question "may we use 1.6 commercially?" (issue #371, Oct 2026) is unanswered.
> MuseTalk inherits the same OpenRAIL++ SyncNet checkpoint. Neither may be a
> default until that is in writing.
>
> Arithmetic detail that matters: the Wan family samples at **16 fps**, EchoMimic
> at 25, InfiniteTalk assumes 25. Choosing Tier 1 avoids a frame-rate fight in the
> render graph on day one.

Known failure modes, honestly: speech-trained models collapse vibrato and melisma
into jaw chatter (inference from the ~50 Hz feature rate, not a published
benchmark); face-crop models (MuseTalk, LatentSync) paint lips over a mic or fail
detection past ~45° profile; long clips drift in colour and need a match pass.
Nobody publishes sync metrics on *sung* audio — if singing is the differentiator,
we have to build that evaluation ourselves.

## 6. Recommended order (value ÷ cost)

1. **Verifier + reshoot loop** (G3, ~0 VRAM) — the largest measured consistency win, and it is free. Character bank → 2–3 candidate plates → score → reshoot rejects.
2. **Avatar provider interface + tier ladder** (G1) — one `generate(image, audio, prompt, duration, fps)` contract with `procedural`, `echomimic_flash`, `infinitetalk`, `wan_s2v`, `longcat_mlx` providers, each carrying `licence`, `territory_blocked[]`, `min_vram_gb`. Route-level licence/territory checks, not a doc note.
3. **Tier-0 viseme timing** — drive the jaw-warp from the forced-alignment word timings we already compute, so the always-available path stops being an envelope guess.
4. **FastMetal Wan 2.2 TI2V-5B** (G2, G5) — native 720p/24 fps; removes the 480p ceiling and most of the interpolation problem.
5. **Reference → keyframe → I2V** with FLUX.2 klein **4B** (G3, G7) — real reference conditioning on this hardware today.
6. **RIFE delivery interpolation** (G5) — replaces the `minterpolate` stopgap.
7. **1080p uplift** (FlashVSR+ on CUDA / SeedVR2-MLX on Mac) — only after 720p native.
8. **Character library objects** (G6) — consistency that survives across songs.
9. **Capability matrix + licence gate in CI** (G8, G9) — cheap, prevents the error class this audit exists for.

Explicitly **not** next: per-character LoRA training (Wan on an M3 Max measured at
~666 s/iter — rent CUDA or skip), 4K delivery, a second cloud provider (D4
stands), buying closed weights (Wan 2.5/3.0 have no weights; Sora's API retired),
and admitting non-commercial models (Sonic, TalkVerse, OmniHuman) as defaults.

## 7. Uncertainty — stated, not hidden

- **FastMetal timings/peak memory are author-reported**, on M4 Max / M5 only. First task is a smoke test that either proves it or records the failure in the capability matrix.
- **SeedVR2 on Apple Silicon conflicts**: the ComfyUI node claims MPS support; two independent reports show `conv3d` failing in the video VAE decode. The image path works; use MLX ports for video.
- **LTX-2 "native 4K"** is marketing for half-res + ×2 latent upscale + refine.
- **Wan 2.2 A14B is 16 fps**, not 24 — many roundups say otherwise.
- **LatentSync weights** are reported as both Apache-2.0 and OpenRAIL++; **InfiniteTalk** as Apache-2.0 and custom. Both are untrusted for a default path.
- **Upscale-vs-native quality studies** are small (E1 single-pair); the QoMEX 2026 paper explicitly warns automatic VQA metrics are unreliable here.

## Related

[Music Video Pipeline Audit](music-video-pipeline-audit-2026-11.md) ·
[Delivery & QC Specs](concepts/music-video-delivery-specs.md) ·
[AI Music Video Studio](entities/video-studio.md) ·
[Remediation Plan](application-remediation-plan-2026-11.md) ·
[Index](index.md)
