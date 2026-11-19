# Video Delivery, QC & Consistency

How a finished music video leaves the studio: takes, delivery presets, aspect variants,
the QC gate, captions, the portable manifest, subject continuity across scenes, and the
lip-sync engine ladder.

> [!NOTE]
> Video generation itself (Wan 2.1 / LTX-Video / hosted backbones, the director, keyframes)
> is documented in [`wiki/entities/video-studio.md`](../wiki/entities/video-studio.md).
> This page is about what happens **after** a master exists.

---

## 1. Takes: nothing is overwritten

Every render is kept as a **take**: a row pointing at its own master, its own clip
workspace and its own plan snapshot (`save_clip_plan`). Retakes, re-renders and exports
create new takes; the previous master is untouched.

| Endpoint | Purpose |
|---|---|
| `GET /videos/{job_id}/renders` | Every take for the track, with state and selected flag |
| `POST /videos/renders/{render_id}/select` | Point `Job.video_path` at exactly one take |
| `DELETE /videos/renders/{render_id}` | Delete a take, its artifacts and its clip workspace |

Two failure modes are handled at the lifecycle level, not by hand:

- **Boot reconciliation** — renders still marked `rendering` when the process died are
  failed on startup, so the UI never shows a phantom job.
- **Stall sweeper** — a render `rendering` longer than any real render takes
  (`MILIMO_RENDER_STALL_SECONDS`, default `3600`) is failed and its reason recorded.

---

## 2. Delivery presets

`GET /videos/presets` lists the static catalogue (cached, ETag). Applying one is
non-destructive: it reads the master and writes a new file plus a new take.

| Preset | Output |
|---|---|
| `default` | H.264 CRF 18, faststart, −14 LUFS |
| `youtube_1080p` | 8 Mbps H.264, max 1920×1080 |
| `youtube_2160p` | 40 Mbps H.264, max 3840×2160 |
| `vevo_hd` | 20 Mbps H.264, 44.1 kHz 320k AAC, master loudness passed through, no slates |
| `canvas` | Silent 9:16 H.264 (Spotify Canvas), ≤8 s |
| `broadcast_prores` | ProRes 422 HQ, 48 kHz PCM, −23 LUFS |

`POST /videos/{job_id}/export/{preset_id}` returns **202** and runs the transcode in a
worker thread.

### Exports are genuinely cancellable

`POST /videos/tasks/{task_id}/cancel` stops the work. This is not cosmetic: an export
runs inside a worker thread where an asyncio cancellation cannot reach it, so the
transcode runs under `Popen` with a per-task `threading.Event`. On cancel FFmpeg is
terminated (killed after 5 s if it resists) and the partial output is removed — no
half-written take is left behind.

---

## 3. Aspect variants

`POST /videos/{job_id}/variants` derives other aspect ratios from **one master**, so a
vertical cut does not cost another diffusion pass.

| Aspect | 720p | 1080p | 2160p |
|---|---|---|---|
| `16:9` | 1280×720 | 1920×1080 | 3840×2160 |
| `9:16` | 720×1280 | 1080×1920 | — |
| `1:1` | 720×720 | 1080×1080 | — |
| `21:9` | 1680×720 | 2560×1080 | — |

Fit mode is `crop` by default (no letterboxing bars); unsupported combinations are
skipped rather than silently resized to the wrong shape.

---

## 4. QC gate

Every delivery is checked before it is presented as finished. `GET /videos/{job_id}/report`
returns the cut list, delivery config, captions and QC verdict for the selected take,
and writes `<stem>_report.json` next to the master.

| Check | Threshold | Why it matters |
|---|---|---|
| Black segments | ≥ **0.10 s** at ≤ 0.10 luma | FFmpeg's default is 2 s, which misses the 1 s and 5-frame gaps a stitched render produces |
| Freeze segments | ≥ **0.05 s** below 0.001 noise | Frozen motion is the signature of a failed clip |
| Silence | ≥ **0.10 s** at ≤ −50 dB | Silent gaps mean a missing audio segment |
| Duration | ± **0.10 s** vs the song | A short assembly padded by the container |
| A/V start drift | ≤ **0.05 s** | Structural sync proxy: per-stream start and duration disagreement |
| Loudness | ± **1.5 LU** vs the target | `ebur128` integrated LUFS + true peak |

`ok` is `False` only for faults that make the file unusable or a lie about the song
(missing/unreadable output, no audio stream). Advisories — a black tail, a long freeze,
loudness drift — are reported but do not fail the render.

> [!WARNING]
> QC **cannot** judge lip sync. It reports the structural proxy and says so. Perceptual
> sync needs eyes on the file.

---

## 5. Captions

Word-level karaoke is generated as ASS with centisecond `\kf` sweeps, then burned in by
FFmpeg (`subtitles=...`) and written as sidecars (`.srt`, `.vtt`, `.ass`) next to the
master. The report lists which sidecars exist.

Six typography presets: `neon` (cyberpunk, default), `spotify` (streaming-clean),
`kinetic_pop`, `cinematic`, `social_vertical` (safe-zone aware) and `retro_vhs`.

Text is escaped for ASS before it is written — raw `{`/`}` open override blocks and
would corrupt the caption line. Bundled fonts are resolved from
`backend/app/services/fonts`; when absent, libass falls back to system fonts and logs a
warning, which is why renders could otherwise look different on Docker vs macOS.

---

## 6. Portable manifest

`GET /jobs/{job_id}/manifest.json` (and `manifest.json` inside the Studio Pack) is the
record a paper, an archive or a bug report can cite:

- seed, sampling parameters, model id, verbatim prompt and lyrics;
- per-stage provenance and the quality verdict;
- every artifact with size + SHA-256;
- app version, git commit, platform;
- the video takes.

Design rule: **never invent**. A missing fact is `null`, not a default.

---

## 7. Subject consistency across scenes

Eight independently seeded scenes read as eight different people. Hosted tools solve it
with reference conditioning; this pipeline bounds the drift with two deterministic
mechanisms that need no extra weights:

| Mode | Mechanism |
|---|---|
| `off` | Independent scenes (previous behaviour) |
| `character` *(default)* | One persisted **character bible** — a single physical description injected into every scene prompt |
| `chain` | Character bible **plus last-frame chaining**: the previous clip's final frame seeds the next scene's image-to-video pass, carrying lighting, grade and framing across the cut |

`chain_scope` controls reach: `section` (default) keeps continuity inside a verse or
chorus and lets a cut into a new section reset the frame; `all` chains across everything.

The description is never invented: explicit config wins, then the director's persisted
character profile, then nothing. Stamping is idempotent, so re-rendering a
already-stamped prompt does not stack the description N times.

Planning accepts the **same** `character_id`, `consistency_mode` and `chain_scope` as the
render (`POST /videos/plan/{job_id}`), so the storyboard the user approves describes the
performer that will actually be rendered.

### Character library

`GET/POST /characters`, `PATCH/DELETE /characters/{id}` store a reference plate plus
description; the plate is copied into the library so the character survives the removal
of the source file. `resolve_for_render` returns exactly what a render needs.

Candidate selection uses dHash (stable under scale and mild grading), Laplacian sharpness
(a blurry plate should not win) and optional CLIP embedding similarity
(`MILIMO_VERIFIER_EMBEDDING=auto|clip|off`).

---

## 8. Lip-sync engine ladder

`GET /videos/lipsync-engines` reports every engine **with real availability**, the reason
it cannot run, and its licence. "Install the weights and it works" is only true if the
studio can tell you what is missing.

| Engine | Kind | Requires | Singing | Licence |
|---|---|---|---|---|
| `jaw_warp` | local, procedural | nothing | n/a | In-house |
| `live_portrait` | local, neural | `models/video/liveportrait` + an audio entry point (`MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT`) | no | MIT code/weights; bundled InsightFace detectors are research-only |
| `fal` | cloud | `FAL_KEY` | no | Provider terms |
| `replicate` | cloud | — | no | Not wired (decision D4: cloud is fal only) |
| `echomimic_flash` | local CLI | `MILIMO_ECHOMIMIC_DIR` → `infer_flash.py` | no | Apache-2.0, ~12 GB VRAM, 25 fps |
| `wan_s2v` | local CLI | `MILIMO_WAN_S2V_DIR` → `generate.py` | **yes** | Apache-2.0, ~80 GB VRAM, 16 fps |
| `infinitetalk` | local CLI | `MILIMO_INFINITETALK_DIR` → `generate_infinitetalk.py` (+ `MILIMO_INFINITETALK_TEMPLATE` request JSON) | **yes** | Apache-2.0, ~24 GB VRAM, streaming, image or video |

Upstream LivePortrait is **video-driven**; audio-driven operation needs a wrapper script
(`inference_audio.py`) or `MILIMO_LIVEPORTRAIT_AUDIO_SCRIPT`. Without it the engine
reports exactly that instead of failing mid-render.

External tiers run in a child process, so a crash in a research repo cannot take the API
down. Their JSON templates are request-shaped: point `MILIMO_INFINITETALK_TEMPLATE` at a
copy of the upstream example rather than editing the repo.

---

## 9. Capability matrix

`GET /videos/capability-matrix` answers "what will this machine actually produce?" — the
device (`mps` / `cuda` / `cpu`), the usable interpolation backend, and each engine's
**effective** resolution, duration and fps after clamping to what the hardware can do,
not what the model could do on a datacentre GPU.

Related: `GET /videos/active-engine`, `GET /videos/models`, `GET /videos/providers`.
