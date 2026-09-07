---
title: Modality Taxonomy (audio | image | video)
type: concept
tags: [modality, taxonomy, model-manager, huggingface, download, hailuo]
created: 2026-09-07
updated: 2026-09-07
sources: []
aliases: [category inference, modality inference, category_source]
---

# Modality Taxonomy (audio | image | video)

The **canonical source of truth** for repo → modality decisions is
[`backend/app/services/modality.py`](../../backend/app/services/modality.py)
(`PIPELINE_TAG_MAP`, `infer_modality`). Every other call site
(`ModelManager.infer_category`, `register_custom_model`,
`POST /models/download`, `search_huggingface`) delegates to it.
There is exactly **one** ordered decision chain — no duplicated keyword lists.

## Precedence (highest wins)

1. **Explicit user override** — `POST /models/download { category }`
   (the modality dropdown in the Direct Repo Downloader). Recorded as
   `category_source: "user"`.
2. **Hugging Face `pipeline_tag`** (exact map). Includes the video family
   `text-to-video`, `image-to-video`, **`image-text-to-video`**,
   `video-to-video`, plus the audio and image families.
3. **HF `tags` / `cardData` keywords** (video / image / audio signals).
4. **Repo-ID keyword rules, video-first with org disambiguation.**
   MiniMax owns *both* music (audio) and H3/Hailuo (video), so bare
   `minimax` is **not** an audio signal. Video disambiguators
   (`minimax-h3`, `hailuo`, `h3-mlx`, …) outrank music markers
   (`minimax-music`, `music3`, `mxfp4`, …).
5. **Filename signals** — video-DiT layouts (transformer blocks, video VAE
   markers) beat image-diffusion layouts (`unet`, `*.vae.safetensors`,
   `text_encoder_2`), which beat audio-codec markers (`codec`, `rvq`, …).
6. **Safe fallback** — `audio` with reason `fallback:needs_review`.
   The invalid fourth value `"custom"` is never produced or accepted
   (`update_custom_model` raises `ValueError` for it).

## Case study: MiniMax-H3-MLX-8bit → `models/video/` (2026-09-07)

`pipenetwork/MiniMax-H3-MLX-8bit` (MLX 8-bit, 35.3 GB, 7 shards) downloaded
to `models/audio/` through **three compounding defects**, all fixed in the
same change:

- `POST /models/download` fetched `model_info` but discarded `pipeline_tag`,
  inferring from filenames alone (`model_manager._infer_category(repo, files)`).
- The tag allowlists did not contain H3's real tag **`image-text-to-video`**,
  so even a correct pass-through fell through to keywords.
- The keyword fallback mapped the bare org name `minimax` → audio, with no
  `h3`/`hailuo` video disambiguator.

The UI dropdown (Audio/Image/Video) compounded it: the selection was never
sent (`startModelDownload(repoId)` only posted `repo_id`). It now posts
`{ repo_id, category }`, and the backend honors it over inference.

## Related rules

- **Recategorize = relocate.** `PATCH /models/custom/{id} { category }`
  moves `models/<old>/<slug>` → `models/<new>/<slug>` and updates
  `local_path`; metadata and bytes cannot split-brain.
- **Delete = unregister + free bytes**, across `models/{audio,image,video,audio_separator}/`.
- **Audio-provider guard.** `HuggingFaceAudioProvider` refuses
  video-diffusion weight dirs (H3/Hunyuan/CogVideoX/Wan markers) with an
  explicit error instead of attempting `text-to-audio` on them.

## Open gap (not fixed here)

> [!WARNING] Correctly placed video weights still have **no DiT inference
> backend** — `video_service.py` is an assembly pipeline (storyboard,
> lip-sync planning, karaoke burn, remux), and `VIDEO_MODEL_PATH` is written
> but never read. See [Video Studio](../entities/video-studio.md).

## Related pages

- [Model Manager](../entities/model-manager.md) | [Video Studio](../entities/video-studio.md) | [Generation Provider](../entities/generation-provider.md)
