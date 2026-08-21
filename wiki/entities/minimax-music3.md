---
title: MiniMax Music 3
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/v2-refactor-plan.md, sources/v2-refactor-plan.md]
tags: [minimax, music3, model, generation, provider]
aliases: [MiniMax-Music3, Music 3]
---

# MiniMax Music 3

**MiniMax Music 3** (`MiniMaxAI/MiniMax-Music3`, MLX build `mlx-community/MiniMax-Music3-bf16`)
is the **default generation model** in Milimo Music. It generates **full songs up to
5 minutes**, lyrics- and description-conditioned, with explicit section tags and
**Structured Captions**.

## Capabilities (as registered)
From `MiniMaxMusic3Provider.get_capabilities()`:
- **provider_id**: `minimax_music3` — version `Music-3-bf16`.
- **max_duration_sec**: 300 (5 min).
- **supports_structured_caption**: yes; **supports_section_tags**: yes; **supports_lora**: yes.
- **recommended_hardware**: `mid_single_gpu` (Apple Silicon MPS / 16GB+).
- **license_class**: `MiniMax Open Weights`; default sample rate **44.1kHz**.

## Structured Captions & section tags
- `parse_structured_caption()` reads `[Global Metadata]` / `[Vocal Details]` / `[Arrangement]`
  headers from the prompt, or constructs them from tags + free text following the official
  prompting guide's three-heading skeleton
  (see [Structured Captions](../concepts/structured-caption.md)).
- `generate(..., structured_caption=...)` **honors caller-provided sections** (composer /
  Ask Producer) and auto-fills missing ones; previously those UI fields never reached the
  model (see [Caption Rewriter](../concepts/caption-rewriter.md)).
- `sanitize_section_tags()` normalizes `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`,
  `[Instrumental]`, `[Solo]`, `[Outro]` etc. **and splits every tag onto its own line**
  (MiniMax drops lyric text sharing a line with a leading tag).

## Model snapshot / loading
- Default snapshot: `~/.cache/huggingface/hub/models--mlx-community--MiniMax-Music3-bf16/snapshots/…`
  (overridable via `MINIMAX_MODEL_PATH`). `is_ready()` returns true if the snapshot dir exists.
- See [Model Manager](model-manager.md) for the adapter/quantization **model tree**
  (BF16 28.5 GB default, INT8 14.2 GB quantized variant).

> [!NOTE] **Real inference — implemented.** `MiniMaxMusic3Provider.generate()` now runs
> **genuine MiniMax Music 3 weight inference** on Apple Silicon via `mlx_audio.music.generate`
> (`mlx-community/MiniMax-Music3-bf16`), conditioned on the prompt / structured caption /
> lyrics / section tags, writing `/audio/<job>.wav`. Requires `mlx` + `mlx-audio` (optional,
> Apple-only). Inference `steps` are clamped to the model's allowed 1–30 range (an earlier
> clamp of 32 made every song ≥62s fail real inference and silently fall back to the synth).
>
> **Fallback is never silent.** When the MLX runtime is missing, the snapshot is absent, or
> inference throws, `generate()` returns `used_fallback_synth=True` + `fallback_reason`, the
> [orchestration pipeline](../concepts/generation-pipeline.md) persists them on the `Job`,
> and the UI shows a visible "Fallback synthesis" badge (hero + AI Provenance tab). The
> procedural synth exists only so the app runs on every platform — users can always tell
> when a track was not actually produced by MiniMax Music 3.
>
> **Self-healing producer:** if the user hands over a bare prompt (e.g. "A smash hit pop song")
> and/or no lyrics, the [producer service](producer-service.md) invokes the real LLM producer to
> enhance the concept and write genuine structured lyrics, so real inference is always
> well-conditioned and never fails (`Lyrics are required`) or falls back to the synthetic
> placeholder (see [Producer Service](producer-service.md)).
>
> **Memory:** the MLX model is loaded **thread-safely** (a `threading.Lock` prevents two racing
> threads from loading two full ~28–40 GB copies), and can be released with
> `unload_minimax_model()`. HTDemucs is unloaded after separation, so both heavy models aren't
> resident between generations.

## In the pipeline
Resolved by the [generation-provider](generation-provider.md) registry and driven through the
[orchestration pipeline](../concepts/generation-pipeline.md) (Step 1).

## Licensing
**MiniMax Open Weights** (per `LICENSES.md`); code/library Apache-2.0. Up to 5-minute full
song generation with Structured Captions; **non-commercial project use only**.

## Related pages
- [Generation provider](generation-provider.md) | [Structured Captions](../concepts/structured-caption.md) | [Caption Rewriter](../concepts/caption-rewriter.md)
- [Model Manager](model-manager.md) | [Orchestration pipeline](../concepts/generation-pipeline.md)
- [Roadmap (v2)](../roadmap.md) | [v2 reference projects](v2-references.md)
