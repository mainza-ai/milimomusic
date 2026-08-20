---
title: HeartMuLa
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/heartlib-bible.md, sources/readme.md]
tags: [heartmula, model, llm, music, generation, provider]
aliases: [HeartMuLa-3B, HeartMuLa-oss-3B]
---

# HeartMuLa

**HeartMuLa** (Hear the Music Language) is the 3B-parameter music **language model** that
is the **legacy/local** generation engine of Milimo Music. It generates high-fidelity music
conditioned on lyrics and stylistic tags.

## Role
HeartMuLa is the semantic/token-prediction engine of the [Heartlib](heartlib.md) framework.
It is not a text model for chat — it predicts **audio tokens** given text (lyrics) and tags.
In the current app it is wrapped as a [generation provider](generation-provider.md)
(`heartmula`, legacy/local — the default is [MiniMax Music 3](minimax-music3.md)).

## Architecture
- **Hierarchical factorization**:
  - **Global Transformer** predicts coarse semantic tokens (Layer 0) conditioned on history.
  - **Local Transformer** predicts fine-grained acoustic details (Layers 1–7) conditioned on the global token.
- **Backbone**: modified **Llama 3.2** — 3B (Global) + 300M (Local).
- **Vocabulary**:
  - Text: 128,256 tokens (Llama 3 tokenizer).
  - Audio: 8,197 tokens.

## As a provider (`HeartMuLaProvider`)
From `get_capabilities()`: `max_duration_sec` 240; **no** structured captions (tag-list only,
`supports_structured_caption=false`); `supports_section_tags=true`; `supports_lora=true`;
default sample rate **48kHz**; license_class Apache-2.0.
- `initialize()` loads `HeartMuLaGenPipeline` from the configured model directory
  (`../heartlib/ckpt` default) on `cuda`/`mps`/`cpu` with fp16/bf16.
- `generate()` builds the tag-style prompt and calls the pipeline's `generate()`.
- `extend()` just delegates to `generate()`; `repair_segment()` returns a metadata-only stub.

## Conditioning
Follows the strict prompt structure `[BOS] <tag> {Style Tags} </tag> [EOS] [MUQ_EMBED] [Lyrics...] [EOS]`
where `[MUQ]` is the placeholder for MuQ-MuLan reference-audio embeddings — see
[Prompt structure & style tags](../concepts/prompt-structure.md).

## In Milimo Music
- Supported style tags that HeartMuLa-3B is fine-tuned for are listed in
  [Prompt structure & style tags](../concepts/prompt-structure.md).
- Selectable alongside MiniMax via [Model Manager](model-manager.md) / registry;
  can be fine-tuned via the [Training Studio](training-studio.md) (LoRA or full).
- Inpainting uses HeartMuLa as Stage 1 (semantic generation) of
  [LM-guided inpainting](../concepts/lm-guided-inpainting.md) (legacy path).

## Related pages
- [Heartlib](heartlib.md) | [HeartCodec](heartcodec.md) | [Generation provider](generation-provider.md)
- [MiniMax Music 3](minimax-music3.md) | [Heartlib Bible source](../sources/heartlib-bible.md)
