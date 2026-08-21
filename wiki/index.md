---
title: Milimo Music Wiki — Index
type: index
created: 2026-08-19
updated: 2026-08-21
---

# Milimo Music Wiki — Index

This is the content catalog for the Milimo Music wiki. Every page is listed below
grouped by kind. Start at [overview](overview.md) for the synthesis, then drill in.

## Top-level pages

- [Overview](overview.md) — what Milimo Music is (AI generation + transcription + DAW).
- [Architecture](architecture.md) — the system: layers, providers, pipeline, data flow.
- [Roadmap (v2)](roadmap.md) — the refactor/upgrade plan and its implementation status.

## Entities — generation & providers

- [Generation Provider Abstraction](entities/generation-provider.md) — pluggable `GenerationProvider` interface + registry + capability manifests.
- [MiniMax Music 3](entities/minimax-music3.md) — the default generation model (structured captions, up to 5 min).
- [HeartMuLa](entities/heartmula.md) — the 3B music language model; now a legacy/local provider.
- [Heartlib](entities/heartlib.md) — the local audio-generation framework wrapping HeartMuLa + HeartCodec.
- [HeartCodec](entities/heartcodec.md) — the 12.5 Hz 8-codebook neural audio codec.
- [HeartCLAP](entities/heartclap.md) — contrastive language–audio pretraining component.
- [HeartTranscriptor](entities/hearttranscriptor.md) — transcription component of Heartlib.
- [HeartMuLaGenPipeline](entities/heartmulagenpipeline.md) — the HeartMuLa generation pipeline.
- [Model Manager](entities/model-manager.md) — model tree, hardware tiers, missing-dependency checks.

## Entities — transcription, DAW & production

- [MuScriptor](entities/muscriptor.md) — multi-instrument transcription → MIDI + MusicXML (git submodule, integrated).
- [Stem Separation (Dual-Engine)](entities/stem-separator.md) — HTDemucs real neural separation + MuScriptor per-instrument parts, user-selectable in the DAW.
- [Matchering Reference Mastering](entities/matchering-mastering.md) — -14 LUFS reference mastering.
- [Karaoke & Lyric Sync](entities/karaoke-lyricsync.md) — timed lyrics, `.lrc`/`.srt` export.
- [Voice Studio (SVC)](entities/voice-service.md) — offline voice conversion + consent-gated profiles.
- [Session Workspace (DAW)](entities/session-workspace.md) — Listen/Arrange/Piano Roll/Notation/Mix/Lyrics.

## Entities — in-app services & agents

- [AI Co-Writer](entities/ai-cowriter.md) — the multi-agent lyrics engine (Coordinator→Lyricist→StructureGuard).
- [Producer Service](entities/producer-service.md) — LLM producer that enhances weak prompts + writes real lyrics so real inference never fakes or fails.
- [Training Studio](entities/training-studio.md) — fine-tune HeartMuLa on custom audio datasets (LoRA/full).
- [Repair Segment / Inpainting Service](entities/inpainting.md) — regenerate a time-range or glitch region.
- [LLM Service & Providers](entities/llm-service.md) — OpenCode, OMLX, Ollama, OpenAI, Gemini, OpenRouter, DeepSeek, LM Studio.
- [Backend & API](entities/backend-api.md) — FastAPI/SQLModel backend, Job/Project models, endpoints, SSE.
- [Frontend](entities/frontend.md) — React 19 + Vite + Tailwind; Suno-class IA + DAW workspace.

## Entities — external dependencies & tools

- [The v2 reference projects](entities/v2-references.md) — MiniMax, MuScriptor, and tools status (🔵/🟡/⚪).

## Concepts

- [Orchestration Pipeline](concepts/generation-pipeline.md) — the 4-step generate → stems → voice → transcribe flow.
- [Structured Captions](concepts/structured-caption.md) — the MiniMax Global Metadata / Vocal Details / Arrangement format.
- [Lyrics conditioning](concepts/lyrics-conditioning.md) — how audio is aligned to lyrics & prosody.
- [Prompt structure & style tags](concepts/prompt-structure.md) — the [BOS] <tag>… format + supported HeartMuLa tags.
- [Track extension](concepts/track-extension.md) — continuing generation from a prior track's tail.
- [LM-guided inpainting](concepts/lm-guided-inpainting.md) — the two-stage repair strategy.
- [LoRA fine-tuning](concepts/lora-finetuning.md) — low-rank adaptation in the Training Studio.
- [AI Co-Writer graph](concepts/co-writer-graph.md) — the pydantic-graph workflow for lyrics editing.

## Sources

- [README (Milimo Music)](sources/readme.md) — product overview, capabilities, setup.
- [Heartlib Bible](sources/heartlib-bible.md) — the definitive Heartlib framework guide.
- [Training Studio Guide](sources/training-studio-guide.md) — UI + API reference for fine-tuning.
- [Inpainting & Glitch Repair Debug Log](sources/inpainting-debug.md) — how repair was built.
- [Milimo Music v2 Refactor Plan](sources/v2-refactor-plan.md) — the upgrade roadmap source.

## Navigation helpers

- [log](log.md) — chronological record of every ingest/query/lint operation.
- [AGENTS.md](../AGENTS.md) — the schema governing this wiki.

---

> [!NOTE] This index is updated on every ingest. If a page is missing here, the wiki is out of date — run a lint pass.
