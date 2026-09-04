---
title: Milimo Music Wiki — Index
type: index
created: 2026-08-19
updated: 2026-09-03
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
- [MiniMax Music 3](entities/minimax-music3.md) — the default generation model (structured captions, up to 5 min; fallback-to-synth now surfaced to the UI).
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
- [Karaoke & Lyric Sync](entities/karaoke-lyricsync.md) — TorchAudio MMS_FA neural forced alignment, sub-100ms word sync, .lrc/.srt export.
- [Voice Studio (SVC)](entities/voice-service.md) — offline voice conversion + consent-gated profiles.
- [Session Workspace (DAW)](entities/session-workspace.md) — Listen/Arrange/Piano Roll/Notation/Mix/Lyrics.

## Entities — in-app services & agents

- [AI Co-Writer](entities/ai-cowriter.md) — the multi-agent lyrics engine (Coordinator→Lyricist→StructureGuard).
- [Producer Service](entities/producer-service.md) — LLM producer that enhances weak prompts + writes real lyrics; captions now come from the [Caption Rewriter](concepts/caption-rewriter.md).
- [Training Studio](entities/training-studio.md) — fine-tune HeartMuLa on custom audio datasets (LoRA/full).
- [Artist Crew Agents](entities/artist-crew-agents.md) — the four registered agents (Experiencer, World Builder, Stylist, Critic) and how they hook into the album pipeline.
- [Durable Task Queue](entities/task-queue.md) — Phase 4 design (locked): SQLite-backed `TaskRecord` queue, GPU/IO lanes, 202 + SSE endpoint conversions, re-enqueue-on-restart.
- [Repair Segment / Inpainting Service](entities/inpainting.md) — regenerate a time-range or glitch region.
- [LLM Service & Providers](entities/llm-service.md) — OpenCode, OMLX, Ollama, OpenAI, Gemini, OpenRouter, DeepSeek, LM Studio.
- [Backend & API](entities/backend-api.md) — FastAPI/SQLModel backend, Job/Project models, endpoints, SSE.
- [Frontend](entities/frontend.md) — React 19 + Vite + Tailwind; Suno-class IA + DAW workspace.

## Entities — external dependencies & tools

- [The v2 reference projects](entities/v2-references.md) — MiniMax, MuScriptor, and tools status (🔵/🟡/⚪).

## Concepts

- [Orchestration Pipeline](concepts/generation-pipeline.md) — the 4-step generate → stems → voice → transcribe flow.
- [Structured Captions](concepts/structured-caption.md) — the MiniMax Global Metadata / Vocal Details / Arrangement format.
- [Caption Rewriter](concepts/caption-rewriter.md) — official music-caption-rewriter port: brief → professional three-heading caption via the real LLM.
- [Lyrics conditioning](concepts/lyrics-conditioning.md) — how audio is aligned to lyrics & prosody.
- [Prompt structure & style tags](concepts/prompt-structure.md) — the [BOS] <tag>… format + supported HeartMuLa tags.
- [Track extension](concepts/track-extension.md) — continuing generation from a prior track's tail; Phase 5 locked design: analysis-conditioned + equal-power crossfade.
- [Singing Voice Conversion](concepts/singing-voice-conversion.md) — Phase 5 locked design: vendored RVC v2 inference (RMVPE + ContentVec), honest DSP fallback, voice-convert bug fixes.
- [Playlists & Studio Profile](concepts/playlists-profiles.md) — Phase 6 locked design: Playlist/PlaylistTrack/StudioUserProfile tables, Alembic baseline, localStorage one-time import.
- [LM-guided inpainting](concepts/lm-guided-inpainting.md) — the two-stage repair strategy.
- [LoRA fine-tuning](concepts/lora-finetuning.md) — low-rank adaptation in the Training Studio.
- [AI Co-Writer graph](concepts/co-writer-graph.md) — the pydantic-graph workflow for lyrics editing.
- [AI Agent Foundation](concepts/agent-foundation.md) — LLM layer investigation + AgentRuntime proposal for multi-agent support.
- [Artist Profiles & Album Agents](concepts/artist-profiles-vision.md) — the ultimate vision: per-project artist identities with assigned agent crews producing full albums.
- [Album Orchestrator Plan](concepts/album-orchestrator-plan.md) — R1–R4 build plan: seed→song mapping, run lifecycle engine, gated album execution.
- [Artist Production Gap Report](concepts/artist-production-gap-report.md) — evidence-based artist-domain audit; its E–H plan is fully shipped (status header inside).
- [Artist Remaining Roadmap](concepts/artist-remaining-roadmap.md) — waves 1–3 shipped (voice identity, World-Builder, observability…); only LoRA links deferred.
- [Artist Domain](concepts/artist-domain.md) — current state: data model, album pipeline with crew hooks, endpoints, frontend surface.

## Reports

- [Production Readiness Audit](production-readiness-audit.md) — security/reliability/frontend/ops findings with file:line refs and fix status.
- [Production Readiness Plan](production-readiness-plan.md) — phased remediation (secrets → security → make-it-real → job engine → CI → packaging) with locked decisions.
- [UI/UX & Design Audit](ui-ux-audit.md) — honesty failures, logic bugs, design-system state, DAW interaction gaps vs pro standards.
- [UI/UX Plan Progress](ui-ux-plan.md) — delivered wave (truth pass, disciplined glass, full piano-roll editor, perf pass, peaks library) + deferred list.

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
