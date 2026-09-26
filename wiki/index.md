---
title: Milimo Music Wiki — Index
type: index
created: 2026-08-19
updated: 2026-09-25
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
- [Stable Audio Open 1.0 Provider](entities/stable-audio-open.md) — cross-platform DiT audio generation (CUDA/MPS/CPU), native 44.1 kHz stereo, immediate VRAM eviction.
- [Meta MusicGen Provider](entities/musicgen.md) — lightweight autoregressive music generation, melody-guided conditioning, CPU-friendly execution.
- [YuE2 48kHz Stereo Music Provider](entities/yue2-music.md) — open-weight 48 kHz stereo music generation, ABC notation guidance, source covers, auto-instrumental LoRA routing, and Auto/Guided My Music training.
- [MuLaCover](entities/mulacover.md) — 3B controllable music cover & remix engine (symbolic cross-attention, dual transcription, composite downloader).
- [HeartMuLa](entities/heartmula.md) — the 3B music language model; now a legacy/local provider.
- [Heartlib](entities/heartlib.md) — the local audio-generation framework wrapping HeartMuLa + HeartCodec.
- [HeartCodec](entities/heartcodec.md) — the 12.5 Hz 8-codebook neural audio codec.
- [HeartCLAP](entities/heartclap.md) — contrastive language–audio pretraining component.
- [HeartTranscriptor](entities/hearttranscriptor.md) — transcription component of Heartlib.
- [HeartMuLaGenPipeline](entities/heartmulagenpipeline.md) — the HeartMuLa generation pipeline.
- [Model Manager](entities/model-manager.md) — model tree, hardware tiers, missing-dependency checks.

## Entities — transcription, DAW & production

- [MuScriptor](entities/muscriptor.md) — multi-instrument transcription → MIDI + MusicXML (git submodule, integrated).
- [Multitrack Timeline Editor](entities/multitrack-editor.md) — non-destructive video and audio multi-track editor, transitions, hardware-accelerated FFmpeg export, and AI round-trip takes.
- [Drum Tracker](entities/drum-tracker.md) — spectral flux sub-band onset extraction for Kick (36), Snare (38), Hi-Hat (42) MIDI conditioning.
- [Stem Separation (Dual-Engine)](entities/stem-separator.md) — HTDemucs real neural separation + MuScriptor per-instrument parts, user-selectable in the DAW.
- [Matchering Reference Mastering](entities/matchering-mastering.md) — -14 LUFS reference mastering.
- [Karaoke & Lyric Sync](entities/karaoke-lyricsync.md) — TorchAudio MMS_FA neural forced alignment, sub-100ms word sync, .lrc/.srt export.
- [Voice Studio (SVC)](entities/voice-service.md) — 3-zone vocal production suite: neural voice conversion, live in-browser vocal booth recording, vocal DSP rack (transposition presets, formant preservation, dry/wet mix), tri-state A/B audition transport, and consent-gated profiles.
- [Neural Singing Voice Conversion (SVC)](entities/neural-svc.md) — zero-shot vocal timbre transfer, formant morphing, pitch transposition, and dry/wet blending.
- [AI Music Video Studio](entities/video-studio.md) — Wan 2.1 & LTX-Video diffusion, LivePortrait neural singing avatar lip-syncing, autonomous musical director, pre-rendered keyframes, and burned karaoke ASS subtitles.
- [Session Workspace (DAW)](entities/session-workspace.md) — Listen/Arrange/Piano Roll/Notation/Mix/Lyrics.
- [Studio Projects](entities/projects.md) — multi-session production workspaces, BPM/Key conditioning, multi-track stems aggregation, and Studio Pack (.zip) export.

## Entities — in-app services & agents

- [Global Hardware Coordinator](entities/hardware-coordinator.md) — centralized GPU device lock, Auto-Tune empirical profiles (1–5), VRAM safety coefficients, scoped CPU memory execution, and live telemetry bar.
- [Durable Task Queue & Job Lifecycle Manager](entities/durable-task-queue.md) — persistent SQLite task queue, input asset vaulting, independent clip checkpointing, and 1-click restart recovery.
- [Sidecar Engine Manager](entities/sidecar-engine-manager.md) — isolated `uv` virtual environments for external audio/video backbones (`backend/engines/<id>/.venv`).
- [AI Co-Writer](entities/ai-cowriter.md) — the multi-agent lyrics engine (Coordinator→Lyricist→StructureGuard).
- [Producer Service](entities/producer-service.md) — LLM producer that enhances weak prompts + writes real lyrics; captions now come from the [Caption Rewriter](concepts/caption-rewriter.md).
- [Training Studio](entities/training-studio.md) — (Deprecated) decommissioned due to MiniMax Music 3 decoder-only architecture and VRAM constraints. `tags: [training, lora, deprecated]`
- [Artist Crew Agents](entities/artist-crew-agents.md) — the four registered agents (Experiencer, World Builder, Stylist, Critic) and how they hook into the album pipeline.
- [Task Queue (Phase 4)](entities/task-queue.md) — Phase 4 design (locked): SQLite-backed `TaskRecord` queue, GPU/IO lanes, 202 + SSE endpoint conversions, re-enqueue-on-restart.
- [Repair Segment / Inpainting Service](entities/inpainting.md) — audio-domain infill regeneration, beat-grid downbeat snapping, equal-power crossfading, and post-repair separation/transcription cascade.
- [LLM Service & Providers](entities/llm-service.md) — OpenCode, Anthropic Claude, OMLX, Ollama, OpenAI, Gemini, OpenRouter, DeepSeek, LM Studio, NVIDIA NIM.
- [Backend & API](entities/backend-api.md) — FastAPI/SQLModel backend, Job/Project models, endpoints, SSE.
- [Docker Deployment](entities/docker-deployment.md) — multi-stage container build, GPU/CPU compose profiles, volume persistence, and single-process web serving.
- [Frontend](entities/frontend.md) — React 19 + Vite + Tailwind; Suno-class IA + DAW workspace.

## Entities — external dependencies & tools

- [The v2 reference projects](entities/v2-references.md) — MiniMax, MuScriptor, and tools status (🔵/🟡/⚪).

## Concepts

- [Director Mode v2](concepts/director-mode-v2.md) — multi-signal audio analysis, hierarchical accent snapping, cut speed pacing ($-2$ to $+2$), visible-cast performance scoping, music-timeline vocal bypass, 0–5 fidelity repair retries, and discrete model frame lattice trimming.
- [Non-Destructive Multitrack Timeline](concepts/non-destructive-multitrack-timeline.md) — atomic project schema, single-pass FFmpeg hardware-accelerated filter graph compilation (NVENC/VideoToolbox), and AI round-trip take workflow.
- [Hardware Auto-Tune, Memory Profiles & OOM Self-Healing](concepts/hardware-autotune-memory-profiles.md) — zero-config empirical profiles 1 to 5, $\le 0.80$ VRAM safety coefficient, scoped CPU execution, kernel benchmarking (PyTorch vs Triton), and self-healing telemetry.
- [Cross-Modal Model Lifecycle & Immediate Eviction Architecture](concepts/cross-modal-model-lifecycle.md) — unified immediate model eviction across audio, image, and video; framework purge protocols (MLX Metal cache, CUDA IPC/caching allocator, glibc malloc_trim); Director phase-decoupled execution; dual eager/TTL memory policies.
- [Orchestration Pipeline](concepts/generation-pipeline.md) — the 4-step generate → stems → voice → transcribe flow.
- [Structured Captions](concepts/structured-caption.md) — the MiniMax Global Metadata / Vocal Details / Arrangement format.
- [Caption Rewriter](concepts/caption-rewriter.md) — official music-caption-rewriter port: brief → professional three-heading caption via the real LLM.
- [Lyrics conditioning](concepts/lyrics-conditioning.md) — how audio is aligned to lyrics & prosody.
- [Track extension](concepts/track-extension.md) — continuing generation from prior track; model-native MLX KV-cache roll-forward, strict lyrics control (never auto-added by default), beat-grid snapping, and verified zero tempo drift ($\Delta = 0.0$ BPM).
- [Singing Voice Conversion](concepts/singing-voice-conversion.md) — Phase 5 locked design: vendored RVC v2 inference (RMVPE + ContentVec), honest DSP fallback, voice-convert bug fixes.
- [Playlists & Studio Profile](concepts/playlists-profiles.md) — Phase 6 locked design: Playlist/PlaylistTrack/StudioUserProfile tables, Alembic baseline, localStorage one-time import.
- [LM-guided inpainting](concepts/lm-guided-inpainting.md) — from legacy token masking to v2 audio-domain infill with beat-grid snapping and equal-power crossfading.
- [LoRA fine-tuning](concepts/lora-finetuning.md) — low-rank adaptation in the Training Studio.
- [AI Co-Writer graph](concepts/co-writer-graph.md) — the pydantic-graph workflow for lyrics editing.
- [AI Agent Foundation](concepts/agent-foundation.md) — LLM layer investigation + AgentRuntime proposal for multi-agent support.
- [Artist Profiles & Album Agents](concepts/artist-profiles-vision.md) — the ultimate vision: per-project artist identities with assigned agent crews producing full albums.
- [Album Orchestrator Plan](concepts/album-orchestrator-plan.md) — R1–R4 build plan: seed→song mapping, run lifecycle engine, gated album execution.
- [Artist Production Gap Report](concepts/artist-production-gap-report.md) — evidence-based artist-domain audit; its E–H plan is fully shipped (status header inside).
- [Artist Remaining Roadmap](concepts/artist-remaining-roadmap.md) — waves 1–3 shipped (voice identity, World-Builder, observability…); only LoRA links deferred.
- [Artist Domain](concepts/artist-domain.md) — current state: data model, album pipeline with crew hooks, endpoints, frontend surface.
- [Naming Contract](concepts/naming-contract.md) — Session/Job/Project titles are independent; rename validation, auto-rename guard. `tags: [naming, sessions, projects]`
- [Modality Taxonomy](concepts/modality-taxonomy.md) — canonical audio|image|video decision chain, H3 misrouting case study, relocate/delete/guard rules. `tags: [modality, taxonomy, model-manager]`
- [Audio Synthesis & Performance Standards](concepts/audio-synthesis-standards.md) — Fletcher-Munson loudness calibration (-13 to -18 dBFS), physical acoustic modeling (electric guitar pick transient, clarinet stopped pipe odd harmonics, drum sub sweeps), crest factor / spectral centroid metrics, and Web Audio transport performance standards. `tags: [dsp, synthesis, loudness, lufs, crest-factor, web-audio, performance-standards]`
- [Vocal Performance Tokens](concepts/vocal-performance-tokens.md) — acoustic performance modifiers (`[breath]`, `[whisper]`, `[belt]`, `[pause]`) and section duet casting. `tags: [lyrics, performance, tokens, composition]`
- [Stem Audio-Reactive Video](concepts/stem-audio-reactive-video.md) — bleed-free vocal envelopes for lip sync and rhythm transient extraction for Wan 2.1 camera zoom/shakes. `tags: [video, audio-reactive, stems, lip-sync]`
- [Modal Store Architecture](concepts/modal-store-architecture.md) — Zustand single-mount modal state management and timeline clip routing. `tags: [frontend, zustand, modal, daw]`
- [Database Integrity Lifecycle](concepts/database-integrity-lifecycle.md) — SQLite text vs SQLAlchemy GUID 32-hex dialect contract, universal multi-format lookup (`get_job_by_id`), boot-time self-healing migrations, relational cascade nullification, and comprehensive filesystem sweeps. `tags: [database, sqlite, sqlalchemy, sqlmodel, uuid, lifecycle, cascade-delete, data-integrity]`
- [Artwork & Static Media Architecture](concepts/artwork-and-static-media-architecture.md) — multi-directory fallback static file serving (`RangedStaticFiles`), bidirectional disk mirroring, auto-cover generation lifecycle, and on-demand manual artwork generation/regeneration. `tags: [media, covers, static-files, ranged-static, pipeline, storage]`

## Reports

- [Production Readiness Audit](production-readiness-audit.md) — security/reliability/frontend/ops findings with file:line refs and fix status.
- [Production Readiness Plan](production-readiness-plan.md) — phased remediation (secrets → security → make-it-real → job engine → CI → packaging) with locked decisions.
- [UI/UX & Design Audit](ui-ux-audit.md) — honesty failures, logic bugs, design-system state, DAW interaction gaps vs pro standards.
- [UI/UX Plan Progress](ui-ux-plan.md) — delivered wave (truth pass, disciplined glass, full piano-roll editor, perf pass, peaks library) + deferred list.

## Sources

- [Maestro Creative Studio Ingest](sources/maestro-creative-studio.md) — architecture, Director v2 (vocal bypass & 0–5 repair retries), Editor mode, YuE2 48kHz audio (Auto/Guided training), H3 Singularity, Qwen 2.1, Hardware Auto-Tune, and universal queue.
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
