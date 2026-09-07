---
title: Source — README
type: source
created: 2026-08-19
updated: 2026-09-07
tags: [source, readme, overview, production, video, models]
aliases: [README]
---

# Source — README

The project `README.md` — canonical product overview, core capabilities, visual studio tour, architectural diagrams, operations, and quickstart setup guide.
**Raw location:** repo root `README.md`.

## Key contents (summary)
- **Positioning**: Next-generation open-source AI music generation, neural transcription, and multitrack production DAW created by Mainza Kangombe.
- **Core Technology & Production Pipelines**:
  - **Generation Engine**: MiniMax Music 3 default engine (native Apple Silicon MLX, PyTorch CUDA INT8, and CPU GGUF) with Structured Caption Rewriter and precision sampling controls (`temperature`, `cfg_scale`, `top_k`).
  - **Multi-Modal Model Hub**: 23-model catalog spanning Audio (MiniMax, HeartMuLa), Image (Black Forest Labs FLUX.2 klein/dev, FLUX.1 schnell, SDXL Turbo), and Video (MiniMax Hailuo H3, Wan 2.1, CogVideoX 1.5, HunyuanVideo).
  - **Live Hugging Face Hub Search**: Real-time querying and on-demand model repository downloading with custom registry integration.
  - **Stem Separation**: BS-Roformer neural 6-stem source separation (`vocals`, `drums`, `bass`, `guitar`, `piano`, `other`) and dual-engine matrix with MuScriptor instrument parts.
  - **MuScriptor Neural Transcription**: Note-level polyphonic transcription into multitrack MIDI and W3C MusicXML 3.1 with Grand Staff engraving.
  - **Neural Acoustic Lyric Sync**: TorchAudio MMS_FA forced alignment with sub-100ms word sync, `.lrc`, and `.srt` export.
  - **AI Music Video Studio**: Model duration constraint management (up to 15.0s on Hailuo H3/Hunyuan), isolated vocal stem viseme lip-syncing via OpenCV facial landmark deformation, animated ASS karaoke subtitle burning, and multi-axis Ken Burns / procedural visual generation.
  - **Voice Training Studio & Neural SVC**: Offline singing voice conversion with RVC `.pth` checkpoint loading and profile-specific acoustic formant/presence equalization chains.
  - **Cover Art Studio**: Real neural diffusion via Diffusers/MFlux and studio-grade 1024x1024 raster PNG generation.
  - **6-Mode Web Audio DAW Workspace**: Listen, Multitrack Arrange, Grand Piano Roll, Notation Editor, Console Mixer, and Lyrics.
  - **Mastering**: Matchering reference mastering with sample-accurate A/B audition toggle (-14.0 LUFS broadcast target).
  - **AI Co-Writer & Multi-Agent Crew**: Multi-provider LLM integration (OpenCode Go, NVIDIA NIM, DeepSeek, local OMLX, Ollama, OpenAI, Gemini) and persistent Artist Crew agents.
  - **Deployment**: Unified single-process serving (`fastapi` hosting `frontend/dist` with SPA fallback) and multi-stage Docker packaging (`docker-compose.yml`, `docker-compose.cpu.yml`).
- **Prerequisites & Setup**: Python 3.12 (Conda), Node 18+, FFmpeg with libx264/AAC.
- **License**: Apache-2.0. Upstream weights governed by respective licenses (MiniMax Open Weights, MuScriptor CC BY-NC 4.0, FLUX.2 Apache 2.0 / Non-Commercial Dev).
- **Creator**: Mainza Kangombe.

## Related pages
- [Overview](../overview.md) | [Architecture](../architecture.md) | [Model Manager](../entities/model-manager.md) | [Video Studio](../entities/video-studio.md) | [Index](../index.md)
