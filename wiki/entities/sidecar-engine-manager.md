---
title: Sidecar Engine Manager
type: entity
created: 2026-09-15
updated: 2026-09-15
sources: [sources/v2-refactor-plan.md, sources/readme.md]
tags: [sidecar, uv, virtualenv, isolation, engines, architecture]
aliases: [Sidecar Engine Manager, SidecarEngineManager, Engine Manager]
---

# Sidecar Engine Manager

The **Sidecar Engine Manager** (`backend/app/services/sidecar/engine_manager.py`) provides isolated runtime environments for neural engines that require incompatible dependency versions or specialized C++/CUDA extensions.

## Problem Context

Milimo Music orchestrates diverse machine learning backbones:
- **MiniMax Music 3**: MLX (Apple Silicon) / PyTorch 2.x
- **Diffusers Wan 2.1 & LTX-Video**: Diffusers `>=0.30`, PyTorch, CUDA
- **MuLaCover & YourMT3**: Specific checkpoint requirements
- **LivePortrait**: Specialized insightface and onnxruntime dependencies

Forcing all neural backbones into a single global Python virtualenv leads to severe pip dependency hell (e.g., conflicting torch/torchaudio/cuda/numpy ABI versions).

## Sidecar Architecture

The `SidecarEngineManager` solves this using fast `uv`-isolated virtual environments located in `backend/engines/<engine_id>/.venv`:

```
backend/
├── app/                  # Main FastAPI Application
└── engines/
    ├── wan2_video/
    │   ├── .venv/        # uv-managed isolated virtualenv
    │   └── runner.py     # Subprocess runner script
    └── liveportrait/
        ├── .venv/
        └── runner.py
```

### Core Responsibilities

1. **Venv Provisioning**: Automatically provisions Python virtual environments using `uv venv backend/engines/<id>/.venv`.
2. **Dependency Resolution**: Installs engine-specific requirements via `uv pip install -r requirements.txt`.
3. **Subprocess Execution**: Launches isolated engine tasks via asynchronous subprocess calls with clean IPC / stdout telemetry pipes, preventing memory leaks or crashes from affecting the main API server.

## Related Pages

- [Architecture](../architecture.md)
- [Video Studio](video-studio.md)
- [Hardware Coordinator](hardware-coordinator.md)
- [Backend & API](backend-api.md)
