---
title: Docker Deployment & Single-Process Architecture
type: entity
created: 2026-09-07
updated: 2026-09-07
tags: [docker, deployment, container, production, nginx, uvicorn, packaging]
aliases: [Docker, Containerization, Docker Deployment, Production Container]
---

# Docker Deployment & Single-Process Architecture

Milimo Music provides a turnkey, multi-stage **production container deployment** strategy
that packages the entire platform—the Python 3.11 backend, DSP audio tools (`ffmpeg`, `libsndfile1`),
and compiled React 19 SPA—into a single, self-contained container image (`milimomusic:latest`).

## 1. Multi-Stage Dockerfile Architecture

The build process is codified in [`Dockerfile`](../../Dockerfile):

### Stage 1: Frontend SPA Compilation (`node:20-alpine`)
- Sets working directory to `/app/frontend`.
- Installs dependencies with `npm install` and compiles the production client-side bundle with `npm run build` (`tsc -b && vite build`).
- Emits optimized static production assets (`index.html`, JavaScript chunks, CSS) into `/app/frontend/dist`.

### Stage 2: Production Python & DSP Audio Runtime (`python:3.11-slim`)
- **System Audio Dependencies**: Installs `ffmpeg`, `libsndfile1`, `git`, `curl`, and `build-essential`.
- **Python Dependencies**: Upgrades `pip` and installs the complete backend requirements from `backend/requirements.txt`.
- **Code & Asset Injection**: Copies `backend`, `muscriptor`, `heartlib`, and compiled frontend assets from Stage 1 into `/app/frontend/dist`.
- **Single-Process Web Serving**: The FastAPI application in `backend/app/main.py` detects `/app/frontend/dist` and mounts it at the root with a client-side routing fallback handler:
  - Non-API routes (`/`, `/tracks`, `/arrange`, `/studio`) serve `dist/index.html`.
  - Static bundles (`/assets/*`) are served directly.
  - API routes (`/jobs`, `/models`, `/voice`, etc.) and static audio mounts (`/audio/*`, `/covers/*`) are handled directly by Uvicorn.
- **Port & Health Check**: Exposes port `8000` with an active Docker health check (`curl -f http://localhost:8000/health || exit 1`).

---

## 2. Docker Compose Configurations

Milimo Music provides dual Docker Compose profiles to support both NVIDIA GPU accelerated environments and CPU / standard runtimes:

### A. GPU Profile (`docker-compose.yml`)
Configured for Linux workstations and cloud VMs with NVIDIA GPUs:
```yaml
services:
  milimo:
    build:
      context: .
      dockerfile: Dockerfile
    image: milimomusic:latest
    container_name: milimo-music
    ports:
      - "8000:8000"
    volumes:
      - milimo-data:/app/data
      - milimo-audio:/app/generated_audio
      - milimo-hf-cache:/root/.cache/huggingface
    environment:
      - MILIMO_IN_DOCKER=1
      - HOST=0.0.0.0
      - PORT=8000
    extra_hosts:
      - "host.docker.internal:host-gateway"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

### B. CPU & Standard Profile (`docker-compose.cpu.yml`)
Configured for systems without NVIDIA Container Toolkit (e.g. CPU-only servers, Apple Silicon Macs running Docker Desktop, or standard CI runners):
- Mirrors `docker-compose.yml` but omits the `deploy.resources.reservations` NVIDIA block.

---

## 3. Persistent Volumes & Data Isolation

Containers are stateless; all generated artifacts, training data, and downloaded model weights are isolated in three named Docker volumes:

| Volume Name | Container Path | Purpose |
|---|---|---|
| `milimo-data` | `/app/data` | SQLite database (`database.db`), voice profiles (`profiles.json`), custom presets, artist lore |
| `milimo-audio` | `/app/generated_audio` | Rendered master tracks, stems (`stems/`), converted vocals, and cover images (`data/covers/`) |
| `milimo-hf-cache` | `/root/.cache/huggingface` | Hugging Face model snapshots (MiniMax Music 3, FLUX, HuBERT, RMVPE weights) |

---

## 4. Host Gateway & Local LLM Networking

Containers include:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```
This enables the container to communicate directly with local LLMs or services running on the host machine:
- **Ollama**: Configure endpoint as `http://host.docker.internal:11434`
- **LM Studio**: Configure endpoint as `http://host.docker.internal:1234/v1`
- **OMLX Server**: Configure endpoint as `http://host.docker.internal:8787/v1`

---

## 5. One-Click Launcher (`docker-start.sh`)

Milimo Music includes an intelligent bash launcher script ([`docker-start.sh`](../../docker-start.sh)):
1. Checks if `docker` and Docker daemon are running.
2. Probes for NVIDIA GPU hardware and the NVIDIA container runtime:
   - If `nvidia-smi` and NVIDIA Docker runtime are detected, uses `docker-compose.yml`.
   - Otherwise, gracefully falls back to `docker-compose.cpu.yml`.
3. Runs `docker compose -f <file> up -d --build`.
4. Polls `http://localhost:8000/health` with a 90-second timeout until the server is ready, then prints direct access and log URLs.

---

## 6. Related Pages

- [Backend & API](backend-api.md) · [Model Manager](model-manager.md)
- [Voice Studio (SVC)](voice-service.md) · [System Architecture](../architecture.md)
