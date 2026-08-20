---
title: Backend & API
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/readme.md, sources/training-studio-guide.md, sources/inpainting-debug.md]
tags: [backend, fastapi, sqlmodel, api, sse, sqlite]
aliases: [Backend, FastAPI backend]
---

# Backend & API

The **backend** is Milimo Music's Python API layer — **FastAPI** with **SQLModel** and
**SQLite** (now `backend/database.db`). It runs on `http://localhost:8000` (`uvicorn app.main:app`).

## Data layer (`app/models.py`)
- **`Job`** — the central record, now carrying v2 assets:
  - Identity/status: id (UUID), status, title, prompt, lyrics, tags, seed, duration, `is_favorite`.
  - Generation/provider: `model_provider` (default `minimax_music3`), `llm_model`,
    `parent_job_id`, `temperature`, `cfg_scale`, `topk`.
  - **v2 multitrack assets**: `midi_path`, `musicxml_path`, `notes_json`, `stems_json`,
    `beat_grid_json`, `timed_lyrics_json`, `structured_caption_json`.
  - **Project association**: `project_id` (indexed).
- **`Project`** — project folders with `name`, `description`, `tags`, `bpm` (default 120),
  `key_signature` (default "C Major"), `color`, `icon`.
- Request models: `GenerationRequest` (adds `model_provider`, `structured_caption`,
  `voice_profile_id`, `project_id`), `LyricsRequest`, `LyricsChatRequest`,
  `EnhancePromptRequest`, `InspirationRequest`, `ProviderConfig`/`LLMConfigUpdate`
  (adds **`opencode`** and **`omlx`** providers), `VoiceProfileCreate`, `MasteringRequest`.

## Core packages (`app/`)
- `providers/` — [generation-provider](generation-provider.md) abstraction + registry
  (`base.py`, `registry.py`, `minimax_provider.py`, `heartmula_provider.py`).
- `orchestration/pipeline.py` — the [orchestration pipeline](../concepts/generation-pipeline.md)
  (generate → stems → voice conversion → transcription + lyric sync).
- `transcription/` — [stem-separator](stem-separator.md), [muscriptor](muscriptor.md),
  [mastering](matchering-mastering.md), [karaoke/lyric-sync](karaoke-lyricsync.md).
- `services/` — `music_service.py`, `llm_service.py`, `lyrics_graph.py`/`lyrics_engine.py`/
  `lyrics_utils.py`/`lyrics_schemas.py` ([AI Co-Writer](ai-cowriter.md)),
  `style_registry.py`, `inpainting_service.py`, `fine_tuning_service.py` + `training/*`
  ([Training Studio](training-studio.md)), `config_manager.py`,
  `model_manager.py` ([Model Manager](model-manager.md)),
  `voice_service.py` ([Voice Studio](voice-service.md)).

## Model management endpoints (`/models/*`)
- `GET /models/tree` — the [model tree](model-manager.md) (`ModelVariant[]`).
- `GET /models/capabilities` — provider capability manifests.
- `GET /models/hardware` — detected `HardwareProfile`.
- `GET /models/check/{model_id}` — missing-dependency check (download-on-demand UX).
- `POST /models/active/{provider_id}` — set the active generation provider.

## Voice endpoints (`/voice/*`)
- `GET/POST /voice/profiles`, `DELETE /voice/profiles/{id}` —
  see [Voice Studio (SVC)](voice-service.md).

## Transcription / workspace endpoints
- `POST /transcribe/upload` — upload user audio, run MuScriptor → creates a `Job`.
- `GET /transcribe/export/{job_id}/{format}` — export `midi`, `musicxml`, `ableton`, `lrc`, `srt`.
- `POST /mastering/match/{job_id}` — [Matchering mastering](matchering-mastering.md).
- `POST /workspace/{job_id}/notes` — persist piano-roll note edits.

## Data / generation endpoints
- `POST /generate/music` (accepts `model_provider`, `voice_profile_id`, `structured_caption`,
  `project_id`), `POST /generate/lyrics`, `POST /generate/lyrics-chat`,
  `POST /generate/enhance_prompt`, `POST /generate/evaluate_inspiration`, `POST /generate/styles`.
- `POST /jobs/{id}/inpaint` — [Repair Segment](inpainting.md).
- `GET /history`, `POST /jobs/{id}/favorite`, `PATCH /jobs/{id}`, `DELETE /jobs/{id}`,
  `POST /jobs/{id}/cancel`, `GET /download_track/{id}`.
- `POST /projects`, `GET /projects`, `GET/PUT/DELETE /projects/{id}` (project folders).
- Training endpoints documented in [Training Studio](training-studio.md).

## Communication
HTTP + JSON; **SSE** (`/events`) for real-time job status + progress; multipart uploads
for audio (datasets, voice datasets, `/transcribe/upload`).

## Related pages
- [Architecture](../architecture.md) | [Generation provider](generation-provider.md)
- [Orchestration pipeline](../concepts/generation-pipeline.md) | [LLM Service](llm-service.md)
- [AI Co-Writer](ai-cowriter.md) | [Training Studio](training-studio.md) | [Repair Segment](inpainting.md)
