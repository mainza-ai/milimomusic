---
title: LLM Service & Providers
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/readme.md]
tags: [llm, ollama, providers, lyrics, title, opencode, omlx]
aliases: [LLM Service, Ollama integration]
---

# LLM Service & Providers

Milimo Music uses a local/remote **LLM Service** for creative text tasks that support
generation. It is provider-agnostic and configurable in-app.

## What the LLM powers
- **Lyrics Generation** — writes structured lyrics (Verse, Chorus, Bridge) from a topic.
- **Prompt Enhancement** — expands simple concepts into detailed musical descriptors.
- **Auto-Titling** — generates creative titles from song content.
- **Inspiration Mode** — brainstorms unique song concepts and style combinations.

## Supported providers
- **OpenCode Go API** — cloud (OpenAI-compatible `https://opencode.ai/zen/go/v1`).
- **OMLX** — **local Apple Silicon** server at `http://localhost:8787/v1` (the Qwen MLX
  vision/chat server).
- **Ollama** (Local) — via `ollama serve` (e.g. Llama 3.2).
- **OpenAI** — GPT models (API key).
- **Google Gemini** — Gemini models (API key).
- **OpenRouter** — Claude, Mistral, Llama via a unified API (API key).
- **DeepSeek** — DeepSeek API.
- **LM Studio** — local inference servers compatible with the OpenAI API.

> [!NOTE] The codebase hardcodes default API keys/base URLs for `opencode` and `omlx` in
> `llm_service.py` (e.g. a default OpenCode key and `api_key="omlx"`). Treat those as
> local-development defaults, not secrets to ship.

## Configuration
Gear icon → provider tab → API key/Base URL → "Save & Set Active"; the app auto-fetches
available models. Persisted in `backend/llm_config.json` via `ConfigManager` (config schema
includes `opencode` and `omlx` in `LLMConfigUpdate`).

## Backend
- `backend/app/services/llm_service.py` — provider clients + orchestration.
- `config_manager.py` — config persistence.
- Integration to the Co-Writer and title/prompt generation flows.

## Related pages
- [AI Co-Writer](ai-cowriter.md) | [Backend & API](backend-api.md)
- [README source](../sources/readme.md)
