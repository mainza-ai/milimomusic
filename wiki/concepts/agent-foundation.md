---
title: AI Agent Foundation — LLM Layer Investigation
type: concept
tags: [agents, llm, architecture, foundation, pydantic-graph]
created: 2026-08-21
updated: 2026-08-21
sources: []
aliases: [agent runtime, AgentRuntime, llm audit]
---

# AI Agent Foundation — LLM Layer Investigation

Investigation of the LLM configuration layer as the foundation for AI agents
(songwriter, world builder, experiencer/critic…), feeding the larger vision in
[Artist Profiles & Album Agents](artist-profiles-vision.md). All findings verified
against source (`llm_service.py` 1258 lines, `config_manager.py`, `lyrics_graph.py`,
surface routes, frontend config UI).

## Executive summary

The layer is a **solid single-shot completion fabric**: 9 providers behind only 3
adapters, hot-swappable config with secret hygiene, pydantic structured outputs,
honest fallback flags, and a **working multi-agent graph already in production**
(the Co-Writer). But it has **none of the four pillars agents require**:
no message/persona API, no tools, no memory, no streaming — and async routes call
blocking SDKs on the event loop.

**Verdict: ~70% agent-ready.** Recommended path: a thin **AgentRuntime** over the
existing `LLMProvider` ABC (Option C below) — not a rewrite.

## Architecture today

```
.env  >  llm_config.json (CWD-relative → backend/)  >  DEFAULT_CONFIG
        ↓ ConfigManager singleton (env applied at read time; hot-switching)
LLMService._get_provider()   ← if/elif factory; fresh instance per call;
                                per-call override (used by failover loops)
        ↓
LLMProvider ABC: generate_text | generate_json | generate_structured(pydantic) | get_models
        ↓ adapters: OllamaProvider (native REST) · OpenAIProvider (openai SDK ×7
          providers via base_url) · GeminiProvider (google-genai)
        ↓
Task methods (all-static): generate_lyrics · chat_with_lyrics · produce_full_track ·
rewrite_caption · enhance_prompt · generate_title · generate_inspiration …
        ↓ ~15 bespoke HTTP routes + Composer / LLMSettingsModal frontend
```

## Provider matrix

| Provider | Adapter | Transport | Notes |
|---|---|---|---|
| nvidia NIM | OpenAIProvider | integrate.api.nvidia.com | timeout 45s, SDK retries=2 |
| deepseek | OpenAIProvider | api.deepseek.com (**hardcoded; `_BASE_URL` env ignored**) | |
| openrouter | OpenAIProvider | openrouter.ai/api/v1 (**hardcoded**) | 400-handling log-only |
| opencode | OpenAIProvider | opencode.ai/zen/go/v1 | |
| openai | OpenAIProvider | official only | **only true schema enforcement** (`beta.parse`) |
| lmstudio / omlx | OpenAIProvider | localhost :1234/:8787 | sentinel keys |
| ollama | OllamaProvider | native REST `/api/generate` | native JSON mode ✓ |
| gemini | GeminiProvider | google-genai SDK | **no timeout set** |

Structured output = JSON-mode + brace-slice + parse everywhere except official
OpenAI. Malformed-JSON recovery lives at graph level (≤3 retries, error-feedback).

## Assets to keep (do not redo)

1. `LLMProvider` ABC + `_get_provider(override)` — provider-agnostic injection works.
2. ConfigManager hygiene: env>file>defaults, blank-key merge protection, secret
   scrubbing on save, masked client payloads (`has_key`). Per-agent model profiles ride on it.
3. **Co-Writer graph** (`lyrics_graph.py`): Coordinator→Lyricist→StructureGuard,
   typed `SongState`, error-feedback retry, `MaxRetriesExceededError`. **This is the
   agent template.**
4. Forgiving schemas (`lyrics_schemas.py`: aliases, extra-ignore, coercions).
5. Honest-degradation contract: `rewritten/fallback_reason`, `producer_skipped`.
6. `_strip_thinking` sanitizer (R1/Qwen leakage) — provider-independent.
7. Grounding systems: StyleRegistry prompt vocabulary + hallucination filtering;
   vendored caption-template library. These are proto-tools.
8. SSE EventManager accepts arbitrary named events → agents emit `agent_progress`
   with zero transport changes.
9. Run-lifecycle precedent: `/generate/music` returns instantly, BackgroundTasks,
   cooperative cancel registry (`MusicService.active_jobs`).

## Gaps blocking agents (prioritized)

| # | Gap | Impact |
|---|-----|--------|
| G1 | No message API — flat `prompt:str`; system role absent; `chat_history` params exist and are **dead code** | Personas hand-roll prose in one user turn |
| G2 | Blocking sync SDK calls inside async coroutines (one lone `to_thread`) | Long chains freeze the server for everyone |
| G3 | Zero tool/function-calling scaffolding | World-builder needs retrieval/world-state ops |
| G4 | No memory: sessions persist messages but nothing feeds them back; attachments decorative | Multi-turn co-writing impossible |
| G5 | Zero token/cost accounting | N agents burn spend invisibly |
| G6 | Failover triplicated folklore (3 diverging hardcoded chains, dead conditions) | Needs one ResiliencePolicy |
| G7 | Untyped errors (auth/quota/timeout indistinguishable) | No smart retry/routing |
| G8 | Prompt sprawl (thousands of inline lines, no registry/versioning) | Adding an agent = editing core service |
| G9 | No streaming anywhere; producer HUD fabricated client-side | Agent UX needs live progress |
| G10 | Config footguns: CWD-relative file, default-provider mismatch (nvidia vs ollama), fake model lists on failure, retired gemini-1.0 fallback | Silent wrong-backend calls |
| G11 | Unauthenticated spend-capable surface (see production-readiness plan Phase 1) | Agents multiply exposure |

## Framework decision

| Option | Verdict |
|---|---|
| A. Adopt `pydantic-ai` fully | Already in requirements (never imported); typed agents/tools/streaming/usage free; migration risk for working Co-Writer graph |
| B. Extend homegrown pydantic-graph | Zero deps, full ownership; rebuild tools/streaming/usage ourselves — the redo-later trap |
| **C. Thin AgentRuntime over existing ABC (recommended)** | ~300-line runtime formalizing messages/tools/memory/policy; pydantic-graph stays as orchestrator; pydantic-ai can slot in later behind the same interface |

## Proposed layout

```
backend/app/agents/
  runtime/ base.py(AgentDefinition{name,persona,tools,schemas,model_profile,max_steps})
           context.py(RunContext{run_id,session_id?,project_id?,history,attachments})
           executor.py(async run_agent; to_thread discipline; SSE agent_progress keyed
                       by run_id; cancel_event)
           policy.py(single ResiliencePolicy: failover order, timeouts, backoff,
                     circuit breaker) usage.py(token capture→attribution/budgets)
           errors.py(AuthError|QuotaError|TimeoutError|BadModel|ParseError)
           registry.py(AGENTS name→definition)
  songwriter/  (extraction of existing lyrics_graph nodes)
  world_builder/ (lore graph + tools[world_state.get/set, style_lookup])
tools/ base.py(Tool protocol: name, pydantic input, run(ctx,input))
       style_lookup.py(wraps StyleRegistry) world_state.py(SQLite-backed store)
Surface: POST /agents/{name}/run → {run_id}; GET runs/{id}; POST cancel;
         reuse EventManager SSE. New tables: agent_runs (+ artist profiles, see vision page).
```

Sequencing: Runtime core (G1/G2/G6/G7) → usage/budgets (G5) → tools+world_state
(G3/G4) → streaming UX (G9) → first agents become thin definitions.

## Related
- [Artist Profiles & Album Agents](artist-profiles-vision.md) — the ultimate vision this serves
- [AI Co-Writer](../entities/ai-cowriter.md) · [Co-Writer graph](co-writer-graph.md) · [LLM Service](../entities/llm-service.md)
