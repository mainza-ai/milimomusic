---
title: Production Readiness Audit (2026-08-21)
type: overview
tags: [audit, production, security, reliability, ops]
created: 2026-08-21
updated: 2026-08-21
sources: []
aliases: [prod-readiness audit, gaps report v2]
---

# Production Readiness Audit (2026-08-21)

Full-codebase audit against the production-not-demo bar. Companion plan:
[production-readiness-plan](production-readiness-plan.md). Extends the raw
`devs/PRODUCTION_GAPS_REPORT.md` (2026-08-20).

**Verdict:** feature-rich and architecturally sound, demo-grade hardening at audit time.
Progress notes below updated as fixes land (see also ui-ux-plan for the frontend wave).

## A. CRITICAL — Security

| # | Finding | Location | Status |
|---|---------|----------|--------|
| A1 | Live API keys in git history (`git show add8d45^:backend/llm_config.json`) | git history | OPEN — rotation is owner action |
| A2 | `GET /config/llm` returned every API key unauthenticated | `main.py:583`→`config_manager.py` | FIXED 2026-08-21 — masked `has_key` payloads |
| A3 | Zero auth + `0.0.0.0` bind + CORS `*` w/ credentials | `main.py:139-145,1803` | OPEN (Phase 1) |
| A4 | Path-traversal file writes (transcribe/upload; dataset upload; voice-convert id) | `main.py:255,1058,1521`; `fine_tuning_service.py:224` | OPEN (Phase 1) |
| A5 | `VITE_OPENCODE_API_KEY` compiled into client bundle | `LLMSettingsModal.tsx` | ✅ RESOLVED by owner 2026-08-21 |
| A6 | Env-overlay keys persisted into `llm_config.json` on save | `config_manager.py:158` | FIXED — save scrubs env-equal keys |
| A7 | SVG cover uploads served from static mount (stored-XSS) | `main.py:713` | OPEN |

## B. CRITICAL — Broken / fake functionality

| # | Finding | Status |
|---|---------|--------|
| B1 | voice-convert called nonexistent `convert_voice` → guaranteed 500 (`main.py:1524`) | OPEN |
| B2 | Model Manager "Download" = `setTimeout` fake | OPEN (Phase 2) |
| B3 | Matchering mastering stub (copies file / 0-byte fallback, fabricated score) | OPEN (Phase 2) |
| B4 | Voice cloning copies stem unchanged; "consent" is a boolean | OPEN (Phase 2) |
| B5 | README CPU/CUDA claim misleading; only fallback = procedural oscillator | OPEN (Phase 2) |
| B6 | Silent LLM failures returned canned HTTP-200 content | OPEN (Phase 2) |
| B7 | `extend()` ignores parent context; `repair_segment()` no-op | OPEN (Phase 2) |
| B8 | Mastering overwrote `job.audio_path` | OPEN |

## C. Reliability & architecture
No durable job system (unmanaged BackgroundTasks; fire-and-forget inpaint task);
`gpu_lock` only used by inpainting; no crash reconciliation on boot; no step timeouts/
retries; minutes-long ML inline in HTTP request; dual ad-hoc migrations (no Alembic);
SSE broadcast-only/no heartbeat/no replay/breaks multi-worker; unbounded disk growth +
incomplete cascade delete; non-reproducible deps (floats, unpinned pydantic-ai,
undeclared demucs, floating submodule); HOST/PORT env ignored.

## D. Frontend
(Condensed here; full detail in [ui-ux-audit](ui-ux-audit.md).) No ErrorBoundary +
unguarded render-time JSON.parse white-screens; SSE stale closures reset filters;
no data layer (raw axios, no cache/abort/debounce); alert()-driven error UX with
silent catches; 60fps context re-render storm; no virtualization (WaveSurfer per
history card); 790KB single bundle; piano roll was insert/delete-only; nothing
survived refresh.

## E. Testing / CI / Ops
Zero CI; tests not hermetic (gitignored fixture, cwd assumptions, live dirs);
zero frontend tests/e2e; no deployment artifacts; shallow `/health`; no SQLite
backup story; pagination gaps; no API versioning.

## Remediation status snapshot (2026-08-21)
- Security code-side quick wins: A2/A6 fixed; A5 owner-fixed. Phase 1 remainder open.
- Frontend resilience/perf/editor wave largely landed same-day via ui-ux plan
  (ErrorBoundary-class guards, memoized parses/layers, real meters/waveforms, undo/redo
  editor, session persistence, deep links, peaks-based library rows).
- Phases 2/3/6/7 remain the active backlog.

## Related
[Production Readiness Plan](production-readiness-plan.md) · [UI/UX Audit](ui-ux-audit.md) · [Roadmap](roadmap.md)
