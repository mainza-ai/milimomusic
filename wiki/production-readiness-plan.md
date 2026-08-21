---
title: Production Readiness — Implementation Plan
type: overview
tags: [plan, production, security, reliability, packaging]
created: 2026-08-21
updated: 2026-08-21
sources: [production-readiness-audit.md]
aliases: [prod plan, remediation plan]
---

# Production Readiness — Implementation Plan

Remediates [production-readiness-audit](production-readiness-audit.md). Nothing cut,
nothing stubbed.

## Locked decisions (owner, 2026-08-21)

- **Model:** open-source self-host product (ComfyUI/Ollama-style). SQLite stays;
  localhost-default bind + optional `MILIMO_AUTH_TOKEN`; no Redis/S3; queue interface
  allows later arq/Redis swap; packaging is first-class.
- **Nothing cut:** RVC voice conversion + Matchering mastering get real implementations.
- **HeartMuLa:** kept for now (decision pending; dropping later = delete provider,
  gate `heartlib/` out of installs).
- **Non-goals:** multi-user tenancy/SaaS, mobile DAW.

## Phases

| Phase | Scope | Est. | Status |
|---|---|---|---|
| 0 Secrets | Rotate keys (OWNER); history purge approval; redact `/config/llm`; stop persisting env keys; remove client key | 0.5d | Code parts ✅; rotation/purge OWNER |
| 1 Security | Optional bearer auth, localhost default, CORS allowlist, upload sanitization+caps+magic bytes, UUID validation, rate limits, global exception handler + error envelope | 2-3d | OPEN |
| 2 Make it real | Fix B1 rename; real RVC v2 SVC (consent SHA-256); real Matchering (`mastered_path`); real model download manager (streamed HF + SSE progress + resume/cancel); honest platform matrix; real `extend()`; dev-flag-only synth fallback | 1.5-2w | OPEN |
| 3 Job engine | Durable in-process queue (GPU lane N=1), step timeouts, backoff retries, cooperative cancellation, boot reconciliation, queued transcribe upload, unified gpu_lock, Alembic baseline, complete cascade delete + TTL janitor, SQLite pragmas + backup, lockfiles + declared demucs + pinned submodule | 1w | OPEN |
| 4 Frontend resilience | ErrorBoundary, safe-parse, TanStack Query, debounced search ✅(HistoryFeed), SSE fix ✅(refs+scope stack), axios instance, Toasts, runtime API URL | 1w | PARTIAL (core landed via ui-ux wave) |
| 5 Performance | Code splitting, virtualization, peaks-based library rows ✅(TrackRowPlayer+server peaks), canvas piano roll, audio-engine store split, N+1 fixes | 3-5d | PARTIAL |
| 6 Testing & CI | Hermetic pytest, pipeline/SSE/job tests, Vitest, Playwright, GitHub Actions, gitleaks | 1w | OPEN |
| 7 Packaging & ops | Dockerfiles+compose, Caddy/nginx SSE-safe, `make install`, JSON logs+request IDs, deep /health ✅(pill)+readyz, HOST/PORT wiring, runbook, versioning | 3-5d | PARTIAL (/health pill live) |

## Quick-deliverable batch status

Landed 2026-08-21: B1-style renames n/a yet; A2 redaction; A6 stop-persist; B8-class
clobber guards pending; boot reconciliation OPEN; gpu_lock unification OPEN; cascade
delete OPEN; SQLite pragmas OPEN; HOST/PORT OPEN; Dockerfile/compose OPEN; CI OPEN.
Frontend batch: see [ui-ux-plan](ui-ux-plan.md) progress log (largely delivered).

Blocked on owner: key rotation · history purge approval · RVC weight-download consent.

## Related
[Audit](production-readiness-audit.md) · [UI/UX Plan](ui-ux-plan.md) · [Agent Foundation](concepts/agent-foundation.md) · [Index](index.md)
