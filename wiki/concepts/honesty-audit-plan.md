---
title: Honesty Audit Remediation Plan
type: concept
tags: [plan, audit, reliability]
created: 2026-08-25
---

# Remediation Plan (post full audit)

## Phase 1 — Truth before features (½ day)
1. Root-cause Streetlight Chant failure (ledger row + backend logs; classify:
   LLM? OOM? pipeline step?) — fix or file precisely.
2. Playwright browser E2E: produce→banner→approve→artifacts visible (first-ever
   frontend verification; runs headless in CI later).
3. Cross-process instance-lock test (two real processes).
4. Mid-generation resume test (kill -9 mid-AR → boot → resume completes track).

## Phase 2 — Guardrails (1 day)
5. CI on push: pytest + tsc + build + Playwright E2E (blocks regressions forever).
6. Alembic baseline migration; retire ad-hoc PRAGMA/ALTER block.
7. Regression test: failed-track propagation (reproduces Streetlight class).

## Phase 3 — Make claimed things real (2–3 days)
8. RVC reality-check: download weights w/ owner consent OR descope honestly.
9. HeartMuLa decision executed (drop = remove provider stubs + boot warning).
10. Training Studio E2E once, honestly documented.
11. Album finishing: per-release mastering pass + simple cover art.

## Phase 4 — Performance truth (1 day)
12. Rolling measured RTF persisted per machine; banner ETA uses it.
13. Quantization benchmark 8bit vs bf16 (speed + listening sample).

## Phase 5 — Product surface (ongoing)
14. R3 polish ledger items (role enum, Toasts, deep-links).
15. Multi-user auth decision; SSE replay design.

Exit criteria per phase listed in review; nothing marked done without evidence link.
