---
title: Memory Management Audit 2026-11
type: overview
tags: [memory, audit, mlx, mps, unified-memory, agents, sqlite, performance]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [MemoryAudit, MemoryReport]
---

# Memory Management Audit — 2026-11

The companion to [Performance Audit 2026-11](performance-audit-2026-11.md). Performance
asks *how fast*; this asks *what the process is holding and whether it can give it back*.
On a unified-memory Mac those are the same question viewed from two ends.

---

## 1. What the app was doing before this pass

| Measurement (this machine) | Value |
|---|---|
| Peak RSS after `import app.main` | **288 MB** |
| Endpoints able to report memory | **none** (`/health` had disk, DB, queue — no memory) |
| `clear_cache()` / `empty_cache()` / `gc.collect()` call sites | **33**, scattered, no policy |
| MLX cache limit | **unset** → MLX default = 1.5 × recommended working set = **161 GiB on a 128 GiB box** |
| `mx.get_cache_limit` probe in our own telemetry | **does not exist** in MLX 0.32 — silently always absent |
| MPS high-watermark | **1.7 × recommended** = **182 GiB cap on a 128 GiB machine** → swaps instead of raising OOM |
| Wan pipeline cache | unbounded by model — switching 14B ↔ 1.3B kept **both** resident |
| Agent conversation history | client-supplied, unbounded, no budget |
| SQLite WAL | `journal_mode=WAL` set, no `journal_size_limit` |

## 2. Findings

### M1 — RSS is the wrong number, and we reported it as "footprint" (fixed)

Measured: with 8 GiB live in MLX, **RSS reads 30 MB while the kernel footprint reads
8 370 MB**. Metal buffers are IOSurface-backed and never enter `resident_size`, so RSS is
blind to accelerator memory — the one thing that can actually kill the machine.

`core/memory.py` now reads `proc_pid_rusage(RUSAGE_INFO_V4)` directly (psutil cannot
report footprint on Darwin) and reports **`footprint_mb`** next to `rss_mb`.

### M2 — The MLX cache could grow past physical RAM (fixed)

MLX keeps freed buffers for reuse; its default cap is the memory limit, which defaults
to 1.5 × the recommended working set. On this box that is **161 GiB**. After unloading a
large model the buffers sit in cache and nothing reports them.

Now: `MILIMO_MLX_CACHE_LIMIT_MB` (default `min(12 GiB, 10 % RAM)`; `0` disables the
cache; `off` keeps MLX's default), applied once at boot.

### M3 — PyTorch MPS would swap rather than raise (fixed)

Default high watermark 1.7 × 107.5 GiB = **182 GiB** on a 128 GiB machine. The cap is
unreachable, so MPS never raises; macOS pages instead, and the whole desktop stalls
during a video render.

The render path now calls `torch.mps.set_per_process_memory_fraction(0.6)`
(`MILIMO_MPS_MEMORY_FRACTION`) before the first tensor — a runtime cap, no relaunch.

### M4 — Deprecated MLX API printed to stderr on every flush (fixed)

`mx.metal.clear_cache()` was used in two places. MLX 0.32 deprecates `mx.metal.*` and
warns **from C++**, so `warnings` filters cannot see it and the message appeared on
every flush. Both call sites use `mx.clear_cache()` now; a test greps for regressions.

### M5 — `flush_memory()` cleared pools before collecting cycles (fixed)

The order was `gc.collect()` → clear pools. Cycles still referenced arrays at that
moment, so the pools had nothing to release. It is now documented and ordered
refs → collect → pools, matching `memory.release()`.

### M6 — The Wan pipeline cache was unbounded by model (fixed)

`_WAN_PIPELINE_CACHE` keyed i2v/t2v per model with no eviction: rendering 14B then 1.3B
kept both alive (multi-GB each). It is now bounded (`MILIMO_WAN_PIPELINE_CACHE`, default
**1**), evicting oldest-first with hooks removed and the pipeline moved off the
accelerator before the reference drops.

### M7 — No memory telemetry, no single release path (fixed)

`GET /system/memory` reports process (RSS **and** footprint), system (available, swap),
MLX (active / cache / **live = active+cache** / peak) and torch (allocated / driver).
`POST /system/memory/release` is the one place that returns pool memory, reporting the
footprint before/after — because RSS will not move.

### M8 — Agent history had no budget (fixed)

`chat_history` was accepted verbatim. A long session grows until the prompt no longer
fits — or the provider silently truncates from the top, dropping the agent's
instructions mid-conversation.

`core/agent_memory.py` enforces a **character + message budget**, always preserves the
leading system/instruction turn, and returns a `report` (`kept`, `dropped`,
`truncated`) so the UI can say turns were dropped instead of pretending.

### M9 — SQLite WAL could grow without bound (fixed)

A checkpoint makes WAL space reusable but does not shrink the file; a single long-lived
reader blocks checkpoints entirely. `PRAGMA journal_size_limit=67108864` is now set on
both the main engine and the task queue.

### M10 — `gc.collect()` was on hot paths (fixed)

Cycle collection walks the whole heap. Measured here: adding it to the release path made
eight tests take 16 s, and the full suite 265 s. It is now **opt-in**
(`MILIMO_MEMORY_GC=1`) and used only on model unload. Tensors are refcounted; cycles and
framework pools are different problems.

## 3. What shipped

| Change | Evidence |
|---|---|
| `core/memory.py` — footprint probe, MLX/MPS caps, one `release()` | `/system/memory` reports footprint ≈ RSS-independent; MPS fraction applied |
| `core/agent_memory.py` — budgeted history | System prompt survives truncation; report returned per request |
| Bounded Wan pipeline cache | Evicts to limit, releases hooks, moves off device |
| `journal_size_limit` on both connections | WAL capped at 64 MiB |
| Modern `mx.clear_cache()`, ordered flush | No deprecated API left (guarded by a test) |
| `GET /system/memory`, `POST /system/memory/release` | Live telemetry + one release path |

A real bug surfaced while building the probe: the first `rusage_info_v4` implementation
declared only the leading fields in a `ctypes.Structure`, so the kernel wrote past the
end of it and a later torch call hung forever. The probe now uses an oversized buffer
and reads documented offsets.

## 4. What to do next (Phase 10)

Ranked by value for a single-user local server:

1. **WAL checkpoint job** — `wal_checkpoint(TRUNCATE)` on a timer, and make sure no
   `Session` is held across an LLM call or an FFmpeg subprocess (the profile also
   blocks page reclaim).
2. **Memory ring buffer + slope alarm** — sample footprint/MLX/MPS every minute; a leak
   is a positive slope *at idle*. Without this, finding M-class bugs stays manual.
3. **Agent memory: per-agent namespaces + episodic store** — the critic should not
   inherit the songwriter's transcript; distilled facts beat raw turns. Research is
   blunt that for one user, structured notes + SQLite FTS5 beat a vector store.
4. **Observability for process hygiene** — fds, thread count, zombie children in
   `/health`; `-nostdin` and `Popen(...).wait()` audits for FFmpeg.
5. **`reset_peak_memory()` on load** — peak is a lifetime max, so one 80 GB model makes
   every later reading meaningless.
6. **Model budget guard** — compare the requested model size against
   `max_recommended_working_set_size` before loading, and refuse with a stated reason
   instead of swapping.

## 5. Uncertainty

- Footprint, MLX and MPS figures were measured on this machine (M3 Max, 128 GiB, MLX
  0.32, torch 2.13); the percentages in Apple's working-set heuristic vary by SKU — read
  the property, not a rule of thumb.
- Whether `mx.array` is weak-referenceable varies by version; `WeakValueDictionary` is
  therefore not used for model caches — bounded caches are.
- The agent-memory literature (compaction, summarisation) is measured on *long*
  sessions; for an eight-turn songwriter conversation none of it matters. Our budget is a
  guardrail, not the product feature.

## Related

[Performance Audit 2026-11](performance-audit-2026-11.md) ·
[Plan — Phase 10](application-remediation-plan-2026-11.md#phase-10--memory-2026-11-18) ·
[Caveat Investigation](caveat-investigation-2026-11.md) ·
[Index](index.md)
