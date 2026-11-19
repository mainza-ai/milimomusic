---
title: Performance Audit 2026-11
type: overview
tags: [performance, audit, latency, sqlite, ffmpeg, fastapi, react, mlx, benchmarking]
created: 2026-11-18
updated: 2026-11-18
sources: []
aliases: [PerfAudit, PerformanceReport]
---

# Performance Audit — 2026-11

Milimo runs on one machine, for one user, with an ML engine on the same silicon as the
web server. That inverts a lot of web-performance canon: **bandwidth is free, CPU and
GPU seconds are not**. This audit measures the current build, separates real costs from
ritual, and records what shipped.

Companions: [Caveat Investigation](caveat-investigation-2026-11.md) ·
[Video Capability Landscape](video-capability-landscape-2026-11.md) ·
[Plan — Phase 9](application-remediation-plan-2026-11.md#phase-9--performance-2026-11-18).

---

## 1. Measured baseline (this machine, 2026-11-18)

| Measurement | Value | How |
|---|---|---|
| Cold import of `app.main` | **4.64 s** | `/usr/bin/time python -c "import app.main"` |
| … of which `app.main` body | **3.34 s** | `-X importtime` |
| … of which LLM SDK clients (`llm_service`, `google.genai`, `openai`, `anthropic`) | **~1.28 s + 0.58 + 0.46 + 0.24** | same |
| … of which `torch` | **0.81 s** | same |
| `GET /models/tree` (cold process) | **131 ms**, 14.2 KB | TestClient |
| `GET /videos/models` (cold) | **982 ms**, 1.8 KB | TestClient |
| `GET /system/licences` | 31 ms, 5.5 KB | TestClient |
| JSON compressibility (gzip -6) | **15–41 % of original** | measured per payload |
| `model_manager.get_model_tree()` | **1 103 ms cold**, 6 ms warm | VFS-cold scan of 32 models |
| FFmpeg 180 s of 1080p24 → H.264 | **libx264 veryfast CRF 18: 5.8 s / 5.7 MB** · **h264_videotoolbox 8 Mbps: 11.1 s / 15 MB** | `time ffmpeg` |
| FFmpeg 3 s 1080p | x264 0.11 s · videotoolbox 0.64 s | same |
| Frontend entry chunk | **575 KB** (163 KB gz), views already `React.lazy` | `vite build` |
| Polling sites | **15** `setInterval` timers, 1–3 s | grep |
| Backend endpoints | 188 routes, 167 sync `def` | grep |

## 2. Findings

### P1 — The API was uncompressed, and *that is correct*

No `Content-Encoding` anywhere; JSON compresses to 15–41 %. The obvious move is gzip —
**and it is wrong here.** gzip's break-even link speed for a JSON response is roughly
400–860 Mbit/s; a loopback socket is far above that, so compression spends CPU to make
the response arrive *later*. Compression is now **opt-in** (`MILIMO_COMPRESS=1`) for a
LAN/remote client, and streaming paths (`/events`, `/audio/`, `/covers/`) are excluded
because gzipping `text/event-stream` breaks its framing.

### P2 — Cold model-tree scan costs ~1.1 s, and three endpoints call it

`/models/tree`, `/videos/models` and the licence audit each walked the models
directory. A 15 s memo + generation counter removes the repeats; anything that changes
installations (download completion, delete, activation) bumps the generation.

Measured after: `/models/tree` **4 ms**, `/videos/models` **4 ms warm** (one-time
~190 ms on first touch), and a conditional GET is **1.1–1.4 ms**.

### P3 — No conditional GET, so polling re-parsed everything

Thirty-to-sixty polls a minute × a JSON parse × a React re-render, for unchanged data.
Read endpoints now return `ETag` + `Cache-Control: private` and answer `304`. The tag
derives from **input state** (catalogue generation, device, env), never from the
finished body — hashing the body pays the cost the 304 exists to avoid.

### P4 — The hardware encoder is *slower* here, and we preferred it

`detect_hardware_encoder()` selected `h264_videotoolbox` because the encoder exists and
can encode a black frame — never because it is faster. On this machine it is **1.9×
slower** (11.1 s vs 5.8 s for 3 min of 1080p) and produced a **2.6× larger** file at
8 Mbps than x264 CRF 18. The lyric path used it by default.

Research explains the discrepancy rather than excusing it: VideoToolbox's win is
**cores**, not wall time (0.9 vs 6.0 cores on an M1 Pro), so it is the right choice for
parallel/background renders and the wrong one when the user is waiting on a single
render.

### P5 — Our encodes saturate the machine

x264 defaults to ~1.5× cores of threads. During a render the editor and the API
compete for the same silicon. Encode commands now carry `-threads 4` by default
(`MILIMO_FFMPEG_THREADS`, `0` disables).

### P6 — Startup spends ~1.3 s importing LLM clients nobody asked for

`llm_service` + `google.genai` + `openai` + `anthropic` are imported eagerly by
`app.main`. They are used only when a director/treatment/lyric agent runs. Deferring
them is a one-line-per-module change for a measurable slice of a 4.6 s boot.

### P7 — 15 polling timers where one SSE stream exists

The backend already implements SSE with replay (`/events`). The video view still polls
at 1 s. Per the research, an idle SSE stream costs ~26 B/min and zero wakeups, while a
1 s poll costs a request, a DB read, a parse and a render **every second**.

### P8 — No performance measurement in CI or in the dev loop

There is no `hyperfine` script, no `py-spy` recipe, no bundle-size budget, and no
endpoint-latency budget. Every number in this report was gathered ad hoc, which means
the next regression will also be discovered ad hoc.

### P9 — SQLite is mostly right, with two unmeasured knobs

`journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=10000` are set. `cache_size`,
`mmap_size` and `temp_store` are not — but the research is unambiguous that on a
single-writer local database their effect is small and must be measured, not assumed.

### P10 — The frontend is already split; the remaining win is parse, not download

The entry is 575 KB with lazy views and vendor chunks. Over loopback that arrives in
single-digit ms; the cost is ~50–100 ms of parse/execute. The real frontend risks are
high-frequency React state (playhead/scrub) and long lists — not chunk size.

## 2b. What the follow-up pass added (same day)

| Change | Measured effect |
|---|---|
| **Lazy LLM SDK imports** (9.4) — `openai` / `google.genai` / `anthropic` load on first agent call, not at boot | cold import **4.64 s → 3.83 s** (−17 %); `llm_service` leaves the import profile entirely |
| **Measured encoder policy** (9.5) — benchmark both encoders once per machine, cache with a fingerprint, `MILIMO_VIDEO_ENCODER` overrides | on this machine **x264 0.21 s vs VideoToolbox 0.31 s** per second of 1080p — the "hardware" default was 48 % slower |
| **Benchmark harness** (9.9) — `scripts/bench_api.py`, `--write-baseline` / `--compare` (ratio-based, 50 % tolerance) + a report-only CI job | warm reads **1.1–1.4 ms**, 304s **1.0–1.2 ms**; JS bundle trackable alongside |
| **Parallel clip normalisation** (9.10) — bounded pool (`MILIMO_NORMALISE_CONCURRENCY`, default 2), order preserved for the stitcher | 5 clips normalise in ~1/3 of serial wall time in the harness test |

## 2c. SQLite knobs, measured (9.8)

`scripts/bench_sqlite.py` runs 20 000 writes and a point read under each pragma set:

```
config                  writes/s  point read   wal KB
defaults                 1657968       0.06ms      0.0
wal+normal               1795500       0.03ms   1665.7
wal+normal+limit         1731645       0.03ms   1665.7
wal+normal+cache64       1661078       0.06ms   1665.7
wal+normal+mmap256       1732933       0.04ms   1665.7
```

**Verdict:** WAL + `synchronous=NORMAL` is the win; `cache_size` (0.93×) and `mmap_size`
(0.97×) are noise or worse on this workload. They stay **unset** — with the numbers
recorded so nobody "tunes" them back in later.

## 3. What shipped in this pass

| Change | Evidence |
|---|---|
| **ETag + 304** on `/models/tree`, `/videos/models`, `/videos/presets`, `/system/licences`, `/videos/capability-matrix` | 304 in **1.1–1.4 ms**; 13 tests |
| **Model-tree memo** (15 s TTL, generation counter, explicit invalidation) | 1 103 ms → 4 ms warm |
| **Boot warm-up** of the catalogue in the background | first UI hit no longer pays the scan |
| **Compression opt-in**, streaming excluded | default off, verified: no `content-encoding` |
| **FFmpeg thread cap** (`-threads 4`, configurable) on delivery + normalise | commands verified to carry the flag |
| **Measurement** of x264 vs VideoToolbox on this machine | recorded above; policy in P4 |

## 4. What to do next (Phase 9)

Ordered by value per hour, with the research's confidence:

1. **Lazy LLM SDK imports** (P6) — ~1.3 s off boot, low risk.
2. **Measured encoder policy** (P4) — benchmark both encoders once per machine, cache the winner, default to x264 when the user is waiting; use VideoToolbox for parallel/background work.
3. **SSE for render progress** (P7) — removes 28 800 requests/day of self-inflicted load.
4. **Profiling harness in CI** (P8) — `hyperfine` for two hot endpoints + a bundle budget; fail loudly on regression.
5. **SQLite knob measurement** (P9) — one benchmark, then keep or drop.
6. **Virtualise the long lists** and move the playhead out of React state (P10).
7. **Parallel normalisation** of clips (independent encodes, then concat-demuxer join) — the research's clearest FFmpeg structural win.

Deliberately **not** next: more uvicorn workers (duplicates model memory and splits the
SQLite writer), `aiosqlite` (31–173× slower per query on a single-writer database),
`orjson`/`ORJSONResponse` (deprecated and slower than annotated Pydantic models), and
`content-visibility` without a measured `contain-intrinsic-size`.

## 5. Uncertainty

- The FFmpeg numbers are one machine (this one) and one content type (synthetic); the
  research's M1 Pro study found VideoToolbox ~parity at 1/7 the CPU. **Re-measure on
  the target chip** before hard-coding a policy — that is why 9.5 benchmarks rather
  than assumes.
- `get_model_tree` cold cost is dominated by filesystem warmth, so it varies with disk
  cache state; the 1.1 s figure is first-touch in a fresh process.
- Endpoint latencies were taken with `TestClient` (in-process), which excludes socket
  and browser costs but is the right basis for comparing our own changes.

## Related

[Caveat Investigation](caveat-investigation-2026-11.md) ·
[Video Capability Landscape](video-capability-landscape-2026-11.md) ·
[Plan — Phase 9](application-remediation-plan-2026-11.md#phase-9--performance-2026-11-18) ·
[Index](index.md)
