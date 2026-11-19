# Operations: Performance, Memory & Telemetry

This is the operator's guide to the tuning knobs that were **measured** rather than
assumed. Every default below is the result of a benchmark run on real hardware
(Apple Silicon, M-series, local single-user deployment). Where a knob is unset, it is
unset on purpose and the measurement is recorded so it does not get "tuned" back in.

> [!NOTE]
> The target deployment is a **local, private, fully functional** installation — not a
> horizontally scaled cloud service. Knobs that help a fleet (multiple workers, blanket
> compression, bigger caches) are net negatives here, and the measurements say so.

---

## 1. Measuring before changing

Two harnesses ship with the repo. Run them on the machine you intend to deploy on;
numbers from another machine do not transfer.

| Tool | What it answers |
|---|---|
| `python3 scripts/bench_api.py` | Request latency + throughput on the live app (`/health`, `/models/tree`, `/jobs`, 304 behaviour) |
| `python3 scripts/bench_sqlite.py` | SQLite write throughput and point-read latency under each pragma set |
| `python3 -m app.services.video.encoder_bench` | Per-encoder encode time for 1 s of 1080p on this GPU/CPU |

`bench_api.py` is also wired into CI as the `perf-smoke` job, so a regression shows up
as a failing gate rather than a slow app.

### What the measurements actually said

| Assumption | Measurement | Decision |
|---|---|---|
| gzip helps | Break-even at ~400–860 Mbit/s on loopback; payloads are already small | **Off by default** (`MILIMO_COMPRESS=1` to enable, for LAN/remote clients) |
| VideoToolbox beats x264 on Apple Silicon | x264 **0.21 s** vs VideoToolbox **0.31 s** per 1 s of 1080p (VideoToolbox wins on core count only: 0.9 vs 6.0) | Encoder is **measured at runtime**, not chosen by name |
| `cache_size` / `mmap_size` speed up SQLite | **0.93×** and **0.97×** vs WAL+NORMAL on 20 000 writes | **Left unset**, with the numbers recorded |
| WAL is worth it | 1 655 868 → 1 795 500 writes/s, point read 0.06 ms → 0.03 ms | **WAL + `synchronous=NORMAL` + `busy_timeout`** |
| More uvicorn workers help | One process owns the GPU lock and the model cache; extra workers duplicate both | **Single process** |
| `aiosqlite` / `orjson` help | Writes are not the bottleneck and response bodies are small | **Not adopted** |

---

## 2. Performance knobs

| Variable | Default | Effect |
|---|---|---|
| `MILIMO_COMPRESS` | unset (off) | Enables gzip response compression. Only worth it for a LAN or remote client; a net loss on loopback. |
| `MILIMO_FFMPEG_THREADS` | `4` | Thread cap for Milimo's own FFmpeg encodes (`0` disables the cap). Uncapped FFmpeg saturates the machine and starves the API. |
| `MILIMO_MODEL_TREE_TTL_SECONDS` | `15` | Memoised model-tree TTL. The uncached tree scan cost **131 ms → 4 ms** memoised; downloads call `invalidate_model_tree()`. |
| `MILIMO_NORMALISE_CONCURRENCY` | `2` | Parallel clip normalisation. Higher concurrency trades latency for memory (each FFmpeg holds a decoded frame buffer). |
| `MILIMO_INTERP_MODE` | `mci` | Frame interpolation: `mci` (motion-compensated, better) or `blend` (fast path). |
| `MILIMO_RIFE_CMD` | unset | External RIFE binary; enables neural frame interpolation for delivery when present. |
| `MILIMO_VIDEO_ENCODER` | unset (measured) | `x264` \| `videotoolbox` \| `nvenc`. Skips the runtime benchmark and pins an encoder. |

### Boot behaviour

- Catalogue is warmed at boot and ETags are precomputed, so the first `/models/tree`
  after a restart is not a cold scan.
- Static catalogue responses carry strong ETags and answer `304` in **1.0–1.4 ms**.
- Lazy LLM SDK imports cut boot from **4.64 s → 3.83 s** (2.46 s on a warm disk).

---

## 3. Memory management

### Truth is `phys_footprint`, not RSS

On macOS with unified memory, RSS under-reports badly: a process holding an 8 GiB MLX
allocation can read **30 MB RSS** against **8 370 MB `phys_footprint`**. The platform
reads `phys_footprint` via `proc_pid_rusage(RUSAGE_INFO_V4)` and reports that
everywhere. If a number disagrees with Activity Monitor, believe the footprint.

> [!WARNING]
> `rusage_info_v4` must be declared at its full size (the probe uses a 512-byte buffer
> with `resident_size` at offset 64 and `phys_footprint` at 72). A truncated `ctypes`
> struct corrupts memory and later torch calls hang — it presents as a mystery deadlock,
> not a crash.

### Accelerator caps

| Variable | Default | Why |
|---|---|---|
| `MILIMO_MLX_CACHE_LIMIT_MB` | `min(12 GiB, 10 % RAM)` | MLX's own default was **161 GiB**, i.e. no ceiling at all. `0` disables the cache. |
| `MILIMO_MPS_MEMORY_FRACTION` | `0.6` | PyTorch's high watermark is 1.7× the recommendation — **182 GiB on this machine**, more than physical RAM. |
| `MILIMO_WAN_PIPELINE_CACHE` | `1` | One resident Wan pipeline. Switching 14B ↔ 1.3B used to keep both alive until an explicit unload. `0` restores unlimited. |
| `MILIMO_AGENT_HISTORY_MAX_CHARS` | `24000` | Trims agent history by characters as well as turns. |
| `MILIMO_AGENT_HISTORY_MAX_MESSAGES` | `24` | Turn cap for the same history. |
| `MILIMO_MEMORY_GC` | unset (off) | Opt-in `gc.collect()` on unload. Measured cost: 8 tests went from instant to **16 s**, suite to 265 s. |

Releasing memory is **one code path** (`memory.release()`), used by the UI button,
the idle sweeper and every unload. `del` alone does not free: MLX keeps buffers in
`cache_memory` until `clear_cache`, and `torch.mps` needs `empty_cache()`.

### Budget guard

`memory.fits(required_mb)` compares a model against `max_recommended_working_set`
(84 % of RAM here). A download larger than that budget is **refused with a 507** and
the real numbers, because exceeding the recommendation does not raise — it swaps the
machine into unresponsiveness.

---

## 4. Telemetry endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness plus DB, disk, queue depth, process hygiene (fds, threads, child/zombie counts) and footprint |
| `GET /system/memory` | Full snapshot: process, system, MLX, torch, hygiene, configured limits |
| `GET /system/memory/history` | Ring buffer of samples (one per sweeper pass, ~1/min), per-minute slope per channel, and a leak verdict |
| `POST /system/memory/release` | Drop cached accelerator memory through the single release path |
| `GET /system/telemetry` | Hardware coordinator telemetry |
| `GET /system/licences` | Model-catalogue licence audit **plus installed dependencies** |

### Leak verdict

A snapshot cannot distinguish "large" from "leaking". The history endpoint answers the
second question: a **positive slope while idle**. Threshold is
`MILIMO_MEMORY_LEAK_MB_PER_MIN` (default **25 MB/min**) across footprint, MLX live,
MPS driver allocation and swap.

```bash
curl -s localhost:8000/system/memory/history | jq '.report'
```

A fresh process reports `samples: 0` until the sweeper has run — that is correct, not a
bug. Give it a few minutes of uptime before trusting the verdict.

---

## 5. Database upkeep

- WAL with `synchronous=NORMAL` and a 64 MiB `journal_size_limit`.
- The sweeper runs `PRAGMA wal_checkpoint(TRUNCATE)` each pass and reports
  `(busy, log_frames, checkpointed)`. `busy=1` means a long-lived reader won; the next
  pass retries.
- Back up with `sqlite3 jobs.db "VACUUM INTO 'backup.db';"` — never by copying active
  files.

---

## 6. Licence gate

```bash
MILIMO_NONCOMMERCIAL_OK=1 python3 scripts/check_licences.py
```

The gate audits the **model catalogue** (including which model each modality actually
resolves to as default) and the **installed dependencies**. It prints `[ACK]` for a
non-commercial default and exits 0 only when `MILIMO_NONCOMMERCIAL_OK=1` acknowledges
the deployment. This is deliberate: `flux_2_klein_9b` is the effective image default
and carries the FLUX.2 Non-Commercial License, while the 4B variant is Apache-2.0.

A flagged dependency (e.g. `insightface` with research-only bundled models) is a
**notice**, not a build failure — the package's licence is independent of which model
is selected, and a model-only gate cannot see it.

The same audit is available at `GET /system/licences` and runs in CI as the
`licence-gate` job.
