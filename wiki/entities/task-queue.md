---
title: Durable Task Queue
type: entity
created: 2026-09-03
updated: 2026-09-03
tags: [queue, concurrency, durability, sse, sqlite, phase-4]
aliases: [TaskQueue, task queue, GPU/IO lanes]
sources: [production-readiness-plan.md]
---

# Durable Task Queue

> [!NOTE] **Status: design locked 2026-09-03, not yet shipped.** This page captures the
> approved Phase 4 design from the [production plan](../production-readiness-plan.md).
> Update this page to "shipped" with file/line references when it lands.

The **Durable Task Queue** (`backend/app/core/queue.py`, planned) replaces ad-hoc
`BackgroundTasks` / fire-and-forget `asyncio.create_task` execution with a SQLite-backed,
two-lane task system so long-running work survives restarts and multi-user load is
arbitrated explicitly instead of by a single global lock.

## Why (audit findings, 2026-09-03)

- Generation already runs under `music_service.gpu_lock` — one global `asyncio.Lock`
  (`orchestration/pipeline.py:72`) — so mastering (CPU DSP) is blocked behind GPU work
  and vice versa; there is no lane separation.
- `reconcile_orphan_jobs()` (`main.py:256`) marks every queued/processing job **FAILED**
  on boot ("Interrupted by server restart") — work is not durable.
- `/transcribe/upload` (`main.py:470`) and `/mastering/match` (`main.py:662`) are fully
  blocking HTTP calls that hold the request open for minutes.
- `/jobs/{id}/inpaint` (`main.py:2781`) uses bare `asyncio.create_task` — untracked,
  not registered for cancellation, lost on restart.
- The model-download ledger is an in-memory dict (`main.py:1039`, `_model_downloads`);
  download state and resumability die with the process.

## Design

### `TaskRecord` (new SQLModel table)

| Column | Notes |
|---|---|
| `id` | UUID PK |
| `type` | `generate` \| `transcribe_upload` \| `mastering` \| `voice_convert` \| `inpaint` \| `model_download` |
| `lane` | `gpu` \| `io` |
| `status` | `queued` \| `running` \| `completed` \| `failed` \| `cancelled` |
| `payload_json` / `result_json` | handler input / output envelopes |
| `job_id` | nullable link to a `Job` row when the task owns one |
| `progress`, `progress_msg` | persisted progress, flushed periodically |
| `attempts` | incremented on re-enqueue (restart recovery) |
| `error_type`, `error_msg` | typed failures (honest-error convention) |
| `cancel_requested` | cooperative-cancel flag |
| `created_at`, `started_at`, `finished_at` | lifecycle timestamps |

### Lanes

| Lane | Workers (default) | Task types |
|---|---|---|
| `gpu` | 1 (`MILIMO_GPU_WORKERS`) | `generate`, `transcribe_upload`, `voice_convert`, `inpaint` |
| `io` | 2 (`MILIMO_IO_WORKERS`) | `mastering`, `model_download` |

The single `gpu_lock` is retained as defense-in-depth; the queue itself provides the
scheduling discipline. CPU-heavy DSP (Matchering mastering) no longer queues behind
neural inference.

### `TaskQueue` singleton

- Worker pool: per-lane asyncio loops that claim the next `queued` row in FIFO order.
- Handler registry: `Dict[TaskType, async (payload, TaskContext) -> result_json]`.
- `TaskContext` carries `task_id`, a `threading.Event` for cooperative cancel
  (reusing the existing `pipeline._abort_if_terminal` guard semantics), and a
  `progress_cb` that publishes SSE `job_progress`/`task_update` and flushes
  `TaskRecord.progress`.
- `cancel(task_id)` merges with `music_service.cancel_job` so `POST /jobs/{id}/cancel`
  keeps working unchanged.

### Restart recovery (`recover()`)

Locked decision: **re-enqueue all**. On boot, `queued` tasks stay queued; `running`
tasks are re-enqueued with `attempts + 1` and an honest log line. This replaces the
fail-honest `reconcile_orphan_jobs()` behavior for queued work — the pipeline is
idempotent, so multi-minute re-runs after a crash are acceptable.

## Endpoint conversions (202 Accepted)

| Endpoint | Today | Planned |
|---|---|---|
| `POST /generate/music` | `BackgroundTasks` + `gpu_lock` | `queue.enqueue(generate)` — Job stays `queued` |
| `POST /transcribe/upload` | blocking separation + transcription | save upload sync → Job(queued) + gpu task → `202 {job_id}` |
| `POST /mastering/match/{id}` | blocking Matchering/LUFS | io task → `202 {task_id, job_id}`; writes `mastered_path` |
| `POST /jobs/{id}/voice-convert` | blocking (+ arg bug, see [SVC](../concepts/singing-voice-conversion.md)) | child Job(queued) + gpu task → `202 {job_id}` |
| `POST /jobs/{id}/inpaint` | bare `asyncio.create_task` | gpu task |
| `POST /models/download` | in-memory ledger | io task; per-file chunk progress in `TaskRecord` (resumable) |

Bodies keep their current shapes (axios treats 202 as success); frontends switch to
job-poll + the existing SSE stream. A `GET /queue/stats` endpoint reports lane depth
for the floating status widget.

## Tests (planned)

`backend/tests/test_queue.py`: enqueue→complete with fake handlers; GPU lane serializes
while IO lane runs in parallel; simulated restart re-enqueues `running`; cancel
propagation; 202 endpoint flows via `TestClient`.

## Related pages

- [Production plan — Phase 4](../production-readiness-plan.md) · [Backend & API](backend-api.md)
- [Orchestration pipeline](../concepts/generation-pipeline.md) · [Singing Voice Conversion](../concepts/singing-voice-conversion.md)
