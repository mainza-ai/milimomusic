---
title: Durable Task Queue & Job Lifecycle Manager
type: entity
tags: [queue, tasks, sqlite, persistence, recovery, checkpointing, lifecycle]
created: 2026-09-16
updated: 2026-09-16
sources: [sources/maestro-creative-studio.md]
aliases: [DurableTaskQueue, JobLifecycleManager, UniversalQueue]
---

# Durable Task Queue & Job Lifecycle Manager

The **Durable Task Queue** (`backend/app/core/task_queue.py`) is Milimo Music's persistent asynchronous job scheduling and execution engine.

In production, heavy generative tasks (such as 10-scene music video rendering, model fine-tuning, and multi-stem audio transcription) can take 5 to 20 minutes to complete. An in-memory dictionary is inadequate: any server restart, system crash, or unhandled exception immediately results in complete loss of progress.

Inspired by [Maestro](../sources/maestro-creative-studio.md)'s universal queue and checkpointing architecture, the Durable Task Queue provides restart recovery, asset isolation, and clip-level checkpointing.

---

## 1. Core Architectural Pillars

```
+-----------------------------------------------------------------------------------+
|                           Frontend Task Queue Manager                             |
|      [Job Status / Pause / Resume / Cancel / Re-order / Progress Telemetry]       |
+-----------------------------------------+-----------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
   [Persistent SQLite Database]                          [Asset Ownership Vault]
    - Job state, parameters, timestamps                   - Deep copy / symlink of input
    - Clip-level completion records                         audio, lyrics, and images
    - Hardware profile assignment                         - Immune to user file moves
                |
                v
   [Crash & Restart Recovery Engine]
    - Scans unfinalized jobs at boot
    - Transitions orphaned jobs to 'PAUSED'
    - 1-Click resumption without re-rendering completed clips
```

---

## 2. Key Production Features

### 2.1 Persistent SQLite State Storage
- Every job submission creates an atomic record in the SQLite persistence store tracking:
  - `job_id`, `job_type` (`music_generation`, `video_director`, `stem_separation`, `voice_training`).
  - `status`: `QUEUED` | `ENHANCING` | `RUNNING` | `PAUSED` | `COMPLETED` | `FAILED` | `CANCELLED`.
  - `progress_pct`, `current_stage`, `completed_clips_json`.
  - `error_message`, `telemetry_snapshot`.

### 2.2 Asset Ownership Vault
- When a user queues a job requiring external assets (e.g. reference face image, custom audio guide, or lyrics file), the queue makes an immutable copy inside `data/job_vault/{job_id}/`.
- If the user renames, moves, or deletes the original file while the job is waiting in queue, the active pipeline executes without missing-file errors.

### 2.3 Independent Clip-Level Checkpointing
- In multi-clip Director video productions, each scene's generation state (scene prompt, keyframe image, rendered video clip) is committed to disk immediately upon completion.
- If clip 8 of 10 encounters an OOM or hardware crash, clips 1 through 7 remain safe. Resuming the job begins directly at clip 8.

### 2.4 Queue Pre-Enhancement
- When multiple jobs are queued, prompt expansion and style enhancement execute asynchronously in the background while the GPU is occupied with synthesis.
- When the GPU completes the preceding job, the next job is already enhanced and begins generation immediately with zero idle latency.

### 2.5 1-Click Restart Recovery
- Upon server startup, `task_queue.recover_orphaned_jobs()` detects any tasks left in `RUNNING` or `ENHANCING` states from a previous session.
- It safely transitions them to `PAUSED` with an explanatory log message, allowing creators to click "Resume" in the UI without losing previously rendered assets.

---

## 3. Related Pages
- [Task Queue](task-queue.md)
- [AI Music Video Studio](video-studio.md)
- [Director Mode v2](../concepts/director-mode-v2.md)
- [Maestro Creative Studio Ingest](../sources/maestro-creative-studio.md)
- [Global Hardware Coordinator](hardware-coordinator.md)
