---
title: Database Integrity, UUID Standards & Track Lifecycle
type: concept
tags: [database, sqlite, sqlalchemy, sqlmodel, uuid, lifecycle, cascade-delete, data-integrity]
created: 2026-09-09
updated: 2026-09-09
sources: [entities/backend-api.md, entities/session-workspace.md]
aliases: [Data Integrity, UUID Standards, Track Deletion Lifecycle, Cascade Deletion]
---

# Database Integrity, UUID Standards & Track Lifecycle

In Milimo Music, tracks (`Job` entities) are the core artifacts linking audio masters, stems, MIDI scores, lyrics, album releases, and studio sessions. Ensuring robust data integrity across SQLite and the SQLAlchemy/SQLModel ORM requires strict adherence to UUID encoding standards, relational cascading rules, and atomic filesystem sweeps.

---

## 1. The SQLite vs. SQLAlchemy UUID Representation Hazard

### 1.1 The Underlying Quirk
- **SQLModel / SQLAlchemy Column Declaration**:
  ```python
  class Job(SQLModel, table=True):
      id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
  ```
- **Dialect Behavior on SQLite**: SQLite lacks a native 128-bit UUID type. When SQLAlchemy compiles queries targeting a `GUID` column on SQLite, its internal type processor automatically coerces `UUID` objects and UUID strings into **unhyphenated 32-character hexadecimal strings** (`u.hex`, e.g. `'188619ef9afd403297d6bf559b0c180e'`).
- **The Problem**: If a row is inserted into SQLite with standard RFC 4122 hyphens (e.g. `'188619ef-9afd-4032-97d6-bf559b0c180e'`), standard ORM lookups like `session.get(Job, u)` or `select(Job).where(Job.id == str_id)` compile to:
  $$\text{SELECT} \dots \text{WHERE job.id} = \text{'188619ef9afd403297d6bf559b0c180e'}$$
  Because SQLite performs strict literal string equality, the comparison returns `False`. The query returns `None`, raising `HTTPException(404, detail="Job not found")`.
- **Delete Failure Hazard**: If an ORM delete is attempted (`session.delete(job)`), SQLAlchemy generates a `DELETE FROM job WHERE job.id = '188619ef9afd403297d6bf559b0c180e'`. Zero rows match in SQLite, triggering an `SAWarning` while leaving the database record completely intact.

---

## 2. Architectural Solutions & Protocols

### 2.1 Universal Multi-Format Lookup Pattern (`get_job_by_id`)
To guarantee that any job can be retrieved regardless of whether the ID is passed as a 32-hex string, a 36-char hyphenated string, or a UUID object, [`get_job_by_id()`](file:///Users/mck/Desktop/milimomusic/backend/app/main.py#L650-L690) executes a direct text SQL clause before falling back to ORM methods:

```python
clean_str = str(job_id_input).strip()
hex_str = clean_str.replace("-", "")
hyphen_str = str(UUID(hex_str)) if len(hex_str) == 32 else clean_str

# Bypasses SQLAlchemy SQLite GUID hex stripping for hyphenated IDs
stmt = select(Job).where(text("id = :c OR id = :h OR id = :hyp")).params(
    c=clean_str, h=hex_str, hyp=hyphen_str
)
job = session.exec(stmt).first()
```

### 2.2 Startup Self-Healing Normalization Migration
On server startup within `init_db()`, an automated migration scans the database for non-canonical hyphenated IDs and updates both primary keys and foreign references in a single transaction:

```python
hyphen_jobs = session.exec(text("SELECT id FROM job WHERE length(id) = 36 AND id LIKE '%-%';")).all()
for (old_id,) in hyphen_jobs:
    new_id = old_id.replace("-", "")
    session.exec(text("UPDATE job SET id = :new WHERE id = :old;").params(new=new_id, old=old_id))
    session.exec(text("UPDATE session SET active_job_id = :new WHERE active_job_id = :old;").params(new=new_id, old=old_id))
    session.exec(text("UPDATE sessionmessage SET generated_job_id = :new WHERE generated_job_id = :old;").params(new=new_id, old=old_id))
    session.exec(text("UPDATE playlisttrack SET job_id = :new WHERE job_id = :old;").params(new=new_id, old=old_id))
    session.exec(text("UPDATE job SET parent_job_id = :new WHERE parent_job_id = :old;").params(new=new_id, old=old_id))
```

### 2.3 Relational Cascade Cleanup Protocol
When a track is deleted via `DELETE /jobs/{job_id}`, foreign references must be cleaned up to prevent dangling pointers:

1. **Active Studio Sessions**: `UPDATE session SET active_job_id = NULL WHERE active_job_id = :tid`
2. **Conversation History**: `UPDATE sessionmessage SET generated_job_id = NULL WHERE generated_job_id = :tid`
3. **Playlists**: `DELETE FROM playlisttrack WHERE job_id = :tid`
4. **Lineage Extensions**: `UPDATE job SET parent_job_id = NULL WHERE parent_job_id = :tid`
5. **Album Track Order**: Parse `release.track_order_json`, filter out the deleted ID, and re-serialize.
6. **Expunged SQL Row Deletion**: Call `session.expunge(job)` to disconnect unit-of-work tracking, followed by `DELETE FROM job WHERE id = :tid`.

### 2.4 Comprehensive Filesystem Artifact Purge
[`_delete_job_artifacts()`](file:///Users/mck/Desktop/milimomusic/backend/app/main.py#L3300-L3345) sweeps both hex and hyphenated ID forms across all storage directories:
- **Master Audio**: `generated_audio/{form}.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`
- **Stems**: `generated_audio/stems/{form}_*.wav`
- **Mastering & Vocals**: `generated_audio/mastered/{form}*`, `generated_audio/converted_vocals/{form}_*.wav`
- **Notation & Scores**: `generated_audio/{form}.mid`, `generated_audio/{form}.musicxml`, `generated_audio/sheets/{form}/`
- **Waveform Peaks & Covers**: `generated_audio/.peaks/{form}.*.json`, `data/covers/{form}*`
- **Tokens**: `generated_tokens/{form}*`

---

## 3. Frontend UX & Performance Standards

1. **Single Source of Confirmation**:
   - Only `handleDeleteJob` in `App.tsx` issues `window.confirm()`. Child components (`HistoryFeed.tsx`, `SongsView.tsx`) forward the delete intent directly without duplicate dialogs.
2. **Event Propagation Protection**:
   - Delete button clicks always call `e.stopPropagation()` to prevent unwanted navigation, track selection, or audio playback triggers.
3. **Toast Notifications**:
   - Success: `toast('Track permanently deleted.', 'info')`
   - Failure: `toast('Failed to delete track: ${errMsg}', 'error')`
4. **Performance Benchmarks**:
   - `get_job_by_id` query latency: **< 1.5ms** on SQLite.
   - `DELETE /jobs/{job_id}` full transaction & disk sweep: **< 25ms**.
   - Frontend optimistic state update to DOM removal: **< 16ms** (within single animation frame).

---

## Related Pages
- [Backend & API](../entities/backend-api.md) — Endpoints, SQLModel schemas, and lifecycle guards.
- [Audio Synthesis Standards](audio-synthesis-standards.md) — Procedural DSP synthesis and acoustic calibration.
- [Session Workspace (DAW)](../entities/session-workspace.md) — Multitrack arrangement and playback.
