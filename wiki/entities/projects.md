---
title: Studio Projects Subsystem
type: entity
tags: [projects, workspace, stems, midi, export, daw, database]
created: 2026-09-07
updated: 2026-09-07
sources: [backend/app/main.py, frontend/src/components/views/ProjectsView.tsx, backend/app/models.py]
aliases: [Projects, Studio Folders, Project Pack]
---

# Studio Projects Subsystem

The **Studio Projects Subsystem** provides multi-session production workspaces in Milimo Music. It organizes individual song generations, multitrack audio stems, neural source separations, MIDI note transcriptions, and lyrics under unified album or EP project containers with shared musical constraints (Tempo/BPM, Musical Key Signature, Style Tags, and Accent Themes).

---

## 1. Architecture & Data Model

### 1.1 Database Schema

Projects are persisted via SQLModel / SQLite in `backend/app/models.py`:

```python
class Project(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    cover_image_path: Optional[str] = Field(default=None)
    image_prompt: Optional[str] = Field(default=None)
    tags: Optional[str] = Field(default=None)
    bpm: Optional[int] = Field(default=120)
    key_signature: Optional[str] = Field(default="C Major")
    color: Optional[str] = Field(default="teal")  # 'teal' | 'cyan' | 'amber' | 'emerald' | 'sky'
    icon: Optional[str] = Field(default="folder")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

Tracks (`Job` records) are associated via the nullable foreign key:
```python
class Job(SQLModel, table=True):
    ...
    project_id: Optional[str] = Field(default=None, index=True)
```

Deleting a project folder unlinks associated generations without deleting the master audio files from the general song library.

---

## 2. Backend REST API Endpoints

All endpoints are hosted at `/projects`:

| Method | Route | Description |
|---|---|---|
| `GET` | `/projects` | Lists all projects sorted by `updated_at DESC` with calculated stats (`track_count`, `total_duration_s`, `stems_count`, `midi_count`). |
| `POST` | `/projects` | Creates a new project with custom name, description, cover image, BPM, key, tags, and theme color. |
| `GET` | `/projects/{project_id}` | Retrieves project details, child session jobs, and calculated track metrics. |
| `PUT` | `/projects/{project_id}` | Updates project metadata and bumps `updated_at`. |
| `DELETE` | `/projects/{project_id}` | Deletes the project container while unlinking child jobs. |
| `POST` | `/projects/{project_id}/tracks` | Associates a `Job` with the project (`{"job_id": "<uuid>"}`) with validation and error handling (400 for invalid UUID, 404 for missing track). |
| `DELETE` | `/projects/{project_id}/tracks/{job_id}` | Unlinks a track from the project container. |
| `POST` | `/projects/{project_id}/duplicate` | Creates an exact clone of the project settings (preserving BPM, Key, Style Tags, Accent Color, and Cover Image) named `{Original Name} (Copy)`. |
| `GET` | `/projects/{project_id}/export` | Packages and streams a comprehensive **Studio Multi-Track Pack** (`.zip`) containing all audio masters, stems, MIDI, notes, lyrics, and JSON metadata. |

---

## 3. Project Studio Pack (.zip) Export Format

When exporting a project via `GET /projects/{project_id}/export`, the backend generates an in-memory streaming ZIP archive with the following production structure:

```
{Project_Name}_studio_pack.zip
├── project_metadata.json
└── tracks/
    ├── 01_{Track_Title}/
    │   ├── master_audio.wav (or .mp3)
    │   ├── score.mid
    │   ├── score.musicxml
    │   ├── notes.json
    │   ├── lyrics.txt
    │   └── stems/
    │       ├── vocals.wav
    │       ├── drums.wav
    │       ├── bass.wav
    │       └── other.wav
    └── 02_{Track_Title}/
        └── ...
```

---

## 4. Frontend Integration

1. **`ProjectsView.tsx`**:
   - **Top-level Browse**: Real-time project search bar (matching title, tags, description), dynamic tag filter pills, total duration, track count, and one-click quick duplicate/export buttons on cards.
   - **2-Column Creation Modal**: Integrated drag-and-drop artwork upload, FLUX.2 AI prompt generator, project name, character-counted description, BPM tempo slider/input (40–240), Musical Key selector, Default Style tags, and Accent Color palette picker.
   - **Inside Project Folder**: Project Header Banner displaying tempo, key, tags, artwork, session checklist (MIDI + Score, Fast 4-Stems), "Play All" continuous playback, "Generate in this Project" (pre-populating composer prompt with BPM and Key), "Export Studio Pack" (.zip download), "Duplicate", "Edit", and "Delete".
2. **`TrackDetailView.tsx`**:
   - Displays project affiliation chip (e.g. `📁 Neon Horizons LP` or `+ Add to Project`).
   - Interactive Project Assignment Modal to move tracks between projects or detach back to general library.
3. **`SongsView.tsx`**:
   - Table and Grid views automatically surface project badges next to track titles, linking song library records with their active project folders.

---

## 5. Verification & Tests

The project lifecycle is thoroughly verified in `backend/tests/test_production_v2.py`:
- `test_project_crud_and_validation`: Verifies create, read, update, delete, and 404 guarantees.
- `test_project_tracks_association_and_stats`: Verifies adding/removing tracks, error handling for invalid UUIDs (400 Bad Request), and aggregate stats computation (`total_duration_s`, `midi_count`, `stems_count`).
- `test_project_duplicate_lifecycle`: Verifies cloning project parameters, name suffixing, and ID isolation.
- `test_project_studio_pack_export_zip`: Verifies multi-track archive assembly, metadata JSON generation, and stem/score directory packaging.
