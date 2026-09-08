---
title: Naming Contract (Session / Job / Project titles)
type: concept
tags: [naming, sessions, projects, jobs, rename, validation]
created: 2026-09-08
updated: 2026-09-08
sources: []
aliases: [rename contract, title validation]
---

# Naming Contract (Session / Job / Project titles)

Milimo has **three independent title fields**. A rename touches exactly one
row and propagates nowhere — deliberate isolation, documented here so nobody
"fixes" it into spooky action.

| Noun | Field | Rename API | Rename UI | Surfaces |
|------|-------|------------|-----------|----------|
| Session (chat thread) | `title` | `PATCH /sessions/{id}` | rail pencil + double-click (optimistic + rollback) | left rail, explore header |
| Job (track) | `title` (nullable) | `PATCH /jobs/{id}` (typed `JobUpdate`) | History inline edit | Recent Sessions list, DAW header, exports, zips |
| Project | `name` | `PUT /projects/{id}` | Projects Edit modal | project cards, Studio Pack |

Consequences that routinely confuse:

- Renaming a **Session** changes nothing in the DAW: `SessionWorkspace`
  renders a **Job**, and "Recent Sessions" lists **Jobs**.
- Renaming a **Job** does not touch its session or project.
- Export filenames sanitize per-export; collisions are callers' concern
  (no uniqueness enforced — see open item below).

## Validation (server, all three paths)

`validate_display_name` (`backend/app/models.py`): strip, require non-blank,
cap at `MAX_NAME_LENGTH = 120`, else **422** (`invalid_input`). Stored
stripped. `Job.title` may be explicitly nulled (legacy nullable semantics);
blank strings may not.

## Auto-rename

`POST /sessions/{id}/chat` renames **only while the title is still the
default**, matched case-insensitively against `DEFAULT_SESSION_TITLE`
(`backend/app/models.py`, mirrored as `DEFAULT_SESSION_TITLE` in
`frontend/src/api.ts`). Deliberate short names ("EP", "V2") survive.
Frontend creation (button + optimistic temp) uses the shared constant.

## Related pages

- [Session Workspace](../entities/session-workspace.md) | [Studio Projects](../entities/projects.md) | [Model Manager](../entities/model-manager.md)
