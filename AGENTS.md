# AGENTS.md — Milimo Music Wiki Schema

This file is the **schema** for the Milimo Music wiki. It tells the LLM how the
wiki is structured, what the conventions are, and what workflows to follow when
ingesting a new source, answering a question, or maintaining the wiki.

Read this file before doing *any* wiki work. When the wiki evolves, update this
schema in the same session so it stays the source of truth for future sessions.

---

## 1. What this project is

**Milimo Music** is an open-source, non-commercial **AI music generation, neural
transcription, and multitrack production platform** (a "DAW") by [Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295).
It generates full stereo tracks conditioned on lyrics and style tags, auto-transcribes
them to note-level MIDI + MusicXML, separates stems, and provides a web DAW workspace
(arrange timeline, piano roll, notation, mixer) plus offline voice cloning.

Beyond generation, it provides: a pluggable **generation-provider** layer (MiniMax Music 3
default, HeartMuLa legacy), a 4-step orchestration pipeline (generate → stem separation →
optional voice conversion → MuScriptor transcription + lyric sync), an AI Co-Writer
(multi-agent lyrics engine), a Training Studio, Model Manager, Projects, and real-time
SSE progress.

The three-layer architecture making this a LLM-wiki (not just a doc dump):

- **Raw sources** (`docs/`, `devs/`, README, backend code) — immutable input. Never edit these.
- **The wiki** (`wiki/`) — LLM-owned markdown. Create, update, cross-reference, keep consistent.
- **The schema** (this file) — governs how the LLM maintains the wiki.

**Rule:** never modify `docs/`, `devs/`, `backend/`, `frontend/`, `heartlib/`,
`muscriptor/`, or `README.md`. The wiki reads from them; it does not write to them.

---

## 2. Directory structure

```
wiki/
  index.md      # content-oriented catalog of every page (the "table of contents")
  log.md        # chronological, append-only record of every operation
  overview.md   # high-level what/why of Milimo Music (the synthesis entry point)
  architecture.md # system architecture: layers, providers, data flow
  roadmap.md    # the v2 refactor / upgrade plan synthesis
  entities/     # noun-like pages: models, agents, services, components, repos
  concepts/     # idea pages: how pieces work and fit together
  sources/      # one page per ingested raw source
```

Any page can live in any subfolder; pick the folder by *kind*:

| Kind | Folder | Example |
|------|--------|---------|
| Model / repo / external thing | `entities/` | `HeartMuLa.md`, `HeartCodec.md`, `muscriptor.md` |
| In-app component / service / agent | `entities/` | `AICoWriter.md`, `TrainingStudio.md` |
| Concept / mechanism / pattern | `concepts/` | `lyrics-conditioning.md`, `inpainting.md` |
| Ingested source doc | `sources/` | `heartlib-bible.md`, `training-studio-guide.md` |

`index.md` and `log.md` are special (see §5). Never delete them.

---

## 3. Page format conventions

Every page begins with YAML frontmatter:

```yaml
---
title: ...
type: entity | concept | source | index | log | overview
tags: [...]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [...]      # ids of source pages feeding this page (if any)
aliases: [...]      # other names people use for this thing
---
```

Body conventions:

- **One idea per page.** If a page grows past ~200 lines, split it and link out.
- **Links are `[[wiki/...]]`-style** relative wiki links: `[HeartMuLa](entities/HeartMuLa.md)`.
  Use relative paths so the wiki is portable (works in Obsidian, editors, plain markdown).
- **Always link**, never inline-copy another page's content. The graph view depends on real cross-links.
- **Citations**: when a claim comes from a specific raw source, cite it as
  `[source](sources/<source-page>.md)`. When a claim is synthesis across sources, say so.
- **Tom** of voice: neutral, technical, specific. Prefer concrete facts (numbers,
  tags, params, file paths) over generalities.
- Mark uncertainty explicitly with `> [!NOTE]` or `> [!WARNING]` blockquotes rather
  than deleting conflicting information.

---

## 4. Workflows

### 4.1 Ingest

When the human drops a new source into `docs/`, `devs/`, or provides one in chat:

1. Read the source fully.
2. Create a page in `sources/<slug>.md` capturing the key content, one summary per section.
3. Update `wiki/index.md` (add the new source + any new entity/concept pages).
4. Update any existing entity/concept pages that the source touches (new facts, corrections).
5. Record **contradictions** — if the new source disagrees with an existing page, note the
   discrepancy on both pages rather than silently overwriting.
6. Append an entry to `wiki/log.md` with the exact prefix format (see §5.2).
7. Discuss key takeaways with the human before making large edits.

### 4.2 Query

1. Read `wiki/index.md` first to locate relevant pages.
2. Read those pages (not raw sources unless depth requires it).
3. Synthesize an answer with citations.
4. If the answer is valuable on its own (a comparison, an analysis, a map),
   offer to file it back into the wiki as a new page — good answers compound.

### 4.3 Lint

Periodically run a health check:

- Contradictions between pages → reconcile or flag both.
- Stale claims superseded by newer sources → update.
- **Orphan pages** (no inbound links) → add inbound links or delete.
- Important concepts mentioned but lacking their own page → create.
- Missing cross-references → add.
- Data gaps that a web search could fill → suggest questions/sources to the human.

---

## 5. Special files

### 5.1 `wiki/index.md` — content catalog

A flat-ish catalog of every wiki page grouped by category. Each entry:

```markdown
- [Page Title](entities/thing.md) — one-line summary. `tags: [...]`
```

Update on every ingest. This is the primary navigation aid; keep it complete and current.

### 5.2 `wiki/log.md` — chronological history

Append-only. Every entry starts with a **consistent, parseable prefix**:

```markdown
## [2026-08-19] ingest | Heartlib Bible
```

Formats we use:
- `## [YYYY-MM-DD] ingest | <source title>`
- `## [YYYY-MM-DD] query | <question>`
- `## [YYYY-MM-DD] lint | <summary>`
- `## [YYYY-MM-DD] create | <new page>`

Because of the consistent prefix, simple tools work:
`grep "^## \[" wiki/log.md | tail -5`.

---

## 6. Conventions to keep consistent

- **Spelling**: HeartMuLa, HeartCodec, HeartCLAP, HeartTranscriptor, Heartlib,
  MuScriptor (capitalized), MiniMax Music 3, Milimo Music, HTDemucs (also “Demucs”),
  LoRA, RVQ, MPS, CUDA, SSE (Server-Sent Events), Co-Writer. MuScriptor is the
  transcription/notation engine; HTDemucs is the real neural source-separation engine —
  the two are **dual stem sources** the user can switch between.
- Link **style tags** inside backticks but not every mention — only the first/defining mention.
- Relative paths only in links.
- Frontmatter `updated` date bumps whenever a page changes.
