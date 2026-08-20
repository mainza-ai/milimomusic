---
title: Frontend
type: entity
created: 2026-08-19
updated: 2026-08-20
sources: [sources/readme.md, sources/v2-refactor-plan.md]
tags: [frontend, react, vite, tailwind, ui, daw]
aliases: [Web UI, Frontend app]
---

# Frontend

The **frontend** is Milimo Music's web UI — **React 19 + Vite + Tailwind CSS** on
`http://localhost:5173` (`npm run dev`). Author: Mainza Kangombe. It implements the
Suno-style **reference IA** plus a full web **DAW workspace**.

## Key dependencies
`axios`, `clsx`, `framer-motion`, `lucide-react`, `react`, `react-dom`, `tailwind-merge`,
`wavesurfer.js`, `zustand`.

## Suno-style IA — dedicated views (`components/views/`)
- **Explore** (default landing) — chat-first "Producer" input, quick-start chips, feed.
- **SongsView** — song library: List/Grid, search + genre pills; per-track play, **View
  Lyrics**, Favorite, **DAW** (open workspace), Extend, Delete; "MIDI Ready + 4 Stems" badges.
- **PlaylistsView** — local (localStorage) playlists/albums with seed playlists; Play All,
  per-track DAW edit (client-side only, no backend playlist API).
- **ProjectsView** — [project folders](backend-api.md) (BPM, key signature, color); project
  stats, "Generate in this Project", "Add Existing Track", DAW open.
- **MusicVideosView** — AI music-video studio (aesthetic presets, simulated storyboard,
  "WhisperX Aligned") — largely a mockup.
- **ProfileView** — artist profile with badges (STUDIO MASTER, RVC/SVC, Note-Level
  Transcription), featured creations.

## Composer (`ComposerSidebar.tsx`)
- **Sound & Style** tab: concept/mood, style pills, **Structured Caption Spec** expander
  (Global Metadata / Vocal Details / Arrangement),
  **Signal & Sampling Controls** (duration 5–300 s, temperature, CFG scale, top-k/top-p,
  DiT diffusion steps, seed lock, master format).
- **Lyrics & Structure** tab: section-tag helper pills (`[Intro] [Verse 1] [Chorus] …`),
  LLM model selector, AI Co-Writer "Write Lyrics".
- **Model provider selector** (MiniMax Music 3 default / HeartMuLa) and **"Sing as…" voice
  selector** (+ "Train New Voice…" → VoiceStudioModal).
- Track-extension banner and Active Project banner. Generate button: "Generate & Transcribe Track".

## DAW workspace (`components/workspace/`)
- **SessionWorkspace** — the [web DAW](session-workspace.md): Listen / Arrange / Piano Roll /
  Notation / Mix / Lyrics modes, multitrack playback engine, export.
- **ArrangeTimeline**, **PianoRoll**, **NotationViewer**, **MultitrackMixer** —
  detailed in [Session Workspace](session-workspace.md).

## Modals & UI kit
- **ModelsManagerModal** — [Model Manager](model-manager.md) (hardware profile + model tree).
- **VoiceStudioModal** — [Voice Studio (SVC)](voice-service.md) with mandatory consent.
- **TrainingStudio**, **StyleManagerModal** — [Training Studio](training-studio.md).
- **LLMSettingsModal**, **PathsSettingsModal** — providers/paths.
- **InpaintModal** — [Repair Segment](inpainting.md).
- **HistoryFeed** — session history with "Open in DAW", stems/MIDI badges, day-grouping.
- UI kit — `GradientButton`, `GlassCard`, `AudioVisualizer`, `Combobox`, `Toast`,
  `GlobalAudioPlayer`, `FloatingStatusWidget` (pipeline progress overlay driven by SSE
  `job_progress` events), `MilimoLogo`; theme context (`ThemeContext`).

## Client API (`api.ts`)
Groups: `api` (generate, lyrics, enhancement, history, jobs, SSE events), `modelsApi`,
`voiceApi`, `workspaceApi` (transcribe upload, export, mastering, save notes), `trainingApi`,
`projectApi`, `styleApi`, `pathsApi`. Full surface in [Backend & API](backend-api.md).

## Related pages
- [Architecture](../architecture.md) | [Session Workspace](session-workspace.md)
- [Backend & API](backend-api.md) | [Roadmap (v2)](../roadmap.md)
