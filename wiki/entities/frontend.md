---
title: Frontend
type: entity
created: 2026-08-19
updated: 2026-08-29
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

## Brand Identity & Design System
- **Official Tagline**: *"Give the silence something worth remembering."* with sub-motto *"Speak it into being. Shape it until it's yours."*
- **Brand Logo (`MilimoLogo.tsx`)**: Official `/milimo_logo.png` rendered within an Apple App Icon glassmorphic container (`rounded-xl bg-gradient-to-tr border border-black/10 dark:border-white/15`). Used across landing hero, navigation rail, audio players, feed thumbnails, DAW workspace, and lock-screen metadata.
- **Apple Glassmorphism UI**: System-wide dark/light palette using clean translucent frosted cards, subtle gradients, and standard status pills.

## Prompt Responsiveness & Production Cancellation
- **Instant 0ms Input Clearing & Optimistic Chat Injection**: Submitting a prompt immediately clears the input field and injects the user's message optimistically into the active conversation thread with zero perceived latency.
- **Live Producer Composing Status Card**: Displays animated stage progression (`"Analyzing musical direction..."` → `"Writing song lyrics..."` → `"Structuring arrangement..."`) with real-time progress bar.
- **Production-Grade Cancellation Architecture**:
  - `AbortController` integration wired through `sessionApi.sendChatMessage` and `api.producerCompose`.
  - **Dynamic Stop Button (`<Square>`)**: In-flight submissions turn the prompt bar send arrow into an animated pulsating Stop button.
  - **Live Composing Card Cancellation**: Inline *"Stop generating"* button and `✕` close control.
  - **Generation Banner & HUD Cancellation**: Dedicated cancel buttons in the active generation banner and floating HUD widget trigger backend `POST /jobs/{id}/cancel` and immediate client state reset.

## Suno-style IA — dedicated views (`components/views/`)
- **Explore** (default landing) — chat-first "Producer" input, quick-start chips, feed, and central brand hero with the official silence tagline.
- **SongsView** — song library: List/Grid, search + genre pills; per-track play, **View
  Lyrics**, Favorite, **DAW** (open workspace), Extend, Delete; "MIDI Ready + 4 Stems" badges.
- **PlaylistsView** — local (localStorage) playlists/albums with seed playlists; Play All,
  per-track DAW edit (client-side only, no backend playlist API).
- **ProjectsView** — [project folders](backend-api.md) (BPM, key signature, color); project
  stats, "Generate in this Project", "Add Existing Track", DAW open.
- **MusicVideosView** — AI music-video studio (aesthetic presets, simulated storyboard,
  "WhisperX Aligned") — marked as "In Dev".
- **ArtistsView** — the [artist domain](../concepts/artist-domain.md) front-end: guided
  4-step create stepper, server-searched/paginated artist grid with stats, artist detail
  (identity editor, singing-voice selector, world-lore editor + World-Builder generation,
  crew management with model overrides, experiencer studio, releases with lifecycle chips +
  art generation, tracklist with play/Studio/retry/reorder/review chips, run history with
  aggregates), deep-links `?view=artists&id=`, run recovery on reload, honest toasts
  throughout.
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
## Centralized Audio Engine & Playback Architecture
- **Single-Node `AudioEngineContext`** (`frontend/src/context/AudioEngineContext.tsx`):
  - Root singleton `<audio>` element with WebAudio `AnalyserNode` connected via `crossOrigin="anonymous"`.
  - Exposes `useAudioEngine()` controlling playback state, buffering, seeking, master volume fader, smart previous/next, shuffle, and loop modes.
  - Contextual dock visibility: floating dock automatically hides when entering dedicated studio environments (`TrackDetailView` and `SessionWorkspace`).
- **Floating Harbor Dock Player (`GlobalAudioPlayer.tsx`)**:
  - Rigid 3-zone flexbox layout (Left Track Info, Centered Non-Collapsing Transport, Right Studio Tools) eliminating all button collisions.
  - Pro media controls suite: Return to Start/Zero (`|<<`), Rewind 10s, Hero Play/Pause, Advance 10s, Next Track (`>>|`), Repeat modes, Speed selector (0.75x–2.0x), Volume slider + Mute, Up Next Queue Drawer, and Synchronized LRC Lyrics Sheet.
- **Track Studio Hero (`TrackDetailView.tsx`)**:
  - Integrated full transport cluster with live frequency equalizer waves, timecode modes (elapsed/remaining), speed selector, and isolated stem auditioning bus.

## Related pages
- [Architecture](../architecture.md) | [Session Workspace](session-workspace.md)
- [Backend & API](backend-api.md) | [Roadmap (v2)](../roadmap.md)
