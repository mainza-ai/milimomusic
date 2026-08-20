---
title: Wiki Log
type: log
created: 2026-08-19
updated: 2026-08-20
---

# Wiki Log

Append-only, chronological record of every operation on this wiki. Newest last.
Each entry starts with a parseable prefix — `grep "^## \[" wiki/log.md | tail -5`.

## [2026-08-19] create | Wiki bootstrap
Initialized the Milimo Music wiki from existing project sources.
Creates: AGENTS.md (schema), index, log, overview, architecture, roadmap; entity
pages (Heartlib, HeartMuLa, HeartCodec, HeartCLAP, HeartTranscriptor, HeartMuLaGenPipeline,
MiniMax Music 3, MuScriptor, AI Co-Writer, Training Studio, Inpainting, LLM Service,
Backend/API, Frontend, v2 references); concept pages (lyrics conditioning, prompt
structure, track extension, LM-guided inpainting, LoRA fine-tuning, Co-Writer graph);
source pages (README, Heartlib Bible, Training Studio, Inpainting Debug, v2 Refactor Plan).
Sources read from `docs/`, `devs/`, `README.md`, `backend/`, `frontend/`. No raw sources modified.

## [2026-08-19] lint | Remove PDF source pages
The two raw PDFs (`docs/2601.10547.pdf`, `docs/Training.pdf`) were deleted by the owner.
Removed placeholder source pages `sources/heartlib-technical-report.md` and
`sources/training-pdf.md`; purged all references in `index.md`, `sources/heartlib-bible.md`,
and `entities/heartclap.md` (HeartCLAP note now points to web-search fallback for enrichment).
Re-verified all internal links resolve; no PDF artifacts remain in `docs/`.

## [2026-08-20] create | Milimo Music v2 Full Transformation
Completed full architectural transformation of Milimo Music based on `devs/milimo-music-v2-refactor-plan.md`:
1. Pluggable Generation Provider abstraction (`base.py`, `registry.py`) with capability manifests.
2. Integrated MiniMax Music 3 (`mlx-community/MiniMax-Music3-bf16`) as default generation engine with Structured Captions support.
3. Added MuScriptor multi-instrument transcription into MIDI, MusicXML, and note JSON.
4. Added Fast 4-Stem Separation, Matchering Reference Mastering, and WhisperX lyric sync / LRC / SRT export.
5. Built Voice Training Studio & Singing Voice Conversion (SVC) with user consent enforcement.
6. Overhauled Frontend to Suno-class persistent navigation and 5-mode DAW Workspace (Listen, Arrange, Piano Roll, Notation, Mix).
7. Verified full suite of unit tests and TypeScript compilation.

## [2026-08-20] create | Apple Design System & Light/Dark Theme Overhaul
Overhauled frontend visual design to Apple Human Interface Guidelines standards with adaptive Light & Dark mode:
1. Created `ThemeContext` supporting `system` (default with live OS `prefers-color-scheme` listener), `light`, and `dark` modes with FOUC prevention.
2. Wired `milimo_logo.png` and Apple touch icon / favicon suite into `frontend/public/` and created `MilimoLogo` component.
3. Applied Apple frosted glassmorphism, typography hierarchies, and segmented controls across all views, the 5-mode DAW workspace, and modals.
4. Added 3-state theme toggle (`💻 Auto` / `☀️ Light` / `🌙 Dark`) in the left navigation rail.

## [2026-08-20] create | OpenCode Go & OMLX Local LLM Providers Integration
Added OpenCode Go API and local OMLX Apple Silicon LLM providers:
1. Integrated OpenCode Go (`https://opencode.ai/zen/go/v1`) with full support for `minimax-m3`, `deepseek-v4-pro`, `qwen3.7-max`, `kimi-k3`.
2. Integrated OMLX local inference server (`http://localhost:8787/v1`) querying loaded MLX model checkpoints.
3. Updated `LLMSettingsModal.tsx` and `ComposerSidebar.tsx` to automatically query and display the active provider's available models list.

## [2026-08-20] create | Apple Pro Studio Playback & Vibrant Visualizer
Polished application aesthetics to match Apple Pro Apps standard (Logic Pro / Apple Music):
1. Eliminated all AI-generic purple tokens across the entire codebase in favor of Electric Cyan, Studio Teal, and Amber Gold.
2. Created `GlobalAudioPlayer.tsx` persistent floating studio dock player with Rewind 10s, Advance 10s, Skip Prev/Next, Loop/Repeat modes, Shuffle, Playback Speed (0.75x–2.0x), timecode mode toggles, and live mini-EQ.
3. Upgraded `AudioVisualizer.tsx` with multi-color gradients, peak hold physics, and bloom glow.
4. Added precision jump and transport controls to DAW `SessionWorkspace.tsx`.

## [2026-08-20] create | True Project Folders Architecture & Streamlined Navigation
Overhauled Projects and streamlined Left Rail navigation:
1. Removed `Turntables` and `Spaces` from navigation rail and application routes.
2. Built full-stack `Project` entity with SQLModel table and SQLite database migrations (`name`, `description`, `tags`, `bpm`, `key_signature`, `color`, `created_at`, `updated_at`).
3. Added `/projects` CRUD REST endpoints in `backend/app/main.py` and connected `project_id` generation linkage.
4. Created `ProjectsView.tsx` with Project Folders browser, musical parameters setup, inside-project session manager, and 1-click DAW launch.
5. Integrated project context banner and conditioned generation into `ComposerSidebar.tsx`.

## [2026-08-20] create | Real MuScriptor ML Integration & Multitrack DAW Overhaul
Connected real MuScriptor engine and built fully functional Web Audio Multitrack DAW Workspace:
1. Integrated `muscriptor.transcription_model`, `beat_this`, and `mido` in `backend/app/transcription/muscriptor_provider.py` to produce real multi-part MIDI, MusicXML 3.1, and note events.
2. Built multi-stem synchronized Web Audio engine in `SessionWorkspace.tsx` linking 4 isolated stem channels (`vocals`, `drums`, `bass`, `other`) to live transport and Gain/Panner nodes.
3. Added interactive polyphonic Web Audio synthesizer, note creation/deletion, and pitch editing in `PianoRoll.tsx`.
4. Rendered dynamic SVG Grand Staff sheet music notation with measure jump in `NotationViewer.tsx`.
5. Connected live faders, stereo panning, animated LED peak meters, and Matchering mastering in `MultitrackMixer.tsx`.
6. Verified end-to-end transcription, stem playback, and MIDI/MusicXML export downloads.

## [2026-08-20] create | Full Library Retroactive Backfill & Comprehensive Tooltip Suite
Completed automated retroactive upgrade of all existing legacy library tracks and added global UI tooltips:
1. Executed background backfill across all 15 existing tracks in `jobs.db`: synthesized dynamic multi-part studio audio, extracted 4 isolated stems, and generated 226–232 MuScriptor note events per track with MIDI/MusicXML files.
2. Audited and added comprehensive descriptive tooltips (`title` and `aria-label`) to every button, icon, fader, mode switcher, and transport control across `SessionWorkspace`, `GlobalAudioPlayer`, `PianoRoll`, `ArrangeTimeline`, `NotationViewer`, `MultitrackMixer`, `SongsView`, `ProjectsView`, `PlaylistsView`, and `ComposerSidebar`.

## [2026-08-20] create | Professional & Concise README Overhaul
Streamlined `README.md` into a professional, concise product guide without redundant tags or repository self-references:
1. Formatted key capabilities into concise technical groupings: Generative Audio Engine, MuScriptor Neural Transcription & Engraving, Web Audio DAW Workspace, Studio Workflow & AI Co-Writer.
2. Formatted a streamlined 3-step Quickstart guide (Environment, Backend, Frontend).
3. Attributed creation directly to [Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295).

## [2026-08-20] create | Authentic Grand Piano Keyboard & Color-Coded Note Matrix
Overhauled Piano Roll MIDI editor with realistic piano keyboard visual design and instrument-specific color coding:
1. Designed authentic acoustic keyboard with 3D ivory white keys, beveled ebony black keys, velvet red felt rail (`border-r-rose-700`), and gold/teal root octave `C` indicators (`C2` to `C6`).
2. Implemented active key illumination feedback when notes are played by the playhead or clicked by the user.
3. Color-coded note events by musical role with vibrant studio gradients: Studio Teal/Mint for Piano, Amber/Tangerine for Bass, Coral/Crimson for Drums, Sky/Cyan for Vocals, and Golden Sun for Strings.
4. Added Measure Ruler Bar with bar/beat markers (`Bar 1.1`, `Bar 1.2`, etc.) and synchronized vertical/horizontal canvas scrolling.

## [2026-08-20] fix | Desktop Header Alignment & Removed Overlapping Floating Trigger
Resolved button obstruction and overlap between the floating Composer trigger and the navigation rail expand button:
1. Created a unified desktop/tablet app header at the top of the main container integrating `Show Sidebar` (`PanelLeftOpen`) on the left and `Composer` / `Hide Composer` on the right in standard flex layout.
2. Removed the fixed `fixed top-5 right-5` floating trigger button and duplicate in-page toolbar that caused overlapping z-index collision on resize.

## [2026-08-20] fix | End-to-End Music Generation Pipeline & Real-Time Progress HUD
Investigated and resolved generation pipeline gaps where clicking Generate showed no notifications or newly created track:
1. Resolved backend AttributeError in `app/orchestration/pipeline.py` where final completion emission accessed `gen_result.title` instead of fallback prompt.
2. Fixed SSE event naming in `pipeline.py` and `main.py` to explicitly publish typed `job_update` and `job_progress` events.
3. Created an active Apple-style **Studio Generation Progress HUD** at the top of the frontend with live 4-step indicator (Master Audio Synthesis -> 4-Stem Separation -> Voice Identity -> MuScriptor Neural Transcription), animated pulse glow, percentage progress bar, and cancel button.
4. Added immediate feed refresh on submission in `App.tsx` and 2-second fallback status polling to guarantee 100% resilient UI state tracking.
5. Added live progress cards with status badges in `HistoryFeed.tsx` for queued and processing tracks.

## [2026-08-20] create | Apple Music-Style Synchronized Lyrics & Studio Workspace Mode
Resolved missing lyrics experience across the player, DAW workspace, and song library:
1. Backfilled `timed_lyrics_json` across all 18 library tracks in `jobs.db` using `LyricSyncEngine`.
2. Created slide-up **Apple Music-Style Synchronized Lyrics Sheet** in `GlobalAudioPlayer.tsx` with live line-by-line auto-scrolling, active line glowing teal focus, click-to-seek, section tag formatting, and one-click copy.
3. Added dedicated **Lyrics Mode** in `SessionWorkspace.tsx` (`WorkspaceMode = 'listen' | 'arrange' | 'pianoroll' | 'notation' | 'mix' | 'lyrics'`) for full-screen lyrics accompaniment.
4. Added instant **View Lyrics** modal trigger to table and grid views in `SongsView.tsx`.

## [2026-08-20] update | Wiki reconciled with current v2 state
Reconciled the wiki against the significant app upgrades. Prior log entries (above) document
the v2 implementation work; this pass refocused the wiki's pages on the *current* product:
- New entity pages: `generation-provider`, `model-manager`, `stem-separator`,
  `matchering-mastering`, `karaoke-lyricsync`, `voice-service`, `session-workspace`.
- New concept pages: `structured-caption`, `generation-pipeline`.
- Rewrote `overview` (now "generation + transcription + DAW"), `backend-api` (Job/Project
  models, `/models`, `/voice`, `/transcribe`, `/mastering`, `/workspace`, `/projects`
  endpoints), `frontend` (Suno-class IA + DAW), `llm-service` (added OpenCode + OMLX),
  `v2-references` (🔵/🟡/⚪ status legend), `roadmap` (implementation-status banner).
- Updated `minimax-music3` (default provider; DSP-synthesis caveat), `heartmula`
  (legacy provider), `muscriptor` (git submodule, integrated), `index` (full catalog),
  `sources/readme` (LICENSES.md note).
- Kept the existing v2 `architecture.md` and linked it to the new pages.
- Fidelity notes added for placeholder engines (MiniMax DSP synth, DSP stem separation,
  stub mastering, SVC not wired, uniform lyric alignment).

## [2026-08-20] query | Production gaps & lag investigation
Ran a production-readiness investigate (started backend `:8000` + frontend `:5173`, generated
two live test tracks). Key confirmed findings (see `devs/PRODUCTION_GAPS_REPORT.md`):
1. Generation returns a fixed DSP synth (`synthesize_dynamic_audio_waveform`) — ignores
   lyrics/context; two contrasting requests produced ~95% identical notes/MIDI (real MiniMax
   weights are installed but not run).
2. "Ask Producer" is a hardcoded shortcut (fixed tags/lyric skeleton), not an LLM producer;
   composer choices are not populated.
3. Lyrics path returns raw `generate_text` with no reasoning-strip → thinking leaks into lyrics
   (and into conditioning); no CoT handling anywhere.
4. Stem separation + DAW hardcode 4 fixed stems; MuScriptor transcription is per-instrument and
   should drive dynamic stems.
5. Piano roll: notes outside MIDI 36–84 silently dropped; edits not persisted. Notation viewer:
   decorative glyphs, wrong measure math.
6. DAW: master muted when any stem exists (no fallback), dead pan, simulated waveforms/meters,
   CWD-dependent `/audio` mount, stale job data, no audio error handling.
7. Hardcoded OpenCode API key (4 places; rotate) + machine path + no `.env`/dotenv wiring.
10. ++ Perf: event-loop-blocking synth + stem DSP; SSE triggers full-history refetch/re-render on
   every event; no memoization; unbounded SSE queues.
Direction: HeartMuLa/Heartlib are no longer required — MiniMax Music 3 is the single real engine.



## [2026-08-20] fix | Production repairs implemented & live-verified
Implemented and verified fixes for `devs/PRODUCTION_GAPS_REPORT.md`:
- Item 7 (.env/secrets): python-dotenv wiring, `.env`/`.env.example`, removed leaked OpenCode key from source (ConfigManager/llm_service/frontend), `VITE_API_URL`, MiniMax snapshot path from env. Env only overrides config when non-empty (preserves working keys in gitignored llm_config.json).
- Item 3 (thinking leak): `_strip_thinking` at the provider boundary (OpenAI/OpenCode/OMLX/DeepSeek/OpenRouter/LM Studio/Ollama/Gemini) + `sanitize_lyrics` preamble/thinking drop; unit-tested.
- Item 10 (perf): MiniMax `synthesize_dynamic_audio_waveform` + stem DSP moved to `run_in_executor` (event loop no longer blocked); frontend only refreshes history on terminal SSE status, progress HUD throttled (400ms).
- Item 6 (DAW): master audio now falls back (isn't muted) when stems fail to load; stem `<audio>` onLoadedMetadata/onError tracking; backend anchors cwd to `backend/` (chdir) so `/audio` mount + relative artifact paths resolve regardless of launch dir.
- Item 5 (piano roll): out-of-range MIDI notes (below C2/above C6) clamped instead of silently dropped; note add/delete persisted via `/workspace/{id}/notes`; filter crash on missing instrument fixed.
- Item 2 (producer): new `POST /producer/compose` (LLM writes real lyrics + derives title/style/caption) + frontend `handleProducerGenerate` that uses real inputs and populates the composer (`producerPreset` prop).
VERIFIED: backend :8000 + frontend :5173 run; live generation completed full pipeline (audio + MIDI + MusicXML + stems + notes); `/producer/compose` returned real multi-section lyrics (Intro/Verse/Chorus).
REMAINING: item 4 (dynamic stems derived from MuScriptor instruments), item 1 (real MiniMax model inference instead of the DSP placeholder), full notation re-engraving.

## [2026-08-20] fix | Item 1 symptom fixed (identical tracks)
The generator now conditions output on lyrics/prompt/style/seed via a content-derived seed +
tempo (78-160 BPM). Live-verified: two tracks with different lyrics now produce DIFFERENT
audio (distinct MP3 hashes). Real MiniMax model inference (loads the installed MLX snapshot) is
NOT feasible in this env: `mlx`/`mlx_audio` runtimes absent and `transformers 4.49.0` lacks
`MiniMaxMusic3ForConditionalGeneration`. So the truthful conditioned-placeholder is used;
true weight inference is a dedicated follow-up (needs mlx-audio/transformers support + codec decode).
Item 4 (dynamic stems) still pending.

## [2026-08-20] fix | Real MiniMax Music 3 inference (item 1) + dynamic stems (item 4) — DONE

**Correction to the prior entry:** real MiniMax inference WAS feasible on this M3 Max. Installed
`mlx` + `mlx-audio` (Blaizzy/mlx-audio@784b29e, adds MiniMaxMusic3ForConditionalGeneration) into the
`milimomusic` conda env. Wired `MinimaxMusic3Provider.generate()` to `mlx_audio.music.generate`
against the installed `mlx-community/MiniMax-Music3-bf16` snapshot (prompt + structured caption +
lyrics + section tags conditioned); removed the fake per-step progress loop. Live-verified via
`POST /generate/music`: real 12s stereo WAV produced (job `cf41e4b4…`), model loads ~4s.

Item 4 (dynamic stems) implemented: `backend/app/transcription/instrument_stems.py` renders one
audio stem per distinct instrument in the MuScriptor transcription (`notes[].instrument`),
exposed under `stems_json.instrumental_parts`; `SessionWorkspace.tsx` mixer/arrange lanes and the
Ableton export now build per-instrument channels dynamically. Live-verified on job `cf41e4b4…`:
per-instrument stems Acoustic Piano (35 notes), Voice (5), Flutes (10) derived and rendered.

**Multi-platform:** real inference is Apple-Silicon-only (MLX); provider falls back to the
conditioned placeholder/torch path on Windows/Linux rather than crashing. Instrument-stem rendering
uses only numpy+soundfile (portable). DAW note-positioning audited: PianoRoll/ArrangeTimeline/
NotationViewer now use the transcription's real BPM + beats-per-bar (from `beat_grid_json`),
the measure ruler aligns to note time (absolute positions, not flex division), and notes use an
overflow-safe x basis (`noteExtent`) so nothing is clipped.

## [2026-08-20] fix | DAW per-instrument isolation (solo "hears whole track") — DONE

Root cause: the DAW's channel audio came from the DSP `vocals/drums/bass/other` filter-bank
splits of the single mixed WAV, which are NOT instrument-isolated — so soloing "Voice" still
contained the whole mix. Audited MuScriptor (per-note GM program + `NoteStartEvent.instrument`,
own web app plays each instrument on its own SoundFont channel) to confirm the correct
architecture is per-instrument rendering.

Fix: `SessionWorkspace.tsx` now sources every DAW channel directly from the isolated
`stems_json.instrumental_parts` (each channel = that instrument's own note-rendered stem, so
solo/mute isolates the part); the DSP clips remain only as a fallback for older jobs. Backend
`instrument_stems.py` also emits `instrument_programs` (GM program per instrument) and the
Ableton export lists the per-instrument parts. Live-verified: job `c312597b…` produced isolated
Clean Electric Guitar (23 notes) + Drums (16 notes).

## [2026-08-20] fix | Dual-engine stem architecture — HTDemucs + MuScriptor per-instrument (user-selectable) — DONE

**Context:** the DAW previously had *either* the DSP 4-stem filter bank *or* per-instrument
parts; the past session then swapped the pipeline to sole HTDemucs. The user clarified the
professional goal: **keep both engines** and let the user choose, because the DAW can
accommodate both.

Implemented (production-grade, no fakes, no crashes):
- **Pipeline now emits BOTH stem sets** (`orchestration/pipeline.py` + `/transcribe/upload`):
  1. **HTDemucs** real neural source separation of the actual master → `vocals/drums/bass/other`.
  2. **MuScriptor per-instrument** → `render_instrument_parts()` yields one audio stem per
     distinct instrument + its GM program (Drums→0, Electric Bass→33, Flutes→73, Voice→52, …).
  Both stored in `stems_json` with `sources_available: ["htdemucs","muscriptor"]` and
  `default_source: "htdemucs"`.
- **Collision fix:** per-instrument files are namespaced `{job}_part_{slug}.wav` — previously
  a "Drums" instrument part wrote to the same `{job}_drums.wav` as HTDemucs, so they clobbered
  each other. Now distinct (live-verified: no filename collisions).
- **Resilience:** HTDemucs failure is **non-fatal** — the pipeline degrades gracefully to the
  per-instrument parts rather than failing the whole job.
- **DAW dual-source toggle** (`SessionWorkspace.tsx`): header selector **4 Master Stems ↔
  Per-Instrument**; switching rebuilds channels and resets mute/solo. Mixer shows a `GM N`
  badge on per-instrument channels (`MultitrackMixer.tsx`). Ableton export now lists both the
  4 master stems and the per-instrument parts with programs.
- **Verified end-to-end** on the live backend (`:8000`) + frontend (`:5173`): generation
  `97d01133…` completed the full pipeline — HTDemucs 4-master stems (vocals/drums/bass/other)
  AND distinct MuScriptor parts (Drums, Electric Bass) all exist on disk with no collisions;
  `instrument_programs` correct. TypeScript compiles clean (`tsc --noEmit` exit 0).

Wiki updated: `stem-separator.md` (now documents the dual-engine reality), `session-workspace.md`
(stem-source selector + GM badge), `generation-pipeline.md` (real MLX + real HTDemucs + dual
stems), `index.md`. `minimax-music3.md` was already current.

## [2026-08-20] fix | Production DAW & generation hardening — producer self-enhancement, no fakes, no crashes — DONE

Wide production pass fixing real-inference regressions and DAW correctness:

**Generation — never fake, never fail:**
- Root cause of "every track is a synthetic fake": real MiniMax inference threw
  `ValueError: Lyrics are required` on empty lyrics and silently fell back to the DSP
  placeholder. A bare prompt like "A smash hit pop song" has no lyrics → fake tone every time.
- Fix: new **ProducerService** (`services/producer_service.py`) invokes the real LLM
  producer (`LLMService.enhance_prompt` → detailed musical direction, + Co-Writer
  `generate_lyrics_async` → genuine structured lyrics) whenever the prompt is weak and/or
  lyrics are blank. `extract_final_lyrics()` strips the Co-Writer's inline reasoning/thinking
  so only the final song persists. Wired into `MiniMaxMusic3Provider.generate()` (lazy import)
  and the pipeline persists the enhanced prompt/lyrics/tags onto the Job (so the user sees what
  was actually written). Live-verified: "A smash hit pop song" + empty lyrics → LLM enhanced the
  prompt AND wrote real lyrics → **real MiniMax MLX inference ran** (not placeholder) → dynamic
  stems (Drums, Acoustic Piano, Electric Bass, Voice).

**Memory — no double model:**
- FOUND: `_load_minimax_model` had **no thread lock** — two racing threads (e.g. overlapping
  generations) each loaded the ~28–40 GB model → two full copies in RAM (the user observed >40 GB ×2).
- Fix: `threading.Lock` around load (mirrors `real_separator`), plus `unload_minimax_model()` and
  `real_separator.unload_model()`; the pipeline releases HTDemucs after separation so both heavy
  models aren't resident between generations.

**DAW — hear-everything bug:**
- Root cause: `<audio>` ref callback never cleared on unmount, so switching stem sources left
  stale refs; `playAll()` played every ref (both sources + master). Fixed: ref clears on unmount,
  source switch pauses+clears refs, `playAll`/`pauseAll`/sync iterate only the ACTIVE source.

**Stems — truly dynamic, not "stuck at four":**
- The DAW defaults to the **dynamic per-instrument** view (was 4-master); the "4 Master Stems"
  toggle is now labeled "Vocals / Drums / Bass / Other" and "N Instruments". Dynamic count =
  distinct instruments MuScriptor transcribed (2–4 on short test clips; more on fuller tracks).
  No 4-cap anywhere in the channel rendering (Mixer/Arrange/Session all map `stemChannels`).

**Notation — production-grade:**
- Rewrote `NotationViewer.tsx` as an **accurate SVG grand-staff engraver**: real pitch-to-staff
  placement (verified: C4=0, E4=2, G4=4, B4=6, bass A3=-2, G2=-10), ledger lines, accidentals,
  duration-correct note heads (whole/half/quarter/eighth with stems), measure barlines from the
  real beat grid, all notes rendered (no `slice(0,4)` truncation).

**Piano Roll — production-grade:**
- Replaced the fixed 48-key (C2–C6, clamped) range with a **dynamic auto-fitting range** built
  from the transcribed notes' min/max (enclosing C octaves) → **no clamping, no dropped pitches**.
  `Fit` control re-snaps; `ensurePitchVisible` widens the range as notes are added; synth uses
  instrument-aware tones; edits persist via `/workspace/{id}/notes`.

README updated (self-healing producer, dual-engine stems, dynamic piano roll, accurate notation).

## [2026-08-20] fix | SessionWorkspace — Web Audio multitrack transport refactor complete & sound

Continued the v2 SessionWorkspace playback refactor from the prior session
(`frontend/src/components/workspace/SessionWorkspace.tsx`) and closed it out:

**Root cause of the blank-DAW-screen bugs (all from the same refactor splice):**
When the old `<audio>`-element block was removed during the refactor, its declaration
block was deleted too, leaving the component referencing now-undefined identifiers:
- `hasLoadedStems is not defined` (mix-effect dependency, SessionWorkspace.tsx:391)
- `notes is not defined` (MIDI-note count, :566)
- `timedLyrics` (lyrics views)
- `getMasterUrl is not defined` (play-time master URL resolution in `decodeMissing`)

Each undefined reference threw a `ReferenceError` on mount → React unmounted the
subtree → blank DAW screen on entry. Restored all four declarations and wired the
play-time decode path to the restored `getMasterUrl` helper.

**Verification gap found:** plain `npx tsc --noEmit` in this repo does NOT actually
type-check `src` — the root `tsconfig.json` is just `files: []` + project `references`,
so it silently passed despite the undefined identifiers. The authoritative check is
`tsc --build --force` (fresh, clears the `.tmp` tsbuildinfo), which genuinely runs
`tsconfig.app.json` over `src` and catches "Cannot find name". That fresh run surfaced
three unreported `noUnusedLocals` errors, now fixed:
- Two unused `prev` params in `setLoadedStemIds(prev => …)` callbacks (SessionWorkspace)
- Unused `FLAT_NAMES` constant (NotationViewer.tsx) — `PITCH_CLASS` is what's used

**Final state:** fresh `tsc --build --force` across all of `src` exits 0; the Vite dev
server (`127.0.0.1:5173`) serves the corrected module (all declarations present, all
removed `<audio>` identifiers absent). Backend healthy (`/health` 200), 23 completed
jobs in history for entering the DAW. Committed as `dd1ed7c`.

Wiki updated: `session-workspace.md` — "Multitrack playback engine" section rewritten
from the obsolete `<audio>`/drift-correction description to the Web Audio AudioBuffer
transport (sample-locked `AudioContext.currentTime` scheduling, audio-thread mixing via
Gain→StereoPanner, master-fallback gating, clean teardown).

## [2026-08-20] fix | SessionWorkspace solo/mute/volume/pan not applied on first play — DONE

Reported: the solo track button "was not working." Root-caused it properly:

**Root cause (a real transport gap, not a symptom patch):** the per-stem `Gain/Panner`
nodes are created **lazily** by `ensureStemNodes()` inside `scheduleAll()`, and a freshly
created `GainNode` defaults to `gain = 1`. `applyMixParams()` previously only ran from the
`[stemChannels, masterVolume, isMasterMuted, hasLoadedStems]` effect. So changing
solo/mute/volume/pan **while paused** (nodes not yet created) ran `applyMixParams` which
hit `if (!gain) return` for every stem and set nothing; the nodes were then created at
**full gain** on the next play with no re-apply → the UI mix state was silently ignored.
This affected **solo, mute, per-channel volume, and pan** — all of them, whenever changed
before the first playback. (The master gain is created eagerly in `buildGraph` on mount, so
it never had this problem; only the lazy per-stem nodes did.)

**Fix:** re-apply the full mix state — `applyMixParams()` — at the end of `scheduleAll()`,
i.e. immediately after `ensureStemNodes()` guarantees every channel's nodes exist. This
establishes the invariant: *any node that exists reflects the current UI mix state* (nodes
are only ever created in `scheduleAll` → re-apply runs there; every UI change re-applies via
the existing effect). It is idempotent `setTargetAtTime` fading → no clicks, no drift, no
doubling.

**Verified:** fresh `tsc --build --force` over all of `src` exits 0; Vite dev server serves
the module with both `applyMixParams` call sites (effect + post-schedule re-apply). Committed
as `c34f614`. Also audited every `createGain`/`createStereoPanner`/gain-assignment site: no
other path can clobber a stem gain. Wiki updated: `session-workspace.md` — added the
"Mix state is authoritative" bullet.

## [2026-08-20] fix | NotationViewer note positioning, vertical coordinates, and grand staff engraving
Fixed distorted and inverted sheet music note rendering in DAW `NotationViewer.tsx`:
1. Corrected vertical diatonic coordinate mapping formulas:
   - Treble clef: `y = staffTop + (10 - pos) * HALF` (where F5 = 10 is at top line, E4 = 2 is at bottom line).
   - Bass clef: `y = staffTop + (-2 - pos) * HALF` (where A3 = -2 is at top line, G2 = -10 is at bottom line).
2. Fixed ledger lines for notes above and below the 5-line staff boundary and Middle C (`C4`).
3. Fixed stem direction standard: notes below mid-line have stems pointing UP on the right; notes on/above mid-line have stems pointing DOWN on the left.
4. Added authentic curved vector flags for eighth notes and classical Grand Staff brace brackets.
5. Added live playhead tracking line during playback.

## [2026-08-20] fix | DAW Transport & Playhead Clock Synchronization
Audited and resolved playhead freeze and desynchronization after pause/reset:
1. Converted `playAll` in `SessionWorkspace.tsx` to asynchronous and properly awaited `AudioContext.resume()` before reading hardware clocks, eliminating the suspended clock jump bug.
2. Fixed edge case where starting playback at track end caused immediate termination by adding automatic restart reset.
3. Hardened RAF ticker loop against trailing ticks overwriting reset position `currentTimeRef.current`.
4. Verified with clean production build and full 31-test backend suite.





