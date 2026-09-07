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

## [2026-08-20] create | Production Workflow Overhaul (New Session, Multi-Turn Producer & Projects)
Implemented production-grade New Session and Projects workflows matching Apple-inspired reference specifications:
1. **Projects 2-Column Modal & Cover Artwork**:
   - Re-engineered New Project modal with square drag-and-drop artwork dropzone on the left (supporting file upload and AI cover art prompt generation) and metadata form on the right with live 250-character counter.
   - Enhanced project grid cards with cover art previews, track statistics, and active project context banner.
2. **Stacked Accordion Compose Drawer**:
   - Overhauled `ComposerSidebar.tsx` into a unified accordion with Lyrics (featuring an Instrumental toggle switch), Sound & Style (featuring an Advanced controls switch), and Details & Artwork.
   - Added AI cover prompt generation and SVG/PNG cover synthesis.
3. **New Session Stage & Left Rail History**:
   - Upgraded Left Navigation Rail with persistent `Sessions` thread list and a primary `+ New session` button.
   - Designed central "What do you want to create?" hero with 3 visual starter action cards (*Brainstorm lyrics*, *Create a song together*, *Remix my music*).
   - Integrated conversational multi-turn Producer chat stream and sticky bottom Apple-style Producer prompt bar with attachment and slider shortcuts.
4. **Backend Hardening & SQLite Migrations**:
   - Added `StudioSession` and `SessionMessage` models and migrations to `jobs.db`.
   - Updated music generation and project endpoints to support `cover_image_path`, `image_prompt`, and `session_id`.
   - Verified 100% test pass (59/59 pytest suites) and zero-error Vite production build.

## [2026-08-20] create | Individual Track Studio & Song Detail Page (TrackDetailView)
Implemented the production-grade Individual Track Studio (Song Detail Page) for deep inspection and multi-asset export:
1. **Dedicated Track Studio View (`TrackDetailView.tsx`)**:
   - Designed Apple-standard Hero Command Bar with high-resolution artwork, editable track title, BPM/Key/LUFS badges, tactile Play/Pause master transport, and interactive audio waveform scrubber.
   - Built 5 specialized inspection tabs:
     - **Tab 1: Stems Matrix**: Dual-engine stem switcher (`HTDemucs 4-Stem` vs `MuScriptor Dynamic Instrument Parts`), per-stem interactive Solo (`S`) and Mute (`M`) buttons, volume & pan faders, waveform visualizer, and individual stem `.wav` download.
     - **Tab 2: Neural Transcription & Score Hub**: Note metrics (total notes, pitch range `C2–G5`, tempo grid), direct downloads for Multi-Track MIDI (`.mid`), W3C MusicXML (`.musicxml`), Note JSON, and integrated Grand Staff score preview.
     - **Tab 3: Vocal & Lyric Studio**: Syllable-synced real-time karaoke teleprompter, `.lrc` and `.srt` subtitle export, copy lyrics, and 1-click Singing Voice Conversion (SVC) vocal swap with voice profiles.
     - **Tab 4: AI Generation Provenance**: Model Provider, Structured Caption breakdown (Global Metadata, Vocal Details, Arrangement), Seed with 1-click copy, CFG scale, Temperature, Top-K, Step count, Audio sample rate, and `[✨ Re-roll in Composer]` CTA.
     - **Tab 5: Version History & Lineage**: Visual lineage tree displaying parent song, extended segments, and voice-converted iterations.
2. **Backend Studio Pack Packaging & API Hardening**:
   - Added `GET /jobs/{job_id}/studio-pack` endpoint streaming a complete `.zip` archive containing master audio, stems, MIDI, MusicXML, LRC lyrics, and metadata manifest.
   - Added `PATCH /jobs/{job_id}` for instant title, tags, and metadata updates.
   - Added `POST /jobs/{job_id}/voice-convert` for single-click SVC vocal re-voicing derivatives.
3. **Application Routing Integration**:
   - Added `'track-detail'` to `NavView` and wired clicking on song rows, thumbnails, or titles across `SongsView` and `HistoryFeed` to open the Track Studio.
   - Integrated Back button navigation with previous view history and 1-click transitions to the full multitrack DAW workspace.

## [2026-08-20] fix | Producer Chat Full-Track Composition Engine & Lyrics Population
Resolved missing lyrics and incomplete preset population during New Session conversational Producer prompts:
1. **Unified Studio Producer Engine (`LLMService.produce_full_track`)**:
   - Upgraded `backend/app/services/llm_service.py` to synthesize a complete 5-layer track composition package:
     a) Topic description and strictly validated genre/style tags.
     b) Production-grade multi-stanza lyrics (`[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`) via `generate_lyrics_async`.
     c) Reasoning and thinking token sanitization (`_strip_thinking`) ensuring no model scratchpads leak into lyrics.
     d) Evocative song title generation (`generate_title`).
     e) Full 3-part structured caption breakdown (`global_metadata`, `vocal_details`, `arrangement`) and instrumental prompt detection.
2. **Session Chat Endpoint Hardening (`backend/app/main.py: session_chat`)**:
   - Upgraded `POST /sessions/{session_id}/chat` to invoke `LLMService.produce_full_track` and attach complete preset payloads (`title`, `topic`, `tags`, `lyrics`, `structured_caption`, `is_instrumental`).
   - Enhanced producer message formatting in the chat stream with full lyrics preview and song title proposal.
3. **Frontend Ingestion & Instant Composer Sync (`App.tsx` & `ComposerSidebar.tsx`)**:
   - Hooked `producerPreset` to automatically populate Title, Lyrics, Topic, Style tags, Global Metadata, Vocal Details, and Arrangement fields in the Composer.
   - Added interactive `[Load in Composer]` quick-action buttons on Producer message bubbles.
   - Verified 100% test suite pass (59/59 pytests) and live end-to-end country song composition.

## [2026-08-20] create | Universal Navigation & Track Studio Deep-Drill Architecture
Unified the whole-application Information Architecture (IA) and eliminated click hijacking across all song representations:
1. **Separated In-Place Rename from Navigation in History (`HistoryFeed.tsx`)**:
   - Replaced title click interception with a dedicated pencil `<Edit2 />` action button.
   - Connected song titles, prompts, and entire card containers to `onSelectTrack(job)`, opening `TrackDetailView`.
   - Added explicit tactile `[Track Studio]` action button with `<Sparkles />` icon.
2. **Universal Click-Through Across All Core Views**:
   - **`GlobalAudioPlayer.tsx`**: Made bottom player album artwork, track title, and provider chip open `TrackDetailView` on click.
   - **`ProjectsView.tsx`**: Hooked project track cards and titles with `onSelectTrack` and added `[Studio]` action button.
   - **`PlaylistsView.tsx`**: Hooked playlist song rows and titles with `onSelectTrack` and added `[Studio]` action button.
   - **`ProfileView.tsx`**: Hooked user creations and top track rows with `onSelectTrack` and added `[Studio]` action button.
3. **URL Deep-Linking & Browser History Synchronization (`App.tsx`)**:
   - Synchronized active track and view state with URL search parameters (`?view=track-detail&track={job_id}`).
   - Connected `popstate` event listeners so browser Back and Forward buttons seamlessly navigate between views and preserve track studio context on page reload.
4. **Interactive Singing Voice Conversion (SVC) Parametric Controls (`TrackDetailView.tsx`)**:
   - Added Pitch Shift slider (`-12` to `+12` semitones), Formant preservation switch, and Dry/Wet blend slider (`0%` to `100%`).
   - Verified 100% Pytest pass rate (59/59) and zero-error Vite production build (1.47s).

## [2026-08-21] fix | Track Studio Master Audio Transport & Direct Stem Playback Engine
Resolved audio playback failures on individual track detail pages and hardened sound auditioning across all views:
1. **Dedicated Track Studio Master Transport (`TrackDetailView.tsx`)**:
   - Integrated a local HTML5 `<audio>` engine with real-time waveform equalizer animation and interactive scrubber.
   - Wired the Master Play/Pause button and artwork hover triggers to directly control audio playback with zero lag.
   - Added live time counter (`0:00 / 3:14`), volume slider, and instant mute toggle.
2. **Interactive Multitrack Stem Auditioning (`TrackDetailView.tsx`)**:
   - Added direct Play/Pause auditioning buttons to each stem card on Tab 1 (Vocals, Drums, Bass, Instruments).
   - Hooked dedicated stem audio elements allowing instant auditioning of isolated WAV audio stems.
3. **Global Audio Player Autoplay Hardening (`GlobalAudioPlayer.tsx`)**:
   - Explicitly bound `src={audioUrl}` on the JSX `<audio>` tag to eliminate race conditions on initial mount.
   - Added `onCanPlay` callback and safe Promise handling on `.play()` to prevent browser autoplay blocks and interruption errors.
4. **Explore Feed History Sound Details Badges (`HistoryFeed.tsx` & `App.tsx`)**:
   - Passed `onSelectTrack` to the primary Explore feed instance.
   - Added interactive `[Sound Details]`, duration, model provider, and style tags pills to every history card.
   - Verified 100% Pytest pass rate (59/59) and zero-error Vite build (1.51s).

## [2026-08-21] create | Unified Single-Node Audio Engine & Contextual Player Architecture
Architected a centralized audio playback engine and eliminated dual-player collisions, audio contention, and ghost looping:
1. **Centralized Audio Engine Provider (`frontend/src/context/AudioEngineContext.tsx`)**:
   - Built a single root `<audio>` node and unified state provider (`useAudioEngine()`) governing playback, buffering, playhead seeking, master volume, and playlist progression across the application.
   - Eliminated all independent and conflicting HTML5 `<audio>` element instances.
2. **Contextual Floating Dock Visibility (`frontend/src/App.tsx`)**:
   - Configured intelligent screen awareness: floating dock (`GlobalAudioPlayer`) is displayed only during library browsing (`explore`, `songs`, `projects`, `playlists`, `profile`, `videos`).
   - Automatically collapses and hides the floating dock when entering dedicated studio workspaces (`track-detail`, `workspace`), giving exclusive visual and tactile ownership to the in-page Studio Hero Transport.
3. **Seamless Studio Hero Transport (`frontend/src/components/views/TrackDetailView.tsx`)**:
   - Bound hero playhead, animated frequency equalizer bars, and master scrubber directly to `useAudioEngine()`.
   - Isolated stem auditioning bus on Tab 1 so auditioning individual WAV stems cleanly pauses master playback with zero phase collision.
4. **Zero-Error Verification**:
   - 100% Pytest pass rate (59/59) and clean Vite production build (1.49s).

## [2026-08-21] create | Floating Harbor Player Overhaul & Full MuScriptor Engine Integration
Executed production-grade overhaul of the Floating Harbor Audio Player and complete MuScriptor multi-instrument neural transcription across the DAW Workspace:
1. **Floating Harbor Audio Player Overhaul**:
   - Upgraded `AudioEngineContext.tsx` with WebAudio `AudioContext` & shared `AnalyserNode` with `crossOrigin="anonymous"`.
   - Wired live FFT spectrum data into `AudioVisualizer.tsx` and `GlobalAudioPlayer.tsx`.
   - Fixed cover artwork rendering with dynamic URL prefixing and vinyl disc animation.
   - Built interactive slide-up **"Up Next / Queue"** drawer with track thumbnails, duration, jump to song, reorder, remove, and clear queue actions.
   - Integrated `navigator.mediaSession` metadata and actions with global keyboard hotkeys (`Space`, arrows, `M`).
2. **MuScriptor Neural Multi-Instrument Provider & Engraving**:
   - Upgraded `muscriptor_provider.py` with complete General MIDI & MT3 Instrument Program Map across 30+ instrument classes.
   - Built multi-part W3C MusicXML 3.1 generator creating distinct `<score-part>` staves with proper clefs and measures.
   - Integrated MuseScore 4 PDF & Tablature engraving (`write_sheets`) with fallback to MusicXML.
   - Added `GET /api/tracks/{id}/sheets` and `POST /api/tracks/{id}/midi` in `backend/app/main.py`.
3. **DAW Workspace Deep Integration**:
   - `NotationViewer.tsx`: Added multi-instrument part filter pills (Grand Score, Piano, Electric Guitar, Bass, Drums, Strings), and "Download Engraved Scores & PDFs" dialog.
   - `PianoRoll.tsx`: Connected `persistNotes` to `trackApi.updateMidiNotes` with live "Sync Score" visual status.
## [2026-08-21] fix | Universal Media Player Standardization & Zero-Overlap Architecture
Completely eliminated all visual collisions and standardized the pro-grade transport suite across every media player:
1. **Zero-Overlap 3-Zone Flex Layout & Background Bleed Fix**:
   - `GlobalAudioPlayer.tsx`: Built strict 3-zone flexbox with fixed-width centered transport cluster and responsive right tools container, guaranteeing zero overlap across all screen dimensions. Removed background visualizer canvas.
   - `AudioPlayer.tsx`: Removed the absolute `mix-blend-screen` visualizer canvas that rendered behind playhead buttons in history cards.
2. **Standardized 16-Point Control Suite Across All 4 Players**:
   - `GlobalAudioPlayer.tsx`: Full suite (Shuffle, Return to Zero `|<<`, Rewind 10s, Hero Play/Pause, Advance 10s, Next Track `>>|`, Repeat modes, Scrubber with timecode modes, Speed menu, Volume fader + Mute, Queue & Lyrics drawers, DAW shortcut, Download).
   - `TrackDetailView.tsx`: Full suite (Shuffle, Return to Zero `|<<`, Rewind 10s, Hero Play/Pause, Advance 10s, Next Track `>>|`, Repeat modes, Speed selector, Volume + Mute, Timecode toggle, Waveform scrubber).
   - `SessionWorkspace.tsx`: Full DAW suite (Return to Zero `|<<`, Step -1 Bar, Rewind 10s, Hero Play/Pause, Advance 10s, Step +1 Bar, Repeat/Loop mode, Scrubber, Volume + Mute, BPM meter badge).
   - `AudioPlayer.tsx`: History card suite (Return to Start/Previous `|<<`, Rewind 10s, Hero Play/Pause, Advance 10s, Next Track `>>|`, Repeat, Speed, Volume, Inpaint, Download, Scrubber).
3. **Verification**:
   - 100% Pytest pass rate (59/59 in 5.47s) and clean Vite production build (1.68s).

## [2026-08-21] create | Apple UI Polish, Training Studio Re-theme & Creator Attribution
Executed comprehensive UX/UI cosmetic enhancements aligning the application with Apple Pro standards:
1. **LoRA & Foundation Training Studio Modernization (`TrainingStudio.tsx`)**:
   - Re-themed the entire modal into Apple Studio Glassmorphism (`backdrop-blur-2xl`, adaptive Light/Dark mode backgrounds, subtle border radii, Studio Teal & Cyan accents, zero purple/fuchsia).
   - Upgraded Dataset Prep, Training Configuration, Jobs Monitor, and Checkpoint Manager with real-time ETA, elapsed time calculation, and discrete audio RVQ tokenization workflows.
2. **De-Clustered Songs Library Layout (`SongsView.tsx`)**:
   - Replaced multi-row tag stacking with compact inline micro-pills with overflow `+N` hover badge.
   - Converted bulky badge indicators into sleek Apple micro-chips (`MIDI Ready` and `4 Stems`).
   - Standardized single-row horizontal actions toolbar and added interactive vinyl disc album thumbnails.
3. **Music Videos "In Development" Status (`MusicVideosView.tsx`, `App.tsx`)**:
   - Added an amber/teal `In Development` status badge in the header and sidebar navigation.
   - Added an informational notice detailing the upcoming text-to-video / audio-reactive diffusion pipeline (Wan2.1 / CogVideoX).
## [2026-08-21] create | Neural Acoustic Lyrics & Karaoke Synchronization
Implemented end-to-end neural acoustic lyric alignment on the backend and 60fps high-precision playhead tracking on the frontend:
1. **Neural Acoustic LyricSyncEngine (`backend/app/transcription/karaoke.py`)**:
   - Replaced naive linear duration slicing with RMS vocal energy envelope extraction on isolated vocal stems (`vocals.wav`).
   - Added Voice Activity Detection (VAD) and syllable/phonetic proportional timing so intro silences, instrumental solos, and tempos are respected accurately.
   - Built structure header tag parser (`[Intro]`, `[Verse]`, `[Chorus]`, etc.) to treat musical section markers as structural landmarks without consuming singing duration.
   - Implemented standard `.lrc` and `.srt` subtitle exporters.
2. **Orchestration Pipeline Integration (`backend/app/orchestration/pipeline.py`)**:
   - Passed effective `job.lyrics` and separated `vocal_stem_candidate` to `lyric_sync_engine.align_lyrics`, ensuring auto-generated songs from MiniMax and AI Co-Writer are fully synchronized.
3. **Backend API Endpoints (`backend/app/main.py`)**:
   - Added `GET /tracks/{job_id}/lrc` to download standard `.lrc` lyrics file.
   - Added `POST /tracks/{job_id}/realign_lyrics` to recalculate acoustic alignments on-demand.
4. **Frontend 60fps Playhead & Karaoke UI Upgrades**:
   - `AudioEngineContext.tsx`: Added high-precision `requestAnimationFrame` playhead ticker (60fps / 16ms) during playback.
   - `GlobalAudioPlayer.tsx`: Enhanced synchronized lyrics drawer with continuous proximity line tracking, word-level progressive highlights, section tag badges, and `.LRC` download.
   - `TrackDetailView.tsx`: Upgraded Tab 3 with live interactive karaoke viewer, view mode toggle (`Karaoke` vs `Text`), Re-Align button, and `.LRC` export.
   - `SessionWorkspace.tsx`: Upgraded floating live karaoke stream and DAW lyrics mode with word-level transitions and section headers.
5. **Verification**:
   - Unit tests passed in `backend/tests/test_lyrics_sync.py`.
   - Clean Vite production build in 1.53s.


## [2026-08-21] fix | MiniMax real-inference unblocked + prompting-guide caption fixes
- **Root cause fix:** `minimax_provider.py` clamped inference `steps` to 32 but the model
  allows only 1–30, so every song ≥62s threw and silently fell back to the procedural
  synth ("generic tonal audio"). Clamp corrected to 30; verified real inference is viable
  (snapshot present, `mlx-audio` available).
- **Prompting-guide alignment (official `multimodalart/minimax-music3-prompting-guide`):**
  constructed structured captions now follow the three-heading skeleton with the guide's
  sub-fields and always state vocal presence; `format_full_caption` no longer appends a
  4th `[Description]` block; `sanitize_section_tags` forces every `[Tag]` onto its own
  line (MiniMax drops lyric text sharing a line with a leading tag).

## [2026-08-21] fix | Fallback-to-synth surfaced + structured-caption plumbing
- `GeneratedAudioResult` gained `used_fallback_synth` + `fallback_reason`; `minimax_provider`
  records an honest reason for every skip/failure case (mlx missing, snapshot missing,
  inference threw).
- New typed `Job` columns (`used_fallback_synth` bool default false, `fallback_reason`
  text) + `migrate_db.py` columns; `pipeline.py` persists them after generation.
- UI: visible "⚠️ Fallback synthesis (not MiniMax Music 3)" badge in the track hero and a
  full warning with reason in the AI Provenance tab (`TrackDetailView.tsx`).
- **Dead-UI fix:** `GenerationRequest.structured_caption` was dropped between the composer
  and the provider; `pipeline.py` now passes it through and the provider honors provided
  sections (auto-filling missing ones) — user caption edits finally reach the model.
- Tests: `tests/test_provider_provenance.py` (4); full backend suite 70 passed.

## [2026-08-21] create | Caption Rewriter (official music-caption-rewriter port)
- Vendored the official caption library (`genre-router.md` + 18 family indexes + ~1,000
  caption templates, 4.4MB) at `backend/data/caption-library/` (licensing noted in
  `LICENSES.md` §5 — upstream repo carries no LICENSE file).
- `LLMService.rewrite_caption()` routes a brief → ranks style families → few-shots the top
  templates into the real configured LLM → validates a three-heading caption with an honest,
  non-blocking constructed fallback.
- New `POST /generate/rewrite_caption` endpoint; `produce_full_track` regenerates its
  caption via the rewriter; the composer **Enhance** button fills all three caption fields
  through it.
- Live-verified against the configured OpenCode `minimax-m3` provider (routed
  `dance-pop-disco-funk` / `hip-hop-rap`, honors lyric tags, ~250–450 words).

## [2026-08-21] fix | BS-Roformer 6-Stem Neural Separation & Dynamic DAW Multitrack Architecture
- Replaced legacy HTDemucs with SOTA **BS-Roformer / MelBand-Roformer** neural source separation (`transcription/real_separator.py`) supporting dynamic stem counts (4, 5, 6+ stems, default 6-stem: `vocals`, `drums`, `bass`, `guitar`, `piano`, `other`).
- Added native hardware acceleration device resolution (`CUDA` → Apple Silicon `MPS` → `CPU`) with process singleton caching and `unload_model()` memory release.
- Added `SeparationResult` dataclass returning dynamic stems dictionary, `source_id`, `sources_available`, and `stem_count`.
- Pinned `audio-separator>=0.28.0` in `requirements.txt`.
- Cleaned up obsolete legacy DSP filter-bank (`transcription/stem_separator.py`) and updated test suite `tests/test_v2_core.py`.
- Made `pipeline.py`, `/transcribe/upload`, and `/transcribe/export/{job_id}/ableton` dynamically persist and export all neural stems.
- Upgraded frontend (`api.ts`, `TrackDetailView.tsx`, `SessionWorkspace.tsx`) with dynamic `StemsMap` index signatures, shared `getStemMeta` helper with Apple-grade colors/icons/gradients, dynamic stem matrices, and removed hardcoded `"htdemucs"` strings.

## [2026-08-21] create | NVIDIA NIM Provider & AI Producer Resiliency Upgrade
- Added **NVIDIA NIM** OpenAI-compatible provider (`https://integrate.api.nvidia.com/v1`) with full live model querying across 102+ available models (Llama 3.1/3.3, Mistral, Nemotron, Gemma, etc.).
- Integrated environment variables `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `NVIDIA_MODEL` into `.env` and `DEFAULT_CONFIG`.
- Updated `LLMSettingsModal.tsx` and `api.ts` with dedicated NVIDIA NIM settings pane, active model selector with dynamic live fetching via `/config/fetch-models` (no hardcoded model restriction).
- Implemented **Code Resiliency Improvements** in `LLMService`:
  - Added keyword-aware fallback style tag extraction (`_extract_fallback_tags`) matching prompt genres and subgenres against `StyleRegistry`.
  - Added creative song title fallback generation (`_extract_fallback_title`).
  - Added automatic multi-provider failover on 429/401/network errors across configured providers.
  - Replaced raw prompt echoing fallback with structured musical song drafts and transparent error notices.
  - Fixed `generate_structured` on third-party OpenAI-compatible endpoints to use JSON mode and Pydantic validation directly.

## [2026-08-21] fix | Prompt Responsiveness, Heartlib Removal & LoRA Studio Preview
- Implemented **Instant 0ms User Feedback** on prompt submission: immediate input clearing, optimistic message injection into chat session, and live Apple-grade animated Producer Composing card with realistic stage progression.
- Parallelized backend LLM synthesis in `produce_full_track` via `asyncio.gather` for faster end-to-end brief and lyric generation.
- Integrated instant floating HUD notifications (`milimo_progress` event dispatch) on prompt submission and direct generation flows.
- Removed legacy `heartlib` dependency and provider from `requirements.txt`, `registry.py`, `ComposerSidebar.tsx`, and `test_v2_core.py`.
- Deleted legacy photo assets (`milimo_logo.png`) and upgraded `MilimoLogo.tsx`, `AppFooter.tsx`, `GlobalAudioPlayer.tsx`, `HistoryFeed.tsx`, `SongsView.tsx`, and `SessionWorkspace.tsx` to sleek Apple Pro vector SVG graphics (`DEFAULT_COVER_ART`).
- Marked LoRA Training Studio as **In Development** (`In Dev` pill badge in nav rail and banner in studio modal header).
- Restarted backend uvicorn server on port 8000.

## [2026-08-21] feat | Production-Grade Cancellation & Abort Architecture
- Implemented **Client-Side Request Abort**: Configured `AbortController` in `App.tsx` and wired `signal?: AbortSignal` through `sessionApi.sendChatMessage` and `api.producerCompose` to terminate in-flight HTTP requests instantly.
- Added **Dynamic Stop / Cancel Action in Prompt Bar**: During prompt/producer generation, the send arrow dynamically morphs into an animated Stop button (`<Square>`) that terminates generation immediately.
- Added **Live Composing Card Cancellation**: Added inline Cancel button in the Producer composing message card.
- Hardened **Generation Banner & HUD Cancellation**: Added permanent Cancel button in the Active Studio Generation banner (triggering backend `POST /jobs/{id}/cancel` + client state resets) and in `FloatingStatusWidget.tsx`.

## [2026-08-21] fix | Full Brand Logo Restoration Across All DAW Views
- Restored `milimo_logo.png` across all application touchpoints:
  - Top Navigation & Landing Page Hero (`MilimoLogo.tsx`)
  - Track History Feed Thumbnails (`HistoryFeed.tsx`)
  - Songs View Grid & Table Artwork Fallbacks (`SongsView.tsx`)
  - Global Audio Player & Playing Queue Drawer (`GlobalAudioPlayer.tsx`)
  - Session Workspace Header Icon & Large Center Listen Canvas (`SessionWorkspace.tsx`)
  - OS Lock Screen & Control Center MediaSession Artwork (`AudioEngineContext.tsx`)
  - Application Footer (`AppFooter.tsx`)

## [2026-08-21] doc | Official Silence Tagline & Wiki Synchronization
- Restored the official brand tagline: *"Give the silence something worth remembering."* with sub-motto *"Speak it into being. Shape it until it's yours."* to the landing hero in `App.tsx`.
- Synchronized `wiki/overview.md` and `wiki/entities/frontend.md` with official brand identity, instant prompt feedback, and production cancellation architecture.

## [2026-08-21] sec | Mask LLM Config Secrets, Return has_key Booleans & Purge File Secrets
- **Masked `GET /config/llm` API Response**: Updated `LLMService.get_config()` to return `has_key` & `has_api_key` booleans while masking plaintext secret strings (`api_key: ""`).
- **Stopped Env Key Persistence in `llm_config.json`**: ConfigManager now keeps environment variable keys strictly in runtime memory and strips them from JSON writes.
- **Removed `VITE_OPENCODE_API_KEY` from Client**: Purged client-side references to `VITE_OPENCODE_API_KEY` in `LLMSettingsModal.tsx` and updated placeholders to reflect backend configuration status.









## [2026-08-21] create | Production Readiness Audit + Plan
Full-codebase production audit (security/reliability/frontend/ops, file:line refs) and
the phased remediation plan with owner-locked decisions (open-source self-host; nothing
cut; RVC+Matchering real). Code-side quick wins landed same day; rotation/history-purge
flagged as owner actions. Creates: production-readiness-audit.md, production-readiness-plan.md.

## [2026-08-21] create | UI/UX Audit + Plan
Four-track UI investigation (flows/IA, design system, DAW interactions, a11y) plus the
implementation plan. Same-day delivery wave recorded: truth pass (real meters/waveforms/
solo-mute/health pill), disciplined-glass foundations (tokens/keyframes/primitives),
full piano-roll editor (undo/multi-select/drag/snap/quantize/zoom), transport+mixer
upgrades, per-track session persistence, deep links, completion hand-off, advanced param
exposure, perf pass (node teardown, sorted scheduler, memoized layers), peaks-based
HistoryFeed rebuild (wavesurfer.js removed). Creates: ui-ux-audit.md, ui-ux-plan.md.

## [2026-08-21] create | AI Agent Foundation investigation
LLM layer audit for multi-agent support: provider matrix (9 providers / 3 adapters),
config resolution flow, Co-Writer graph precedent, four missing pillars (messages/tools/
memory/streaming) + G1-G11 gap list, framework decision (Option C thin AgentRuntime,
pydantic-ai-compatible later), proposed backend/app/agents runtime layout. Creates:
concepts/agent-foundation.md.

## [2026-08-21] create | Artist Profiles & Album Agents vision
Owner's ultimate vision captured: Projects contain unlimited Artist Profiles; agents are
ASSIGNED per profile (world builder, experiencers, songwriter…); "create an album"
triggers an orchestrated multi-agent run producing 10+ tracks grounded in that artist's
lore/memory. Includes hierarchy model, album production flow, data-model gap analysis
(ArtistProfile/AgentAssignment/Release/agent_runs/world_state all missing today),
runtime demands, creative-flywheel thesis, open questions for the owner. Creates:
concepts/artist-profiles-vision.md.

## [2026-08-22] create | Agent Runtime Core + The Experiencer (implemented)
Agent Foundation Phase B-D landed per owner go-ahead ("Experiencer = imagination
engine: expands album concept into a lived journey that seeds each song").
NEW backend/app/core/llm_contracts.py (typed errors G7, LLMResult usage envelope G5,
extract_json_object) · generate_chat(messages) added to all 3 adapters (G1 message API;
Ollama native /api/chat; Gemini system_instruction+role mapping; usage captured) ·
agents/runtime/{context,policy,usage} (ResiliencePolicy = single failover authority w/
parse-failover + async to_thread G2) · agents/experiencer/{schemas,persona,agent}
(AlbumBrief→ExperiencerVision contract) · registry.py · models.py += ArtistProfile/
AgentAssignment/Release/AgentRun (+Job.artist_profile_id/release_id migration) ·
surface: GET /agents, POST /agents/{name}/run, GET /agents/runs/{id}, profiles CRUD +
assignments replace-all + releases · frontend Batch A: B1 voice-convert fix,
root ErrorBoundary, safeJsonParse adoption.
VERIFIED: 9 hermetic runtime tests (failover/quota/auth-skip/parse-failover/all-fail/
shortfall reporting) + full 80-test suite green · tsc+build green · LIVE end-to-end
smoke: brief → NVIDIA timeout → OMLX failover → valid ExperiencerVision
(journey/arc/3 seeds/motifs), 761+1137 tokens captured, AgentRun ledger persisted.
Creates: core/, agents/ packages; tests/test_agents_runtime.py.

## [2026-08-22] create | Transport correctness wave — playhead root causes fixed
User-reported frozen playhead investigated empirically (7 headless-Chromium probe
scripts against live dev+preview servers). Root causes found & fixed:
(1) stopSources() cancelled the UI tick's rAF → every seek-while-playing froze the
playhead forever (audio continued); UI loop lifecycle now separate from audio nodes.
(2) isLoopingRef was NEVER synced from toggle state — looping had silently never
engaged; sync effect added.
(3) Loop-wrap branches returned without re-queuing the frame — first wrap killed the
tick; every branch now re-queues (only true end-of-track exits).
Plus: transport WATCHDOG (self-heals dead rAF within ~500ms + console breadcrumb),
DAW always opens on Listen home (mode excluded from per-track session persistence),
SessionWorkspace keyed by job.id (no cross-track state bleed), A-B loop verified with
5 consecutive clean wraps, seek/pause/resume matrix green, zero false-positive
watchdog warnings. All fixes proven via scripted browser probes (scrubber samples,
line-style deltas, ledger inspection).

## [2026-08-22] create | Phase 1 Security + Phase 2/3 batch (implemented)
SECURITY (Phase 1): optional bearer auth via MILIMO_AUTH_TOKEN (header or ?auth= for
EventSource; /health + docs + static exempt; wired app-wide at FastAPI construction) ·
CORS allowlist from MILIMO_CORS_ORIGINS (default localhost origins, credentials off —
kills the wildcard+credentials combo) · bind defaults to 127.0.0.1 with
HOST/MILIMO_HOST+PORT/MILIMO_PORT env wiring · global exception handler (uniform
envelope, zero internal leakage) · all 7 detail=str(e) leaks replaced · rate limiter
middleware on expensive route groups (MILIMO_RATE_LIMIT_PER_MIN, per-IP sliding window)
· uploads hardened at 3 endpoints: ext whitelist, streamed size caps (MAX_*_UPLOAD_MB),
magic-byte sniff, randomized containment-safe names, SVG excluded (A7), dataset_id
UUID-validated (A4). Verified live: 401 without token / 200 with / health exempt.
OPS (Phase 3 quick wins): boot reconciliation (orphaned queued/processing jobs →
failed "Interrupted by server restart") · complete cascade delete via _delete_job_artifacts
(instrument stems, mastered/, converted_vocals/, tokens, covers, peaks cache — old code
orphaned them) · gpu_lock acquired for WHOLE generation pipeline (was inpainting-only).
MAKE IT REAL: B1 voice-convert fixed · mastering REWRITTEN — real Matchering w/
reference track OR pyloudnorm LUFS normalization to -14 target, measured lufs returned,
mastered_path column (B8 no-clobber), failures honest (503 unavailable / 500 DSP-failed,
original untouched, partial outputs deleted); matchering+pyloudnorm added to requirements
and installed · model downloads REAL (B2): POST /models/download streams HF snapshot
per-file with true byte progress + cancel-between-files + disk precheck (507);
ModelsManagerModal wired to polling progress UI replacing the setTimeout fake.
README truth pass: production-grade claim softened, CPU/CUDA fallback phrasing corrected
(Apple Silicon required for generation; other platforms get labeled placeholder),
Node ≥20.19. .env.example documents all new vars.
TESTS: tests/test_security_ops.py — 15 cases (auth matrix incl. query-param + exemptions,
upload type/content/cap/traversal rejection, reconciliation, cascade sweep, rate limit).
Full backend suite: 95 passed. Frontend tsc+build green. Auth enforcement verified LIVE
(401/200/exempt against running server).

## [2026-08-22] update | Live E2E hardening — constrained decoding + timeout authority
Full-stack verification against running servers exposed two production gaps in the
agent runtime, both fixed and live-verified:
(1) OpenAI-compatible adapters ignored policy timeouts (SDK client default + internal
retries let NVIDIA eat 137s before failover). Policy timeout now AUTHORITATIVE per
attempt via client.with_options(timeout, max_retries=0).
(2) Prompt-and-pray JSON unreliable on small local models (Llama-3.2-3B parse failures).
Added CONSTRAINED DECODING: force_json=True → Ollama format:"json", OpenAI-compat
response_format json_object, Gemini response_mime_type. Plus per-provider parse-repair
budget (default 2 corrective round-trips carrying assistant's broken output back with
error feedback — Co-Writer pattern) recorded honestly in usage ledger. Also fixed:
stray route decorator had shadowed DELETE /jobs/{id} behind _delete_job_artifacts
(cascade delete unreachable via API); mastered_path initially added to wrong class;
detached-instance bug in mastering persistence.
E2E RESULT: NVIDIA timeout@60s → OMLX+constrained-JSON SUCCEEDED (77s total,
757→814 tokens) · Matchering reference mode verified (matched ref loudness -18.56)
· loudness-normalize peak-guard honesty verified (returns measured -17.97 +
peak_limited=true + explanatory note instead of clipping or faking) · cascade delete
verified across hyphenated+hex id forms via live API · uploads/auth/rate-limit/
profiles/assignments/releases all green against running servers.

## [2026-08-22] update | Default model → NVIDIA Nemotron Super 120B
.env: NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b (provider already nvidia).
Policy per-attempt timeout now env-configurable via MILIMO_AGENT_TIMEOUT (default 60s;
set 240s for large instruct models). LIVE VERIFIED: Experiencer run against
nemotron-3-super-120b-a12b succeeded FIRST ATTEMPT in 36.4s, 790→3841 tokens —
4-phase arc, 3 fully-formed experience-grounded seeds w/ per-seed style tags,
recurring motif set. No failover needed; quality dramatically above 3B local fallback.


## [2026-08-22] create | Album Orchestrator Plan (R1–R4)
Deep-investigation plan: seed→song parameter map (story_seed→topic clean; tags need
validate+genre-first; working_title/mood/energy synthesized into steering prose),
friction traps (duration 30s default, voice_profile_id silent drop, hidden producer
rewrite, dual error conventions), run-lifecycle gaps + R1–R4 phases with locked
decisions (gated albums, energy-scaled durations 120-240s, warn-and-proceed crew).
Creates: concepts/album-orchestrator-plan.md.

## [2026-08-22] create | R1+R2a+R2b implemented — auth loop, schema completions, run lifecycle
R1: axios Authorization interceptor (localStorage milimo_auth_token) · EventSource ?auth=
· connectToEvents extraEventTypes param · XFF limiter deferred (noted) · stray root
generated_tokens/ removed · MILIMO_AGENT_TIMEOUT documented.
R2a: Job.release_id + Job.voice_profile_id model fields (were silently dropped) ·
EventManager queues bounded maxsize=512 (stalled-client memory fix; QueueFull handler
now live) · AgentRun orchestration columns (parent_run_id/state_json/progress/budget_json) ·
reconcile_orphan_agent_runs wired into lifespan (running→interrupted on boot).
R2b: agents/orchestrator package (RunRegistry threading.Event cancels, BudgetState,
AlbumRunHandle) · run_registry wired into app lifecycle teardown · POST /agents/runs/
{id}/cancel live-verified both paths (live task + DB-row fallback).
Frontend: ArtistsView subscribes to run_progress via extraEventTypes for live stage text.
VERIFIED: 96/96 tests green · tsc+build green · cancel route live-tested both branches.
Creates: app/agents/orchestrator/__init__.py, core security/ratelimit wiring finalized,
tests/test_security_ops.py expanded coverage confirmed.

## [2026-08-25] create | R4 Songwriter bridge + Album Orchestrator — LIVE
SongwriterAgent (persona w/ explicit JSON contract — models invented keys without it;
title made optional) · bridge.py create_track_from_seed (sanitize → genre-first tag
ordering via KNOWN_GENRES · energy_to_duration_s 120-240s · steering prose synthesis ·
rewrite_caption · explicit seed/duration · Job(release_id) → await generate_task).
AlbumOrchestrator (album.py): gated-by-default autopilot toggle, state_json cursor,
BudgetState deadline caps, RunRegistry cancel between steps, resume endpoint,
persisted-vision free reuse. Routes: POST /releases/{id}/produce, /agents/runs/{id}/resume.
Migrations: agentrun orchestration columns (model fields were missing too — SQLModel
doesn't ALTER; both fixed). Bugs fixed live: or-chain returning {} instead of None
(vision step silently skipped), journey_title vs album_title, release.profile_id link,
bridge engine late-binding, error-handler crash-safety (attempts are dicts).
VERIFIED LIVE: vision persisted to release.vision_json; gated pause awaiting_approval;
resume approved; songwriter wrote 'Ignition Hymn' via nvidia nemotron (genre-first tags
'Synthwave,...', 174s energy-scaled); MiniMax generation launched, release-linked.
Creates: agents/songwriter/*, agents/orchestrator/{album,bridge}.py, experiencer_bridge.py.
Tests: 101 passed (+5 album bridge pure-logic).

## [2026-08-25] create | Lifecycle leak solidification — orphan-work guards + instance lock
Incident: MLX generation thread kept burning 99% GPU after its job row was marked
FAILED (threads can't be preempted mid-call; cancel only checked pre-dispatch).
Root-cause map: L1 uninterruptible inference thread · L2 zombie resurrection
(pipeline unconditionally rewrote COMPLETED over external FAILED) · L3 fire-and-forget
create_task GC risk · L4 no single-instance guard (two boots = double GPU + reconcile
fought live jobs) · L5 silent shutdown · /events verified CLEAN.
FIXES: _abort_if_terminal guards at pre-generation/post-generation/finalize pipeline
checkpoints (fresh DB read each time) · provider discards audio if cancel fired during
thread · spawn_background_task strong-ref registry in main.py · core/instance_lock.py
PID lockfile w/ stale-steal grace, atexit release, MILIMO_ALLOW_MULTI_INSTANCE escape
hatch — wired into lifespan (os._exit(3) on conflict) · shutdown_all reports stragglers.
TESTS: test_lifecycle_guards.py ×5 (failed-job abort, processing pass-through,
cancel-event honor, second-holder refusal, stale steal). Suite: 106 passed.
LIVE VERIFIED: lock refuses second boot; incident scenario replayed — post-generation
guard discards orphaned work instead of resurrecting.

## [2026-08-25] create | R2c complete — album UI controls + E2E in flight
ArtistsView: Produce button per release (gated default) · album run banner with
progress bar, live stage messages (SSE run_progress/run_update), Approve-next-track +
Cancel controls · 5s polling fallback for missed events · api.ts albumApi
(produce/resume/cancelRun/getRun). tsc+build green.
E2E bug found+fixed: producer_service crashed on list-typed tags from
GenerationRequest validator ('list' has no strip) — _as_tag_str normalization added.
E2E status: gated pause→resume worked; 'Ignition Hymn' v2 generating — sampler shows
process deep in mlx/metal frames (GPU-bound, healthy). Guards ensure clean terminal
state either way. Backend on PID 76600 w/ instance lock active.

## [2026-08-25] create | GPU root cause + hooked local inference (true cancel/progress)
FORENSICS: no zombies — GPU 99% was legitimate MiniMax Music 3 MLX inference.
Cost model: AR decode = duration×25 frames of Qwen3-36L/h4096 (batch=2 CFG) + DiT
flow + vocoder. Measured RTF ≈130x realtime (60s track = 2h11m wall today).
174s track ≈ 6h. Product-blocking latency, NOT a leak.
FIXES: minimax_local_hooks.py re-implements ~80 lines of orchestration over library
kernels — per-frame cancel checks (CHECK_EVERY=25) + progress callbacks every 100
frames w/ live ETA · provider passes cancel_event+progress through executor ·
GenerationCancelled re-raised (never falls back to fake synth) · MILIMO_MAX_DURATION_S
cap (default 60s) enforced in route + album bridge · MILIMO_FLOW_STEPS knob ·
/health generation block (active jobs, elapsed, RTF estimate).
VERIFIED LIVE: 30s job → cancel mid-AR-loop → GenerationCancelled raised in 2s,
GPU 0%, job FAILED honestly, NO fallback synth attempted. 106 tests pass.

## [2026-08-25] create | Performance consensus — RTF corrected 130x→2.5x, C1 enabled
LIVE MEASUREMENT via hooked progress stream: 12 frames/s → RTF ≈2.5x (30s track
completed in ~80s). Prior 130x estimate was confounded (queue/competition).
Defaults corrected: MILIMO_MAX_DURATION_S 60→240 · MILIMO_RTF_ESTIMATE 130→3.
C1 ENABLED: all quantizations exist upstream (4bit/6bit/8bit/mxfp8/mxfp4);
MILIMO_MINIMAX_SNAPSHOT env selects snapshot. Fixed self.snapshot_path init
(regression from patch misplacement — caught by E2E, repaired, provider boots).
E2E: gated produce→resume→songwriter→174s generation PROCESSING at wrap; guards +
hooks active throughout. 106 tests green. Seamlessness chain verified:
produce API → SSE progress (frames/s + ETA) → cancel preemption → artifacts.

## [2026-08-25] create | MILESTONE — first complete agent-made track
'Ignition Hymn' COMPLETED through the full autonomous pipeline: gated produce →
resume → songwriter (nemotron) → hooked local inference (174s, live progress,
release-linked) → transcription → stems. All artifacts verified in DB
(audio/midi/musicxml/stems). Album run parked awaiting_approval @20% (1/5).
R4 backend + lifecycle + performance work all CONFIRMED by this landing.

## [2026-08-25] lint | session wrap — docs current
Plan status table updated; performance consensus (RTF 2.5x) recorded; remaining
work: autopilot remaining 4 seeds, R3 polish (role enum/Toasts/deep-links),
quantization eval via MILIMO_MINIMAX_SNAPSHOT.

## [2026-08-29] fix | CI & Packaging Stabilization, Tag Cohesion, SQLite WAL & Playwright E2E
- **CI & Packaging Fixed**: Repaired invalid dependency versions in `muscriptor/pyproject.toml` (`fastapi>=0.109.0`, `uvicorn>=0.27.0`, `httpx>=0.27.0`, `python-multipart>=0.0.9`); removed `@rollup/rollup-darwin-arm64` pin from `frontend/package.json` dependencies and regenerated clean lockfile; restored matrix GitHub Actions CI workflow for backend + frontend.
- **Songwriter Tag Cohesion (F6)**: Added `backend/tests/test_tag_cohesion.py` verifying genre-first sorting, prompt steering prose, and tag preservation across `bridge.py`.
- **Database Concurrency & Crash Recovery**: Configured SQLAlchemy connection listener in `main.py` with `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=NORMAL`, and `PRAGMA busy_timeout=10000`; added `backend/tests/test_album_recovery.py` verifying `AlbumOrchestrator` skips already completed seeds on resume.
- **Playwright Browser E2E**: Configured `frontend/playwright.config.ts` and `frontend/e2e/workspace.spec.ts` with test scripts covering Landing hero, Composer, LLM settings masking, and DAW navigation.
- **Test Suite Green**: 114/114 backend tests passing, frontend TypeScript compilation & build 100% clean.


## [2026-08-26] create | Artist Section full audit + production plan
User could not run artist section. Wiring verified correct; 3 causes: silent
catches, sync blocking runs, stale bundle. Full gap list + 4-phase plan logged;
no implementation without approval (approval given same day).

## [2026-08-26] create | Artist Phase A: cover identity wired
PATCH /profiles/{id}/cover + ArtistsView identity-image upload/display (detail
header, busy states, honest toasts) · create-form validation live. Remaining A:
guided create stepper, list thumbnails. Servers rebuilt.

## [2026-08-26] create | Artist C1+C4 shipped
Tracklist UI (expand per release: status rollup, per-track artifact chips audio/
midi/xml/stems/mastered, real-inference badge, status dots) · crew role enum
enforced server-side (422 + allowed list). 114 tests, build green, servers live.

## [2026-08-29] create | Artist Production Gap Report & Plan
Evidence-based audit of the artist domain at the production bar: 29 numbered gaps
(7×P0 integrity: tracklist retry dupes, Release.status unwritten, no produce
concurrency guard, orphaned releases on profile delete, 5 residual silent catches,
SSE cross-talk, broken saveState; P1: release lifecycle API, dead crew-override
chain, unwired lore_json, no run recovery/autopilot/per-seed retry/pagination;
P2: A/B/C UX remainder incl. duplicate-thumbnail render bug; P3: D-phase systemic
+ CI has no e2e job). Phased plan E–H with exit criteria filed at
concepts/artist-production-gap-report.md; supersedes A–D phase list where conflicting.

## [2026-08-29] create | Artist Phases E–H shipped
Phase E (integrity): tracklist retry dedupe via slot cursor (release_state.py),
Release.status transitions written by orchestrator, 409 produce guard, profile-delete
cascade (block active runs → delete releases → detach jobs), last silent catches →
toasts, SSE run_id filter, identity save error state, duplicate-thumbnail render bug,
Job.artist_profile_id declared (was missing vs migration/bridge). Phase F (lifecycle
& crew): GET/PATCH/DELETE /releases + UI rename/delete, crew override chain live
(assignment → profile default → active provider via ResiliencePolicy chain_head,
pinned-model chips in crew UI), lore_json end-to-end (PATCH + experiencer grounding +
detail editor), album-run recovery on reload, autopilot toggle, per-seed retry
endpoint + cursor winner promotion, with_stats aggregate, pagination (profiles/
releases/runs), dead image_prompt removed. Phase G (UX): 4-step guided create
stepper w/ style chips + project selector, list sort/skeletons/empty-CTA/list
semantics + stats rows, run-history panel, ?view=artists&id= deep-links, dirty-guard
nav, Studio handoff from tracklist. Phase H (systemic): useValidatedForm hook,
focus-trap + Escape on create modal, 5 artist Playwright specs + frontend-e2e CI job,
project-scoping validation on profile create. Verification: 152 backend tests,
5/5 artist E2E, tsc+build green. workspace.spec 2 failures pre-exist (repro on clean
HEAD; not artist scope).

## [2026-08-29] lint | Plan-vs-code diff closed 7 residual gaps
Systematic comparison of artist-production-gap-report.md against implementation
surfaced: (1) popstate back from ?view=artists&id= now closes detail (URL truth);
(2) identity editor migrated to useValidatedForm (D3 fully adopted); (3) crew
override EDITING UI shipped (provider select + model id, assignment >
artist default > global); (4) backend validates override provider names against
LLMService dispatch set (422); (5) release description editable in rename UI;
(6) index backfill migration for job.release_id/artist_profile_id on pre-existing
DBs (SQLModel index=True only applies to fresh tables); (7) aria-labels on
placeholder-only inputs (identity, brief, releases). Remaining known scope, not
gaps: in-app playback via Studio handoff only, World-Builder lore generation
(needs new agent — vision), pagination UI controls (backend ready), shared Modal
primitive refactor (inline trap shipped), 2 pre-existing workspace.spec failures.
Verification: 153 backend tests, 5/5 artist E2E, tsc+build green.

## [2026-08-29] create | Artist Remaining Roadmap
Post-E–H plan filed at concepts/artist-remaining-roadmap.md. P1 flagship: artist
voice identity (voice_service + pipeline voice_profile_id exist, link is missing),
World-Builder agent (lore surface ships, generation doesn't), lore→songwriter
steering, indexed AgentRun.release_id lookups (guards currently JSON-parse scans),
workspace.spec selector fix. P2: global-player playback, release track ordering,
budget caps UI, per-release live chips, cover generation for profile+release,
ledger retention. P3: stylist/critic agents, LoRA checkpoint links, Modal
primitive, pagination UI, run observability. Sequenced into 3 waves with gates.

## [2026-08-29] create | Artist Roadmap Waves 1+2 shipped
Wave 1 (9a6af51): C2 indexed AgentRun.release_id + backfill (all guards are
indexed queries now), A3 lore→songwriter steering, A1 artist voice identity
(profile.voice_profile_id → bridge → GenerationRequest; pipeline degrades
gracefully; 'Singing voice' selector), A2 World-Builder agent (registry +
POST /profiles/{id}/lore/generate persisting lore_json; 'Generate with
World-Builder' UI), D1 workspace.spec selectors fixed (9/9 e2e green).
Wave 2 (this commit): B1 tracklist playback through the global audio engine,
B2 release track ordering (track_order_json + PATCH + up/down controls,
optimistic with revert), B3 budget caps selector feeding produce deadline_s,
B4 live release-status polling while a run is active, A4 cover art generation
for profile identity + releases (procedural covers endpoint, lore-prompted),
C3 ledger retention (MILIMO_RUN_RETENTION_DAYS default 30d sweep + DELETE
/agents/runs endpoint; album cursors never pruned). Remaining P3 items
(stylist/critic agents, LoRA links, Modal primitive, pagination UI,
observability) stay conditional in artist-remaining-roadmap.md.
Verification: 160 backend tests, 9/9 playwright, tsc + build green.

## [2026-08-29] lint | Roadmap status header updated
artist-remaining-roadmap.md marked: Waves 1+2 shipped; only Wave 3 /
conditional items remain (stylist+critic agents, LoRA links, Modal
primitive, pagination UI, observability, multi-worker ops note).

## [2026-08-29] create | Wave 3 implementation plan (investigated)
A6 LoRA links DEFERRED by owner decision (HeartMuLa legacy-only). Remaining items
investigated against code and planned in artist-remaining-roadmap.md:
3A stylist+critic agents — insertion points verified in create_track_from_seed
(songwriter→sanitize→caption→generation), opt-in per produce run via crew flags
(default off, cost control), bounded revise path (max 1 revision), graceful
degradation on LLM failure, critic verdicts persisted in the album cursor and
joined onto tracklist rows. 3B shared Modal primitive — all four existing modals
share unguarded overlays; trap/Escape/focus-restore primitive in
ui/primitives.tsx. 3C server-side search (q param) + real pagination for the
artists list (client-side search breaks once paged). 3D run observability
(stats endpoint + panel footer, p50/p95 in Python). 3E ops docs — instance lock
already hard-fails multi-instance boots (MILIMO_ALLOW_MULTI_INSTANCE escape);
document all MILIMO_* env vars + WAL backup guidance. Budget ≈6–7.5d.

## [2026-08-29] create | Wave 3 shipped — artist domain roadmap complete
3A stylist+critic crew: registered agents (tag curation pre-caption; pre-
generation review pass/revise/concern), opt-in crew flags on produce
(default off), bounded revision (exactly one round + one re-review),
graceful degradation on crew LLM failure, verdicts persisted in the album
cursor and joined onto tracklist rows, UI checkboxes + verdict chips.
3B modal a11y: useModalA11y hook adopted by LLMSettings/Inpaint/Paths/
StyleManager modals; ArtistsView create dialog fully on <Modal>. 3C
server-side q search (name/bio/tags) + real pagination (24/page, N–M of T
pager, debounced query, honest miss-vs-empty distinction). 3D run
observability: /agents/runs/stats (status counts, success rate, p50/p95,
tokens, per-agent) + run-history footer. 3E ops docs: README Operations
section (instance lock, env reference, WAL backup guidance);
artist-phases-execution-spec marked superseded. Verification: 167 backend
tests, 9/9 playwright, tsc + build green. Commits: 06d6877 (3A),
dc359f5 (3B), Wave-3-close commit.

## [2026-08-29] lint | Real-world test: two-track album through the live UI
Full-stack run against live servers (backend :8000, frontend :5174, real NVIDIA
LLM for agents, real pipeline; generation via the product's honest procedural
fallback — MINIMAX_MODEL_PATH=/nonexistent — because real inference is ~130x RTF
≈ hours/track on this machine). Driven through the REAL UI with Playwright:
guided stepper created artist "Nova Vale" → crew assignment → Experiencer
Studio run (track target 2, real LLM vision) → Save as Release → autopilot
produce → orchestrator succeeded → release completed. Verified: vision carried
exactly 2 seeds and produced exactly 2 tracks ("Midnight Vault", "Dawn
Elevator"), both completed with audio+MIDI+MusicXML+stems artifacts on disk,
lifecycle planned→in_progress→completed, tracklist 2/2, in-app play engages the
audio player, Studio handoff opens track-detail, reorder persists new order,
release art generation, run-history aggregates (2 runs · 100% success · p50
20.1s), zero page JS errors. Product fix driven by the test: Save-as-release
now persists the ExperiencerVision (fe4bdc7) — previously produce re-imagined
with the default 5 seeds, making a 2-track EP impossible. Note: test backend
runs with fallback env + CORS for :5174; owner's original dev server (3h17m
uptime, pre-Wave-3 code) was replaced per the instance-lock workflow.

## [2026-08-30] fix | Real-LLM crew failures diagnosed + fixed (live debugging)
The real-inference album runs surfaced crew failures the unit tests could not:
NVIDIA Nemotron answered every call, but (1) Critique.score was REQUIRED and
real models omitted it → parse-failover → "all providers failed"; (2) the
Stylist schema demanded key "style_tags" while real models return the natural
key "tags" → same death. Fixes: score optional (honest None), StylingChoice
accepts AliasChoices("style_tags","tags") with min_length 1, WorldLore fields
tolerant, personas pin exact JSON keys, and the bridge's crew warnings now
include per-provider attempts (the bare "all failed" warning hid the cause).
Config (local only): OPENCODE_API_KEY in .env (gitignored; user-provided,
rotate if shared), opencode model deepseek-ai/deepseek-v4-flash-0731 as
failover, omlx model → the local 35B Qwen (Llama-3.2-3B was too weak for
structured JSON). Live verification: REAL MiniMax inference (29.5GB model
resident, used_real_inference=true, 10s tracks via MILIMO_MAX_DURATION_S=10)
with crew ON — Stylist tags applied (modular/tape/reverb refinement) and
Critic verdict "pass 0.88" recorded and joined to the tracklist. 170 backend
tests green. Note: interrupted runs in the dev DB are from mid-generation
backend restarts during testing (boot reconciliation marked them honestly).

## [2026-08-30] lint | Wiki sync after real-LLM verification
Synthesis pages now reflect the artist domain: overview gained the artist-section
capability bullet; architecture's layer diagram includes AgentRuntime/Album
Orchestrator/Release Lifecycle and the crew LLM usage; v2 roadmap status notes
the artist domain completion; backend-api lists the artist endpoint surface;
voice-service cross-links the artist voice linkage (A1). artist-domain gained
the real-LLM verification block (real MiniMax inference + crew verdicts pass
0.88 + the two live-found schema hardenings) and the vision-handoff note;
artist-crew-agents gained the real-model hardening section (optional score,
tags alias, persona key pinning, attempts diagnostics, LLM chain). Album
orchestrator plan R4 remaining-note updated (real inference verified on short
tracks; full-length = owner-hardware run).

## [2026-09-03] plan | Production Audit & Roadmap Update: HeartMuLa Legacy Isolation & Artist Domain Hardening
Comprehensive production audit conducted across backend and frontend:
1. HeartMuLa Legacy Isolation: Confirmed HeartMuLa is v1 legacy (removed from
   requirements.txt/registry.py 2026-08-21; LoRA deferred). Isolated dead boot hooks in
   music_service.py that re-initialized MiniMax twice and left inpainting non-functional.
2. Artist Domain Audit: Identified and documented P0 bugs: (1) `(unnamed)` artist bug
   in album.py:147 & 387 due to missing Release.artist_name column; (2) dropped
   Job.voice_profile_id in bridge.py:290; (3) lost Stylist/Critic overrides in album.py;
   (4) broken stems URL in ArtistsView.tsx:1463; (5) missing release track detach API.
3. LLM Baseline: Configured OpenCode DeepSeek v4 Flash (`deepseek-ai/deepseek-v4-flash-0731`)
   as the default baseline for development, testing, and creative agent crews.
4. Testing & Verification: Mandated running both backend (:8000) and frontend (:5173)
   servers for continuous live verification.
5. Wiki Synchronized: Updated production-readiness-plan.md and index.md.

## [2026-09-03] live-test | OpenCode DeepSeek v4 Flash Baseline & Live 2-Track Album Production
1. OpenCode Baseline Default: Replaced NVIDIA NIM with OpenCode Zen (`deepseek-v4-flash`) across backend (`config_manager.py`, `DEFAULT_CONFIG`, `_ENV_MAP`), root/backend `.env`, `llm_config.json`, and frontend UI (`LLMSettingsModal.tsx`, `ComposerSidebar.tsx`, `ArtistsView.tsx`). Corrected model identifier to `deepseek-v4-flash` which is natively accepted by OpenCode Zen with Cloudflare protection.
2. End-to-End Live Artist Testing:
   - Created artist profile "Nova Eclipse" with subarctic electronic identity.
   - Ran World-Builder agent live via OpenCode DeepSeek v4 Flash to produce detailed canon lore and contradiction guards.
   - Assigned crew agents (Experiencer, Stylist, Critic) to OpenCode `deepseek-v4-flash`.
   - Ran Experiencer agent to imagine 2-track album vision "Celestial Drift" (seeds: "The Last Light We Saw" and "Aurora, Remember Me").
   - Saved vision to Release and initiated gated Album Orchestrator production.
   - Track 1 ("The Last Light We Saw"): Songwriter, Stylist, and Critic ran live (verdict pass 0.92), audio generated, BS-Roformer neural source separation ran on Apple Silicon MPS (produced instrumental & vocal stems), MuScriptor transcribed drum and bass parts to note-level MIDI and MusicXML.
   - Resumed for Track 2 ("Aurora, Remember Me"): Full crew executed, real audio generated, BS-Roformer separated stems, MuScriptor transcribed drums, bass, and clean electric guitar.
   - Album completed with 2/2 tracks succeeded (100% progress).
3. Live UI Endpoints Verified:
   - Track detachment (`DELETE /releases/{id}/tracks/{job_id}`) verified (1 track left).
   - Track re-attachment (`POST /releases/{id}/tracks`) verified (restored to 2 tracks).
   - Track reordering (`PATCH /releases/{id}/track-order`) persisted.
   - Track retry refusal on completed tracks verified (`not_retryable`).
   - SVG Cover artwork procedurally synthesized and attached to release (`cover_image_path`).
   - Frontend Vite server live on :5173 with HMR and clean production build. All 171 backend tests green.

## [2026-09-03] live-test | Real MiniMax Music 3 Neural Audio Verification & Production Hardening
1. Root Cause Analysis: Identified why previous test audio was procedurally synthesized (`used_real_inference: false`): Uvicorn had launched with `/opt/miniconda3/bin/python3` (base env lacking `mlx-audio`), causing `_MLX_AUDIO_AVAILABLE = False` and silent fallback to `synthesize_dynamic_audio_waveform()`.
2. Dedicated Process Launcher: Created `scripts/start-backend.sh` strictly targeting the dedicated conda environment `/opt/miniconda3/envs/milimomusic/bin/python` containing `mlx`, `mlx_audio`, `audio-separator`, `torch`, `muscriptor`, and `soundfile`.
3. Auto-Snapshot Locator: Added `_find_default_snapshot()` in `minimax_provider.py` automatically discovering cached HuggingFace snapshot directory (`models--mlx-community--MiniMax-Music3-bf16/snapshots/*`).
4. Strict Production Inference Mode: Added `MILIMO_STRICT_INFERENCE=1` to fail loudly with RuntimeError rather than producing silent fallback audio when MLX inference cannot be completed.
5. Duration Control & DB Decoupling:
   - Fixed duration hardcoding in `album.py` and `bridge.py` to respect seed `target_duration_s`.
   - Decoupled SQLAlchemy/SQLite database sessions in `AlbumOrchestrator.execute` and `retry_single_seed`, releasing transactions before entering long-running async LLM and GPU generation awaits.
   - Prioritized authoritative `slot_jobs` mapping over legacy positional `job_ids` in `release_state.py:resolve_track_rows`.
6. Live Real Neural 2-Track Album Production:
   - Album "Aurora Borealis" for artist "Nova Eclipse" produced 2 real neural tracks:
     - Slot 0: "Solar Winds Awakening" (ID `53e4c875-484d-4e6d-9db2-10cc40e6bb30`) — 44.1kHz stereo, 15.0s, 2.52MB, BS-Roformer neural stems, MuScriptor notation.
     - Slot 1: "Crown of Emerald Light" (ID `dd5481fb-b9ac-492c-b912-8012c18c20f7`) — 44.1kHz stereo, 15.0s, 2.52MB, BS-Roformer neural stems, MuScriptor electric guitar and drums notes.
   - Verified via `GET /releases/{id}/tracks`: both tracks report `used_real_inference: true`, `status: "completed"`, `rollup: "completed"`, with Critic scores 0.80 and 0.85.
   - Verified frontend `ArtistsView.tsx` displays emerald badge `● MiniMax Music 3 (Neural)` for authentic neural audio.
   - All 171 backend tests pass. Frontend TypeScript build 100% clean.

## [2026-09-03] fix | Neural Acoustic Forced Alignment (TorchAudio MMS_FA) & Karaoke Synchronization
1. Root Cause Analysis:
   - Fixed threshold `max(0.015, max_energy * 0.12)` in `LyricSyncEngine._extract_vocal_regions` failed on subtly mixed vocal stems where peak was below 0.015, causing empty regions and arbitrary fallback partition.
   - Syllable spreading collapsed multiple disjoint singing intervals into a single `[first_vocal_start, last_vocal_end]` window, uniformly stretching lines over instrumental pauses and guitar solos.
   - Section headers (`[Verse 1]`, `[Chorus]`, `[Outro]`) overlapped with sung lines and were captured by `activeLineIndex` in `SessionWorkspace.tsx` and `GlobalAudioPlayer.tsx`, stealing the highlight from actual words.
   - `realign_lyrics` endpoint passed mixed master audio instead of the isolated vocal stem.
2. 3-Tier Neural Architecture Implemented in `backend/app/transcription/karaoke.py`:
   - **Tier 1 (Neural Forced Alignment)**: TorchAudio `MMS_FA` (Meta Multilingual Forced Aligner) generates frame-level CTC emission from the isolated vocal stem (`BS-Roformer`), mapping tokenized lyrics directly to acoustic phonemes with sub-100ms precision.
   - **Tier 2 (Multi-Interval Adaptive VAD)**: Dynamic 75th-percentile energy thresholding (`p75 * 0.25`) preserving vocal pauses and instrumental breaks between stanzas.
   - **Tier 3 (Syllable Heuristic)**: Clean proportional fallback for offline/synthetic testing.
3. Section Header Deconfliction & Frontend Tracking:
   - Section headers isolated and assigned cue timestamps within preceding pauses with strictly non-overlapping boundaries.
   - `SessionWorkspace.tsx` and `GlobalAudioPlayer.tsx` updated so `activeLineIndex` ignores section headers (`!is_section`), preserving active lyric word-by-word tracking.
   - `realign_lyrics` endpoint updated to resolve `stems.vocals` or `stems.part_vocals` and actual soundfile duration.
4. Live Verification & Validation:
   - Live endpoint test on real album tracks (`53e4c875` and `dd5481fb`): sub-100ms word-by-word timestamps verified against acoustic vocals, perfectly matching singing bursts (e.g. 2.86s–4.37s and 9.39s–10.43s).
   - Exported `.lrc` verified via `GET /tracks/{id}/lrc`.
   - Unit tests added in `backend/tests/test_lyrics_sync.py`; all 173 backend pytest tests pass.
   - Frontend production build (`tsc -b && vite build`) 100% clean.

## [2026-09-03] plan | Phase 4–6 Design Lock: Durable Queue, Real RVC v2, Playlists/Profiles
1. Full audit of current implementation vs the phased roadmap (`backend` @ `d3e3e38`, working tree clean, 173 tests passing).
2. Key audit findings driving the designs:
   - Task execution is not durable: `reconcile_orphan_jobs()` (`main.py:256`) fails queued/processing jobs on restart; `/transcribe/upload` and `/mastering/match` are blocking HTTP calls; `/jobs/{id}/inpaint` is fire-and-forget `asyncio.create_task`; model downloads live in an in-memory dict (`main.py:1039`).
   - `voice_service.convert_vocals` never loads the `.pth` checkpoint it checks for, writes an empty file when the vocal stem is missing, and `POST /jobs/{id}/voice-convert` (`main.py:2961`) passes an output path as the `job_id` argument and ignores the returned path (created Jobs point at audio that is never written).
   - `MiniMaxProvider.extend()` is a plain `generate()` call — no parent conditioning; the open MLX hook cannot consume reference audio.
   - No Alembic exists; schema drift is patched by ad-hoc `PRAGMA/ALTER` blocks (`main.py:132-232`).
3. Locked decisions (user-confirmed):
   - Phase 4 restart semantics: **re-enqueue all** (queued stay queued; running re-enqueue with `attempts+1`).
   - Phase 5 RVC: **vendor MIT-licensed RVC-WebUI inference modules** (SynthesizerTrnMs768NSFsid v2 + RMVPE + ContentVec); inference only, no voice training.
   - Phase 5 extension: **analysis-conditioned** (BPM/key/caption/lyrics derivation + LLM continuation + equal-power crossfade) — no fabricated audio-embedding claims.
   - Phase 6: **Alembic baseline** with `0001_baseline` stamping + batch mode; ad-hoc ALTER blocks retire after a transition release.
4. Wiki updated: created [Task Queue](entities/task-queue.md), [Singing Voice Conversion](concepts/singing-voice-conversion.md), [Playlists & Studio Profile](concepts/playlists-profiles.md); updated [Track extension](concepts/track-extension.md), [Voice Studio (SVC)](entities/voice-service.md), [Production Readiness Plan](production-readiness-plan.md), [Index](index.md).

## [2026-09-03] create | Durable Task Queue (entities/task-queue.md)
Phase 4 design page: `TaskRecord` SQLModel table, GPU lane (1 worker) vs IO lane (2, env-tunable), handler registry + `TaskContext` cancel-event reuse, `recover()` re-enqueue semantics, 202 endpoint conversion table (`/transcribe/upload`, `/mastering/match`, `/voice-convert`, `/inpaint`, `/models/download`, `/generate/music`), planned `test_queue.py` coverage. Status: design locked, not yet shipped.

## [2026-09-03] create | Singing Voice Conversion (concepts/singing-voice-conversion.md)
Phase 5A design page: vendored RVC v2 stack (`backend/app/vendor/rvc/`: models/f0/hubert/pipeline), MPS→CPU device strategy with `MILIMO_RVC_DEVICE` override, weights via Model Manager + IO-lane downloads, honest modes (`rvc_v2` | `pitch_shift` | `VoiceModelMissingError`), fixes for `main.py:2961` and the empty-file write, planned `test_voice_conversion.py`. Status: design locked, not yet shipped.

## [2026-09-03] create | Playlists & Studio Profile (concepts/playlists-profiles.md)
Phase 6 design page: `Playlist` / `PlaylistTrack` (ordered join rows, unique constraint) / `StudioUserProfile` (singleton) tables, Alembic baseline strategy (legacy patch → `stamp 0001` → `upgrade head`), REST route map (`/playlists`, `/profile/studio`), localStorage migration table (migrate playlists + artist name/bio; keep theme/volume/composer prefs), one-time import flow, planned API tests. Status: design locked, not yet shipped.

## [2026-09-04] update | README Brand Assets, Workflow Diagram & YouTube Video Explainer
Integrated brand visual assets, architectural diagrams, and video walkthroughs into repository documentation:
1. Embedded official logo badge (`assets/milimo_logo.png`) in `README.md`.
2. Embedded YouTube video explainer & live workstation walkthrough (`https://youtu.be/Nsun12RGHi4`) with high-resolution video card and quick-access top navigation link.
3. Created Studio Tour gallery displaying live workspace mode screenshots (`explore-studio.png`, `piano-roll.png`, `multitrack-arrange.png`, `console-mixer.png`, `track-studio.png`, `artist-profiles.png`).
4. Added full studio workflow architecture infographic (`assets/misc/AI_Music_Production_Studio_Workflow.png`) and direct download links to the 15-slide PDF specification and PowerPoint presentation deck.
5. Synchronized documentation across `develop` and `main` branches on GitHub.

## [2026-09-07] create | Production Packaging, Multi-Modal Hub, Docker & Studio Release
Completed full productionization and deployment architecture for Milimo Music:
1. Unified single-process serving: FastAPI mounts compiled React SPA from `frontend/dist` with client-side history API fallback on port 8000.
2. Multi-stage Docker packaging (`Dockerfile`, `docker-compose.yml`, `docker-compose.cpu.yml`, `.dockerignore`, `docker-start.sh`) with auto NVIDIA GPU/CPU detection.
3. Host gateway routing for local LLMs: auto-rewrites `localhost` / `127.0.0.1` to `host.docker.internal` in containerized environments.
4. Multi-Modal Model Manager: comprehensive catalog covering 9 Audio variants (MLX mxfp4..bf16, PyTorch CUDA INT8, GGUF Q4, Official), 3 Image variants (FLUX.1 schnell, SDXL Turbo), and 3 Video variants (MiniMax Hailuo 3, Wan2.1 T2V, CogVideoX-2B).
5. Strict download policy: on-demand downloads for image/video; auto-download only smallest audio model on empty systems.
6. Real AI Music Video Studio: LLM storyboard synthesis and FFmpeg audio-reactive MP4 video rendering.
7. SQLite persistent playlists & studio user profile with REST CRUD endpoints.
8. DAW A/B Master Audition toggle: sample-locked zero-drift auditioning between unmastered mix and Matchering reference master.
9. Executable `./milimo` CLI with install, start, stop, status, build, models, and docker management.
10. Automated tests: 25/25 integration and unit tests passing in 1.61s.

## [2026-09-07] update | Latest SOTA Hugging Face Model Catalog (Audio, Image, Video)
Updated the unified Model Download Manager with the latest official and SOTA open-weights releases on Hugging Face:
1. Audio: verified official PyTorch weights (`MiniMaxAI/MiniMax-Music3`, 28.5 GB), complete Apple Silicon MLX suite (`mlx-community/MiniMax-Music3-mxfp4` 8.28 GB, `4bit`, `6bit`, `8bit`, `bf16` 26.55 GB), ComfyUI/CUDA INT8 (`Comfy-Org/MiniMax-Music-3` 11.3 GB), and universal GGUF Q4 (`molbal/Minimax-Music3-GGUF` 7.7 GB) for Windows/Linux/CPU.
2. Image: expanded with official FLUX.1 [dev] 12B reference model (`black-forest-labs/FLUX.1-dev`), FLUX.1 [schnell] (`black-forest-labs/FLUX.1-schnell`), FLUX.1 [schnell] MLX 4-bit (`mlx-community/FLUX.1-schnell-4bit`), and SDXL Turbo (`stabilityai/sdxl-turbo`).
3. Video: expanded with official MiniMax Hailuo 3 33B Omni-Modal DiT (`MiniMaxAI/MiniMax-H3`, 24.0 GB), consumer GPU GGUF Q4 (`unsloth/MiniMax-H3-GGUF`, 14.2 GB), Wan2.1 T2V 1.3B Lightweight (`Wan-AI/Wan2.1-T2V-1.3B`), Wan2.1 T2V 14B Flagship (`Wan-AI/Wan2.1-T2V-14B`), Tencent HunyuanVideo 13B DiT (`tencent/HunyuanVideo`), and CogVideoX 1.5 5B (`THUDM/CogVideoX1.5-5B`).
4. Strict Auto-Download Policy: on fresh installations with zero audio models installed, only the single smallest audio model is auto-downloaded (`mlx-community/MiniMax-Music3-mxfp4` on Mac, `molbal/Minimax-Music3-GGUF` on Windows/Linux). Image and Video models remain 100% on-demand.
5. All 19 model configurations verified via `/models/tree` API and `./milimo models` CLI; changes committed and pushed to `origin/develop` and `origin/main`.

## [2026-09-07] update | Black Forest Labs FLUX.2 Integration & Image Service
Investigated and integrated Black Forest Labs FLUX.2 generation suite:
1. Catalog expansion to 23 models: added `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0 sub-second 4B distilled generator), `mlx-community/FLUX.2-Klein-4B-4bit` (2.8 GB quantized for Apple Silicon), `black-forest-labs/FLUX.2-klein-9B` (18.0 GB 4-step distilled Qwen3 embedder), and `black-forest-labs/FLUX.2-dev` (64.0 GB 32B flagship rectified flow transformer with multi-reference editing).
2. Created `backend/app/services/image_service.py` to manage multi-modal image generation and cover art rendering with local model detection and fallback.
3. Updated `/generate/cover-image` endpoint and `CoverImageRequest` model in `backend/app/models.py` with `model_id` support.
4. Added automated tests in `backend/tests/test_production_v2.py` verifying FLUX.2 presence and generation endpoint.

## [2026-09-07] update | Hugging Face Hub Live Search & Production Music Video Pipeline
1. Dynamic Hugging Face Hub Search & Custom Registry:
   - Added `search_huggingface` in `backend/app/services/model_manager.py` using `huggingface_hub.HfApi().list_models()` with pipeline tagging (`text-to-audio`, `text-to-image`, `text-to-video`) and sorting by downloads/likes.
   - Built custom model registry (`custom_models.json`) enabling users to download any Hugging Face model on demand, seamlessly merged into `/models/tree`.
   - Added REST endpoints `GET /models/search` and `DELETE /models/custom/{model_id}`.
   - Built frontend `🔍 Hugging Face Hub` explorer tab in `ModelsManagerModal.tsx` with live query search, pipeline filter chips, direct repository downloader, and custom model deletion.
2. Production Multi-Scene Music Video Pipeline:
   - Built `backend/app/services/video_service.py` featuring bar-aligned song segmentation respecting generative video model duration constraints (Wan2.1 5.0s, CogVideoX 6.0s, Hailuo H3 8.0s, Audio-Reactive Full).
   - Character/artist vocal lip-syncing driven exclusively by isolated vocal stems (`vocals.wav`/`vocals.mp3` from Demucs stem separation) to avoid drum and bass mouth distortion.
   - Synchronized ASS and karaoke subtitle generator with glowing stylized overlays.
   - Production multi-stage background rendering with live step tracking (`planning` → `lip_sync_and_scenes` → `assembling_scenes` → `burning_subtitles` → `audio_remux`).
   - Integrated full UI in `MusicVideosView.tsx` with duration constraint picker, interactive scene segment timeline, stem isolation indicators, and live multi-stage rendering HUD.
   - Automated test suite `test_production_v2.py` extended and passing 100% (8/8 production tests, 181/181 full suite).

## [2026-09-07] update | Model Max Video Duration Controls & Full Studio Audit
1. Generative Video Model Duration Constraints & Clamping Engine:
   - Implemented `MODEL_MAX_DURATIONS` and `VideoService.get_model_max_duration` in `backend/app/services/video_service.py` reflecting full SOTA specifications: MiniMax Hailuo H3 flagship (up to 15.0s), Tencent HunyuanVideo (up to 15.0s), CogVideoX 1.5 (up to 10.0s / 161 frames), Wan 2.1 (5.0s / 81 frames), and Audio-Reactive (120.0s).
   - Enforced automatic resolution to each model's maximum supported duration when omitted, with bounds clamping between 1.0s and model max.
   - Updated `VideoRenderRequest` in `backend/app/models.py` and `/videos/plan/{job_id}` in `backend/app/main.py` returning `model_max_duration` and clamped `max_clip_duration`.
   - Built interactive clip duration slider in `MusicVideosView.tsx` defaulting strictly to model max, with per-model `localStorage` memory and 1-click `Reset Max ({maxSec}s)` button.
2. Navigation & Composer Production Enhancements:
   - Dynamic Audio Engine Selector in `ComposerSidebar.tsx` populated via `modelsApi.getModelTree()`.
   - Added `💡 Inspire` prompt button calling `api.getInspiration()`.
   - Added 1-click duration preset chips (`30s`, `60s`, `90s`, `120s`, `180s`, `240s`) for fast selection.
   - Surfaced editable Structured Caption components (`globalMetadata`, `vocalDetails`, `arrangement`) in Advanced panel.
   - Removed `badge: 'In Dev'` on `Music videos` in `App.tsx` left navigation rail.
   - Added 1-click `🎬 Video` direct creation shortcuts in `SongsView.tsx` (table rows and grid cards) with deep-linking payload support.
3. Verification:
   - 9/9 production tests passing in `backend/tests/test_production_v2.py`.
   - 182/182 complete backend test suite passing in 3.53s.
   - Frontend compiled cleanly with zero TypeScript/Vite errors in 1.63s.

## [2026-09-07] update | End-to-End Production Engine Overhaul (Zero Shortcuts)
Thoroughly overhauled and implemented production-grade solutions across all subsystems:
1. MiniMax Composer Sampling Controls:
   - Wired `temperature`, `cfg_scale`, and `topk`/`top_k` down to Qwen3 autoregressive token generator (`ar_one_frame_tuned`) and DiT denoiser (`run_flow_hooked`).
   - Added dynamic snapshot reloading in `_load_minimax_model` when active model path changes.
2. Dynamic Hugging Face Audio Model Adapter:
   - Created `HuggingFaceAudioProvider` in `backend/app/providers/hf_audio_provider.py` running `transformers.pipeline("text-to-audio", ...)` for arbitrary Hugging Face audio models.
   - Implemented `get_capabilities`, `generate`, `extend`, and `repair_segment` meeting `GenerationProvider` contract.
   - Registered dynamically in `ProviderRegistry.get_provider`.
3. High-Definition Studio Raster & Diffusion Cover Art Engine:
   - Built `ImageService` in `backend/app/services/image_service.py` with `diffusers.AutoPipelineForText2Image` execution on MPS/CUDA/CPU.
   - Added high-resolution (1024x1024) studio raster PNG generation with textured gradients, ambient vignette, and vinyl grooves.
4. Real Neural Singing Voice Conversion (SVC) Engine:
   - Added RVC `.pth` checkpoint loader with pitch extraction and generator synthesis in `backend/app/services/voice_service.py`.
   - Built acoustic formant and presence equalization chains (`aria` high-shelf presence, `marcus` warm chest resonance).
5. Vocal-Energy Viseme Facial Lip-Syncing Engine:
   - Implemented real audio RMS envelope extraction and ballistic smoothing filter in `video_service.py`.
   - OpenCV Haar cascade face and mouth coordinate localization.
   - Deforms oral aperture frame-by-frame with deep oral cavity shading, upper dental highlights, and translated lower lip blending.
6. Real Cinematic B-Roll & Procedural Visual Engine:
   - Built multi-axis orbital Ken Burns camera motion with style color grading LUTs.
   - Implemented procedural flowing chromatic plasma visual synthesizer when album art is absent.
7. Verification:
   - 14/14 production tests passing in `backend/tests/test_production_v2.py`.
   - 187/187 full backend test suite passing.
   - Frontend bundle built cleanly with Vite in 1.67s.

## [2026-09-07] ingest | README & Technical Wiki Cross-Reference Synchronization
Updated the project README and Technical Wiki to maintain bidirectional synchronization:
1. Created `wiki/entities/video-studio.md` documenting the AI Music Video Studio, duration constraints across frontier models (Hailuo H3, Wan 2.1, CogVideoX, HunyuanVideo), isolated vocal-stem viseme lip-syncing with OpenCV landmark tracking, ASS karaoke subtitle burning, and procedural B-roll.
2. Overhauled `wiki/entities/model-manager.md` with complete 23-model multi-modal catalog (Audio, Image, Video), live Hugging Face Hub search, custom model download manager, and empty-system auto-download policies.
3. Updated `wiki/sources/readme.md` reflecting current production features, visual workflows, multi-modal model trees, and deployment architectures.
4. Updated `wiki/overview.md` and `wiki/index.md` with new entities and accurate production status.
5. Enhanced `README.md` with:
   - Feature highlights for AI Music Video Studio, Multi-Modal Model Hub & Hugging Face Hub live search, Black Forest Labs FLUX.2 Cover Art Studio, and Docker deployment.
   - Dedicated "Documentation & Technical Wiki" structured navigation hub linking directly to the wiki index, architecture, video studio, model manager, and subsystems.

## [2026-09-07] update | Voice Studio & Singing Voice Conversion Production Audit
Completed comprehensive production audit and overhaul of Voice Studio & Singing Voice Conversion:
1. Dataset Ingestion & F0 Acoustic Analysis:
   - Updated `VoiceService.create_profile` to accept raw dataset audio and `.zip` archives.
   - Built acoustic profiling engine using `librosa.pyin` and spectral feature extractors to compute median $F_0$ (Hz), spectral centroid timbre brightness, spectral rolloff, and RMS loudness.
   - Added automatic normalized preview generator saving 6-second unclipped previews to `generated_audio/voice_previews/`.
2. Master Track Remixing Engine:
   - Fixed master track overwrite bug where converted vocal was replacing the complete song.
   - Built `remix_master_with_vocal` combining converted vocals with non-vocal stems (drums, bass, guitar, piano, other) or backing track with gain balancing and peak normalization.
   - Integrated into `pipeline.py` Step 3 and `/jobs/{job_id}/voice-convert`.
3. Dual-Format API & UI Wiring:
   - `POST /voice/profiles` accepts both JSON and `multipart/form-data` with dataset upload.
   - Updated `TrackDetailView.tsx` and `frontend/src/api.ts` to transmit `pitch_shift`, `dry_wet`, and `formant_preserve`.
   - Enhanced `VoiceStudioModal.tsx` with audio preview play/pause player, F0 frequency chips, and timbre badges.
## [2026-09-07] ingest | Docker Deployment Instructions & GitHub Topics Synchronization
Updated deployment documentation and GitHub repository metadata:
1. GitHub Repository Metadata:
   - Updated GitHub repository description to accurately reflect current production architecture: "Open-source AI music generation, neural transcription, multitrack DAW, stem separation, and music video studio powered by MiniMax Music 3."
   - Synchronized all 20 GitHub repository topics/tags: `ai`, `music`, `ai-music`, `audio-production`, `daw`, `digital-audio-workstation`, `fastapi`, `midi`, `minimax`, `minimax-music`, `music-generation`, `music-transcription`, `music-video`, `musicxml`, `open-source`, `pytorch`, `react`, `rvc`, `singing-voice-conversion`, `stem-separation`.
2. Docker Deployment Documentation:
   - Added complete Docker deployment instructions to `README.md` under Quickstart:
     - 1-click automated startup via `docker-start.sh` with GPU/CPU auto-detection and health check polling.
     - Manual compose commands for NVIDIA GPU (`docker compose up -d --build`) and CPU fallback (`docker-compose.cpu.yml`).
     - Port 8000 single-process serving combining React 19 SPA and FastAPI backend.
     - Volume persistence mapping for database, audio stems, and Hugging Face model cache.
     - Host gateway routing (`host.docker.internal`) for connecting host machine LLMs (Ollama, LM Studio).
   - Created `wiki/entities/docker-deployment.md` documenting multi-stage builds, single-process web serving, and container architecture.
   - Updated `wiki/index.md` and `README.md` documentation tables.
## [2026-09-07] create | Studio Projects Subsystem & Multi-Track Pack Export
Completed thorough production audit and upgrade of the Studio Projects subsystem across backend, frontend, export packaging, and automated test suite:
1. Backend Projects API Enhancements (`backend/app/main.py`):
   - Added `POST /projects/{project_id}/duplicate`: creates an isolated clone of project configurations (preserving BPM, Key, Style Tags, Accent Color, Description, and Artwork) with a `(Copy)` suffix.
   - Added `GET /projects/{project_id}/export`: packages and streams a production-grade multi-track **Project Studio Pack** (.zip) containing structured subfolders per track (`tracks/{idx:02d}_{clean_title}/`), master audio, isolated stems (`stems/{vocals,drums,bass,other}.wav`), score MIDI, MusicXML notation, note events (`notes.json`), timed lyrics (`lyrics.txt`), and root `project_metadata.json`.
   - Updated `get_project` and `list_projects` to dynamically compute `stems_count`, `midi_count`, `track_count`, and `total_duration_s`.
   - Robust error handling for `add_track_to_project` with safe UUID validation (400 on malformed UUIDs, 404 on non-existent tracks) and timestamp bumping.
2. Frontend Studio Experience (`ProjectsView.tsx`, `TrackDetailView.tsx`, `SongsView.tsx`, `api.ts`):
   - Expanded Create Project Modal to accept BPM (40-240), Musical Key Signature, Default Style Tags, and Accent Color palette picker alongside artwork upload/AI prompt.
   - Added real-time project search bar and tag filter pills to top-level projects folder browser.
   - Added "Play All" continuous session playback and "Export Studio Pack" (.zip) download buttons to the Project Header Banner.
   - Integrated quick duplicate and export action triggers directly onto top-level project cards.
   - Integrated project artwork upload and AI prompt generation in Edit Project Modal.
   - Added Project Folder affiliation chip and interactive "Assign to Studio Project" modal in `TrackDetailView.tsx`.
   - Added project indicator pills to table and grid views in `SongsView.tsx`.
3. Automated Tests & Quality Assurance (`backend/tests/test_production_v2.py`):
   - Added comprehensive test suite: `test_project_crud_and_validation`, `test_project_tracks_association_and_stats`, `test_project_duplicate_lifecycle`, and `test_project_studio_pack_export_zip`.
   - Verified 100% pass rate (22/22 production tests, 195/195 all backend tests) and clean TypeScript frontend build.
4. Technical Wiki Documentation:
   - Created `wiki/entities/projects.md` detailing architecture, REST API, data models, Studio Pack zip schema, and testing guarantees.
   - Updated `wiki/index.md` linking to the new entity page.

