---
title: Session Workspace (DAW)
type: entity
created: 2026-08-20
updated: 2026-08-21
tags: [daw, workspace, piano-roll, notation, mixer, arrange, multitrack, web-audio]
aliases: [SessionWorkspace, DAW, Web Audio DAW]
---

# Session Workspace (DAW)

The **Session Workspace** (`frontend/src/components/workspace/SessionWorkspace.tsx`) is
Milimo's in-browser **DAW**. Any completed, transcribed track can be opened here (via
Songs, Playlists, Projects, Profile, or History "Open in DAW"). It is the frontend
realization of the v2 "session grows a workspace" idea.

## Stem source selector (dual-engine)
When a track carries both stem sets, the DAW shows a **stem-source toggle** in the header:

- **4 Master Stems** — the real HTDemucs neural separation
  (`vocals` / `drums` / `bass` / `other`) of the actual master audio.
- **Per-Instrument** — the MuScriptor-derived parts, one channel per distinct instrument
  (e.g. Drums, Electric Bass), each labeled with its **General MIDI (GM) program** number in
  the Mixer.

Switching rebuilds the DAW channels and resets solo/mute state. Solo/mute always isolates
the selected channel's *own* audio. The mixer shows a `GM N` badge on per-instrument
channels (see [Stem Separation](stem-separator.md) for the dual-engine detail).

## Workspace modes (`WorkspaceMode`)
`listen` (default) → `arrange` → `pianoroll` → `notation` → `mix` → `lyrics`

- **Listen** — album art, title, badges ("48kHz Stereo …", "N MIDI Notes Transcribed"),
  and a **synchronized karaoke lyrics stream** from `timed_lyrics_json` highlighting the
  current line (see [Karaoke & Lyric Sync](karaoke-lyricsync.md)).
- **Arrange** — [ArrangeTimeline](session-workspace.md): dynamic stem tracks (per-instrument parts
  **or** the master source group, per the selector) with **Solo (S) / Mute (M)** matrix, measure
  ruler (from the track's real BPM/beat grid), zoom, click-to-seek, playhead overlay.
- **Piano Roll** — [PianoRoll](session-workspace.md): parses `notes_json` into `NoteEvent[]`,
  live Web Audio multi-harmonic synth playback, track-filter solo pills, note color-coding by
  instrument role, click/delete + pitch-edit note editing (persisted via `/workspace/{id}/notes`),
  **Download MIDI** export, and a **Fit** control. Features an authentic **144px Apple Studio Grand
  Keyboard** (ivory white keys, 3D raised matte ebony black keys, illuminated `C4 Middle C` badge),
  full-height vertical measure/beat division grid lines across the canvas, adaptive note pill
  rendering without text clipping, and compact percussion hit triggers (`🥁`) that eliminate 16th-note pileups.
- **Notation** — [NotationViewer](session-workspace.md): **accurate SVG grand-staff engraving** —
  real diatonic pitch-to-staff placement (`y = staffTop + (10 - pos) * HALF`), dynamic ledger lines,
  strict stem direction rules (up on right for lower notes, down on left for higher notes), curved
  vector flags, grand staff left brace bracket, **Chord Grouping (`ChordGlyph`)** with unified stems
  for concurrent notes, and **Measure Accidental Deduplication** (only drawing ♯/♭ on the first note
  instance per bar). All notes are rendered; **Export MusicXML** / Print.
- **Mix** — [MultitrackMixer](session-workspace.md): stem strips (4 Master or per-instrument)
  with volume fader, stereo pan, Solo/Mute, animated LED peak meters, **GM program badge** on
  per-instrument channels, MASTER bus at **-14.0 LUFS**, and a
  **Matchering DNS Reference Master** button (see [Matchering](matchering-mastering.md)).
- **Lyrics** — synchronized karaoke view with section-header pills, click-to-seek lines,
  Copy Text, fallback to plain lyrics.

## Multitrack playback engine (Web Audio transport)
Since the v2 playback refactor the DAW uses a **Web Audio multitrack transport** instead of
mixed `<audio>` elements (each `<audio>` had its own independent clock, so stems drifted and
needed glitch-prone seek-based resync; the old `<audio>` refs also leaked across stem-source
switches, causing "hear everything" bugs):
- **Sample-locked scheduling & Clock Awaiting** — the master and every stem are decoded into `AudioBuffer`s
  and scheduled against a single `AudioContext.currentTime` master clock. `playAll()` strictly awaits
  `AudioContext.resume()` before sampling hardware timestamps, eliminating clock jumping after pause/idle.
  When starting playback at track end, the engine automatically rewinds to 0:00.
- **Mix on the audio thread** — each channel routes through its own `GainNode → StereoPannerNode`
  into a master gain; volume/pan/mute/solo are applied as smooth `setTargetAtTime` ramps (no DOM
  writes, no clicks). Solo mutes all non-soloed channels; mute zeroes that channel.
- **Master-mix fallback gating** — when decoded stems are active the master-mix gain is set to 0;
  if stems fail to load or decode the clean master is heard instead of a dead-stem multitrack
  (`hasLoadedStems` drives this). Live seeking stops + reschedules all sources at the new position.
- **Clean teardown** — unmount or `switchStemSource` stops all in-flight sources, disconnects the
  per-channel graph nodes, drops the deactivated source's buffers, and (on unmount) closes the
  shared `AudioContext`; an init flag guards StrictMode's dev double-mount from decoding twice.
  Only the selected source (htDemucs 4-master or MuScriptor per-instrument) is ever audible.
- **Mix state is authoritative** — per-stem `GainNode`s are created lazily (default gain 1) inside
  `scheduleAll()`, so the full mix state (`applyMixParams`) is re-applied at the end of `scheduleAll()`
  immediately after the nodes exist. This makes solo/mute/volume/pan take effect even when changed
  *while paused before first play*.
- **Transport bar**: Return to Zero / Start (`|<<`), Step Back 1 Measure (`-1 Bar`), Rewind 10s (`RotateCcw`),
  Master Play/Pause hero button, Advance 10s (`RotateCw`), Step Forward 1 Measure (`+1 Bar`), Loop Playback
  Toggle (`Repeat`), Timecode (elapsed/duration), Dynamic BPM indicator, Full Multitrack Scrubber, and Master Volume slider.
- **Export DAW Assets** dropdown → `GET /transcribe/export/{job_id}/{format}` for
  `midi`, `musicxml`, `lrc`.

## Data inputs
Reads the v2 assets the [orchestration pipeline](../concepts/generation-pipeline.md) writes
onto `Job`: `stems_json`, `notes_json`, `timed_lyrics_json`, `midi_path`, `musicxml_path`.

## Related pages
- [Frontend](frontend.md) | [Backend & API](backend-api.md)
- [Stem separator](stem-separator.md) | [MuScriptor](muscriptor.md)
- [Matchering mastering](matchering-mastering.md) | [Karaoke & Lyric Sync](karaoke-lyricsync.md)
- [Orchestration pipeline](../concepts/generation-pipeline.md)
