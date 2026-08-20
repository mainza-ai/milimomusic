---
title: Session Workspace (DAW)
type: entity
created: 2026-08-20
updated: 2026-08-20
tags: [daw, workspace, piano-roll, notation, mixer, arrange, multitrack]
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
  **Download MIDI** export, and a **Fit** control. The pitch range is **dynamic** — auto-built from
  the transcribed notes' min/max (enclosing C octaves) so **no pitch is ever clamped or dropped**.
- **Notation** — [NotationViewer](session-workspace.md): **accurate SVG grand-staff engraving** —
  real pitch-to-staff placement, ledger lines, accidentals, duration-correct note heads
  (whole/half/quarter/eighth), and measure barlines from the track's true beat grid. All notes are
  rendered (no truncation); **Export MusicXML** / Print.
- **Mix** — [MultitrackMixer](session-workspace.md): stem strips (4 Master or per-instrument)
  with volume fader, stereo pan, Solo/Mute, animated LED peak meters, **GM program badge** on
  per-instrument channels, MASTER bus at **-14.0 LUFS**, and a
  **Matchering DNS Reference Master** button (see [Matchering](matchering-mastering.md)).
- **Lyrics** — synchronized karaoke view with section-header pills, click-to-seek lines,
  Copy Text, fallback to plain lyrics.

## Multitrack playback engine
- **Active-source-only playback** — only the currently-selected stem source's `<audio>` elements
  play; stale refs from the other source are cleared on switch, so you never hear the per-instrument
  parts *and* the master group *and* the full mix at once.
- One master `<audio>` (transport clock) + one `<audio>` per stem from `stems_json`
  (`http://localhost:8000{stem path}`); when stems are loaded the master is muted and stems
  carry 100% of the mix; drift correction > 50 ms; solo/mute honored per channel.
- Transport bar: Rewind −5s, Play/Pause, Advance +5s, timecode, scrubber, master volume.
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
